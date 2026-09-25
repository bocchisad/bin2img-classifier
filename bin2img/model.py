"""CatBoost malware-family classifier wrapper.

Training / inference with:
  - temperature scaling (validation-tuned when possible)
  - Unknown reject when confidence or margin is too low
  - sidecar ``model_config.json`` for thresholds + feature names
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.model_selection import train_test_split

from bin2img.features import FeatureExtractor, FeatureVector

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("artifacts/model.cbm")
DEFAULT_LABELS_PATH = Path("artifacts/labels.json")
DEFAULT_METRICS_PATH = Path("artifacts/metrics.json")
DEFAULT_CONFIG_PATH = Path("artifacts/model_config.json")

UNKNOWN_LABEL = "Unknown"


def _sharpen_probabilities(
    proba: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Temperature-scale a probability vector (T<1 sharpens, T>1 softens)."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    out = np.asarray(proba, dtype=np.float64)
    if abs(temperature - 1.0) < 1e-9:
        return out / out.sum()
    logits = np.log(np.clip(out, 1e-12, 1.0)) / temperature
    logits -= np.max(logits)
    exp = np.exp(logits)
    return exp / exp.sum()


def _fit_temperature(
    raw_proba: np.ndarray,
    y_true: Sequence[str],
    class_names: Sequence[str],
) -> float:
    """Grid-search temperature minimizing NLL on a holdout."""
    label_to_idx = {c: i for i, c in enumerate(class_names)}
    y_idx = np.asarray([label_to_idx[y] for y in y_true], dtype=np.int64)
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.3, 2.5, 23):
        calibrated = np.vstack(
            [_sharpen_probabilities(row, float(t)) for row in raw_proba]
        )
        nll = float(log_loss(y_idx, calibrated, labels=list(range(len(class_names)))))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    # Avoid extreme sharpening — overconfident OOD predictions break Unknown reject.
    return float(np.clip(best_t, 0.75, 1.8))


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Single-sample classification output."""

    label: str
    confidence: float
    probabilities: dict[str, float]
    is_unknown: bool = False
    margin: float = 0.0
    raw_label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "probabilities": dict(self.probabilities),
            "is_unknown": self.is_unknown,
            "margin": self.margin,
            "raw_label": self.raw_label or self.label,
        }


@dataclass
class TrainResult:
    """Artifacts produced by a training run."""

    model_path: Path
    labels_path: Path
    metrics: dict[str, Any] = field(default_factory=dict)
    class_names: list[str] = field(default_factory=list)
    config_path: Path | None = None


class MalwareFamilyClassifier:
    """CatBoost wrapper with calibration + Unknown reject."""

    def __init__(
        self,
        *,
        feature_extractor: FeatureExtractor | None = None,
        iterations: int = 500,
        depth: int = 7,
        learning_rate: float = 0.07,
        l2_leaf_reg: float = 1.5,
        random_seed: int = 42,
        verbose: bool = False,
        probability_temperature: float = 1.0,
        confidence_threshold: float = 0.55,
        margin_threshold: float = 0.12,
    ) -> None:
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed
        self.verbose = verbose
        self.probability_temperature = probability_temperature
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self._model: Any | None = None
        self.class_names: list[str] = []

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.feature_extractor.feature_names

    @property
    def is_fitted(self) -> bool:
        return self._model is not None and bool(self.class_names)

    def train(
        self,
        X: np.ndarray,
        y: Sequence[str],
        *,
        test_size: float = 0.25,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        labels_path: str | Path = DEFAULT_LABELS_PATH,
        metrics_path: str | Path | None = DEFAULT_METRICS_PATH,
        config_path: str | Path | None = DEFAULT_CONFIG_PATH,
        calibrate_temperature: bool = True,
    ) -> TrainResult:
        """Fit CatBoost, optionally calibrate temperature, persist artifacts."""
        from catboost import CatBoostClassifier, Pool

        X = np.asarray(X, dtype=np.float64)
        labels = list(y)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if X.shape[0] != len(labels):
            raise ValueError("X and y length mismatch")
        if X.shape[0] < 4:
            raise ValueError("Need at least 4 samples to train")

        class_names = sorted(set(labels))
        if len(class_names) < 2:
            raise ValueError("Need at least 2 distinct classes")

        counts = Counter(labels)
        stratify: Sequence[str] | None = labels if min(counts.values()) >= 2 else None
        if stratify is None:
            logger.warning(
                "Unstratified train/test split: some classes have <2 samples (%s)",
                dict(counts),
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            labels,
            test_size=test_size,
            random_state=self.random_seed,
            stratify=stratify,
        )

        model = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            random_seed=self.random_seed,
            verbose=self.verbose,
            allow_writing_files=False,
            bootstrap_type="Bernoulli",
            subsample=0.85,
            random_strength=0.3,
        )
        train_pool = Pool(X_train, y_train, feature_names=list(self.feature_names))
        eval_pool = Pool(X_test, y_test, feature_names=list(self.feature_names))
        model.fit(
            train_pool,
            eval_set=eval_pool,
            use_best_model=True,
            early_stopping_rounds=80,
        )

        self._model = model
        self.class_names = [str(c) for c in model.classes_]

        raw_proba = np.asarray(model.predict_proba(X_test))
        if calibrate_temperature and len(y_test) >= 4:
            self.probability_temperature = _fit_temperature(
                raw_proba, y_test, self.class_names
            )
        else:
            self.probability_temperature = max(self.probability_temperature, 0.3)

        y_pred_labels = []
        unknown_count = 0
        for row in raw_proba:
            pred = self._proba_to_prediction(row)
            y_pred_labels.append(pred.raw_label)
            if pred.is_unknown:
                unknown_count += 1

        metrics: dict[str, Any] = {
            "accuracy": float(accuracy_score(y_test, y_pred_labels)),
            "f1_macro": float(
                f1_score(y_test, y_pred_labels, average="macro", zero_division=0)
            ),
            "report": classification_report(
                y_test, y_pred_labels, output_dict=True, zero_division=0
            ),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "iterations": self.iterations,
            "feature_names": list(self.feature_names),
            "probability_temperature": self.probability_temperature,
            "confidence_threshold": self.confidence_threshold,
            "margin_threshold": self.margin_threshold,
            "unknown_rate_on_test": float(unknown_count / max(1, len(y_test))),
            "tree_count": int(getattr(model, "tree_count_", self.iterations)),
        }
        logger.info(
            "Train done: accuracy=%.3f f1_macro=%.3f T=%.3f unknown_rate=%.3f",
            metrics["accuracy"],
            metrics["f1_macro"],
            self.probability_temperature,
            metrics["unknown_rate_on_test"],
        )

        model_path = Path(model_path)
        labels_path = Path(labels_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_path))
        labels_path.write_text(json.dumps(self.class_names, indent=2), encoding="utf-8")

        cfg_path: Path | None = None
        if config_path is not None:
            cfg_path = Path(config_path)
            self._save_config(cfg_path)

        if metrics_path is not None:
            metrics_path = Path(metrics_path)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        return TrainResult(
            model_path=model_path,
            labels_path=labels_path,
            metrics=metrics,
            class_names=list(self.class_names),
            config_path=cfg_path,
        )

    def train_from_bytes(
        self,
        samples: Sequence[tuple[bytes, str]],
        **kwargs: Any,
    ) -> TrainResult:
        """Extract features from ``(raw_bytes, label)`` pairs and train."""
        if not samples:
            raise ValueError("samples is empty")
        X_list: list[np.ndarray] = []
        y_list: list[str] = []
        for raw, label in samples:
            vec = self.feature_extractor.extract(raw)
            X_list.append(vec.as_row())
            y_list.append(label)
        X = np.vstack(X_list)
        return self.train(X, y_list, **kwargs)

    def load(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        """Load a previously saved ``.cbm`` model (+ optional config sidecar)."""
        from catboost import CatBoostClassifier

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}")
        model = CatBoostClassifier()
        model.load_model(str(path))
        self._model = model
        classes = getattr(model, "classes_", None)
        self.class_names = [str(c) for c in classes] if classes is not None else []

        labels_sidecar = path.with_name("labels.json")
        if labels_sidecar.is_file():
            loaded = json.loads(labels_sidecar.read_text(encoding="utf-8"))
            if isinstance(loaded, list) and loaded:
                sidecar = [str(c) for c in loaded]
                if not self.class_names:
                    self.class_names = sidecar
                elif set(sidecar) != set(self.class_names):
                    logger.warning(
                        "labels.json %s differs from model.classes_ %s; preferring model",
                        sidecar,
                        self.class_names,
                    )

        config_sidecar = path.with_name("model_config.json")
        if config_sidecar.is_file():
            self._load_config(config_sidecar)

    def predict_features(self, features: np.ndarray | FeatureVector) -> PredictionResult:
        """Classify a single feature row or :class:`FeatureVector`."""
        self._ensure_fitted()
        assert self._model is not None

        if isinstance(features, FeatureVector):
            row = features.as_row().reshape(1, -1)
        else:
            arr = np.asarray(features, dtype=np.float64)
            row = arr.reshape(1, -1) if arr.ndim == 1 else arr

        # Feature-length guard (stale model vs new extractor).
        expected = len(self.feature_names)
        if row.shape[1] != expected:
            raise ValueError(
                f"Feature length {row.shape[1]} != model expectation {expected}. "
                "Retrain the model after feature changes."
            )

        proba = np.asarray(self._model.predict_proba(row))[0]
        return self._proba_to_prediction(proba)

    def predict(
        self,
        source: str | Path | bytes | bytearray | memoryview,
    ) -> PredictionResult:
        """Extract features from a binary and classify."""
        vec = self.feature_extractor.extract(source)
        return self.predict_features(vec)

    def predict_batch_features(self, X: np.ndarray) -> list[PredictionResult]:
        """Classify a design matrix row-wise."""
        self._ensure_fitted()
        assert self._model is not None
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        expected = len(self.feature_names)
        if X.shape[1] != expected:
            raise ValueError(
                f"Feature length {X.shape[1]} != model expectation {expected}. "
                "Retrain the model after feature changes."
            )
        proba = np.asarray(self._model.predict_proba(X))
        return [self._proba_to_prediction(row) for row in proba]

    def _proba_to_prediction(self, proba: np.ndarray) -> PredictionResult:
        assert self._model is not None
        calibrated = _sharpen_probabilities(proba, self.probability_temperature)
        classes = [str(c) for c in self._model.classes_]
        order = np.argsort(-calibrated)
        best_idx = int(order[0])
        second_idx = int(order[1]) if len(order) > 1 else best_idx
        confidence = float(calibrated[best_idx])
        margin = float(calibrated[best_idx] - calibrated[second_idx])
        raw_label = classes[best_idx]
        probs = {classes[i]: float(calibrated[i]) for i in range(len(classes))}

        is_unknown = (
            confidence < self.confidence_threshold or margin < self.margin_threshold
        )
        label = UNKNOWN_LABEL if is_unknown else raw_label
        return PredictionResult(
            label=label,
            confidence=confidence,
            probabilities=probs,
            is_unknown=is_unknown,
            margin=margin,
            raw_label=raw_label,
        )

    def _save_config(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "probability_temperature": self.probability_temperature,
            "confidence_threshold": self.confidence_threshold,
            "margin_threshold": self.margin_threshold,
            "feature_names": list(self.feature_names),
            "class_names": list(self.class_names),
            "unknown_label": UNKNOWN_LABEL,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_config(self, path: Path) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load model config %s: %s", path, exc)
            return
        self.probability_temperature = float(
            payload.get("probability_temperature", self.probability_temperature)
        )
        self.confidence_threshold = float(
            payload.get("confidence_threshold", self.confidence_threshold)
        )
        self.margin_threshold = float(
            payload.get("margin_threshold", self.margin_threshold)
        )

    def _ensure_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "Model is not loaded/fitted. Call train() or load() first."
            )


def build_xy_from_bytes(
    samples: Sequence[tuple[bytes, str]],
    extractor: FeatureExtractor | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Convert ``(bytes, label)`` samples into ``(X, y)``."""
    ext = extractor or FeatureExtractor()
    rows = [ext.extract(raw).as_row() for raw, _ in samples]
    y = [label for _, label in samples]
    return np.vstack(rows), y
