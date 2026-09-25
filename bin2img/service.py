"""Core analysis orchestration used by API / CLI / jobs."""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path
from typing import Any

from bin2img.constants import MAX_UPLOAD_BYTES, SUSPICIOUS_SECTION_HINTS
from bin2img.explain import explain_prediction
from bin2img.features import FeatureExtractor
from bin2img.model import DEFAULT_MODEL_PATH, MalwareFamilyClassifier, PredictionResult
from bin2img.parser import BinaryMetadata, BinaryParser, HIGH_ENTROPY_THRESHOLD
from bin2img.risk import assess_risk
from bin2img.rules import evaluate_rules
from bin2img.store import AnalysisStore
from bin2img.visualizer import BinaryVisualizer

logger = logging.getLogger(__name__)


def build_risk_tags(metadata: BinaryMetadata) -> list[str]:
    """Derive human-readable risk assessment tags from parse metadata."""
    tags: list[str] = []
    if metadata.is_valid_executable:
        tags.append("VALID_EXECUTABLE")
    if metadata.high_entropy_alert or metadata.overall_entropy > HIGH_ENTROPY_THRESHOLD:
        tags.append("HIGH_ENTROPY_PACKED")
    elif any(s.entropy > HIGH_ENTROPY_THRESHOLD for s in metadata.sections):
        tags.append("HIGH_ENTROPY_PACKED")

    suspicious = False
    for section in metadata.sections:
        name_u = section.name.upper()
        if any(hint in name_u for hint in SUSPICIOUS_SECTION_HINTS):
            suspicious = True
            break
        if section.raw_size > 0 and section.virtual_size > section.raw_size * 4:
            suspicious = True
            break
        if section.raw_size == 0 and section.virtual_size > 0 and section.is_executable:
            suspicious = True
            break
    if suspicious:
        tags.append("SUSPICIOUS_SECTION")
    return tags


def sha256_hex(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class AnalysisService:
    """Shared parser / visualizer / classifier / store used by API routes."""

    def __init__(
        self,
        model_path: Path | None = None,
        *,
        db_path: str | Path = "artifacts/bin2img.db",
    ) -> None:
        self.parser = BinaryParser()
        self.visualizer = BinaryVisualizer()
        self.features = FeatureExtractor(parser=self.parser, visualizer=self.visualizer)
        self.classifier = MalwareFamilyClassifier(feature_extractor=self.features)
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.model_loaded = False
        self.store = AnalysisStore(db_path)
        self._predict_lock = threading.Lock()

    def load_model(self) -> bool:
        try:
            if self.model_path.is_file():
                self.classifier.load(self.model_path)
                self.model_loaded = True
                logger.info("Loaded model from %s", self.model_path)
                return True
            logger.warning("Model not found at %s — classification disabled", self.model_path)
            self.model_loaded = False
            return False
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load model: %s", exc)
            self.model_loaded = False
            return False

    def analyze_bytes(self, raw: bytes, filename: str) -> dict[str, Any]:
        """Full analysis pipeline for an uploaded binary."""
        if not raw:
            raise ValueError("Empty file upload")
        if len(raw) > MAX_UPLOAD_BYTES:
            raise ValueError(
                f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit"
            )

        metadata = self.parser.parse(raw)
        metadata.file_path = filename or metadata.file_path
        render = self.visualizer.render(raw)
        feature_vec = self.features.extract_from_parts(
            metadata, render.grayscale, raw=raw
        )
        feat_dict = feature_vec.to_dict()

        classification: dict[str, Any] | None = None
        pred: PredictionResult | None = None
        explanation: dict[str, Any] = {"method": "none", "top_features": []}
        risk_tags = build_risk_tags(metadata)

        if self.model_loaded and self.classifier.is_fitted:
            with self._predict_lock:
                try:
                    pred = self.classifier.predict_features(feature_vec)
                    classification = pred.to_dict()
                    if pred.is_unknown:
                        risk_tags.append("LOW_CONFIDENCE_UNKNOWN")
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Classification failed: %s", exc)
                    classification = {"error": str(exc)}
                    pred = None
                try:
                    if pred is not None:
                        explanation = explain_prediction(self.classifier, feature_vec)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Explain failed: %s", exc)
                    explanation = {"method": "error", "top_features": [], "error": str(exc)}

        rule_hits = evaluate_rules(metadata, raw, features=feat_dict)
        risk = assess_risk(
            classification=pred,
            rule_hits=rule_hits,
            risk_tags=risk_tags,
        )
        risk_tags.append(f"VERDICT_{risk.verdict.upper()}")

        payload: dict[str, Any] = {
            "filename": filename,
            "sha256": sha256_hex(raw),
            "metadata": metadata.to_dict(),
            "classification": classification,
            "risk_tags": risk_tags,
            "risk": risk.to_dict(),
            "explanation": explanation,
            "images": {
                "grayscale_png_base64": render.grayscale_base64(),
                "heatmap_png_base64": render.heatmap_base64(),
            },
            "features": feat_dict,
        }
        analysis_id = self.store.save_analysis(
            filename=filename,
            sha256=payload["sha256"],
            payload=payload,
        )
        payload["analysis_id"] = analysis_id
        return payload
