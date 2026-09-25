"""Unit tests for Stage 4 — synthetic data + CatBoost classifier."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bin2img.model import UNKNOWN_LABEL, MalwareFamilyClassifier, build_xy_from_bytes
from bin2img.parser import calculate_entropy
from scripts.generate_synthetic_data import (
    CLASS_LABELS,
    generate_dataset,
    generate_sample,
    iter_dataset_bytes,
)


class TestSyntheticGenerators:
    def test_labels_complete(self) -> None:
        assert len(CLASS_LABELS) == 4

    def test_ransomware_high_entropy(self) -> None:
        blob = generate_sample("Ransomware_Encrypted", seed=0, size=4096)
        from bin2img.parser import BinaryParser

        meta = BinaryParser().parse(blob)
        assert meta.is_valid_executable
        assert meta.high_entropy_alert
        assert meta.max_section_entropy > 7.5

    def test_benign_lower_entropy_than_ransomware(self) -> None:
        benign = generate_sample("Benign_Code", seed=1, size=4096)
        ransom = generate_sample("Ransomware_Encrypted", seed=1, size=4096)
        assert calculate_entropy(benign) < calculate_entropy(ransom)

    def test_downloader_mostly_printable(self) -> None:
        blob = generate_sample("Downloader_TextHeavy", seed=2, size=8192)
        from bin2img.features import FeatureExtractor

        feats = FeatureExtractor().extract(blob).to_dict()
        assert feats["printable_ratio"] > 0.45
        assert feats["is_valid_executable"] == 1.0

    def test_packed_has_mz_prefix(self) -> None:
        blob = generate_sample("Packed_PE", seed=3, size=4096)
        assert blob[:2] == b"MZ"
        from bin2img.parser import BinaryParser

        meta = BinaryParser().parse(blob)
        assert meta.is_valid_executable is True
        assert any("UPX" in s.name.upper() for s in meta.sections)

    def test_benign_is_valid_pe_low_entropy(self) -> None:
        blob = generate_sample("Benign_Code", seed=4, size=4096)
        from bin2img.parser import BinaryParser, calculate_entropy

        meta = BinaryParser().parse(blob)
        assert meta.format.value == "PE"
        assert meta.is_valid_executable is True
        assert calculate_entropy(blob) < 6.0

    def test_write_dataset(self, tmp_path: Path) -> None:
        manifest = generate_dataset(per_class=2, output_dir=tmp_path, seed=0)
        assert len(manifest) == 8
        assert (tmp_path / "manifest.json").is_file()
        assert (tmp_path / "Benign_Code").is_dir()
        assert any(tmp_path.joinpath("Benign_Code").glob("*.bin"))

    def test_iter_dataset_bytes_counts(self) -> None:
        samples = iter_dataset_bytes(per_class=3, seed=0)
        assert len(samples) == 12
        labels = {lab for _, lab in samples}
        assert labels == set(CLASS_LABELS)


class TestMalwareFamilyClassifier:
    @pytest.fixture(scope="module")
    def trained(self, tmp_path_factory: pytest.TempPathFactory) -> MalwareFamilyClassifier:
        out = tmp_path_factory.mktemp("model")
        samples = iter_dataset_bytes(per_class=25, seed=7)
        clf = MalwareFamilyClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            l2_leaf_reg=1.5,
            random_seed=7,
            verbose=False,
            probability_temperature=0.55,
        )
        clf.train_from_bytes(
            samples,
            model_path=out / "model.cbm",
            labels_path=out / "labels.json",
            metrics_path=out / "metrics.json",
            config_path=out / "model_config.json",
            test_size=0.3,
        )
        clf._tmp_out = out  # type: ignore[attr-defined]
        return clf

    def test_train_metrics_reasonable(self, trained: MalwareFamilyClassifier) -> None:
        out: Path = trained._tmp_out  # type: ignore[attr-defined]
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        # Synthetic families are separable — expect strong accuracy.
        assert metrics["accuracy"] >= 0.7
        assert metrics["f1_macro"] >= 0.7
        assert (out / "model.cbm").is_file()

    def test_predict_known_family(self, trained: MalwareFamilyClassifier) -> None:
        ransom = generate_sample("Ransomware_Encrypted", seed=99, size=5000)
        pred = trained.predict(ransom)
        assert pred.label == "Ransomware_Encrypted"
        assert pred.confidence >= 0.55
        assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-5

    def test_predict_benign(self, trained: MalwareFamilyClassifier) -> None:
        benign = generate_sample("Benign_Code", seed=100, size=5000)
        pred = trained.predict(benign)
        assert pred.label == "Benign_Code"
        assert pred.confidence >= 0.55

    def test_predict_crafted_sample_exe_like(self, trained: MalwareFamilyClassifier) -> None:
        from tests.test_parser import _craft_pe_with_pefile

        pe = _craft_pe_with_pefile()
        if pe is None:
            pytest.skip("PE craft unavailable")
        pred = trained.predict(pe)
        assert pred.label == "Benign_Code"
        assert pred.confidence >= 0.45

    def test_load_and_predict(self, trained: MalwareFamilyClassifier) -> None:
        out: Path = trained._tmp_out  # type: ignore[attr-defined]
        fresh = MalwareFamilyClassifier()
        fresh.load(out / "model.cbm")
        blob = generate_sample("Downloader_TextHeavy", seed=11, size=4096)
        pred = fresh.predict(blob)
        assert pred.label in CLASS_LABELS

    def test_build_xy(self) -> None:
        samples = iter_dataset_bytes(per_class=2, seed=0)
        X, y = build_xy_from_bytes(samples)
        assert X.shape[0] == 8
        assert X.shape[1] == len(MalwareFamilyClassifier().feature_names)
        assert len(y) == 8
        assert np.all(np.isfinite(X))

    def test_predict_before_fit_raises(self) -> None:
        clf = MalwareFamilyClassifier()
        with pytest.raises(RuntimeError):
            clf.predict(b"\x00" * 100)

    def test_unknown_reject_with_strict_thresholds(
        self, trained: MalwareFamilyClassifier
    ) -> None:
        # Force Unknown on a copy of thresholds — do not mutate shared fixture.
        out: Path = trained._tmp_out  # type: ignore[attr-defined]
        strict = MalwareFamilyClassifier(
            confidence_threshold=0.999,
            margin_threshold=0.99,
        )
        strict.load(out / "model.cbm")
        strict.confidence_threshold = 0.999
        strict.margin_threshold = 0.99
        pred = strict.predict(generate_sample("Benign_Code", seed=1, size=4096))
        assert pred.is_unknown is True
        assert pred.label == UNKNOWN_LABEL
        assert pred.raw_label in CLASS_LABELS

    def test_config_sidecar_roundtrip(self, trained: MalwareFamilyClassifier) -> None:
        out: Path = trained._tmp_out  # type: ignore[attr-defined]
        assert (out / "model_config.json").is_file()
        cfg = json.loads((out / "model_config.json").read_text(encoding="utf-8"))
        fresh = MalwareFamilyClassifier(
            confidence_threshold=0.1,
            margin_threshold=0.0,
        )
        fresh.load(out / "model.cbm")
        assert fresh.confidence_threshold == pytest.approx(cfg["confidence_threshold"])
