"""Unit tests for Stage 3 — GLCM / LBP / unified feature vectors."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from bin2img.features import (
    FEATURE_NAMES,
    FeatureExtractor,
    extract_glcm_features,
    extract_lbp_histogram,
    feature_matrix,
)
from bin2img.parser import BinaryParser
from bin2img.visualizer import bytes_to_matrix
from tests.test_parser import _craft_pe_with_pefile


class TestGLCM:
    def test_length_and_finite(self) -> None:
        img = np.arange(64, dtype=np.uint8).reshape(8, 8)
        feats = extract_glcm_features(img)
        assert feats.shape == (5,)
        assert np.all(np.isfinite(feats))

    def test_empty_returns_zeros(self) -> None:
        feats = extract_glcm_features(np.zeros((0, 32), dtype=np.uint8))
        assert feats.shape == (5,)
        assert np.allclose(feats, 0.0)

    def test_uniform_vs_noise_contrast(self) -> None:
        flat = np.full((32, 32), 128, dtype=np.uint8)
        noise = np.random.default_rng(0).integers(0, 256, size=(32, 32), dtype=np.uint8)
        c_flat = extract_glcm_features(flat)[0]  # contrast
        c_noise = extract_glcm_features(noise)[0]
        assert c_noise > c_flat


class TestLBP:
    def test_histogram_normalized(self) -> None:
        img = np.arange(256, dtype=np.uint8).reshape(16, 16)
        hist = extract_lbp_histogram(img)
        assert hist.shape == (10,)
        assert hist.sum() == pytest.approx(1.0, abs=1e-9)
        assert np.all(hist >= 0.0)

    def test_tiny_image_zeros(self) -> None:
        hist = extract_lbp_histogram(np.zeros((2, 2), dtype=np.uint8))
        assert hist.shape == (10,)
        assert np.allclose(hist, 0.0)

    def test_structured_vs_noise_differs(self) -> None:
        structured = np.tile(
            np.arange(32, dtype=np.uint8),
            (32, 1),
        )
        noise = np.random.default_rng(1).integers(0, 256, size=(32, 32), dtype=np.uint8)
        h1 = extract_lbp_histogram(structured)
        h2 = extract_lbp_histogram(noise)
        # Histograms should not be identical for structured vs noise.
        assert not np.allclose(h1, h2, atol=1e-3)


class TestFeatureExtractor:
    def test_vector_length_matches_names(self) -> None:
        ext = FeatureExtractor()
        vec = ext.extract(b"\x90" * 2048)
        assert vec.values.shape == (len(ext.feature_names),)
        assert list(vec.names) == list(ext.feature_names)
        assert len(FEATURE_NAMES) == len(ext.feature_names)

    def test_to_dict_keys(self) -> None:
        vec = FeatureExtractor().extract(os.urandom(1024))
        d = vec.to_dict()
        assert "overall_entropy" in d
        assert "glcm_contrast" in d
        assert "lbp_0" in d
        assert "printable_ratio" in d
        assert "has_mz_header" in d
        assert "overlay_ratio" in d
        assert "suspicious_api_hits" in d
        assert "string_density" in d
        assert d["high_entropy_alert"] in (0.0, 1.0)

    def test_high_entropy_flag_on_noise(self) -> None:
        vec = FeatureExtractor().extract(os.urandom(4096))
        assert vec.to_dict()["high_entropy_alert"] == 1.0
        assert vec.to_dict()["overall_entropy"] > 7.0

    def test_low_entropy_zeros(self) -> None:
        vec = FeatureExtractor().extract(b"\x00" * 2048)
        assert vec.to_dict()["high_entropy_alert"] == 0.0
        assert vec.to_dict()["overall_entropy"] == pytest.approx(0.0, abs=1e-9)

    def test_pe_sample_executable_flag(self) -> None:
        from tests.test_parser import _craft_pe_with_pefile

        pe = _craft_pe_with_pefile()
        if pe is None:
            pytest.skip("PE craft unavailable")
        vec = FeatureExtractor().extract(pe)
        d = vec.to_dict()
        assert d["is_valid_executable"] == 1.0
        assert d["section_count"] >= 1.0
        assert d["executable_section_ratio"] > 0.0

    def test_extract_from_parts(self) -> None:
        raw = bytes(range(256)) * 8
        meta = BinaryParser().parse(raw)
        gray = bytes_to_matrix(raw, 32)
        vec = FeatureExtractor().extract_from_parts(meta, gray)
        assert vec.values.shape[0] == len(FEATURE_NAMES)

    def test_extract_batch_and_feature_matrix(self) -> None:
        samples = [b"\x00" * 512, os.urandom(512), b"ABCD" * 128]
        X, names = feature_matrix(samples)
        assert X.shape == (3, len(names))
        assert names[0] == "mean_section_entropy"

    def test_from_file_path(self, tmp_path: Path) -> None:
        path = tmp_path / "blob.bin"
        path.write_bytes(os.urandom(800))
        vec = FeatureExtractor().extract(path)
        assert vec.metadata["file_size"] == 800

    def test_extract_from_stream_not_double_consumed(self) -> None:
        from io import BytesIO

        pe = _craft_pe_with_pefile()
        if pe is None:
            pytest.skip("PE craft unavailable")
        from_bytes = FeatureExtractor().extract(pe).values
        from_stream = FeatureExtractor().extract(BytesIO(pe)).values
        assert from_bytes.shape == from_stream.shape
        assert np.allclose(from_bytes, from_stream)

    def test_all_finite(self) -> None:
        vec = FeatureExtractor().extract(os.urandom(3000))
        assert np.all(np.isfinite(vec.values))
