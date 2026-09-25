"""Tests for Stage 5 — FastAPI analyze endpoint & risk tags."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bin2img.api import AnalysisService, build_risk_tags, create_app
from bin2img.parser import BinaryMetadata, BinaryFormat, SectionInfo
from scripts.generate_synthetic_data import generate_sample


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    import os

    db = tmp_path_factory.mktemp("api") / "test.db"
    os.environ["BIN2IMG_DB"] = str(db)
    os.environ.pop("BIN2IMG_API_KEY", None)
    model = Path("artifacts/model.cbm")
    app = create_app(model_path=model if model.is_file() else Path("nonexistent.cbm"))
    with TestClient(app) as c:
        yield c


class TestRiskTags:
    def test_valid_and_high_entropy(self) -> None:
        meta = BinaryMetadata(
            file_path="x",
            file_size=100,
            format=BinaryFormat.PE,
            architecture="x86",
            entry_point=0x1000,
            sections=[
                SectionInfo(".text", 100, 100, 7.8, is_executable=True),
            ],
            overall_entropy=7.8,
            high_entropy_alert=True,
            is_valid_executable=True,
        )
        tags = build_risk_tags(meta)
        assert "VALID_EXECUTABLE" in tags
        assert "HIGH_ENTROPY_PACKED" in tags

    def test_suspicious_upx_name(self) -> None:
        meta = BinaryMetadata(
            file_path="x",
            file_size=100,
            format=BinaryFormat.PE,
            architecture="x86",
            entry_point=None,
            sections=[SectionInfo("UPX0", 1000, 10, 3.0, is_executable=True)],
            overall_entropy=3.0,
            high_entropy_alert=False,
            is_valid_executable=True,
        )
        assert "SUSPICIOUS_SECTION" in build_risk_tags(meta)


class TestAPI:
    def test_health(self, client: TestClient) -> None:
        res = client.get("/api/v1/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "ok"

    def test_index_html(self, client: TestClient) -> None:
        res = client.get("/")
        assert res.status_code == 200
        assert "bin2img" in res.text
        assert "/api/v1/analyze" in res.text

    def test_analyze_ransomware(self, client: TestClient) -> None:
        blob = generate_sample("Ransomware_Encrypted", seed=5, size=4096)
        res = client.post(
            "/api/v1/analyze",
            files={"file": ("ransom.bin", blob, "application/octet-stream")},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["filename"] == "ransom.bin"
        assert "grayscale_png_base64" in data["images"]
        assert "heatmap_png_base64" in data["images"]
        assert data["images"]["grayscale_png_base64"]
        assert "HIGH_ENTROPY_PACKED" in data["risk_tags"]
        assert "overall_entropy" in data["metadata"]
        assert "glcm_contrast" in data["features"]
        assert data.get("risk") is not None
        assert "score" in data["risk"]
        assert data.get("analysis_id")
        # Classification present only if model loaded.
        if data["classification"] and "label" in data["classification"]:
            label = data["classification"]["label"]
            raw = data["classification"].get("raw_label") or label
            assert raw in {
                "Benign_Code",
                "Ransomware_Encrypted",
                "Packed_PE",
                "Downloader_TextHeavy",
            }
            assert label in {
                "Benign_Code",
                "Ransomware_Encrypted",
                "Packed_PE",
                "Downloader_TextHeavy",
                "Unknown",
            }

    def test_analyze_empty_rejected(self, client: TestClient) -> None:
        res = client.post(
            "/api/v1/analyze",
            files={"file": ("empty.bin", b"", "application/octet-stream")},
        )
        assert res.status_code == 400

    def test_service_analyze_bytes_direct(self, tmp_path: Path) -> None:
        svc = AnalysisService(
            model_path=Path("artifacts/model.cbm"),
            db_path=tmp_path / "svc.db",
        )
        svc.load_model()
        blob = generate_sample("Benign_Code", seed=1, size=2048)
        out = svc.analyze_bytes(blob, "benign.bin")
        assert out["filename"] == "benign.bin"
        assert out["metadata"]["file_size"] == 2048
        assert out["risk"]["verdict"] in {"Benign", "Suspicious", "Malicious", "Unknown"}
