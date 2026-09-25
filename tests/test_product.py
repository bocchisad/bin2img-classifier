"""Tests for product layers: rules, risk, store, jobs, auth, reports."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bin2img.api import create_app
from bin2img.jobs import JobRunner
from bin2img.model import PredictionResult
from bin2img.parser import BinaryFormat, BinaryMetadata, SectionInfo
from bin2img.report import render_html_report, write_html_report, write_json_report
from bin2img.risk import assess_risk
from bin2img.rules import evaluate_rules
from bin2img.service import AnalysisService
from bin2img.store import AnalysisStore
from scripts.generate_synthetic_data import generate_sample


def _meta(**kwargs) -> BinaryMetadata:
    defaults = dict(
        file_path="x",
        file_size=100,
        format=BinaryFormat.PE,
        architecture="x86",
        entry_point=0x1000,
        sections=[SectionInfo(".text", 100, 100, 5.0, is_executable=True)],
        overall_entropy=5.0,
        high_entropy_alert=False,
        is_valid_executable=True,
    )
    defaults.update(kwargs)
    return BinaryMetadata(**defaults)


class TestRules:
    def test_high_entropy_and_upx(self) -> None:
        meta = _meta(
            overall_entropy=7.8,
            high_entropy_alert=True,
            sections=[SectionInfo("UPX0", 1000, 10, 7.5, is_executable=True)],
        )
        hits = evaluate_rules(meta, b"MZ" + b"\x00" * 50)
        ids = {h.rule_id for h in hits}
        assert "high_file_entropy" in ids
        assert "suspicious_section_name" in ids

    def test_string_match(self) -> None:
        meta = _meta()
        hits = evaluate_rules(meta, b"xxx CreateRemoteThread yyy")
        assert any(h.rule_id == "api_crt" for h in hits)


class TestRisk:
    def test_benign_low_score(self) -> None:
        pred = PredictionResult(
            label="Benign_Code",
            confidence=0.95,
            probabilities={"Benign_Code": 0.95},
            is_unknown=False,
            raw_label="Benign_Code",
            margin=0.9,
        )
        risk = assess_risk(classification=pred, rule_hits=[], risk_tags=["VALID_EXECUTABLE"])
        assert risk.verdict == "Benign"
        assert risk.score < 35

    def test_ransomware_high_score(self) -> None:
        pred = PredictionResult(
            label="Ransomware_Encrypted",
            confidence=0.9,
            probabilities={"Ransomware_Encrypted": 0.9},
            is_unknown=False,
            raw_label="Ransomware_Encrypted",
            margin=0.8,
        )
        from bin2img.rules import RuleHit

        hits = [RuleHit("high_file_entropy", "high", 25, "entropy")]
        risk = assess_risk(
            classification=pred,
            rule_hits=hits,
            risk_tags=["HIGH_ENTROPY_PACKED"],
        )
        assert risk.score >= 70
        assert risk.verdict == "Malicious"


class TestStore:
    def test_save_and_list(self, tmp_path: Path) -> None:
        store = AnalysisStore(tmp_path / "t.db")
        aid = store.save_analysis(
            filename="a.bin",
            sha256="abc",
            payload={
                "filename": "a.bin",
                "risk": {"verdict": "Suspicious", "score": 40},
                "classification": {"label": "Packed_PE"},
            },
        )
        row = store.get_analysis(aid)
        assert row is not None
        assert row["filename"] == "a.bin"
        assert row["analysis_id"] == aid
        listed = store.list_analyses(limit=10)
        assert any(r["id"] == aid for r in listed)

    def test_jobs(self, tmp_path: Path) -> None:
        store = AnalysisStore(tmp_path / "j.db")
        jid = store.create_job("x.bin")
        store.update_job(jid, status="done", result_id="rid")
        job = store.get_job(jid)
        assert job is not None
        assert job["status"] == "done"
        assert job["result_id"] == "rid"


class TestJobs:
    def test_async_runner(self, tmp_path: Path) -> None:
        store = AnalysisStore(tmp_path / "jobs.db")

        def analyze(raw: bytes, filename: str) -> dict:
            return {
                "analysis_id": "aid1",
                "filename": filename,
                "size": len(raw),
            }

        runner = JobRunner(store, analyze, max_workers=1)
        jid = runner.submit(b"hello", "h.bin")
        for _ in range(50):
            job = store.get_job(jid)
            if job and job["status"] in {"done", "error"}:
                break
            time.sleep(0.05)
        runner.shutdown()
        job = store.get_job(jid)
        assert job is not None
        assert job["status"] == "done"
        assert job["result_id"] == "aid1"


class TestReport:
    def test_html_and_json(self, tmp_path: Path) -> None:
        payload = {
            "filename": "t.bin",
            "analysis_id": "abc",
            "metadata": {"format": "PE", "sections": []},
            "risk": {"verdict": "Suspicious", "score": 42, "rule_hits": []},
            "classification": {"label": "Packed_PE", "confidence": 0.8},
            "risk_tags": ["VALID_EXECUTABLE"],
            "explanation": {"top_features": [{"feature": "x", "contribution": 1.2}]},
            "images": {},
        }
        html = render_html_report(payload)
        assert "Suspicious" in html
        assert "Packed_PE" in html
        p = write_html_report(payload, tmp_path / "r.html")
        assert p.is_file()
        j = write_json_report(payload, tmp_path / "r.json")
        assert j.is_file()


class TestProductAPI:
    @pytest.fixture()
    def client(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
        monkeypatch.setenv("BIN2IMG_DB", str(tmp_path / "api.db"))
        monkeypatch.delenv("BIN2IMG_API_KEY", raising=False)
        model = Path("artifacts/model.cbm")
        app = create_app(model_path=model if model.is_file() else Path("nonexistent.cbm"))
        with TestClient(app) as c:
            yield c

    def test_analyze_has_risk_and_history(self, client: TestClient) -> None:
        blob = generate_sample("Ransomware_Encrypted", seed=9, size=4096)
        res = client.post(
            "/api/v1/analyze",
            files={"file": ("r.bin", blob, "application/octet-stream")},
        )
        assert res.status_code == 200
        data = res.json()
        assert "risk" in data
        assert data["risk"]["score"] >= 0
        assert data["analysis_id"]
        assert any(t.startswith("VERDICT_") for t in data["risk_tags"])

        hist = client.get("/api/v1/analyses")
        assert hist.status_code == 200
        assert any(r["id"] == data["analysis_id"] for r in hist.json())

        report = client.get(f"/api/v1/analyses/{data['analysis_id']}/report.html")
        assert report.status_code == 200
        assert "text/html" in report.headers["content-type"]

    def test_async_job(self, client: TestClient) -> None:
        blob = generate_sample("Benign_Code", seed=2, size=2048)
        res = client.post(
            "/api/v1/analyze/async",
            files={"file": ("b.bin", blob, "application/octet-stream")},
        )
        assert res.status_code == 200
        job_id = res.json()["job_id"]
        status = None
        for _ in range(80):
            job = client.get(f"/api/v1/jobs/{job_id}").json()
            status = job["status"]
            if status in {"done", "error"}:
                break
            time.sleep(0.05)
        assert status == "done"

    def test_auth_required(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BIN2IMG_DB", str(tmp_path / "auth.db"))
        monkeypatch.setenv("BIN2IMG_API_KEY", "secret-key")
        app = create_app(model_path=Path("nonexistent.cbm"))
        with TestClient(app) as c:
            denied = c.post(
                "/api/v1/analyze",
                files={"file": ("x.bin", b"MZ\x00\x00", "application/octet-stream")},
            )
            assert denied.status_code == 401
            assert c.get("/api/v1/analyses").status_code == 401
            assert c.get("/api/v1/analyses/nope").status_code == 401
            ok = c.post(
                "/api/v1/analyze",
                headers={"X-API-Key": "secret-key"},
                files={"file": ("x.bin", b"MZ\x00\x00", "application/octet-stream")},
            )
            assert ok.status_code == 200
            aid = ok.json()["analysis_id"]
            assert (
                c.get(f"/api/v1/analyses/{aid}", headers={"X-API-Key": "secret-key"}).status_code
                == 200
            )
            # Wrong-length key must be 401, not 500
            assert (
                c.get(
                    "/api/v1/analyses",
                    headers={"X-API-Key": "short"},
                ).status_code
                == 401
            )

    def test_negative_limit_clamped(self, client: TestClient) -> None:
        res = client.get("/api/v1/analyses?limit=-1")
        assert res.status_code == 200
        assert isinstance(res.json(), list)

    def test_compare(self, client: TestClient) -> None:
        a = generate_sample("Benign_Code", seed=1, size=2048)
        b = generate_sample("Ransomware_Encrypted", seed=2, size=2048)
        res = client.post(
            "/api/v1/compare",
            files=[
                ("file_a", ("a.bin", a, "application/octet-stream")),
                ("file_b", ("b.bin", b, "application/octet-stream")),
            ],
        )
        assert res.status_code == 200
        body = res.json()
        assert "feature_deltas_top" in body
        assert body["a"]["filename"] == "a.bin"


def test_service_persists(tmp_path: Path) -> None:
    svc = AnalysisService(model_path=Path("nonexistent.cbm"), db_path=tmp_path / "s.db")
    out = svc.analyze_bytes(b"MZ" + b"\x00" * 200, "tiny.bin")
    assert out["sha256"]
    assert out["analysis_id"]
    assert out["risk"]["verdict"] in {"Benign", "Suspicious", "Malicious", "Unknown"}
