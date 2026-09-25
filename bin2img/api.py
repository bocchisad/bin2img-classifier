"""FastAPI application — product API + interactive HTML viewer.

Endpoints:
  GET  /                              UI
  GET  /api/v1/health                 Liveness
  POST /api/v1/analyze                Sync analysis
  POST /api/v1/analyze/async          Queue analysis job
  GET  /api/v1/jobs/{id}              Job status
  GET  /api/v1/analyses               History list
  GET  /api/v1/analyses/{id}          Stored analysis
  GET  /api/v1/analyses/{id}/report.html
  POST /api/v1/compare                Compare two uploads
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from bin2img.constants import MAX_UPLOAD_BYTES, __version__
from bin2img.auth import require_api_key
from bin2img.jobs import JobRunner
from bin2img.model import DEFAULT_MODEL_PATH
from bin2img.report import render_html_report
from bin2img.service import AnalysisService, build_risk_tags

logger = logging.getLogger(__name__)

__all__ = ["AnalysisService", "build_risk_tags", "create_app"]

_STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"
_AUTH = [Depends(require_api_key)]


def _default_model_path() -> Path:
    env = os.environ.get("BIN2IMG_MODEL")
    return Path(env) if env else DEFAULT_MODEL_PATH


class AnalyzeResponse(BaseModel):
    filename: str
    sha256: str | None = None
    analysis_id: str | None = None
    metadata: dict[str, Any]
    classification: dict[str, Any] | None
    risk_tags: list[str]
    risk: dict[str, Any] | None = None
    explanation: dict[str, Any] | None = None
    images: dict[str, str] = Field(default_factory=dict)
    features: dict[str, float] = Field(default_factory=dict)


async def _read_upload_capped(file: UploadFile) -> bytes:
    """Read upload in chunks; reject before buffering beyond ``MAX_UPLOAD_BYTES``."""
    cl = file.headers.get("content-length") if file.headers else None
    if cl is not None:
        try:
            if int(cl) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
                )
        except ValueError:
            pass

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
            )
        chunks.append(chunk)
    raw = b"".join(chunks)
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file upload")
    return raw


def _safe_filename(name: str | None, fallback: str) -> str:
    raw = (name or fallback).replace("\\", "/").split("/")[-1].strip()
    return (raw[:200] if raw else fallback)


def create_app(model_path: str | Path | None = None) -> FastAPI:
    """Application factory (used by uvicorn and tests)."""
    resolved = Path(model_path) if model_path is not None else _default_model_path()
    db = os.environ.get("BIN2IMG_DB", "artifacts/bin2img.db")
    service = AnalysisService(model_path=resolved, db_path=db)
    runner = JobRunner(service.store, service.analyze_bytes)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        service.load_model()
        yield
        runner.shutdown()

    app = FastAPI(
        title="bin2img-classifier",
        description="Binary Visualizer & Malware Family Classifier",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.service = service
    app.state.runner = runner

    @app.get("/api/v1/health")
    async def health() -> dict[str, Any]:
        auth_on = bool(os.environ.get("BIN2IMG_API_KEY", "").strip())
        return {
            "status": "ok",
            "model_loaded": service.model_loaded,
            "auth_required": auth_on,
            # Keep path/classes for local UI; omit sensitive detail when auth is on
            # unless callers are authenticated — health stays public & minimal then.
            **(
                {}
                if auth_on
                else {
                    "model_path": str(service.model_path),
                    "classes": (
                        list(service.classifier.class_names)
                        if service.model_loaded
                        else []
                    ),
                }
            ),
            **(
                {
                    "classes_count": (
                        len(service.classifier.class_names) if service.model_loaded else 0
                    )
                }
                if auth_on
                else {}
            ),
        }

    @app.post(
        "/api/v1/analyze",
        dependencies=_AUTH,
        response_model=AnalyzeResponse,
    )
    async def analyze(file: UploadFile = File(...)) -> JSONResponse:
        raw = await _read_upload_capped(file)
        name = _safe_filename(file.filename, "upload.bin")
        try:
            payload = await asyncio.to_thread(service.analyze_bytes, raw, name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(payload)

    @app.post("/api/v1/analyze/async", dependencies=_AUTH)
    async def analyze_async(file: UploadFile = File(...)) -> dict[str, Any]:
        raw = await _read_upload_capped(file)
        name = _safe_filename(file.filename, "upload.bin")
        try:
            job_id = runner.submit(raw, name)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        job = service.store.get_job(job_id) or {"id": job_id, "status": "queued"}
        return {"job_id": job_id, "status": job.get("status", "queued")}

    @app.get("/api/v1/jobs/{job_id}", dependencies=_AUTH)
    async def get_job(job_id: str) -> dict[str, Any]:
        job = service.store.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.get("/api/v1/analyses", dependencies=_AUTH)
    async def list_analyses(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 200))
        safe_offset = max(0, int(offset))
        return service.store.list_analyses(limit=safe_limit, offset=safe_offset)

    @app.get("/api/v1/analyses/{analysis_id}", dependencies=_AUTH)
    async def get_analysis(analysis_id: str) -> dict[str, Any]:
        row = service.store.get_analysis(analysis_id)
        if not row:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return row

    @app.get("/api/v1/analyses/{analysis_id}/report.html", dependencies=_AUTH)
    async def analysis_report(analysis_id: str) -> Response:
        row = service.store.get_analysis(analysis_id)
        if not row:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return Response(content=render_html_report(row), media_type="text/html")

    @app.post("/api/v1/compare", dependencies=_AUTH)
    async def compare(
        file_a: UploadFile = File(...),
        file_b: UploadFile = File(...),
    ) -> dict[str, Any]:
        raw_a = await _read_upload_capped(file_a)
        raw_b = await _read_upload_capped(file_b)
        name_a = _safe_filename(file_a.filename, "a.bin")
        name_b = _safe_filename(file_b.filename, "b.bin")
        a, b = await asyncio.gather(
            asyncio.to_thread(service.analyze_bytes, raw_a, name_a),
            asyncio.to_thread(service.analyze_bytes, raw_b, name_b),
        )
        fa, fb = a.get("features") or {}, b.get("features") or {}
        keys = sorted(set(fa) | set(fb))
        deltas = {k: float(fa.get(k, 0.0)) - float(fb.get(k, 0.0)) for k in keys}
        top = sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)[:12]
        return {
            "a": {
                "analysis_id": a.get("analysis_id"),
                "filename": a.get("filename"),
                "risk": a.get("risk"),
                "classification": a.get("classification"),
            },
            "b": {
                "analysis_id": b.get("analysis_id"),
                "filename": b.get("filename"),
                "risk": b.get("risk"),
                "classification": b.get("classification"),
            },
            "feature_deltas_top": [{"feature": k, "delta": v} for k, v in top],
        }

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        if _STATIC_INDEX.is_file():
            return HTMLResponse(_STATIC_INDEX.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>bin2img</h1><p>UI missing.</p>")

    return app
