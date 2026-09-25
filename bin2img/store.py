"""SQLite persistence for analysis history."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AnalysisStore:
    """Thread-safe SQLite store for analyses and async jobs."""

    def __init__(self, db_path: str | Path = "artifacts/bin2img.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.blob_dir = self.db_path.parent / "analysis_blobs"
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS analyses (
                        id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        sha256 TEXT NOT NULL,
                        verdict TEXT,
                        risk_score INTEGER,
                        label TEXT,
                        payload_json TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_analyses_created
                        ON analyses(created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_analyses_sha256
                        ON analyses(sha256);
                    CREATE TABLE IF NOT EXISTS jobs (
                        id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        status TEXT NOT NULL,
                        filename TEXT,
                        error TEXT,
                        result_id TEXT
                    );
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def _strip_images(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        slim = dict(payload)
        images = slim.pop("images", None) or {}
        return slim, dict(images) if isinstance(images, dict) else {}

    def save_analysis(
        self,
        *,
        filename: str,
        sha256: str,
        payload: dict[str, Any],
    ) -> str:
        analysis_id = uuid.uuid4().hex
        risk = payload.get("risk") or {}
        classification = payload.get("classification") or {}
        slim, images = self._strip_images(payload)
        blob_path = self.blob_dir / f"{analysis_id}.json"
        blob_path.write_text(json.dumps({"images": images}), encoding="utf-8")

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO analyses
                    (id, created_at, filename, sha256, verdict, risk_score, label, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        analysis_id,
                        _utc_now(),
                        filename,
                        sha256,
                        risk.get("verdict"),
                        risk.get("score"),
                        classification.get("label"),
                        json.dumps(slim),
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        return analysis_id

    def get_analysis(self, analysis_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
                ).fetchone()
            finally:
                conn.close()
        if row is None:
            return None
        payload = json.loads(row["payload_json"])
        payload["analysis_id"] = row["id"]
        payload["created_at"] = row["created_at"]
        blob_path = self.blob_dir / f"{analysis_id}.json"
        if blob_path.is_file():
            try:
                blob = json.loads(blob_path.read_text(encoding="utf-8"))
                payload["images"] = blob.get("images") or {}
            except (OSError, json.JSONDecodeError):
                payload.setdefault("images", {})
        else:
            payload.setdefault("images", {})
        return payload

    def list_analyses(self, *, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT id, created_at, filename, sha256, verdict, risk_score, label
                    FROM analyses
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                ).fetchall()
            finally:
                conn.close()
        return [dict(r) for r in rows]

    def count_active_jobs(self) -> int:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT COUNT(*) AS n FROM jobs
                    WHERE status IN ('queued', 'running')
                    """
                ).fetchone()
            finally:
                conn.close()
        return int(row["n"] if row else 0)

    def create_job(self, filename: str) -> str:
        job_id = uuid.uuid4().hex
        now = _utc_now()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO jobs (id, created_at, updated_at, status, filename)
                    VALUES (?, ?, ?, 'queued', ?)
                    """,
                    (job_id, now, now, filename),
                )
                conn.commit()
            finally:
                conn.close()
        return job_id

    def update_job(
        self,
        job_id: str,
        *,
        status: str,
        error: str | None = None,
        result_id: str | None = None,
        clear_error: bool = False,
    ) -> None:
        sets = ["updated_at = ?", "status = ?"]
        args: list[Any] = [_utc_now(), status]
        if error is not None or clear_error:
            sets.append("error = ?")
            args.append(error)
        if result_id is not None:
            sets.append("result_id = ?")
            args.append(result_id)
        args.append(job_id)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?",
                    args,
                )
                conn.commit()
            finally:
                conn.close()

    def cancel_active_jobs(self, reason: str = "server shutdown") -> int:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    UPDATE jobs
                    SET updated_at = ?, status = 'error', error = ?
                    WHERE status IN ('queued', 'running')
                    """,
                    (_utc_now(), reason),
                )
                conn.commit()
                return int(cur.rowcount or 0)
            finally:
                conn.close()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM jobs WHERE id = ?", (job_id,)
                ).fetchone()
            finally:
                conn.close()
        return dict(row) if row else None
