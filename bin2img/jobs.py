"""In-process async job runner for large uploads."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from bin2img.constants import MAX_PENDING_JOBS
from bin2img.store import AnalysisStore

logger = logging.getLogger(__name__)


class JobRunner:
    """Background worker pool backed by :class:`AnalysisStore`."""

    def __init__(
        self,
        store: AnalysisStore,
        analyze_fn: Callable[[bytes, str], dict[str, Any]],
        *,
        max_workers: int = 2,
        max_pending: int = MAX_PENDING_JOBS,
    ) -> None:
        self.store = store
        self.analyze_fn = analyze_fn
        self.max_pending = max_pending
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="bin2img")
        self._lock = threading.Lock()
        self._shutting_down = False

    def submit(self, raw: bytes, filename: str) -> str:
        with self._lock:
            if self._shutting_down:
                raise RuntimeError("Job runner is shutting down")
            if self.store.count_active_jobs() >= self.max_pending:
                raise RuntimeError(
                    f"Too many pending jobs (max {self.max_pending}); try again later"
                )
            job_id = self.store.create_job(filename)
            self._pool.submit(self._run, job_id, raw, filename)
            return job_id

    def _run(self, job_id: str, raw: bytes, filename: str) -> None:
        self.store.update_job(job_id, status="running", clear_error=True)
        try:
            payload = self.analyze_fn(raw, filename)
            result_id = payload.get("analysis_id")
            self.store.update_job(job_id, status="done", result_id=result_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Job %s failed", job_id)
            self.store.update_job(job_id, status="error", error=str(exc))

    def shutdown(self) -> None:
        with self._lock:
            self._shutting_down = True
        self.store.cancel_active_jobs("server shutdown")
        self._pool.shutdown(wait=False, cancel_futures=True)
