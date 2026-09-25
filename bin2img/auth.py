"""Optional API-key authentication dependency."""

from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException


def configured_api_key() -> str | None:
    """Return API key from env, or None if auth is disabled."""
    key = os.environ.get("BIN2IMG_API_KEY", "").strip()
    return key or None


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """FastAPI dependency: enforce ``X-API-Key`` when ``BIN2IMG_API_KEY`` is set."""
    expected = configured_api_key()
    if expected is None:
        return
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
    # compare_digest raises ValueError on length mismatch on some Python versions.
    if len(x_api_key) != len(expected) or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
