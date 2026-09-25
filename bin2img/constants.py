"""Shared product constants."""

from __future__ import annotations

__version__ = "0.2.0"

# Packer / protector section name fragments (uppercase match).
SUSPICIOUS_SECTION_HINTS: tuple[str, ...] = (
    "UPX",
    "PACK",
    "THEMIDA",
    "VMP",
    "ASPACK",
    "ENIGMA",
    "NSP",
    "PEC",
    "CRYPT",
)

# Hard upload / analyze size limit (bytes).
MAX_UPLOAD_BYTES: int = 50 * 1024 * 1024

# Cap heuristic string scan window (bytes) to bound CPU/RAM.
RULE_SCAN_BYTES: int = 2 * 1024 * 1024

# Max concurrent async jobs (queued + running).
MAX_PENDING_JOBS: int = 32
