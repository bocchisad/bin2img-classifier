"""Heuristic rule engine (YARA-lite) for binary risk signals."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from bin2img.constants import RULE_SCAN_BYTES, SUSPICIOUS_SECTION_HINTS
from bin2img.parser import HIGH_ENTROPY_THRESHOLD, BinaryMetadata

_SUSPICIOUS_SECTION_HINTS = SUSPICIOUS_SECTION_HINTS

_SUSPICIOUS_STRINGS: tuple[tuple[str, str, int], ...] = (
    # (pattern, rule_id, weight)
    (r"VirtualAlloc", "api_virtualalloc", 8),
    (r"WriteProcessMemory", "api_wpm", 12),
    (r"CreateRemoteThread", "api_crt", 12),
    (r"URLDownloadToFile", "api_urldownload", 10),
    (r"WinExec", "api_winexec", 6),
    (r"powershell", "str_powershell", 8),
    (r"cmd\.exe", "str_cmd", 5),
    (r"This program cannot be run in DOS mode", "dos_stub", 0),
)


@dataclass(frozen=True, slots=True)
class RuleHit:
    rule_id: str
    severity: str  # info | low | medium | high
    weight: int
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_rules(
    metadata: BinaryMetadata,
    raw: bytes,
    *,
    features: dict[str, float] | None = None,
) -> list[RuleHit]:
    """Run static heuristic rules; return ordered hits."""
    hits: list[RuleHit] = []
    features = features or {}

    if metadata.is_valid_executable:
        hits.append(
            RuleHit("valid_executable", "info", 0, f"Parsed as {metadata.format.value}")
        )

    if metadata.overall_entropy > HIGH_ENTROPY_THRESHOLD:
        hits.append(
            RuleHit(
                "high_file_entropy",
                "high",
                25,
                f"Overall entropy {metadata.overall_entropy:.3f} > {HIGH_ENTROPY_THRESHOLD}",
            )
        )

    for section in metadata.sections:
        name_u = section.name.upper()
        if any(h in name_u for h in _SUSPICIOUS_SECTION_HINTS):
            hits.append(
                RuleHit(
                    "suspicious_section_name",
                    "high",
                    20,
                    f"Section name looks packed/protected: {section.name!r}",
                )
            )
        if section.entropy > HIGH_ENTROPY_THRESHOLD:
            hits.append(
                RuleHit(
                    "high_section_entropy",
                    "medium",
                    12,
                    f"Section {section.name!r} entropy={section.entropy:.3f}",
                )
            )
        if section.raw_size > 0 and section.virtual_size > section.raw_size * 4:
            hits.append(
                RuleHit(
                    "inflated_virtual_size",
                    "medium",
                    10,
                    f"{section.name}: virt={section.virtual_size} raw={section.raw_size}",
                )
            )
        chars = section.characteristics or 0
        if (chars & 0x20000000) and (chars & 0x80000000):
            hits.append(
                RuleHit(
                    "writable_executable_section",
                    "high",
                    18,
                    f"WX section {section.name!r}",
                )
            )

    if features.get("overlay_ratio", 0.0) > 0.15:
        hits.append(
            RuleHit(
                "large_overlay",
                "medium",
                10,
                f"Overlay ratio={features['overlay_ratio']:.3f}",
            )
        )
    if features.get("url_hit_ratio", 0.0) > 0.05:
        hits.append(
            RuleHit(
                "url_dense",
                "medium",
                8,
                f"URL density={features['url_hit_ratio']:.3f}",
            )
        )
    if features.get("suspicious_api_hits", 0.0) >= 3:
        hits.append(
            RuleHit(
                "suspicious_api_cluster",
                "high",
                15,
                f"Suspicious API hits={int(features['suspicious_api_hits'])}",
            )
        )

    text = raw[:RULE_SCAN_BYTES].decode("latin-1", errors="ignore")
    for pattern, rule_id, weight in _SUSPICIOUS_STRINGS:
        if weight <= 0:
            continue
        if re.search(pattern, text, flags=re.IGNORECASE):
            hits.append(
                RuleHit(
                    rule_id,
                    "medium" if weight < 10 else "high",
                    weight,
                    f"Matched pattern /{pattern}/",
                )
            )

    # Deduplicate by rule_id keeping strongest weight
    best: dict[str, RuleHit] = {}
    for hit in hits:
        prev = best.get(hit.rule_id)
        if prev is None or hit.weight > prev.weight:
            best[hit.rule_id] = hit
    return sorted(best.values(), key=lambda h: (-h.weight, h.rule_id))
