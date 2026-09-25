"""String / PE structural helpers for feature extraction."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from bin2img.parser import HIGH_ENTROPY_THRESHOLD, BinaryMetadata

# Suspicious Win32 API / packer-adjacent strings often seen in malware.
_SUSPICIOUS_API_PATTERNS: tuple[str, ...] = (
    "VirtualAlloc",
    "VirtualProtect",
    "WriteProcessMemory",
    "CreateRemoteThread",
    "NtUnmapViewOfSection",
    "URLDownloadToFile",
    "InternetOpen",
    "InternetReadFile",
    "WinExec",
    "ShellExecute",
    "CryptEncrypt",
    "CryptDecrypt",
    "IsDebuggerPresent",
    "CheckRemoteDebuggerPresent",
    "GetProcAddress",
    "LoadLibrary",
    "WinHttpOpen",
    "socket",
    "WSAStartup",
    "powershell",
    "cmd.exe",
)

_URL_RE = re.compile(rb"https?://[^\x00-\x1f]{4,120}", re.IGNORECASE)
_IP_RE = re.compile(
    rb"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)


def extract_byte_string_stats(raw: bytes) -> dict[str, float]:
    """Cheap printable-string statistics over a raw blob."""
    if not raw:
        return {
            "url_hit_ratio": 0.0,
            "ip_hit_ratio": 0.0,
            "suspicious_api_hits": 0.0,
            "avg_string_len": 0.0,
            "string_density": 0.0,
        }

    urls = _URL_RE.findall(raw)
    ips = _IP_RE.findall(raw)
    api_hits = 0
    # Decode latin-1 once for substring search (binary-safe).
    text = raw.decode("latin-1", errors="ignore")
    for name in _SUSPICIOUS_API_PATTERNS:
        if name.lower() in text.lower():
            api_hits += 1

    # Extract ASCII runs length >= 4
    strings = re.findall(rb"[\x20-\x7e]{4,}", raw)
    avg_len = float(np.mean([len(s) for s in strings])) if strings else 0.0
    string_bytes = sum(len(s) for s in strings)
    return {
        "url_hit_ratio": min(1.0, len(urls) / max(1, len(raw) / 4096)),
        "ip_hit_ratio": min(1.0, len(ips) / max(1, len(raw) / 4096)),
        "suspicious_api_hits": float(api_hits),
        "avg_string_len": avg_len,
        "string_density": string_bytes / len(raw),
    }


def extract_pe_structural_features(
    metadata: BinaryMetadata,
    raw: bytes | None,
) -> dict[str, float]:
    """Section geometry / overlay / import-lite features."""
    sections = metadata.sections
    n = len(sections)
    if n == 0:
        virt_raw_max = 0.0
        high_ent_ratio = 0.0
        wx_ratio = 0.0
        mean_raw = 0.0
    else:
        ratios = []
        high = 0
        wx = 0
        raw_sizes = []
        for s in sections:
            raw_sizes.append(s.raw_size)
            if s.raw_size > 0:
                ratios.append(s.virtual_size / s.raw_size)
            elif s.virtual_size > 0:
                ratios.append(8.0)  # cap — fully virtual section
            if s.entropy > HIGH_ENTROPY_THRESHOLD:
                high += 1
            # IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_WRITE, or exec+write via flags
            chars = s.characteristics or 0
            pe_wx = bool((chars & 0x20000000) and (chars & 0x80000000))
            # ELF/Mach-O: characteristics often unset; treat executable + inflated
            # writable heuristic via characteristic write bit OR virt>>raw with exec.
            if pe_wx or (s.is_executable and chars & 0x80000000):
                wx += 1
            elif s.is_executable and s.raw_size == 0 and s.virtual_size > 0:
                # Common packer pattern on non-PE: exec virtual-only section
                wx += 1
        virt_raw_max = float(min(max(ratios) if ratios else 0.0, 32.0))
        high_ent_ratio = high / n
        wx_ratio = wx / n
        mean_raw = float(np.mean(raw_sizes)) if raw_sizes else 0.0

    entry_present = 1.0 if metadata.entry_point is not None else 0.0

    # Overlay: bytes after last section's raw end (PE-oriented heuristic).
    overlay_ratio = 0.0
    import_count = 0.0
    if raw and len(raw) >= 2 and raw[:2] == b"MZ":
        overlay_ratio, import_count = _pefile_extras(raw)

    return {
        "virt_raw_ratio_max": virt_raw_max,
        "high_entropy_section_ratio": float(high_ent_ratio),
        "wx_section_ratio": float(wx_ratio),
        "mean_section_raw_size_log10": math.log10(mean_raw) if mean_raw > 0 else 0.0,
        "entry_point_present": entry_present,
        "overlay_ratio": float(overlay_ratio),
        "import_count_log1p": math.log1p(import_count),
    }


def _pefile_extras(raw: bytes) -> tuple[float, float]:
    """Return ``(overlay_ratio, import_count)`` via pefile when possible."""
    try:
        import pefile
    except ImportError:
        return 0.0, 0.0

    pe: Any = None
    try:
        pe = pefile.PE(data=raw, fast_load=True)
        try:
            pe.parse_data_directories(
                directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]]
            )
        except Exception:  # noqa: BLE001
            pass

        # Overlay
        overlay_ratio = 0.0
        try:
            offset = pe.get_overlay_data_start_offset()
            if offset is not None and len(raw) > 0:
                overlay_ratio = max(0.0, (len(raw) - offset) / len(raw))
        except Exception:  # noqa: BLE001
            overlay_ratio = 0.0

        import_count = 0.0
        try:
            if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
                for entry in pe.DIRECTORY_ENTRY_IMPORT:
                    import_count += len(getattr(entry, "imports", []) or [])
        except Exception:  # noqa: BLE001
            import_count = 0.0

        return float(overlay_ratio), float(import_count)
    except Exception:  # noqa: BLE001
        return 0.0, 0.0
    finally:
        if pe is not None:
            try:
                pe.close()
            except Exception:  # noqa: BLE001
                pass
