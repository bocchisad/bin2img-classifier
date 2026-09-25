"""Dataset ingest for real + synthetic binaries.

Expected layout (recommended)::

    data/
      benign/                 → label Benign_Code
        putty.exe
        ...
      malware/
        Ransomware_Encrypted/
          sample1.bin
        Packed_PE/
          ...
        Downloader_TextHeavy/
          ...
        CustomFamily/         → arbitrary family name OK
          ...

Also accepts flat per-class dirs (legacy synthetic)::

    data/synthetic/Benign_Code/*.bin
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

BINARY_SUFFIXES: tuple[str, ...] = (
    ".exe",
    ".dll",
    ".bin",
    ".elf",
    ".so",
    ".dylib",
    ".sys",
    "",  # extensionless OK if file is regular
)

# Map folder names under data/benign → canonical label.
BENIGN_LABEL = "Benign_Code"


def _is_binary_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name.startswith("."):
        return False
    suffix = path.suffix.lower()
    if suffix in {".json", ".txt", ".md", ".csv", ".ds_store", ".png", ".jpg", ".jpeg", ".gif", ".py", ".html"}:
        return False
    # Allowlist known binary suffixes; extensionless files OK (ELF often has none).
    if suffix and suffix not in BINARY_SUFFIXES:
        # Peek magic for unknown suffixes
        try:
            head = path.read_bytes()[:4]
        except OSError:
            return False
        if head[:2] == b"MZ" or head[:4] == b"\x7fELF" or head[:4] in {
            b"\xfe\xed\xfa\xce",
            b"\xce\xfa\xed\xfe",
            b"\xfe\xed\xfa\xcf",
            b"\xcf\xfa\xed\xfe",
            b"\xca\xfe\xba\xbe",
        }:
            return True
        return False
    return True


def iter_files(directory: Path) -> Iterable[Path]:
    """Yield binary-like files under ``directory`` (non-recursive one level or rglob)."""
    if not directory.is_dir():
        return
    for path in sorted(directory.rglob("*")):
        if _is_binary_file(path):
            yield path


def load_dataset_dir(root: Path) -> list[tuple[bytes, str, str]]:
    """Load ``(raw_bytes, label, source_path)`` from a dataset root.

    Supports:
      - ``root/benign/**``
      - ``root/malware/<Family>/**``
      - ``root/<Label>/**`` (direct class folders)
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    samples: list[tuple[bytes, str, str]] = []

    benign_dir = root / "benign"
    malware_dir = root / "malware"

    if benign_dir.is_dir() or malware_dir.is_dir():
        if benign_dir.is_dir():
            for path in iter_files(benign_dir):
                try:
                    samples.append((path.read_bytes(), BENIGN_LABEL, str(path)))
                except OSError as exc:
                    logger.warning("Skip %s: %s", path, exc)
        if malware_dir.is_dir():
            loose = [p for p in malware_dir.iterdir() if p.is_file()]
            if loose:
                logger.warning(
                    "Ignoring %d file(s) directly under %s — put samples in "
                    "malware/<FamilyName>/",
                    len(loose),
                    malware_dir,
                )
            for family_dir in sorted(p for p in malware_dir.iterdir() if p.is_dir()):
                label = family_dir.name
                for path in iter_files(family_dir):
                    try:
                        samples.append((path.read_bytes(), label, str(path)))
                    except OSError as exc:
                        logger.warning("Skip %s: %s", path, exc)
    else:
        # Flat: root/Label/*.bin
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            label = class_dir.name
            if label.startswith("."):
                continue
            for path in iter_files(class_dir):
                try:
                    samples.append((path.read_bytes(), label, str(path)))
                except OSError as exc:
                    logger.warning("Skip %s: %s", path, exc)

    if not samples:
        raise FileNotFoundError(
            f"No binary samples under {root}. Expected data/benign + "
            f"data/malware/<Family> or data/<Label>/ files."
        )
    return samples


def summarize_dataset(samples: list[tuple[bytes, str, str]]) -> dict[str, int]:
    """Count samples per label."""
    counts: dict[str, int] = {}
    for _, label, _ in samples:
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def merge_samples(
    *groups: list[tuple[bytes, str]] | list[tuple[bytes, str, str]],
) -> list[tuple[bytes, str]]:
    """Flatten heterogeneous sample tuples to ``(bytes, label)``."""
    out: list[tuple[bytes, str]] = []
    for group in groups:
        for item in group:
            out.append((item[0], item[1]))
    return out
