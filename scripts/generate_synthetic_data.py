"""Synthetic binary sample generator for demo training.

Simulates four behavioural families via byte-level entropy / structure
patterns (not real malware). Benign / Packed / Downloader produce
**parseable PE** samples so the classifier learns the same feature space
as real uploads (e.g. sample.exe).

    python scripts/generate_synthetic_data.py --out data/synthetic --per-class 50
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Callable

import numpy as np

# Family labels used by the CatBoost pipeline.
CLASS_LABELS: tuple[str, ...] = (
    "Benign_Code",
    "Ransomware_Encrypted",
    "Packed_PE",
    "Downloader_TextHeavy",
)

GeneratorFn = Callable[[np.random.Generator, int], bytes]

# PE section characteristics
_SCN_CODE_EXEC_READ = 0x60000020
_SCN_INITIALIZED_READ = 0x40000040
_SCN_INITIALIZED_READ_WRITE = 0xC0000040


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def build_pe(
    sections: list[tuple[str, bytes, int]],
    *,
    entry_point: int = 0x1000,
) -> bytes:
    """Build a minimal PE32 that ``pefile`` can parse.

    Args:
        sections: ``(name, raw_data, characteristics)`` — data is padded to
            file alignment (0x200) internally.
        entry_point: RVA stored in OptionalHeader.
    """
    file_align = 0x200
    section_align = 0x1000
    num_sections = len(sections)
    if num_sections < 1:
        raise ValueError("need at least one section")

    # Headers: DOS(64) + PE/COFF/Opt(4+20+0xE0) + section table
    headers_size = 0x40 + 4 + 20 + 0xE0 + 40 * num_sections
    headers_raw = _align(headers_size, file_align)

    pe = bytearray()
    pe += b"MZ" + b"\x00" * 58
    pe += struct.pack("<I", 0x40)
    pe += b"PE\x00\x00"
    pe += struct.pack("<HHIIIHH", 0x14C, num_sections, 0, 0, 0, 0xE0, 0x0103)

    opt = bytearray(0xE0)
    struct.pack_into("<H", opt, 0, 0x10B)
    struct.pack_into("<I", opt, 16, entry_point)
    struct.pack_into("<I", opt, 28, 0x400000)
    struct.pack_into("<I", opt, 32, section_align)
    struct.pack_into("<I", opt, 36, file_align)
    struct.pack_into("<H", opt, 40, 6)
    struct.pack_into("<I", opt, 56, section_align * (num_sections + 2))
    struct.pack_into("<I", opt, 60, headers_raw)
    struct.pack_into("<H", opt, 68, 3)
    struct.pack_into("<I", opt, 92, 16)
    pe += opt

    # Section table
    padded_payloads: list[bytes] = []
    next_va = section_align
    next_raw = headers_raw
    for name, data, chars in sections:
        raw_size = _align(max(len(data), 1), file_align)
        payload = data[:raw_size] + b"\x00" * max(0, raw_size - len(data))
        padded_payloads.append(payload)
        virt_size = max(len(data), 1)
        name_bytes = name.encode("ascii", errors="replace")[:8].ljust(8, b"\x00")
        pe += name_bytes
        pe += struct.pack(
            "<IIIIIIHHI",
            virt_size,
            next_va,
            raw_size,
            next_raw,
            0,
            0,
            0,
            0,
            chars,
        )
        next_va += _align(virt_size, section_align)
        next_raw += raw_size

    if len(pe) < headers_raw:
        pe += b"\x00" * (headers_raw - len(pe))
    else:
        # Truncate only if somehow oversized (should not happen).
        pe = pe[:headers_raw]

    for payload in padded_payloads:
        pe += payload

    return bytes(pe)


def _opcode_code(rng: np.random.Generator, nbytes: int) -> bytes:
    """Low-entropy x86-ish code texture."""
    alphabet = np.array(
        [0x90, 0x55, 0x8B, 0xEC, 0x83, 0xC0, 0x01, 0xC3, 0x48, 0x89, 0x5D, 0xC9],
        dtype=np.uint8,
    )
    base = rng.choice(alphabet, size=nbytes)
    pattern = np.tile(np.array([0x90, 0x90, 0x55, 0x8B, 0xEC, 0xC3, 0x00, 0x00], dtype=np.uint8), 8)
    for offset in range(0, max(0, nbytes - len(pattern)), 64):
        end = min(offset + len(pattern), nbytes)
        base[offset:end] = pattern[: end - offset]
    # Tiny noise so samples are not bitwise identical.
    n_flip = max(2, nbytes // 200)
    idx = rng.choice(nbytes, size=n_flip, replace=False)
    base[idx] = rng.choice(alphabet, size=n_flip)
    return base.tobytes()


def _ascii_blob(rng: np.random.Generator, nbytes: int) -> bytes:
    chunks = [
        b"http://cdn.example.com/update?id=",
        b"GET /download/payload.bin HTTP/1.1\r\n",
        b"User-Agent: Mozilla/5.0 (compatible)\r\n",
        b"Content-Type: application/octet-stream\r\n",
        b"cmd.exe /c powershell -nop -w hidden -enc ",
        b"Software\\Microsoft\\Windows\\CurrentVersion\\Run\x00",
        b"https://mirror.example.org/agent/",
    ]
    out = bytearray()
    while len(out) < nbytes:
        part = chunks[int(rng.integers(0, len(chunks)))]
        out.extend(part)
        out.extend(f"{int(rng.integers(0, 99999)):05d}".encode("ascii"))
        out.extend(b"\r\n")
    data = np.frombuffer(bytes(out[:nbytes]), dtype=np.uint8).copy()
    nulls = rng.choice(nbytes, size=max(4, nbytes // 100), replace=False)
    data[nulls] = 0
    return data.tobytes()


def generate_benign_code(rng: np.random.Generator, size: int = 4096) -> bytes:
    """Parseable PE with low-entropy ``.text`` (optionally + tiny ``.data``)."""
    size = max(size, 768)
    payload_budget = max(256, size - 0x400)
    # ~50% single-section PE (matches tiny sample.exe style).
    if float(rng.random()) < 0.5 or payload_budget < 400:
        text = _opcode_code(rng, payload_budget)
        return build_pe([(".text", text, _SCN_CODE_EXEC_READ)])

    text_len = int(payload_budget * 0.85)
    data_len = max(32, payload_budget - text_len)
    text = _opcode_code(rng, text_len)
    data = bytes([0x00, 0x01, 0x02, 0x00]) * (data_len // 4 + 1)
    data = data[:data_len]
    return build_pe(
        [
            (".text", text, _SCN_CODE_EXEC_READ),
            (".data", data, _SCN_INITIALIZED_READ_WRITE),
        ]
    )


def generate_ransomware_encrypted(rng: np.random.Generator, size: int = 4096) -> bytes:
    """Parseable PE with a dominant high-entropy section (encrypted lookalike)."""
    size = max(size, 2048)
    payload_budget = max(1024, size - 0x400)
    stub_len = max(64, payload_budget // 16)
    enc_len = payload_budget - stub_len
    stub = _opcode_code(rng, stub_len)
    encrypted = rng.integers(0, 256, size=enc_len, dtype=np.uint8).tobytes()
    return build_pe(
        [
            (".text", stub, _SCN_CODE_EXEC_READ),
            (".rdata", encrypted, _SCN_INITIALIZED_READ_WRITE),
        ]
    )


def generate_packed_pe(rng: np.random.Generator, size: int = 4096) -> bytes:
    """Parseable PE: small low-entropy stub + high-entropy ``UPX1`` section."""
    size = max(size, 2048)
    payload_budget = max(1024, size - 0x400)
    stub_len = max(128, payload_budget // 10)
    packed_len = payload_budget - stub_len
    stub = _opcode_code(rng, stub_len)
    packed = rng.integers(0, 256, size=packed_len, dtype=np.uint8).tobytes()
    # UPX-like names trigger SUSPICIOUS_SECTION and mark packing.
    return build_pe(
        [
            (".text", stub, _SCN_CODE_EXEC_READ),
            ("UPX0", b"\x00" * 64, _SCN_INITIALIZED_READ_WRITE),
            ("UPX1", packed, _SCN_CODE_EXEC_READ),
        ]
    )


def generate_downloader_text_heavy(rng: np.random.Generator, size: int = 4096) -> bytes:
    """Parseable PE with tiny code stub and large ASCII/URL ``.rdata``."""
    size = max(size, 2048)
    payload_budget = max(1024, size - 0x400)
    text_len = max(128, payload_budget // 12)
    rdata_len = payload_budget - text_len
    text = _opcode_code(rng, text_len)
    rdata = _ascii_blob(rng, rdata_len)
    return build_pe(
        [
            (".text", text, _SCN_CODE_EXEC_READ),
            (".rdata", rdata, _SCN_INITIALIZED_READ),
        ]
    )


GENERATORS: dict[str, GeneratorFn] = {
    "Benign_Code": generate_benign_code,
    "Ransomware_Encrypted": generate_ransomware_encrypted,
    "Packed_PE": generate_packed_pe,
    "Downloader_TextHeavy": generate_downloader_text_heavy,
}


def generate_sample(
    label: str,
    *,
    rng: np.random.Generator | None = None,
    size: int | None = None,
    seed: int | None = None,
) -> bytes:
    """Generate one synthetic sample for ``label``."""
    if label not in GENERATORS:
        raise ValueError(f"Unknown label {label!r}; choose from {CLASS_LABELS}")
    gen = rng or _rng(seed)
    if size is None:
        size = int(gen.integers(3072, 12289))
    return GENERATORS[label](gen, size)


def generate_dataset(
    *,
    per_class: int = 40,
    output_dir: str | Path | None = None,
    seed: int = 42,
    size_range: tuple[int, int] = (1024, 12288),
) -> list[dict[str, str | int]]:
    """Create synthetic samples (optionally write to disk)."""
    rng = _rng(seed)
    out: Path | None = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, str | int]] = []
    for label in CLASS_LABELS:
        class_dir = out / label if out is not None else None
        if class_dir is not None:
            class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            size = int(rng.integers(size_range[0], size_range[1] + 1))
            blob = generate_sample(label, rng=rng, size=size)
            rel = ""
            if class_dir is not None:
                name = f"{label.lower()}_{i:04d}.bin"
                path = class_dir / name
                path.write_bytes(blob)
                rel = str(path)
            manifest.append(
                {
                    "path": rel,
                    "label": label,
                    "size": len(blob),
                    "index": i,
                }
            )

    if out is not None:
        (out / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        (out / "labels.json").write_text(
            json.dumps(list(CLASS_LABELS), indent=2),
            encoding="utf-8",
        )
    return manifest


def iter_dataset_bytes(
    *,
    per_class: int = 40,
    seed: int = 42,
    size_range: tuple[int, int] = (1024, 12288),
) -> list[tuple[bytes, str]]:
    """In-memory ``(raw_bytes, label)`` pairs for training without disk I/O."""
    rng = _rng(seed)
    samples: list[tuple[bytes, str]] = []
    for label in CLASS_LABELS:
        for _ in range(per_class):
            size = int(rng.integers(size_range[0], size_range[1] + 1))
            samples.append((generate_sample(label, rng=rng, size=size), label))
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic binary families for bin2img demos."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/synthetic"),
        help="Output directory (default: data/synthetic)",
    )
    parser.add_argument("--per-class", type=int, default=40, help="Samples per class")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--min-size", type=int, default=1024)
    parser.add_argument("--max-size", type=int, default=12288)
    args = parser.parse_args(argv)

    manifest = generate_dataset(
        per_class=args.per_class,
        output_dir=args.out,
        seed=args.seed,
        size_range=(args.min_size, args.max_size),
    )
    print(f"Wrote {len(manifest)} samples → {args.out.resolve()}")
    print(f"Classes: {', '.join(CLASS_LABELS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
