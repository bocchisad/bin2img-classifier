#!/usr/bin/env python3
"""Fetch a small *safe* real-ish demo dataset (no live malware).

What it does:
  1. Downloads official PuTTY PE builds (benign).
  2. UPX-packs copies → data/malware/Packed_PE (real packer).
  3. XOR-encrypts copies → data/malware/Ransomware_Encrypted (high-entropy).
  4. Appends URL-heavy overlays → data/malware/Downloader_TextHeavy.

Requires: curl, and ``upx`` on PATH (``brew install upx``).

Usage::

    python scripts/fetch_demo_dataset.py
    python scripts/train.py --real-data data --mix-synthetic --per-class 40
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUTTY_BASE = "https://the.earth.li/~sgtatham/putty/latest/w64"
PUTTY_FILES = ("putty.exe", "puttygen.exe", "pscp.exe", "plink.exe", "pageant.exe")


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"GET {url}")
    urllib.request.urlretrieve(url, dest)


def _which_upx() -> str | None:
    return shutil.which("upx")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch safe demo PE dataset")
    parser.add_argument("--root", type=Path, default=ROOT / "data")
    parser.add_argument("--workdir", type=Path, default=ROOT / "artifacts" / "downloads")
    args = parser.parse_args(argv)

    upx = _which_upx()
    if not upx:
        print("error: upx not found. Install with: brew install upx", file=sys.stderr)
        return 1

    work = args.workdir
    work.mkdir(parents=True, exist_ok=True)
    benign = args.root / "benign"
    packed = args.root / "malware" / "Packed_PE"
    ransom = args.root / "malware" / "Ransomware_Encrypted"
    downloader = args.root / "malware" / "Downloader_TextHeavy"
    for d in (benign, packed, ransom, downloader):
        d.mkdir(parents=True, exist_ok=True)

    local_exes: list[Path] = []
    for name in PUTTY_FILES:
        dest = work / name
        if not dest.is_file():
            _download(f"{PUTTY_BASE}/{name}", dest)
        else:
            print(f"cached {dest}")
        shutil.copy2(dest, benign / name)
        local_exes.append(dest)

    # Packed via UPX
    for src in local_exes:
        tmp = work / f"{src.stem}.to_pack.exe"
        shutil.copy2(src, tmp)
        before = tmp.stat().st_size
        proc = subprocess.run([upx, "-9", "-q", str(tmp)], check=False)
        after = tmp.stat().st_size if tmp.is_file() else 0
        packed_bytes = tmp.read_bytes() if tmp.is_file() else b""
        looks_upx = b"UPX" in packed_bytes[: min(len(packed_bytes), 4096)] or after != before
        if proc.returncode != 0 or not looks_upx:
            print(f"SKIP pack {src.name}: upx exit={proc.returncode} size {before}->{after}")
            continue
        shutil.copy2(tmp, packed / f"{src.stem}_upx.exe")
        print(f"packed → {packed / (src.stem + '_upx.exe')}")

    # High-entropy encrypted lookalikes (NOT malware)
    for src in local_exes:
        raw = src.read_bytes()
        keystream = os.urandom(len(raw))
        enc = bytes(a ^ b for a, b in zip(raw, keystream))
        out = ransom / f"{src.stem}_enc.bin"
        out.write_bytes(enc)
        print(f"encrypted → {out}")

    # Text/URL-heavy overlays on real PE
    urls = b"\r\n".join(
        [
            b"http://cdn.example.com/update?id=1",
            b"https://mirror.example.org/agent/payload.bin",
            b"GET /download/ HTTP/1.1",
            b"User-Agent: Mozilla/5.0",
            b"cmd.exe /c powershell -nop -w hidden",
            b"Software\\Microsoft\\Windows\\CurrentVersion\\Run",
        ]
        * 800
    )
    for src in local_exes:
        out = downloader / f"{src.stem}_urls.exe"
        out.write_bytes(src.read_bytes() + b"\x00\x00" + urls)
        print(f"downloader → {out}")

    print("\nDone. Train with:")
    print(
        "  python scripts/train.py --real-data data --mix-synthetic "
        "--per-class 40 --out artifacts/model.cbm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
