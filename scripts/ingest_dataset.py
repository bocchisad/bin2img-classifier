#!/usr/bin/env python3
"""Validate / summarize a real binary dataset layout.

Layout::

    data/
      benign/**                 → Benign_Code
      malware/<FamilyName>/**   → FamilyName

Example::

    python scripts/ingest_dataset.py --root data --write-manifest artifacts/dataset_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bin2img.dataset import load_dataset_dir, summarize_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest / summarize binary dataset")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data"),
        help="Dataset root (default: data/)",
    )
    parser.add_argument(
        "--write-manifest",
        type=Path,
        default=None,
        help="Optional JSON manifest path",
    )
    args = parser.parse_args(argv)

    samples = load_dataset_dir(args.root)
    counts = summarize_dataset(samples)
    total = sum(counts.values())
    print(f"Root: {args.root.resolve()}")
    print(f"Total samples: {total}")
    for label, n in counts.items():
        print(f"  {label}: {n}")

    if args.write_manifest is not None:
        manifest = [
            {"path": path, "label": label, "size": len(raw)}
            for raw, label, path in samples
        ]
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Manifest → {args.write_manifest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
