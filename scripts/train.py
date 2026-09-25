#!/usr/bin/env python3
"""Train CatBoost malware-family classifier.

Sources (can combine)::

    # synthetic only (default)
    python scripts/train.py

    # real dataset
    python scripts/train.py --real-data data

    # synthetic + real
    python scripts/train.py --real-data data --mix-synthetic --per-class 40

Real layout: see data/README.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bin2img.dataset import load_dataset_dir, merge_samples, summarize_dataset  # noqa: E402
from bin2img.model import MalwareFamilyClassifier  # noqa: E402
from scripts.generate_synthetic_data import (  # noqa: E402
    CLASS_LABELS,
    generate_dataset,
    iter_dataset_bytes,
)


def _load_legacy_class_dirs(data_dir: Path) -> list[tuple[bytes, str]]:
    samples: list[tuple[bytes, str]] = []
    for label in CLASS_LABELS:
        class_dir = data_dir / label
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".bin", ".exe", ".dll", ""}:
                try:
                    samples.append((path.read_bytes(), label))
                except OSError:
                    continue
    if not samples:
        # Fallback to generic ingest
        loaded = load_dataset_dir(data_dir)
        return merge_samples(loaded)
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train bin2img CatBoost classifier")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Legacy/synthetic per-class directory (Benign_Code/, ...)",
    )
    parser.add_argument(
        "--real-data",
        type=Path,
        default=None,
        help="Real dataset root (data/benign + data/malware/<Family>)",
    )
    parser.add_argument(
        "--mix-synthetic",
        action="store_true",
        help="When using --real-data, also mix in synthetic samples",
    )
    parser.add_argument(
        "--write-synthetic",
        type=Path,
        default=None,
        help="Also write synthetic dataset to this directory before training",
    )
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=350)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.07)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.55,
        help="Below this → Unknown",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.12,
        help="Best−second probability margin; below → Unknown",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/model.cbm"),
        help="Output CatBoost model path",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.write_synthetic is not None:
        generate_dataset(
            per_class=args.per_class,
            output_dir=args.write_synthetic,
            seed=args.seed,
        )
        print(f"Synthetic dataset → {args.write_synthetic.resolve()}")

    samples: list[tuple[bytes, str]] = []

    if args.real_data is not None:
        real = load_dataset_dir(args.real_data)
        samples.extend(merge_samples(real))
        print(f"Real data {args.real_data}: {summarize_dataset(real)}")
        if args.mix_synthetic:
            synth = iter_dataset_bytes(per_class=args.per_class, seed=args.seed)
            samples.extend(synth)
            print(f"Mixed +{len(synth)} synthetic samples")
    elif args.data_dir is not None:
        samples = _load_legacy_class_dirs(args.data_dir)
        print(f"Loaded {len(samples)} samples from {args.data_dir}")
    else:
        samples = iter_dataset_bytes(per_class=args.per_class, seed=args.seed)
        print(f"Generated {len(samples)} in-memory synthetic samples")

    if len(samples) < 4:
        print("error: need at least 4 samples to train", file=sys.stderr)
        return 1

    labels = {lab for _, lab in samples}
    if len(labels) < 2:
        print(f"error: need ≥2 classes, got {labels}", file=sys.stderr)
        return 1

    clf = MalwareFamilyClassifier(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        random_seed=args.seed,
        verbose=args.verbose,
        confidence_threshold=args.confidence_threshold,
        margin_threshold=args.margin_threshold,
    )
    result = clf.train_from_bytes(
        samples,
        model_path=args.out,
        labels_path=args.out.parent / "labels.json",
        metrics_path=args.out.parent / "metrics.json",
        config_path=args.out.parent / "model_config.json",
    )

    print(f"Saved model → {result.model_path.resolve()}")
    print(f"Config     → {result.config_path}")
    print(f"Classes: {', '.join(result.class_names)}")
    print(
        f"accuracy={result.metrics['accuracy']:.3f}  "
        f"f1_macro={result.metrics['f1_macro']:.3f}  "
        f"T={result.metrics['probability_temperature']:.3f}  "
        f"unknown_rate={result.metrics['unknown_rate_on_test']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
