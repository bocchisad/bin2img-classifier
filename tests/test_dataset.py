"""Tests for dataset ingest helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from bin2img.dataset import load_dataset_dir, summarize_dataset
from scripts.generate_synthetic_data import generate_sample


def test_load_benign_and_malware_layout(tmp_path: Path) -> None:
    benign = tmp_path / "benign"
    mal = tmp_path / "malware" / "Packed_PE"
    benign.mkdir(parents=True)
    mal.mkdir(parents=True)
    (benign / "a.exe").write_bytes(generate_sample("Benign_Code", seed=1, size=2048))
    (mal / "b.bin").write_bytes(generate_sample("Packed_PE", seed=2, size=2048))

    samples = load_dataset_dir(tmp_path)
    counts = summarize_dataset(samples)
    assert counts["Benign_Code"] == 1
    assert counts["Packed_PE"] == 1


def test_empty_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset_dir(tmp_path)
