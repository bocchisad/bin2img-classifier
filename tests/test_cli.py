"""Smoke tests for Stage 6 CLI entry point."""

from __future__ import annotations

from pathlib import Path

from main import build_parser, main
from scripts.generate_synthetic_data import generate_sample
from tests.test_parser import _craft_pe_with_pefile


def test_cli_help() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "analyze" in help_text
    assert "serve" in help_text
    assert "train" in help_text


def test_analyze_command(tmp_path: Path, capsys) -> None:
    pe = _craft_pe_with_pefile()
    assert pe is not None
    sample = tmp_path / "demo.exe"
    sample.write_bytes(pe)
    out = tmp_path / "out"
    code = main(
        [
            "analyze",
            str(sample),
            "-o",
            str(out),
            "--no-model",
            "--json",
        ]
    )
    assert code == 0
    assert (out / "demo_gray.png").is_file()
    assert (out / "demo_heatmap.png").is_file()
    assert (out / "demo_report.html").is_file()
    assert (out / "demo_report.json").is_file()
    captured = capsys.readouterr().out
    assert "format:" in captured
    assert "PE" in captured


def test_analyze_missing_file() -> None:
    assert main(["analyze", "/no/such/file.exe", "--no-model"]) == 1
