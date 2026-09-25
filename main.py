#!/usr/bin/env python3
"""bin2img-classifier CLI entry point.

Commands:
    python main.py analyze <filepath>   Terminal summary + PNG / HTML reports
    python main.py serve                Launch FastAPI UI / REST API
    python main.py train                Train CatBoost on synthetic data
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from bin2img.model import DEFAULT_MODEL_PATH
from bin2img.parser import BinaryParser
from bin2img.report import write_html_report, write_json_report
from bin2img.service import AnalysisService, build_risk_tags
from bin2img.visualizer import BinaryVisualizer


def _cmd_analyze(args: argparse.Namespace) -> int:
    path = Path(args.filepath)
    if not path.is_file():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 1

    from bin2img.constants import MAX_UPLOAD_BYTES

    raw = path.read_bytes()
    if len(raw) > MAX_UPLOAD_BYTES:
        print(
            f"error: file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
            file=sys.stderr,
        )
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or path.stem
    model_path = Path(args.model)

    parser = BinaryParser()
    visualizer = BinaryVisualizer()
    meta = parser.parse(raw)
    meta.file_path = str(path.resolve())
    render = visualizer.render(raw)
    gray_path = visualizer.save_png(render.grayscale, out_dir / f"{stem}_gray.png")
    heat_path = visualizer.save_png(render.heatmap, out_dir / f"{stem}_heatmap.png")

    if args.no_model:
        risk_tags = build_risk_tags(meta)
        classification = None
        risk = None
        analysis_id = None
        payload = {
            "filename": path.name,
            "metadata": meta.to_dict(),
            "classification": None,
            "risk_tags": risk_tags,
            "risk": None,
            "images": {
                "grayscale_png_base64": render.grayscale_base64(),
                "heatmap_png_base64": render.heatmap_base64(),
            },
        }
    else:
        db = out_dir / "bin2img.db"
        svc = AnalysisService(
            model_path=model_path if model_path.is_file() else None,
            db_path=db,
        )
        if model_path.is_file():
            svc.load_model()
        payload = svc.analyze_bytes(raw, path.name)
        classification = payload.get("classification")
        risk_tags = payload.get("risk_tags") or []
        risk = payload.get("risk")
        analysis_id = payload.get("analysis_id")

    print("=" * 60)
    print(f"file:     {path.resolve()}")
    print(f"format:   {meta.format.value}")
    print(f"arch:     {meta.architecture}")
    print(f"size:     {meta.file_size} bytes")
    print(f"entry:    {meta.entry_point}")
    print(f"entropy:  {meta.overall_entropy:.4f}  (alert={meta.high_entropy_alert})")
    print(f"valid:    {meta.is_valid_executable}")
    print(f"risk:     {', '.join(risk_tags) if risk_tags else '—'}")
    if risk:
        print(f"verdict:  {risk.get('verdict')}  score={risk.get('score')}/100")
    if classification and "label" in classification:
        unk = "  [UNKNOWN/LOW CONF]" if classification.get("is_unknown") else ""
        print(
            f"class:    {classification['label']}  "
            f"({classification.get('confidence', 0) * 100:.1f}%)"
            f"{unk}"
        )
        if classification.get("raw_label") and classification.get("is_unknown"):
            print(f"raw top:  {classification['raw_label']}")
    else:
        print("class:    (model not loaded)")
    if analysis_id:
        print(f"id:       {analysis_id}")
    print("-" * 60)
    print(f"{'section':<16} {'virt':>8} {'raw':>8} {'entropy':>8} {'exec':>4}")
    for sec in meta.sections:
        print(
            f"{sec.name[:16]:<16} {sec.virtual_size:8d} {sec.raw_size:8d} "
            f"{sec.entropy:8.4f} {'Y' if sec.is_executable else '':>4}"
        )
    print("-" * 60)
    print(f"saved:    {gray_path}")
    print(f"saved:    {heat_path}")
    html_path = write_html_report(payload, out_dir / f"{stem}_report.html")
    print(f"report:   {html_path}")
    if args.json:
        json_path = write_json_report(payload, out_dir / f"{stem}_report.json")
        print(f"json:     {json_path}")
    print("=" * 60)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    model_path = Path(args.model).resolve()
    os.environ["BIN2IMG_MODEL"] = str(model_path)

    print(f"Serving bin2img UI on http://{args.host}:{args.port}")
    print(f"Model: {model_path} ({'found' if model_path.is_file() else 'MISSING'})")
    if os.environ.get("BIN2IMG_API_KEY", "").strip():
        print("Auth: BIN2IMG_API_KEY is set (X-API-Key required)")

    uvicorn.run(
        "bin2img.api:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=bool(args.reload),
        reload_dirs=(
            [str(Path(__file__).resolve().parent / "bin2img")] if args.reload else None
        ),
    )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from scripts.train import main as train_main

    argv = [
        "--per-class",
        str(args.per_class),
        "--iterations",
        str(args.iterations),
        "--out",
        str(args.out),
        "--seed",
        str(args.seed),
        "--confidence-threshold",
        str(args.confidence_threshold),
        "--margin-threshold",
        str(args.margin_threshold),
    ]
    if args.write_synthetic:
        argv.extend(["--write-synthetic", str(args.write_synthetic)])
    if args.real_data:
        argv.extend(["--real-data", str(args.real_data)])
    if args.mix_synthetic:
        argv.append("--mix-synthetic")
    if args.verbose:
        argv.append("--verbose")
    return train_main(argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bin2img",
        description="Binary Visualizer & Malware Family Classifier",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="Analyze a binary and save PNG images")
    p_analyze.add_argument("filepath", help="Path to .exe / .dll / .elf / .bin")
    p_analyze.add_argument(
        "-o",
        "--output-dir",
        default="artifacts",
        help="Directory for PNG / JSON / HTML outputs (default: artifacts)",
    )
    p_analyze.add_argument("--stem", default=None, help="Output filename stem")
    p_analyze.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to model.cbm",
    )
    p_analyze.add_argument(
        "--no-model",
        action="store_true",
        help="Skip classification (visuals + metadata only)",
    )
    p_analyze.add_argument(
        "--json",
        action="store_true",
        help="Also write a JSON report next to the PNGs",
    )
    p_analyze.set_defaults(func=_cmd_analyze)

    p_serve = sub.add_parser("serve", help="Launch FastAPI web UI + REST API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to model.cbm",
    )
    p_serve.add_argument(
        "--reload",
        action="store_true",
        help="Auto-reload on bin2img/ changes (excludes .venv)",
    )
    p_serve.set_defaults(func=_cmd_serve)

    p_train = sub.add_parser("train", help="Train CatBoost (synthetic and/or real data)")
    p_train.add_argument("--per-class", type=int, default=50)
    p_train.add_argument("--iterations", type=int, default=350)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    p_train.add_argument(
        "--write-synthetic",
        type=Path,
        default=None,
        help="Also dump synthetic .bin samples to this directory",
    )
    p_train.add_argument(
        "--real-data",
        type=Path,
        default=None,
        help="Real dataset root (data/benign + data/malware/<Family>)",
    )
    p_train.add_argument(
        "--mix-synthetic",
        action="store_true",
        help="Mix synthetic samples when using --real-data",
    )
    p_train.add_argument("--confidence-threshold", type=float, default=0.55)
    p_train.add_argument("--margin-threshold", type=float, default=0.12)
    p_train.add_argument("--verbose", action="store_true")
    p_train.set_defaults(func=_cmd_train)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
