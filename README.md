# bin2img-classifier

Binary Visualizer & Malware Family Classifier.

Converts executables (`.exe`, `.dll`, `.elf`, `.macho` / raw blobs) into grayscale byteplots and thermal entropy heatmaps, extracts GLCM/LBP + section-entropy features, and classifies family patterns with CatBoost — plus heuristic rules, risk score / verdict, analysis history, async jobs, and HTML reports.

> Demo classifier can be trained on **synthetic** byte patterns and/or a safe demo set (PuTTY + UPX overlays). Not a replacement for a full AV/EDR stack.

## Features

- PE / ELF / Mach-O parsing (`pefile` + `lief`) with Shannon entropy
- Adaptive-width byteplot + blue→red thermal entropy map (OpenCV)
- GLCM + LBP texture features (`scikit-image`) + byte-stat / overlay / URL / API signals
- CatBoost multi-class family prediction with Unknown reject
- Heuristic rules + unified risk score (0–100) and verdict
- FastAPI REST + interactive HTML UI (history, async, compare, reports)
- Optional `X-API-Key` auth (`BIN2IMG_API_KEY`)
- SQLite analysis history + background jobs
- CLI (`analyze` / `serve` / `train`), Docker, Compose, CI

## Project layout

```
bin2img-classifier/
├── bin2img/
│   ├── parser.py / visualizer.py / features.py / model.py
│   ├── rules.py / risk.py / explain.py / service.py
│   ├── store.py / jobs.py / auth.py / report.py
│   ├── api.py + static/index.html
├── scripts/          # train, synthetic, demo dataset
├── tests/
├── main.py
├── Dockerfile / docker-compose.yml
└── .github/workflows/ci.yml
```

## Quick start (local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python main.py train --per-class 40 --out artifacts/model.cbm
python main.py analyze sample.exe -o artifacts --json
python main.py serve --port 8000
# optional: python main.py serve --reload
```

Python **3.11–3.13** recommended (3.14 needs pinned wheels from `requirements.txt`).

## CLI

```bash
python main.py analyze <filepath> [-o artifacts] [--model …] [--json] [--no-model]
python main.py serve [--host 127.0.0.1] [--port 8000] [--model …] [--reload]
python main.py train [--per-class 40] [--iterations 150] [--real-data data] [--mix-synthetic]
```

`analyze` writes PNGs, an HTML report, and optional JSON. With a model loaded it also persists to SQLite under the output dir.

## REST API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Interactive upload UI |
| `GET` | `/api/v1/health` | Liveness + model status |
| `POST` | `/api/v1/analyze` | Sync analysis |
| `POST` | `/api/v1/analyze/async` | Queue job → `{job_id}` |
| `GET` | `/api/v1/jobs/{id}` | Job status |
| `GET` | `/api/v1/analyses` | History list |
| `GET` | `/api/v1/analyses/{id}` | Stored analysis |
| `GET` | `/api/v1/analyses/{id}/report.html` | HTML report |
| `POST` | `/api/v1/compare` | Compare two uploads |

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/analyze \
  -F "file=@sample.exe;type=application/octet-stream" | jq .
```

Optional auth: set `BIN2IMG_API_KEY` and send header `X-API-Key` on **all** API routes except `GET /` and `GET /api/v1/health`. The UI has an API key field (stored in `localStorage`).

Response includes: `metadata`, `risk_tags`, `risk` (score/verdict/rules), `explanation`, Base64 PNGs, `classification`, `features`, `analysis_id`, `sha256`.

### Risk tags / verdict

| Tag / field | Meaning |
|-------------|---------|
| `VALID_EXECUTABLE` | PE/ELF/Mach-O parsed successfully |
| `HIGH_ENTROPY_PACKED` | File/section entropy > 7.2 |
| `SUSPICIOUS_SECTION` | Packer-like section name or odd virt/raw sizes |
| `LOW_CONFIDENCE_UNKNOWN` | Model rejected to Unknown |
| `VERDICT_*` | Product verdict tag |
| `risk.score` | 0–100 combined model + rules |
| `risk.verdict` | Benign / Suspicious / Malicious / Unknown |

## Synthetic families

| Label | Pattern |
|-------|---------|
| `Benign_Code` | Low/medium entropy, structured opcode-like bytes |
| `Ransomware_Encrypted` | PE with dominant high-entropy section |
| `Packed_PE` | PE stub + high-entropy `UPX*` sections |
| `Downloader_TextHeavy` | ASCII / URL-heavy low entropy |

## Real / demo data

See [`data/README.md`](data/README.md). Safe demo (no live malware):

```bash
brew install upx   # optional
python scripts/fetch_demo_dataset.py
python scripts/train.py --real-data data --mix-synthetic --per-class 40 \
  --out artifacts/model.cbm
```

### Unknown reject

If confidence `< 0.55` or margin `< 0.12` → `label: "Unknown"`, `is_unknown: true`.

## Docker / Compose

```bash
docker build -t bin2img-classifier .
docker run --rm -p 8000:8000 bin2img-classifier

# or
docker compose up --build
```

Open http://127.0.0.1:8000. Persist DB via the Compose volume (`BIN2IMG_DB=/data/bin2img.db`).

## Tests / CI

```bash
pytest -q
```

GitHub Actions runs train + pytest on Python 3.11 and 3.12.

## License / disclaimer

For defensive research and education only. Do not use to develop or deploy malware. Handle real samples in an isolated lab.
