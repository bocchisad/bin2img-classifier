# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BIN2IMG_MODEL=/app/artifacts/model.cbm \
    BIN2IMG_DB=/app/artifacts/bin2img.db

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY bin2img ./bin2img
COPY scripts ./scripts
COPY main.py ./

# Demo model (synthetic) — replace via volume for real use.
RUN mkdir -p artifacts /data \
    && python scripts/train.py --per-class 30 --iterations 80 --out artifacts/model.cbm \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin bin2img \
    && chown -R bin2img:bin2img /app /data

USER bin2img

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health')" || exit 1

CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8000", "--model", "artifacts/model.cbm"]
