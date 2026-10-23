# syntax=docker/dockerfile:1

# ---------- base: shared toolchain and production dependency cache ----------
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/models \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev --extra ml

# ---------- dev: tests, lint and local tooling ----------
FROM base AS dev

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --all-extras

CMD ["pytest"]

# ---------- runtime: API and UI only ----------
FROM base AS runtime

COPY src/ ./src/
COPY api/ ./api/
COPY ui/ ./ui/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY data/ ./data/
COPY .streamlit/ ./.streamlit/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --extra ml

RUN mkdir -p /models \
    && useradd --create-home --uid 10001 app \
    && chown -R app:app /app /models

USER app

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["bash", "scripts/start_space.sh"]

# ---------- runtime-gpu: opt-in CUDA build for local GPU runs ----------
# Only built by the compose `gpu` profile; the default runtime stays CPU-only
# so the Hugging Face Space image (and CI) does not carry ~5 GB of CUDA wheels.
FROM runtime AS runtime-gpu

USER root

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --reinstall torch

USER app
