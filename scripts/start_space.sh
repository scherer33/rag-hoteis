#!/usr/bin/env bash
# Single-container entrypoint for Hugging Face Spaces (Docker SDK).
# The API listens on localhost and the Streamlit UI is the public port.
set -euo pipefail

export QDRANT_URL="${QDRANT_URL:-:memory:}"
export API_URL="${API_URL:-http://127.0.0.1:8000}"

uvicorn api.main:create_default_app --factory --host 127.0.0.1 --port 8000 &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true' EXIT

exec streamlit run ui/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true
