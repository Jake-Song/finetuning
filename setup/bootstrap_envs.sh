#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TRAINER_PROJECT="$ROOT_DIR/envs/trainer"
SERVER_PROJECT="$ROOT_DIR/envs/vllm-server"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

echo "Syncing trainer environment..."
uv sync --project "$TRAINER_PROJECT"

echo "Syncing vLLM server environment..."
uv sync --project "$SERVER_PROJECT"

echo "Installing trainer-side vLLM sync shim (no dependencies)..."
uv pip install --python "$TRAINER_PROJECT/.venv/bin/python" --no-deps "vllm==0.19.0"

echo "Bootstrap complete."
