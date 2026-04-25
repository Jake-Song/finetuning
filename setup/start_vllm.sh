#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

MODEL="${1:-Qwen/Qwen3.6-35B-A3B-FP8}"
MAX_MODEL_LEN="${2:-20480}"
HOST="${3:-0.0.0.0}"
PORT="${4:-8000}"
SYNC_BACKEND="${5:-nccl}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

cd "$ROOT_DIR"
VLLM_SERVER_DEV_MODE=1 uv --project "$ROOT_DIR/envs/vllm-server" run vllm serve "$MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --host "$HOST" \
    --port "$PORT" \
    --enforce-eager \
    --reasoning-parser deepseek_r1 \
    --weight-transfer-config "{\"backend\": \"$SYNC_BACKEND\"}" \
    --load-format dummy
