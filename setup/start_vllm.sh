#!/bin/bash
set -e

cd "$(dirname "$0")/.."

MODEL="${1:-Qwen/Qwen3-4B-Thinking-2507-FP8}"
MAX_MODEL_LEN="${2:-8192}"
HOST="${3:-0.0.0.0}"
PORT="${4:-8000}"
SYNC_BACKEND="${5:-nccl}"

VLLM_SERVER_DEV_MODE=1 uv run vllm serve "$MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --host "$HOST" \
    --port "$PORT" \
    --enforce-eager \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 \
    --weight-transfer-config "{\"backend\": \"$SYNC_BACKEND\"}" \
    --load-format dummy
