#!/bin/bash
set -e

cd "$(dirname "$0")/.."

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
MAX_MODEL_LEN="${2:-8192}"
GPU_UTIL="${3:-0.4}"

uv run trl vllm-serve \
    --model "$MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --host 0.0.0.0 \
    --port 8000
