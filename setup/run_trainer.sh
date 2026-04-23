#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <python-args...>"
    echo "Example: $0 python scripts/if_train.py --dry-run"
    exit 1
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

cd "$ROOT_DIR"
uv --project "$ROOT_DIR/envs/trainer" run "$@"
