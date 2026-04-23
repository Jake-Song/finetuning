#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TRAINER_PROJECT="$ROOT_DIR/envs/trainer"
TRAINER_VENV="$TRAINER_PROJECT/.venv"
TRAINER_PYTHON="$TRAINER_VENV/bin/python"
TRAINER_TORCHRUN="$TRAINER_VENV/bin/torchrun"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <python-args...>"
    echo "Example: $0 python scripts/if_train.py --dry-run"
    exit 1
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

if [[ ! -x "$TRAINER_PYTHON" ]]; then
    echo "Trainer environment is missing. Run ./setup/bootstrap_envs.sh first."
    exit 1
fi

if ! "$TRAINER_PYTHON" -c "import torch" >/dev/null 2>&1; then
    echo "Trainer environment exists but is incomplete (missing torch)."
    echo "Re-run ./setup/bootstrap_envs.sh to sync envs/trainer before launching training."
    exit 1
fi

cd "$ROOT_DIR"

case "$1" in
    python|python3)
        shift
        exec "$TRAINER_PYTHON" "$@"
        ;;
    torchrun)
        shift
        if [[ ! -x "$TRAINER_TORCHRUN" ]]; then
            echo "torchrun is missing from the trainer environment. Re-run ./setup/bootstrap_envs.sh."
            exit 1
        fi
        exec "$TRAINER_TORCHRUN" "$@"
        ;;
    *)
        exec uv --project "$TRAINER_PROJECT" run "$@"
        ;;
esac
