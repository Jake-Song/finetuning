#!/bin/bash
set -e

cd "$(dirname "$0")/.."

DRY_RUN=""
CONFIG=""
RESUME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)  DRY_RUN="--dry-run"; shift ;;
        --config)   CONFIG="--config $2"; shift 2 ;;
        --resume)   RESUME="--resume-from-checkpoint $2"; shift 2 ;;
        *)          echo "Unknown arg: $1"; exit 1 ;;
    esac
done

uv run python grpo_train.py $DRY_RUN $CONFIG $RESUME
