#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TRAINER_PROJECT="$ROOT_DIR/envs/trainer"
WORK_DIR="${TMPDIR:-/tmp}/flash-attention-fa3"
REPO_DIR="$WORK_DIR/flash-attention"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"
export MAX_JOBS="${MAX_JOBS:-4}"

mkdir -p "$WORK_DIR"

if ! command -v git >/dev/null 2>&1; then
    echo "git is required to install FlashAttention-3."
    exit 1
fi

if ! command -v ninja >/dev/null 2>&1; then
    echo "ninja is required to build FlashAttention-3. Install it with setup/setup.sh first."
    exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc is required to build FlashAttention-3."
    exit 1
fi

if [[ -d "$REPO_DIR/.git" ]]; then
    git -C "$REPO_DIR" fetch --depth=1 origin main
    git -C "$REPO_DIR" reset --hard origin/main
else
    git clone --depth=1 https://github.com/Dao-AILab/flash-attention.git "$REPO_DIR"
fi

cd "$REPO_DIR/hopper"
uv run --project "$TRAINER_PROJECT" python setup.py install

echo "FlashAttention-3 install complete in trainer environment."
