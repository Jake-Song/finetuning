#!/bin/bash
set -e

export UV_CACHE_DIR="/workspace/.cache/uv"
if [ ! -d "$UV_CACHE_DIR" ]; then
    mkdir -p "$UV_CACHE_DIR"
fi

echo "Environment setup complete. uv cache dir: $UV_CACHE_DIR"