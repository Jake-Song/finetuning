#!/bin/bash
set -e

apt update && apt install -y tmux python3-dev

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv sync

echo "Setup complete."
