#!/bin/bash
set -e

apt update && apt install -y git ninja-build tmux python3-dev vim

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "Setup complete."
