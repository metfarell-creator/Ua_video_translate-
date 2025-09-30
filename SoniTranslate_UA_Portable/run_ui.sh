#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

source .venv/bin/activate
export HF_HOME="${HF_HOME:-$(pwd)/.cache/huggingface}"
python ui/gradio_app.py
