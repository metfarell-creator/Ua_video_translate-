#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

mkdir -p ".cache/huggingface"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

echo "[INFO] Base deps (no Torch)"
pip install -r requirements.txt --no-deps

PT_OK=
set +e
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121 && PT_OK=1
if [ -z "$PT_OK" ]; then
  pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118 && PT_OK=1 || true
fi
if [ -z "$PT_OK" ]; then
  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu && PT_OK=1 || true
fi
set -e
[ -z "$PT_OK" ] && echo "[ERROR] Torch install failed" && exit 1

echo "[INFO] Full deps"
pip install -r requirements.txt

export HF_HOME="$(pwd)/.cache/huggingface"
python tools/prefetch_models.py

echo "Done. Run: bash run_ui.sh"
