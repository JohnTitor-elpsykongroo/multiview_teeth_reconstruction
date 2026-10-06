#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
test "$(uname -s)" = Linux || { echo 'Run inside Ubuntu/WSL2.' >&2; exit 1; }
test ! -e .venv || { echo '.venv already exists; refusing to replace it.' >&2; exit 1; }
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r configs/training_handoff_v1/requirements.txt
python -m pip check
python -m pip freeze > .venv/installed-requirements.txt
echo 'Environment installed. Next: bash scripts/run_training_wsl.sh preflight'
