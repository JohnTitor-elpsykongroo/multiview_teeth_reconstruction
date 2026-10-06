#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
test "$(uname -s)" = Linux || { echo 'Run inside Ubuntu/WSL2.' >&2; exit 1; }
test ! -e .venv || { echo '.venv already exists; refusing to replace it.' >&2; exit 1; }
# Accept either system Python or a Linux Conda interpreter, without changing base.
is_python310() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.platform == "linux" and sys.version_info[:2] == (3, 10) else 1)' >/dev/null 2>&1
}
if [ -n "${PYTHON_BIN:-}" ]; then
  is_python310 "$PYTHON_BIN" || { echo 'PYTHON_BIN must point to Linux Python 3.10.' >&2; exit 1; }
else
  for candidate in python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && is_python310 "$candidate"; then
      PYTHON_BIN=$(command -v "$candidate")
      break
    fi
  done
fi
if [ -z "${PYTHON_BIN:-}" ]; then
  echo 'Linux Python 3.10 was not found. With Conda: conda create -n dental-bootstrap310 python=3.10 pip; conda activate dental-bootstrap310' >&2
  echo 'On Ubuntu 22.04 you can instead install python3.10 and python3.10-venv using apt.' >&2
  exit 1
fi
"$PYTHON_BIN" -c 'import venv, ensurepip' || { echo 'Install python3.10-venv, or use a Conda Python 3.10 interpreter.' >&2; exit 1; }
echo "Creating project .venv with $PYTHON_BIN"
"$PYTHON_BIN" -m venv --copies .venv
source .venv/bin/activate
python -m pip install pip==25.3
python -m pip install --only-binary=:all: torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install --only-binary=:all: -r configs/training_handoff_v1/requirements.txt
python -m pip check
python -m pip freeze > .venv/installed-requirements.txt
echo 'Environment installed. Next: bash scripts/run_training_wsl.sh preflight'
