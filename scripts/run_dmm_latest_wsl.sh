#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
test "$(uname -s)" = Linux || { echo 'Run inside WSL/Linux.' >&2; exit 1; }
runtime=${DMM_PYTHON:-"$PWD/.venv/bin/python"}
test -x "$runtime" || { echo 'Set DMM_PYTHON to an installed Linux Python with CUDA PyTorch.' >&2; exit 1; }
mode=${1:-validate}; arch=${2:-upper}
case "$arch" in upper|lower) ;; *) echo 'arch must be upper or lower' >&2; exit 1;; esac
shift "$(( $# >= 2 ? 2 : $# ))"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
config="configs/dmm_latest_v2/${arch}.json"
case "$mode" in
  validate) exec "$runtime" third_party/DMM/dmm_cli.py validate-latest --config "$config" "$@" ;;
  train) exec "$runtime" third_party/DMM/dmm_cli.py train-latest --config "$config" --device cuda "$@" ;;
  *) echo 'Usage: bash scripts/run_dmm_latest_wsl.sh validate|train upper|lower [CLI options]' >&2; exit 1 ;;
esac
