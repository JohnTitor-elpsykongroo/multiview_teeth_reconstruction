#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/third_party/DMM${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
stamp=$(date -u +%Y%m%dT%H%M%SZ)
mode=${1:-preflight}
mkdir -p runs/logs
# Keep stderr and pipeline failures, including failures before the trainer starts.
exec > >(tee "runs/logs/${mode}_${2:-all}_${stamp}.log") 2>&1
case "$mode" in
  preflight)
    python scripts/verify_training_package.py --data-inventory
    python scripts/training_target_preflight.py --output "runs/target_preflight_$stamp"
    ;;
  smoke|pilot|formal)
    arch=${2:?Specify upper or lower}
    case "$arch" in upper|lower) ;; *) echo 'arch must be upper or lower' >&2; exit 1;; esac
    # Require an actual successful target preflight bound to the current source.
    proof=${3:?Pass the target preflight report.json}
    python scripts/verify_training_package.py --preflight "$proof"
    kind=$mode
    if [ "$mode" = formal ]; then
      kind=formal_candidate
      echo 'Formal training requested explicitly. Pilot quality review is still your responsibility.'
    fi
    python third_party/DMM/dmm_cli.py train --config "configs/training_handoff_v1/${arch}_${kind}.json" \
      --output "runs/${arch}_${mode}_$stamp" --device cuda
    ;;
  *) echo 'Usage: bash scripts/run_training_wsl.sh preflight | smoke|pilot|formal upper|lower <preflight/report.json>' >&2; exit 1;;
esac
