#!/bin/bash

set -euo pipefail

: "${SCENE:?export SCENE=hetero5 or homo3}"
: "${METHOD:?export a public method key}"
: "${SEED:?export a user-selected integer seed}"
: "${RUN_ID:?export a seed-free opaque run label}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR}"
mkdir -p .codex_hpc/jobs

module load miniconda3/4.10.3
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi
source activate fedbridge-easyrl4rec
which python
python -V

asset_root="${ASSET_ROOT:-${SLURM_SUBMIT_DIR}}"
output_root="${OUTPUT_ROOT:-${SLURM_SUBMIT_DIR}/results}"

extra_args=()
if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
    extra_args+=(--smoke-test)
fi

python launcher.py \
    --scene "${SCENE}" \
    --method "${METHOD}" \
    --seed "${SEED}" \
    --run-id "${RUN_ID}" \
    --asset-root "${asset_root}" \
    --output-root "${output_root}" \
    --device cpu \
    "${extra_args[@]}"
