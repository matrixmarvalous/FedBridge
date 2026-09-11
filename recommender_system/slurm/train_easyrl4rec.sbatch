#!/bin/bash
# Codex-added 2026-08-14: 8-rank, 16-core Siyuan-1 release template.
# Codex-modified 2026-08-19: activate the environment created by environment.yml.
#SBATCH --job-name=fb-easyrl4rec
#SBATCH --partition=64c512g
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=.codex_hpc/jobs/%x-%j.out
#SBATCH --error=.codex_hpc/jobs/%x-%j.err

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

# Codex-modified 2026-08-14: use this template for either a scientific run or
# a non-scientific one-epoch structural check.
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
