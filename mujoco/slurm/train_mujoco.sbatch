#!/usr/bin/env bash
# Codex-added 2026-08-14: Siyuan-1 8-client / 16-core Slurm template.
# Codex-modified 2026-08-18: activate the public FedBridge environment name.
# Codex-modified 2026-08-19: forward the optional structural smoke-test flag.
#SBATCH --job-name=fedbridge-mujoco
#SBATCH --partition=64c512g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=.codex_hpc/jobs/%x-%j.out
#SBATCH --error=.codex_hpc/jobs/%x-%j.err

set -euo pipefail

: "${ENV_NAME:?Pass ENV_NAME with sbatch --export}"
: "${METHOD:?Pass METHOD with sbatch --export}"
: "${SEED:?Pass your own non-negative integer SEED with sbatch --export}"

cd "${SLURM_SUBMIT_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export I_MPI_FABRICS="${I_MPI_FABRICS:-shm}"

module load miniconda3/4.10.3
if [[ -f "${HOME}/.bashrc" ]]; then
  source "${HOME}/.bashrc"
fi
source activate fedbridge-mujoco
which python
python -V

extra_args=()
if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  extra_args+=(--smoke-test)
fi

mpirun -n "${SLURM_NTASKS:-8}" python launcher.py \
  --env-name "${ENV_NAME}" \
  --method "${METHOD}" \
  --seed "${SEED}" \
  --output-root "${OUTPUT_ROOT:-results}" \
  "${extra_args[@]}"
