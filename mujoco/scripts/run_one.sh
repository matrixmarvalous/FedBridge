#!/usr/bin/env bash
# Codex-added 2026-08-14: thin portable wrapper around the copied MPI launcher.
set -euo pipefail

: "${ENV_NAME:?Set ENV_NAME to HalfCheetah-v5, Hopper-v4, or Humanoid-v4}"
: "${METHOD:?Set METHOD to a launcher method key}"
: "${SEED:?Set SEED to your own non-negative integer}"

MPI_RANKS="${MPI_RANKS:-8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results}"
if [[ "${MPI_RANKS}" != "8" ]]; then
  echo "FedBridge MuJoCo requires exactly eight MPI ranks." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${RELEASE_DIR}"

exec mpirun -n "${MPI_RANKS}" python launcher.py \
  --env-name "${ENV_NAME}" \
  --method "${METHOD}" \
  --seed "${SEED}" \
  --output-root "${OUTPUT_ROOT}" \
  "$@"

