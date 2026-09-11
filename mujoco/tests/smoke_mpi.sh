#!/usr/bin/env bash
# Codex-added 2026-08-14: explicit-seed end-to-end smoke command.
set -euo pipefail

: "${SEED:?Set a non-paper synthetic SEED for this smoke test}"
ENV_NAME="${ENV_NAME:-HalfCheetah-v5}"
METHOD="${METHOD:-fedbridge_singlecritic}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/fedbridge_mujoco_smoke}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${RELEASE_DIR}"

exec mpirun -n 8 python launcher.py \
  --env-name "${ENV_NAME}" \
  --method "${METHOD}" \
  --seed "${SEED}" \
  --output-root "${OUTPUT_ROOT}" \
  --smoke-test

