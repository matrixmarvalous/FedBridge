# MuJoCo experiments

<!-- Codex-modified 2026-09-10: replace the internal release notes with a
concise usage guide; the previous version is preserved in the private archive. -->

This directory contains the FedBridge training-performance and held-out
zero-shot experiments for `HalfCheetah-v5`, `Hopper-v4`, and `Humanoid-v4`.
Each experiment uses eight MPI ranks.

## Installation

```bash
conda env create -f environment.yml
conda activate fedbridge-mujoco
```

## Algorithms

The main comparison methods are:

| Algorithm | Launcher key |
| --- | --- |
| FedBridge | `fedbridge_singlecritic` |
| Individual PPO | `individual` |
| Push-Avg | `push_avg` |
| FedAvg | `fedavg` |
| PerFedDC | `perfeddc` |
| pFedMe | `pfedme` |

## Training and zero-shot evaluation

The launcher trains one method on one environment and automatically evaluates
the final policies on held-out environment clusters D, E, and F:

```bash
SEED=YOUR_INTEGER_SEED
mpirun -n 8 python launcher.py \
  --env-name HalfCheetah-v5 \
  --method fedbridge_singlecritic \
  --seed "${SEED}"
```

Substitute another environment or method key to run a different condition. A
small end-to-end check is available by adding `--smoke-test`.

## Re-evaluating checkpoints

```bash
SEED=YOUR_INTEGER_SEED
mpirun -n 8 python evaluate_mujoco.py \
  --env-name HalfCheetah-v5 \
  --method fedbridge_singlecritic \
  --seed "${SEED}" \
  --run-dir results/halfcheetah_v5__fedbridge_singlecritic__seed_${SEED}
```

## Plotting

After completing the desired runs:

```bash
python scripts/plot_results.py \
  --results-root results \
  --output-dir figures
```

The experiment settings are defined in `experiment_spec.py` and summarized in
[`docs/HYPERPARAMETERS.md`](docs/HYPERPARAMETERS.md).
