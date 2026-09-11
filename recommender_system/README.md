# Recommender-system experiments

<!-- Codex-modified 2026-09-10: replace the internal release notes with a
concise usage guide; the previous version is preserved in the private archive. -->

This directory contains the FedBridge experiments built on EasyRL4Rec for
`MovieLensEnv-v0`. The supported settings are `hetero5` and `homo3`, and each
experiment uses eight MPI ranks.

## Installation

```bash
conda env create -f environment.yml
conda activate fedbridge-easyrl4rec
```

The repository includes the required EasyRL4Rec, Tianshou, and DeepCTR-Torch
source code. The prepared MovieLens and DeepFM assets must be staged separately;
their directory layout is documented in [`docs/DATA.md`](docs/DATA.md).

Validate the asset layout before training:

```bash
python scripts/validate_assets.py --asset-root /path/to/staged-assets
```

## Algorithms

| Algorithm | Launcher key |
| --- | --- |
| FedBridge | `fedbridge` |
| Individual PPO | `individual` |
| Push-Pull | `pushpull` |
| FedAvg | `fedavg` |
| PerFedDC | `perfeddc` |
| pFedMe | `pfedme` |

## Training

```bash
SEED=YOUR_INTEGER_SEED
python launcher.py \
  --scene hetero5 \
  --method fedbridge \
  --seed "${SEED}" \
  --run-id run-a \
  --asset-root /path/to/staged-assets \
  --output-root results
```

Use `homo3` for the second setting or substitute another method key. Add
`--smoke-test` for a one-epoch end-to-end check, or `--dry-run` to inspect the
constructed command without launching MPI.

## Aggregation and plotting

```bash
python scripts/aggregate_results.py \
  --log-root results/MovieLensEnv-v0/PPO/logs \
  --output-dir results/aggregated

python scripts/plot_results.py \
  --input-dir results/aggregated \
  --output-dir figures
```

The experiment settings and runner map are defined in `experiment_spec.py` and
summarized in [`docs/HYPERPARAMETERS.md`](docs/HYPERPARAMETERS.md).
