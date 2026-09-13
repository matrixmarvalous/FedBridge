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
source code. Run the following commands from `recommender_system/` on Linux
with OpenMPI. The launcher starts all eight MPI ranks; do not wrap it in
another `mpirun` command.

## Download the representative-experiment assets

The [hetero5 asset release](https://github.com/matrixmarvalous/FedBridge/releases/tag/recommender-hetero5-v1)
provides 54 runtime files (1.46 GB uncompressed): all three MovieLens user
groups, their processed data and three five-model DeepFM ensembles. The
repository is currently private, so downloading requires repository access.
Authenticate GitHub CLI with your own account before running:

```bash
gh release download recommender-hetero5-v1 \
  --repo matrixmarvalous/FedBridge \
  --pattern 'fedbridge-hetero5-assets-v1.tar.gz*' \
  --dir assets
(cd assets && sha256sum -c fedbridge-hetero5-assets-v1.tar.gz.sha256)
tar -xzf assets/fedbridge-hetero5-assets-v1.tar.gz -C assets
python scripts/validate_assets.py \
  --asset-root assets/fedbridge-hetero5-assets \
  --scene hetero5 \
  --verify-manifest docs/hetero5_asset_manifest.csv
```

Alternatively, download the archive and its `.sha256` file from the release
page into `assets/`, then run the checksum, extraction and validation commands.
Only load pickle/PyTorch files obtained from a trusted source. The asset
inventory, dataset terms and reproduction scope are in [`docs/DATA.md`](docs/DATA.md).

## Algorithms

| Algorithm | Launcher key |
| --- | --- |
| FedBridge | `fedbridge` |
| Individual PPO | `individual` |
| Push-Avg | `push_avg` |
| FedAvg | `fedavg` |
| PerFedDC | `perfeddc` |
| pFedMe | `pfedme` |

## Run one complete representative experiment

The example uses base seed `0` as a documented representative choice, not as
a claim about the original paper seeds. It keeps the full `hetero5` FedBridge
configuration: eight clients, A/B/C user groups, 100 epochs and the training
and evaluation settings in `experiment_spec.py`.

```bash
SEED=0
python launcher.py \
  --scene hetero5 \
  --method fedbridge \
  --seed "${SEED}" \
  --run-id representative \
  --asset-root assets/fedbridge-hetero5-assets \
  --output-root results
```

Evaluation runs during training and is recorded in the eight rank logs under
`results/MovieLensEnv-v0/PPO/logs/`. A successful full run produces
`RUN_STATUS.json` with `status: completed`, `final_common_epoch: 100`, and
`mpi_ranks: 8`. The default configuration does not save policy checkpoints.

For a quick environment check first, run the same command with `--smoke-test`,
`--run-id environment-check`, and `--output-root smoke-results`. This uses
reduced settings for one epoch and is not a complete scientific experiment.
Keep these logs separate from the full run. Add `--dry-run` instead to check
asset availability and inspect the command without launching MPI.

The same `hetero5` assets can be used with the other method keys. Running
`homo3` requires its separate prepared assets, which are not included in this
representative bundle.

## Aggregation and plotting

```bash
python scripts/aggregate_results.py \
  --log-root results/MovieLensEnv-v0/PPO/logs \
  --scene hetero5 --method fedbridge \
  --output-dir results/aggregated

python scripts/plot_results.py \
  --input-dir results/aggregated \
  --scene hetero5 --method fedbridge \
  --output-dir figures
```

Aggregation still requires all 100 epochs and all eight ranks for the selected
experiment. It writes `completeness.csv`, `epoch_mean_ci95.csv` and
`tail10_summary_ci95.csv`; plotting writes `learning_curves.png`, `.pdf` and
`.svg` for reward, interaction length and coverage. For a single run, `n=1`
and the confidence-interval fields are undefined; no confidence band is drawn.
This example does not reproduce the paper's multi-seed comparisons or every
figure. Omit the scene/method filters only when aggregating a complete grid of
both scenes and all six methods. Do not use `--allow-incomplete` to certify a
full run.

Release verification covers asset hashes, archive integrity, launcher dry-run
and automated tests of validation and analysis. A full 100-epoch training run
with this packaged bundle has not been re-executed as part of release preparation.

The experiment settings and runner map are defined in `experiment_spec.py` and
summarized in [`docs/HYPERPARAMETERS.md`](docs/HYPERPARAMETERS.md).
