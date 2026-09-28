# FedBridge

<!-- Codex-modified 2026-09-10: concise code-only README following the
organization of Networked-MB-MARL; the previous version is in the private
publication archive. -->

Code for the paper *Towards Personalized Collaborative Reinforcement Learning
with Decoupled Bridge Policy Distillation*.

## Algorithms

The repository contains FedBridge and five comparison methods. The launcher
keys are:

| Algorithm | MuJoCo | Recommender system |
| --- | --- | --- |
| FedBridge | `fedbridge_singlecritic` | `fedbridge` |
| Individual PPO | `individual` | `individual` |
| Push-Avg | `push_avg` | `push_avg` |
| FedAvg | `fedavg` | `fedavg` |
| PerFedDC | `perfeddc` | `perfeddc` |
| pFedMe | `pfedme` | `pfedme` |

## Environments

- MuJoCo: `HalfCheetah-v5`, `Hopper-v4`, and `Humanoid-v4`.
- Recommender system: `MovieLensEnv-v0` with `hetero5` and `homo3` settings.

## Software requirements

The experiments require Linux, Python, and an MPI implementation. MuJoCo and
the recommender-system experiments use separate Conda environments.

## Installation

MuJoCo:

```bash
cd mujoco
conda env create -f environment.yml
conda activate fedbridge-mujoco
```

Recommender system:

```bash
cd recommender_system
conda env create -f environment.yml
conda activate fedbridge-easyrl4rec
```

The recommender experiments also require the prepared MovieLens and DeepFM
assets. The assets for one representative heterogeneous (`hetero5`) FedBridge
experiment are provided in the
[hetero5 asset release](https://github.com/matrixmarvalous/FedBridge/releases/tag/recommender-hetero5-v1).
Download, integrity-check, training and plotting instructions are in
[`recommender_system/README.md`](recommender_system/README.md). The bundle
contains 54 runtime files (1.46 GB uncompressed), including all three user
groups and their pretrained models. It does not include `homo3` assets.
The file inventory and scope are described in
[`recommender_system/docs/DATA.md`](recommender_system/docs/DATA.md).

## Running experiments

Run one MuJoCo training and zero-shot evaluation experiment with eight MPI
ranks:

```bash
cd mujoco
SEED=YOUR_INTEGER_SEED
mpirun -n 8 python launcher.py \
  --env-name HalfCheetah-v5 \
  --method fedbridge_singlecritic \
  --seed "${SEED}"
```

Run one recommender-system experiment:

First download and extract the asset bundle using the component guide.

```bash
cd recommender_system
SEED=0  # Representative run, not an embedded paper seed.
python launcher.py \
  --scene hetero5 \
  --method fedbridge \
  --seed "${SEED}" \
  --run-id run-a \
  --asset-root assets/fedbridge-hetero5-assets \
  --output-root results
```

Use the corresponding component guide for evaluation and plotting commands:

- [`mujoco/README.md`](mujoco/README.md)
- [`recommender_system/README.md`](recommender_system/README.md)

## Citation

Citation details are omitted during double-blind review and will be added after the review process.
