# FedBridge

<!-- Codex-modified 2026-09-10: concise code-only README following the
organization of Networked-MB-MARL; the previous version is in the private
publication archive. -->

Code for the paper *Bridge reinforcement learning enables collaborative
learning of personalized policies across heterogeneous environments*.

## Algorithms

The repository contains FedBridge and five comparison methods. The launcher
keys are:

| Algorithm | MuJoCo | Recommender system |
| --- | --- | --- |
| FedBridge | `fedbridge_singlecritic` | `fedbridge` |
| Individual PPO | `individual` | `individual` |
| Push-Avg / Push-Pull | `push_avg` | `pushpull` |
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
assets described in
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

```bash
cd recommender_system
SEED=YOUR_INTEGER_SEED
python launcher.py \
  --scene hetero5 \
  --method fedbridge \
  --seed "${SEED}" \
  --run-id run-a \
  --asset-root /path/to/staged-assets \
  --output-root results
```

Use the corresponding component guide for evaluation and plotting commands:

- [`mujoco/README.md`](mujoco/README.md)
- [`recommender_system/README.md`](recommender_system/README.md)

## Citation

```bibtex
@misc{wang2026fedbridge,
  title  = {Bridge Reinforcement Learning Enables Collaborative Learning of Personalized Policies across Heterogeneous Environments},
  author = {Dazhong Wang and Weidong Liu and Xiaojun Mao},
  year   = {2026}
}
```
