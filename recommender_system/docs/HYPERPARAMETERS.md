# EasyRL4Rec reference configuration

<!-- Codex-added 2026-08-14: record the paper-facing configuration without disclosing author seed values. -->
<!-- Codex-modified 2026-08-18: use the final Bridge terminology. -->

This file records the configuration used for the MovieLens-1M experiments. The executable source of truth is `experiment_spec.py`; command-line arguments shown by `launcher.py --dry-run` should agree with this document.

## Seed policy

The authors' experiment seed values are intentionally not embedded in this release. Every run requires a non-negative user-selected `--seed`, and public run identifiers must not contain seed values.

The preserved runners use two historical, deterministic rank conventions: some derive rank-specific streams from the supplied base seed, while others pass the same base seed to all ranks. `experiment_spec.py` records the convention for every scene-method pair. Keep that convention fixed when making paired comparisons; do not infer the authors' seeds from it.

## Dataset and evaluation protocols

| Setting | Heterogeneous (`hetero5`) | Homogeneous (`homo3`) |
|---|---|---|
| Local populations | Genre-controlled A, B and C datasets with 50% preference heterogeneity over Action, Drama and Sci-Fi | Three local training datasets; all policies use one common mixed test population |
| Agent assignment | A, B, C, A, B, C, A, C | A, B, C, A, B, C, A, C |
| Test environment | Held-out split matched to each agent's local population | Common held-out split after cluster-specific user IDs are mapped to original IDs |
| FedBridge coefficient, `alpha` / `lambda_kl` | 0.005 | 0.001 |
| FedBridge exchanges per epoch | 5 | 10 |

The paper-facing split counts are:

| Protocol | Dataset | Users | Movies | Training ratings | Test ratings |
|---|---:|---:|---:|---:|---:|
| Heterogeneous | A | 2,416 | 3,491 | 355,009 | 39,433 |
| Heterogeneous | B | 2,416 | 3,491 | 359,757 | 39,966 |
| Heterogeneous | C | 2,416 | 3,491 | 358,360 | 39,801 |
| Homogeneous | A | 5,701 | 3,540 | 379,932 | 39,994 |
| Homogeneous | B | 5,696 | 3,540 | 377,432 | 39,994 |
| Homogeneous | C | 5,692 | 3,540 | 384,269 | 39,994 |

During training, the reward is the DeepFM ensemble prediction shifted by the minimum of the prediction matrix. During testing, it is the matrix-completed rating clipped to `[0, 5]`. An episode contains at most 30 recommendations. A simulated user leaves when a new item's distance is below 75 from any of the seven most recent items.

## Frozen DeepFM user models

| Setting | Value |
|---|---:|
| Models per local dataset | 5 |
| Negative samples per positive interaction | 5 |
| Embedding dimension | 8 |
| Hidden layers | `[128, 128]` |
| Optimizer | Adam |
| Batch size | 2,048 |
| L2 coefficient | 0.1 |
| Training epochs | 5 |

The ensemble is trained before policy optimization and then frozen. Its prepared weights are external assets; their training seed values are not distributed.

## Common PPO settings

| Setting | Value |
|---|---:|
| MPI agents | 8 |
| Policy-training epochs | 100 |
| Local interactions per agent and epoch | 100,000 |
| Parallel training environments per agent | 100 |
| Parallel test environments per agent | 100 |
| Maximum episode length | 30 |
| Personal and bridge hidden layers | `[64, 64]` |
| State tracker | Average item embedding |
| State window | 3 interactions |
| Replay buffer size | 100,000 |
| PPO batch size | 1,024 |
| Update repeats per collect | 1 |
| Steps per collect | 100 |
| Learning rate | 0.001 |
| Discount factor | 0.90 |
| GAE coefficient | 0.95 |
| PPO clipping coefficient | 0.20 |
| Value-loss coefficient | 0.50 |
| Entropy coefficient | 0 |
| Gradient-norm limit | 0.50 |
| Reward normalization | Disabled |
| Advantage normalization | Enabled |
| Evaluation | Deterministic; 100 episodes per agent after every epoch |

## Method-specific settings

| Method | Coupling | Exchanges per epoch | Release implementation note |
|---|---:|---:|---|
| FedBridge | `alpha=0.005` (`hetero5`); `alpha=0.001` (`homo3`) | 5; 10 | Personal PPO plus separately updated and exchanged bridge actor |
| Individual PPO | `alpha=0` | 25 historical trainer synchronizations | Preserved BridgePPO runner with an exact-zero short circuit: KL evaluation and bridge optimization are skipped, and the personal update equals standard PPO; the trainer synchronization remains and must not be counted as an efficient no-communication implementation |
| Push-Pull* | None | 10 | Dispatches the copied `PushPullBaseTrainer` overwrite path |
| pFedMe | `lambda_l2=0.001`, `beta=0.05` | 10 | Same values in both protocols |
| PerFedDC | `lambda_l2=0.05`, `beta=0.005` | 10 | Same values in both protocols |
| FedAvg | None | 10 | Exact population aggregation through the server process |

All methods use the same PPO backbone and local interaction budget. The reported final summaries first average the eight agents within a run, use the last ten available training epochs, and then summarize five independent runs with a two-sided 95% Student-*t* interval.
