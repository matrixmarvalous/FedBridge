# Public MuJoCo configuration

`experiment_spec.py` is the executable source of truth. The values below match
the current supplementary-information table and are included here for quick
inspection.

## Common settings

| Setting | Value |
| --- | ---: |
| Clients / MPI ranks | 8 |
| Local environment steps | 3,000,000 |
| Maximum episode length | 1,000 |
| PPO update interval | 2,000 steps |
| PPO epochs per update | 20 |
| Minibatch size | 4,000 |
| Actor / critic hidden layers | [64, 64] |
| Actor learning rate | 0.0003 |
| Critic learning rate | 0.001 |
| Discount factor | 0.99 |
| PPO clipping | 0.20 |
| Entropy coefficient | 0.01 |
| Value coefficient | 0.50 |
| Initial / minimum action std. | 0.60 / 0.10 |
| Action-std. decay | 0.05 every 250,000 steps |
| Communication interval | 5 PPO updates |
| pFedMe / PerFedDC beta | 0.05 |
| D/E/F evaluation | 10 repeats x 10 episodes per client and cluster |

## Task-specific coefficients

| Environment | FedBridge `lambda_kl` | PerFedDC `lambda_l2` | pFedMe `lambda_l2` |
| --- | ---: | ---: | ---: |
| HalfCheetah-v5 | 0.05 | 0.003 | 0.0005 |
| Hopper-v4 | 0.01 | 0.0005 | 0.005 |
| Humanoid-v4 | 0.05 | 0.0005 | 0.0005 |

Command-line overrides exist for controlled diagnostics. Runs using an override
record the effective value in `run_config.json` and should not be mixed with
the default main comparison without an explicit label.

## Training dynamics

Clients follow `ABCABCAC`:

| Cluster | Mass | Friction | Gear |
| --- | ---: | ---: | ---: |
| A | 0.75 | 0.80 | 0.70 |
| B | 1.00 | 1.00 | 1.00 |
| C | 1.30 | 1.20 | 1.40 |

Held-out D/E/F values differ by task and are kept in
`HELDOUT_CLUSTERS` in `experiment_spec.py` to avoid duplicating executable
configuration. No paper seed values are stored in this package.

