#!/usr/bin/env python3
# Codex-added 2026-07-23: single source of truth for the audited experiment matrix.
# Codex-modified 2026-08-14: make the copied publication configuration portable
# and remove the private paper-seed matrix while preserving experiment logic.
# Codex-modified 2026-08-18: expose the algorithms directory with final naming.
from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Set, Tuple


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
ALGORITHMS_DIR = THIS_DIR / "algorithms"
RESULT_ROOT = THIS_DIR / "results"
SLURM_ROOT = THIS_DIR / "slurm"
HPC_LOG_ROOT = THIS_DIR / "logs"

ENVIRONMENTS: Tuple[str, ...] = ("HalfCheetah-v5", "Hopper-v4", "Humanoid-v4")
N_CLIENTS = 8
CLUSTER_SEQUENCE = "ABCABCAC"

TRAIN_CLUSTERS: Mapping[str, Mapping[str, float]] = {
    "A": {"mass": 0.75, "friction": 0.80, "gear": 0.70},
    "B": {"mass": 1.00, "friction": 1.00, "gear": 1.00},
    "C": {"mass": 1.30, "friction": 1.20, "gear": 1.40},
}

HELDOUT_CLUSTERS: Mapping[str, Mapping[str, Mapping[str, float]]] = {
    "HalfCheetah-v5": {
        "D": {"mass": 0.65, "friction": 0.75, "gear": 0.60, "incline_deg": 0.0},
        "E": {"mass": 1.45, "friction": 1.35, "gear": 1.60, "incline_deg": 0.0},
        "F": {"mass": 1.00, "friction": 1.50, "gear": 1.00, "incline_deg": 5.0},
    },
    "Hopper-v4": {
        "D": {"mass": 0.75, "friction": 1.05, "gear": 0.85, "incline_deg": 0.0},
        "E": {"mass": 1.45, "friction": 1.35, "gear": 1.60, "incline_deg": 0.0},
        "F": {"mass": 1.05, "friction": 1.60, "gear": 0.80, "incline_deg": 7.0},
    },
    "Humanoid-v4": {
        "D": {"mass": 0.65, "friction": 0.75, "gear": 0.80, "incline_deg": 0.0},
        "E": {"mass": 1.45, "friction": 1.35, "gear": 1.60, "incline_deg": 0.0},
        "F": {"mass": 1.00, "friction": 1.50, "gear": 1.00, "incline_deg": 5.0},
    },
}

# Eleven fixed levels make the response curve dense without using test outcomes
# to choose interesting regions. Only one factor is changed at a time.
SHIFT_GRID: Mapping[str, Tuple[float, ...]] = {
    "mass": tuple(round(0.6 + 0.1 * i, 2) for i in range(11)),
    "friction": tuple(round(0.6 + 0.1 * i, 2) for i in range(11)),
    "gear": tuple(round(0.6 + 0.1 * i, 2) for i in range(11)),
    "incline_deg": tuple(float(-10 + 2 * i) for i in range(11)),
}

METHOD_LABELS: Mapping[str, str] = {
    "fedbridge_singlecritic": "FedBridge",
    "no_exchange_singlecritic": "No Exchange",
    "one_way_kl_singlecritic": "One-way KL",
    "distill_only_singlecritic": "KL Distill-only",
    "mutual_l2_singlecritic": "Mutual L2",
    "fedbridge_legacy_dualcritic": "FedBridge legacy dual critic",
    "individual": "Individual PPO",
    "push_avg": "Push-Avg",
    "fedavg": "FedAvg",
    "perfeddc": "PerFedDC",
    "pfedme": "pFedMe",
}

SINGLE_CRITIC_METHODS: Set[str] = {
    "fedbridge_singlecritic",
    "no_exchange_singlecritic",
    "one_way_kl_singlecritic",
    "distill_only_singlecritic",
    "mutual_l2_singlecritic",
}
BRIDGE_METHODS: Set[str] = SINGLE_CRITIC_METHODS | {"fedbridge_legacy_dualcritic"}

PROTOCOL_METHODS: Tuple[str, ...] = (
    "fedbridge_singlecritic",
    "no_exchange_singlecritic",
    "fedbridge_legacy_dualcritic",
)
PROTOCOL_ENVS: Tuple[str, ...] = ("Hopper-v4", "Humanoid-v4")

ABLATION_METHODS: Tuple[str, ...] = (
    "fedbridge_singlecritic",
    "no_exchange_singlecritic",
    "one_way_kl_singlecritic",
    "distill_only_singlecritic",
    "mutual_l2_singlecritic",
)
ABLATION_ENVS: Tuple[str, ...] = ("HalfCheetah-v5", "Humanoid-v4")

SHIFT_METHODS: Tuple[str, ...] = (
    "fedbridge_singlecritic",
    "no_exchange_singlecritic",
    "individual",
    "push_avg",
    "fedavg",
    "perfeddc",
    "pfedme",
)
SHIFT_ENVS: Tuple[str, ...] = ENVIRONMENTS

HYPERPARAMETERS = {
    "total_local_steps": 3_000_000,
    "max_episode_length": 1_000,
    "update_timestep": 2_000,
    "ppo_epochs": 20,
    "minibatch_size": 4_000,
    "actor_learning_rate": 3e-4,
    "critic_learning_rate": 1e-3,
    "gamma": 0.99,
    "eps_clip": 0.20,
    "entropy_coefficient": 0.01,
    "value_coefficient": 0.50,
    "lambda_kl": 0.05,
    "lambda_l2_ablation": 0.05,
    "pfl_lambda_l2": 0.0005,
    "pfl_beta": 0.05,
    "pfedme_local_steps": 1,
    "action_std_init": 0.60,
    "action_std_decay": 0.05,
    "action_std_decay_frequency": 250_000,
    "action_std_min": 0.10,
    "communication_interval_updates": 5,
    "checkpoint_frequency_steps": 300_000,
    "training_log_frequency_steps": 100_000,
    "local_eval_episodes": 10,
    "heldout_eval_repeats": 10,
    "heldout_eval_episodes": 10,
    "shift_eval_episodes": 10,
}

# Public task-specific coefficients reported for the main MuJoCo experiments.
# Random seeds are deliberately not stored in this release configuration.
TASK_METHOD_HYPERPARAMETERS: Mapping[str, Mapping[str, Mapping[str, float]]] = {
    "HalfCheetah-v5": {
        "fedbridge_singlecritic": {"lambda_kl": 0.05},
        "perfeddc": {"pfl_lambda_l2": 0.003},
        "pfedme": {"pfl_lambda_l2": 0.0005},
    },
    "Hopper-v4": {
        "fedbridge_singlecritic": {"lambda_kl": 0.01},
        "perfeddc": {"pfl_lambda_l2": 0.0005},
        "pfedme": {"pfl_lambda_l2": 0.005},
    },
    "Humanoid-v4": {
        "fedbridge_singlecritic": {"lambda_kl": 0.05},
        "perfeddc": {"pfl_lambda_l2": 0.0005},
        "pfedme": {"pfl_lambda_l2": 0.0005},
    },
}


def task_hyperparameters(environment: str, method: str) -> Dict[str, float]:
    """Return public task/method overrides without mutating common defaults."""
    return dict(TASK_METHOD_HYPERPARAMETERS.get(environment, {}).get(method, {}))


def communication_protocol(method: str) -> str:
    if method in {"no_exchange_singlecritic", "individual"}:
        return "none"
    if method in SINGLE_CRITIC_METHODS | {"fedbridge_legacy_dualcritic", "push_avg"}:
        return "exponential_rotating_ring_replacement"
    if method in {"fedavg", "perfeddc", "pfedme"}:
        return "centralized_server_average"
    raise KeyError(method)


def policy_types(method: str) -> Tuple[str, ...]:
    if method in BRIDGE_METHODS:
        return ("personal", "bridge")
    if method == "pfedme":
        return ("personal", "global")
    if method == "fedavg":
        return ("global",)
    if method == "perfeddc":
        return ("personalized",)
    return ("local",)


def validate_spec() -> None:
    assert CLUSTER_SEQUENCE == "ABCABCAC"
    assert len(CLUSTER_SEQUENCE) == N_CLIENTS
    assert set(TRAIN_CLUSTERS) == {"A", "B", "C"}
    assert set(HELDOUT_CLUSTERS) == set(ENVIRONMENTS)
    assert set(SHIFT_GRID) == {"mass", "friction", "gear", "incline_deg"}
    assert all(len(levels) == 11 for levels in SHIFT_GRID.values())
    assert all(1.0 in SHIFT_GRID[k] for k in ("mass", "friction", "gear"))
    assert 0.0 in SHIFT_GRID["incline_deg"]
    assert set(TASK_METHOD_HYPERPARAMETERS) == set(ENVIRONMENTS)


validate_spec()
