#!/usr/bin/env python3
# Codex-added 2026-08-14: one public source of truth for EasyRL4Rec runs.
# Codex-modified 2026-08-18: point final algorithm entries to BridgePPO runners.
from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Tuple


THIS_DIR = Path(__file__).resolve().parent
N_CLIENTS = 8
CLUSTER_SEQUENCE = "ABCABCAC"
SCENES: Tuple[str, ...] = ("hetero5", "homo3")
METHODS: Tuple[str, ...] = (
    "fedbridge",
    "individual",
    "push_avg",
    "pfedme",
    "perfeddc",
    "fedavg",
)

METHOD_LABELS: Mapping[str, str] = {
    "fedbridge": "FedBridge",
    "individual": "Individual",
    # Push-Avg retains the original actor-parameter communication update.
    # PushAvgBaseTrainer overwrites with received actor parameters.
    "push_avg": "Push-Avg",
    "pfedme": "pFedMe",
    "perfeddc": "PerFedDC",
    "fedavg": "FedAvg",
}

RUNNERS: Mapping[Tuple[str, str], str] = {
    ("hetero5", "fedbridge"): "run_FedBridgePPO_hetero5_seed.py",
    ("homo3", "fedbridge"): "homo_run_FedBridgePPO_seed.py",
    ("hetero5", "individual"): "run_FedBridgePPO_hetero5_seed.py",
    ("homo3", "individual"): "homo_run_FedBridgePPO_seed.py",
    ("hetero5", "push_avg"): "run_FedPPO_push_avg_seed.py",
    ("homo3", "push_avg"): "homo_run_FedPPO_push_avg_seed.py",
    ("hetero5", "pfedme"): "run_pFedMe_seed.py",
    ("homo3", "pfedme"): "homo_run_pFedMe_seed.py",
    ("hetero5", "perfeddc"): "run_PerFedDC_seed.py",
    ("homo3", "perfeddc"): "homo_run_PerFedDC_seed.py",
    ("hetero5", "fedavg"): "run_FedAvgPPO_seed.py",
    ("homo3", "fedavg"): "homo_run_FedAvgPPO_seed.py",
}

COMMON_ARGS: Tuple[str, ...] = (
    "--env", "MovieLensEnv-v0",
    "--epoch", "100",
    "--which_tracker", "avg",
    "--reward_handle", "cat",
    "--window_size", "3",
    "--read_message", "pointneg",
    "--quiet_progress",
    "--no_save",
)

METHOD_ARGS: Mapping[Tuple[str, str], Tuple[str, ...]] = {
    ("hetero5", "fedbridge"): ("--lambda_kl", "0.005", "--comm_times_per_epoch", "5"),
    ("homo3", "fedbridge"): ("--lambda_kl", "0.001", "--comm_times_per_epoch", "10"),
    # Codex-modified 2026-09-09: Individual historically reused the bridge
    # runner with zero KL. BridgePPO now short-circuits its inactive KL/update
    # path at exactly zero, making the personal update standard PPO while the
    # preserved trainer still performs the historical bridge synchronization.
    ("hetero5", "individual"): ("--lambda_kl", "0", "--comm_times_per_epoch", "25"),
    ("homo3", "individual"): ("--lambda_kl", "0", "--comm_times_per_epoch", "25"),
    ("hetero5", "push_avg"): ("--comm_times_per_epoch", "10"),
    ("homo3", "push_avg"): ("--comm_times_per_epoch", "10"),
    ("hetero5", "pfedme"): ("--comm_times_per_epoch", "10", "--lambda_l2", "0.001", "--beta", "0.05"),
    ("homo3", "pfedme"): ("--comm_times_per_epoch", "10", "--lambda_l2", "0.001", "--beta", "0.05"),
    ("hetero5", "perfeddc"): ("--comm_times_per_epoch", "10", "--lambda_l2", "0.05", "--beta", "0.005"),
    ("homo3", "perfeddc"): ("--comm_times_per_epoch", "10", "--lambda_l2", "0.05", "--beta", "0.005"),
    ("hetero5", "fedavg"): ("--comm_times_per_epoch", "10"),
    ("homo3", "fedavg"): ("--comm_times_per_epoch", "10"),
}

PAPER_HYPERPARAMETERS: Mapping[str, object] = {
    "epochs": 100,
    "local_steps_per_client_per_epoch": 100_000,
    "training_environments_per_client": 100,
    "test_episodes_per_client_per_epoch": 100,
    "max_episode_length": 30,
    "hidden_sizes": (64, 64),
    "state_tracker": "average item embedding",
    "state_window": 3,
    "batch_size": 1_024,
    "repeat_per_collect": 1,
    "learning_rate": 1e-3,
    "discount_factor": 0.9,
    "gae_lambda": 0.95,
    "ppo_clip": 0.2,
    "value_coefficient": 0.5,
    "entropy_coefficient": 0.0,
    "max_gradient_norm": 0.5,
    "deterministic_evaluation": True,
}

# This describes transformations applied to a user-provided base seed. It does
# not contain the authors' paper seed values.
RANK_SEED_MODE: Mapping[Tuple[str, str], str] = {
    (scene, method): (
        "base_plus_rank_offset"
        if (scene, method) in {
            ("hetero5", "fedbridge"),
            ("hetero5", "individual"),
            ("hetero5", "push_avg"),
            ("homo3", "push_avg"),
            ("hetero5", "fedavg"),
            ("homo3", "fedavg"),
        }
        else "shared_base"
    )
    for scene in SCENES
    for method in METHODS
}


def runner_path(scene: str, method: str) -> Path:
    return THIS_DIR / "examples" / "policy" / RUNNERS[(scene, method)]


def command_args(scene: str, method: str) -> Tuple[str, ...]:
    return COMMON_ARGS + METHOD_ARGS[(scene, method)]


def validate_spec() -> None:
    expected = {(scene, method) for scene in SCENES for method in METHODS}
    assert len(CLUSTER_SEQUENCE) == N_CLIENTS
    assert set(CLUSTER_SEQUENCE) == {"A", "B", "C"}
    assert set(RUNNERS) == expected
    assert set(METHOD_ARGS) == expected
    assert set(RANK_SEED_MODE) == expected
    assert set(METHOD_LABELS) == set(METHODS)


validate_spec()
