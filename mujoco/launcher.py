#!/usr/bin/env python3
# Codex-added 2026-07-23: one method x environment x seed audited MPI runner.
# Codex-modified 2026-08-14: portable public launcher copied from the audited
# runner; paper seeds are not embedded and algorithm logic is unchanged.
# Codex-modified 2026-08-18: consume the BridgePPO module and Bridge metric keys.
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch

from experiment_spec import (
    ENVIRONMENTS,
    HPC_LOG_ROOT,
    HYPERPARAMETERS,
    METHOD_LABELS,
    N_CLIENTS,
    ALGORITHMS_DIR,
    RESULT_ROOT,
    SHIFT_METHODS,
    SINGLE_CRITIC_METHODS,
    TRAIN_CLUSTERS,
    communication_protocol,
    policy_types,
    task_hyperparameters,
)
from runtime_support import (
    apply_canonical_initialization,
    architecture_manifest,
    build_agent,
    client_cluster,
    communicate,
    deterministic_action,
    evaluation_seed,
    initialization_seed,
    make_dynamics_env,
    make_training_env,
    payload_manifest,
    primary_policy_type,
    runtime_contract,
    training_seed,
    update_agent,
)


THIS_DIR = Path(__file__).resolve().parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(THIS_DIR), "rev-parse", "HEAD"],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except Exception:
        # Avoid serializing a private absolute checkout path into run metadata.
        return "not_available"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest() -> List[Dict[str, str]]:
    relative_sources = [
        THIS_DIR / "experiment_spec.py",
        ALGORITHMS_DIR / "single_critic_fedbridge.py",
        THIS_DIR / "runtime_support.py",
        THIS_DIR / "launcher.py",
        ALGORITHMS_DIR / "PPO.py",
        # Codex-modified 2026-08-19: final public module name.
        ALGORITHMS_DIR / "BridgePPO.py",
        ALGORITHMS_DIR / "perfeddc_ppo.py",
        ALGORITHMS_DIR / "pfedme_ppo.py",
    ]
    return [
        {"path": str(path.relative_to(THIS_DIR)), "sha256": file_sha256(path)}
        for path in relative_sources
        if path.exists()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one audited method x environment x training-seed tuple."
    )
    parser.add_argument("--env-name", choices=ENVIRONMENTS, required=True)
    parser.add_argument("--method", choices=sorted(METHOD_LABELS), required=True)
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="User-selected seed. The paper's random seeds are not embedded.",
    )
    parser.add_argument("--n-clients", type=int, default=N_CLIENTS)
    parser.add_argument("--experiments", default="main")
    parser.add_argument("--evaluate-shift", action="store_true")
    parser.add_argument(
        "--total-local-steps",
        type=int,
        default=HYPERPARAMETERS["total_local_steps"],
    )
    parser.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Construct and audit the agent, then exit before rollouts.",
    )
    parser.add_argument(
        "--lambda-kl",
        type=float,
        default=None,
        help="Optional override for the task-specific FedBridge coefficient.",
    )
    parser.add_argument(
        "--pfl-lambda-l2",
        type=float,
        default=None,
        help="Optional override for the task-specific PerFedDC/pFedMe coefficient.",
    )
    parser.add_argument(
        "--communication-interval-updates",
        type=int,
        default=None,
        help="Optional override; the paper setting is five PPO updates.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny end-to-end training, MPI communication and D--F evaluation test.",
    )
    return parser.parse_args()


def configure_run(args: argparse.Namespace) -> None:
    """Apply public task settings and optional non-paper smoke overrides."""
    if args.seed < 0:
        raise ValueError("--seed must be a non-negative integer")
    if args.total_local_steps < 1:
        raise ValueError("--total-local-steps must be positive")
    HYPERPARAMETERS["total_local_steps"] = int(args.total_local_steps)
    HYPERPARAMETERS.update(task_hyperparameters(args.env_name, args.method))
    if args.lambda_kl is not None:
        HYPERPARAMETERS["lambda_kl"] = float(args.lambda_kl)
    if args.pfl_lambda_l2 is not None:
        HYPERPARAMETERS["pfl_lambda_l2"] = float(args.pfl_lambda_l2)
    if args.communication_interval_updates is not None:
        if args.communication_interval_updates < 1:
            raise ValueError("--communication-interval-updates must be positive")
        HYPERPARAMETERS["communication_interval_updates"] = int(
            args.communication_interval_updates
        )
    if args.smoke_test:
        HYPERPARAMETERS.update(
            {
                "total_local_steps": 64,
                "max_episode_length": 32,
                "update_timestep": 32,
                "ppo_epochs": 1,
                "minibatch_size": 32,
                "communication_interval_updates": 1,
                "checkpoint_frequency_steps": 64,
                "training_log_frequency_steps": 32,
                "action_std_decay_frequency": 1_000_000_000,
                "local_eval_episodes": 1,
                "heldout_eval_repeats": 1,
                "heldout_eval_episodes": 1,
                "shift_eval_episodes": 1,
            }
        )
        args.total_local_steps = HYPERPARAMETERS["total_local_steps"]


def run_slug(args: argparse.Namespace) -> str:
    env = args.env_name.replace("-", "_").lower()
    return f"{env}__{args.method}__seed_{args.seed}"


def evaluate_episodes(
    agent,
    method: str,
    env_name: str,
    factors: Mapping[str, float],
    policy_type: str,
    run_seed: int,
    client_id: int,
    family_id: int,
    condition_id: int,
    repeat: int,
    episodes: int,
    condition_fields: Mapping[str, object],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    first_seed = evaluation_seed(
        run_seed,
        client_id,
        family_id,
        condition_id,
        repeat,
        0,
    )
    env = make_dynamics_env(env_name, factors, first_seed)
    for episode in range(int(episodes)):
        seed = evaluation_seed(
            run_seed,
            client_id,
            family_id,
            condition_id,
            repeat,
            episode,
        )
        state, _ = env.reset(seed=seed)
        episode_return = 0.0
        episode_length = 0
        for _ in range(HYPERPARAMETERS["max_episode_length"]):
            action = deterministic_action(agent, method, state, policy_type)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            episode_length += 1
            if terminated or truncated:
                break
        row = {
            "environment": env_name,
            "method_key": method,
            "method": METHOD_LABELS[method],
            "training_seed": run_seed,
            "client_id": client_id,
            "source_cluster": client_cluster(client_id),
            "policy_type": policy_type,
            "primary_eval_policy": str(
                policy_type == primary_policy_type(method)
            ).lower(),
            "evaluation_seed": seed,
            "repeat": repeat,
            "episode": episode,
            "return": episode_return,
            "episode_length": episode_length,
        }
        row.update(condition_fields)
        rows.append(row)
    env.close()
    return rows


def final_evaluation(
    agent,
    args: argparse.Namespace,
    client_id: int,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    # Held-out constants are imported only after optimization and final checkpointing.
    from experiment_spec import HELDOUT_CLUSTERS, SHIFT_GRID

    local_rows: List[Dict[str, object]] = []
    heldout_rows: List[Dict[str, object]] = []
    shift_rows: List[Dict[str, object]] = []
    source_cluster = client_cluster(client_id)

    for policy_type in policy_types(args.method):
        local_rows.extend(
            evaluate_episodes(
                agent=agent,
                method=args.method,
                env_name=args.env_name,
                factors=TRAIN_CLUSTERS[source_cluster],
                policy_type=policy_type,
                run_seed=args.seed,
                client_id=client_id,
                family_id=1,
                condition_id={"A": 0, "B": 1, "C": 2}[source_cluster],
                repeat=0,
                episodes=HYPERPARAMETERS["local_eval_episodes"],
                condition_fields={
                    "evaluation_family": "matched_local",
                    "target_cluster": source_cluster,
                    "mass": TRAIN_CLUSTERS[source_cluster]["mass"],
                    "friction": TRAIN_CLUSTERS[source_cluster]["friction"],
                    "gear": TRAIN_CLUSTERS[source_cluster]["gear"],
                    "incline_deg": 0.0,
                },
            )
        )

        for cluster_index, (target_cluster, factors) in enumerate(
            HELDOUT_CLUSTERS[args.env_name].items()
        ):
            for repeat in range(HYPERPARAMETERS["heldout_eval_repeats"]):
                heldout_rows.extend(
                    evaluate_episodes(
                        agent=agent,
                        method=args.method,
                        env_name=args.env_name,
                        factors=factors,
                        policy_type=policy_type,
                        run_seed=args.seed,
                        client_id=client_id,
                        family_id=2,
                        condition_id=cluster_index,
                        repeat=repeat,
                        episodes=HYPERPARAMETERS["heldout_eval_episodes"],
                        condition_fields={
                            "evaluation_family": "heldout_cluster",
                            "target_cluster": target_cluster,
                            "mass": factors["mass"],
                            "friction": factors["friction"],
                            "gear": factors["gear"],
                            "incline_deg": factors["incline_deg"],
                        },
                    )
                )

        if args.evaluate_shift:
            condition_id = 0
            for shift_factor, levels in SHIFT_GRID.items():
                for level in levels:
                    factors = {
                        "mass": 1.0,
                        "friction": 1.0,
                        "gear": 1.0,
                        "incline_deg": 0.0,
                    }
                    factors[shift_factor] = float(level)
                    shift_rows.extend(
                        evaluate_episodes(
                            agent=agent,
                            method=args.method,
                            env_name=args.env_name,
                            factors=factors,
                            policy_type=policy_type,
                            run_seed=args.seed,
                            client_id=client_id,
                            family_id=3,
                            condition_id=condition_id,
                            repeat=0,
                            episodes=HYPERPARAMETERS["shift_eval_episodes"],
                            condition_fields={
                                "evaluation_family": "one_factor_shift",
                                "shift_factor": shift_factor,
                                "shift_value": float(level),
                                "mass": factors["mass"],
                                "friction": factors["friction"],
                                "gear": factors["gear"],
                                "incline_deg": factors["incline_deg"],
                            },
                        )
                    )
                    condition_id += 1
    return local_rows, heldout_rows, shift_rows


def save_agent(agent, checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(checkpoint_path))


def train(args: argparse.Namespace) -> None:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != args.n_clients or size != N_CLIENTS:
        raise RuntimeError(
            f"Expected exactly {N_CLIENTS} MPI ranks, got {size}; "
            f"--n-clients={args.n_clients}"
        )
    if args.evaluate_shift and args.method not in SHIFT_METHODS:
        raise RuntimeError(
            f"Continuous-shift evaluation was not pre-registered for {args.method}"
        )

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    device = torch.device("cpu")
    local_training_seed = training_seed(args.seed, rank)
    torch.manual_seed(local_training_seed)
    np.random.seed(local_training_seed)

    run_dir = Path(args.output_root) / run_slug(args)
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    # Only A--C training dynamics are constructed before optimization.
    env = make_training_env(args.env_name, rank, local_training_seed)
    state, _ = env.reset(seed=local_training_seed)
    state_dim = int(np.asarray(state).shape[0])
    action_dim = int(env.action_space.shape[0])
    agent = build_agent(args.method, state_dim, action_dim, device)
    apply_canonical_initialization(
        agent, args.method, state_dim, action_dim, args.seed, rank
    )
    contract = runtime_contract(agent, args.method)
    payload_rows, payload_bytes, communicated_object = payload_manifest(
        agent, args.method
    )

    if rank == 0:
        experiments = tuple(
            item.strip() for item in args.experiments.split(",") if item.strip()
        )
        seed_rows = [
            {
                "client_id": client_id,
                "cluster": client_cluster(client_id),
                "training_seed": training_seed(args.seed, client_id),
                "personal_actor_init_seed": initialization_seed(
                    args.seed, client_id, 1
                ),
                "bridge_actor_init_seed": initialization_seed(
                    args.seed, client_id, 2
                ),
                "personal_critic_init_seed": initialization_seed(
                    args.seed, client_id, 3
                ),
                "legacy_bridge_critic_init_seed": initialization_seed(
                    args.seed, client_id, 4
                )
                if args.method == "fedbridge_legacy_dualcritic"
                else "",
                "evaluation_seed_namespace_start": evaluation_seed(
                    args.seed, client_id, 1, 0, 0, 0
                ),
            }
            for client_id in range(size)
        ]
        write_json(
            run_dir / "run_config.json",
            {
                "audit_schema_version": 1,
                "created_at": utc_now(),
                "environment": args.env_name,
                "method_key": args.method,
                "method": METHOD_LABELS[args.method],
                "training_seed": args.seed,
                "n_clients": size,
                "cluster_sequence": "".join(client_cluster(i) for i in range(size)),
                "training_clusters": TRAIN_CLUSTERS,
                "experiments": experiments,
                "evaluate_shift": bool(args.evaluate_shift),
                "heldout_evaluation_during_training": False,
                "checkpoint_selection": "none_final_checkpoint_only",
                "hyperparameters": HYPERPARAMETERS,
                "communication_protocol": communication_protocol(args.method),
                "communicated_object": communicated_object,
                "payload_bytes": payload_bytes,
                "git_revision": git_revision(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "source_manifest_file": "source_manifest.json",
            },
        )
        write_json(run_dir / "architecture_manifest.json", contract)
        write_json(run_dir / "source_manifest.json", {"files": source_manifest()})
        write_csv(
            run_dir / "seed_manifest.csv",
            seed_rows,
            [
                "client_id",
                "cluster",
                "training_seed",
                "personal_actor_init_seed",
                "bridge_actor_init_seed",
                "personal_critic_init_seed",
                "legacy_bridge_critic_init_seed",
                "evaluation_seed_namespace_start",
            ],
        )
        write_csv(
            run_dir / "communicated_payload_manifest.csv",
            payload_rows,
            [
                "method_key",
                "communication_enabled",
                "object_name",
                "tensor_name",
                "num_scalars",
                "dtype",
                "bytes",
                "contains_critic",
            ],
        )
        write_json(
            run_dir / "STATUS.json",
            {"status": "audit_passed_training_pending", "updated_at": utc_now()},
        )
    comm.Barrier()

    if args.audit_only:
        env.close()
        if rank == 0:
            write_json(
                run_dir / "STATUS.json",
                {"status": "audit_only_completed", "updated_at": utc_now()},
            )
        return

    max_episode_length = HYPERPARAMETERS["max_episode_length"]
    update_timestep = HYPERPARAMETERS["update_timestep"]
    communication_interval = HYPERPARAMETERS["communication_interval_updates"]
    total_local_steps = int(args.total_local_steps)
    time_step = 0
    update_count = 0
    communication_round = 0
    next_log_step = HYPERPARAMETERS["training_log_frequency_steps"]
    next_checkpoint_step = HYPERPARAMETERS["checkpoint_frequency_steps"]
    next_action_decay_step = HYPERPARAMETERS["action_std_decay_frequency"]
    completed_episode_returns: List[float] = []
    completed_episode_lengths: List[int] = []
    training_rows: List[Dict[str, object]] = []
    communication_rows: List[Dict[str, object]] = []
    latest_metrics: Dict[str, float] = {}
    training_started = time.time()

    while time_step < total_local_steps:
        episode_return = 0.0
        episode_length = 0
        for _ in range(max_episode_length):
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            time_step += 1
            episode_length += 1
            episode_return += float(reward)
            budget_terminal = time_step >= total_local_steps
            done = bool(terminated or truncated or budget_terminal)
            agent.buffer.rewards.append(float(reward))
            agent.buffer.is_terminals.append(done)

            if time_step % update_timestep == 0:
                latest_metrics = update_agent(agent, args.method)
                update_count += 1
                if (
                    update_count % communication_interval == 0
                    and communication_protocol(args.method) != "none"
                ):
                    comm.Barrier()
                    started = time.perf_counter()
                    communication_stats = communicate(
                        agent,
                        args.method,
                        comm,
                        rank,
                        communication_round,
                        device,
                    )
                    elapsed = time.perf_counter() - started
                    communication_rows.append(
                        {
                            "environment": args.env_name,
                            "method_key": args.method,
                            "method": METHOD_LABELS[args.method],
                            "training_seed": args.seed,
                            "client_id": rank,
                            "communication_round": communication_round,
                            "local_step": time_step,
                            "protocol": communication_protocol(args.method),
                            "payload_bytes": payload_bytes,
                            "bytes_sent": communication_stats["bytes_sent"],
                            "bytes_received": communication_stats["bytes_received"],
                            "messages_sent": communication_stats["messages_sent"],
                            "messages_received": communication_stats[
                                "messages_received"
                            ],
                            "system_bytes": communication_stats["system_bytes"],
                            "communication_time_seconds": elapsed,
                        }
                    )
                    communication_round += 1

            if time_step >= next_action_decay_step:
                current_std = float(getattr(agent, "action_std"))
                new_std = max(
                    HYPERPARAMETERS["action_std_min"],
                    round(
                        current_std - HYPERPARAMETERS["action_std_decay"],
                        4,
                    ),
                )
                agent.set_action_std(new_std)
                next_action_decay_step += HYPERPARAMETERS[
                    "action_std_decay_frequency"
                ]

            if time_step >= next_log_step:
                training_rows.append(
                    {
                        "environment": args.env_name,
                        "method_key": args.method,
                        "method": METHOD_LABELS[args.method],
                        "training_seed": args.seed,
                        "client_id": rank,
                        "cluster": client_cluster(rank),
                        "local_step": time_step,
                        "updates": update_count,
                        "communication_rounds": communication_round,
                        "completed_episodes_in_window": len(
                            completed_episode_returns
                        ),
                        "return_mean": float(np.mean(completed_episode_returns))
                        if completed_episode_returns
                        else "",
                        "return_std": float(np.std(completed_episode_returns))
                        if completed_episode_returns
                        else "",
                        "episode_length_mean": float(
                            np.mean(completed_episode_lengths)
                        )
                        if completed_episode_lengths
                        else "",
                        "action_std": float(getattr(agent, "action_std")),
                        "personal_policy_loss": latest_metrics.get(
                            "Loss/personal_policy",
                            latest_metrics.get("Loss/priv_policy", ""),
                        ),
                        "value_loss": latest_metrics.get(
                            "Loss/value",
                            latest_metrics.get("Loss/priv_value", ""),
                        ),
                        "bridge_policy_loss": latest_metrics.get(
                            "Loss/bridge_policy",
                            latest_metrics.get("Loss/bridge_policy", ""),
                        ),
                        "personal_to_bridge_regularizer": latest_metrics.get(
                            "Regularizer/personal_to_bridge",
                            latest_metrics.get("KL/priv_vs_bridge", ""),
                        ),
                        "bridge_to_personal_regularizer": latest_metrics.get(
                            "Regularizer/bridge_to_personal",
                            latest_metrics.get("KL/bridge_vs_priv", ""),
                        ),
                    }
                )
                completed_episode_returns.clear()
                completed_episode_lengths.clear()
                next_log_step += HYPERPARAMETERS[
                    "training_log_frequency_steps"
                ]

            if time_step >= next_checkpoint_step:
                save_agent(
                    agent,
                    run_dir
                    / "checkpoints"
                    / f"client_{rank}"
                    / f"step_{next_checkpoint_step}.pth",
                )
                next_checkpoint_step += HYPERPARAMETERS[
                    "checkpoint_frequency_steps"
                ]

            if done:
                break

        completed_episode_returns.append(episode_return)
        completed_episode_lengths.append(episode_length)
        if time_step < total_local_steps:
            state, _ = env.reset()

    env.close()
    final_checkpoint = (
        run_dir / "checkpoints" / f"client_{rank}" / "final_policy.pth"
    )
    save_agent(agent, final_checkpoint)
    comm.Barrier()

    # Test environments are first constructed here, after the final checkpoint.
    local_rows, heldout_rows, shift_rows = final_evaluation(agent, args, rank)

    gathered_training = comm.gather(training_rows, root=0)
    gathered_communication = comm.gather(communication_rows, root=0)
    gathered_local = comm.gather(local_rows, root=0)
    gathered_heldout = comm.gather(heldout_rows, root=0)
    gathered_shift = comm.gather(shift_rows, root=0)

    if rank == 0:
        flatten = lambda parts: [row for part in parts for row in part]
        training_output = flatten(gathered_training)
        communication_output = flatten(gathered_communication)
        if not communication_output:
            communication_output = [
                {
                    "environment": args.env_name,
                    "method_key": args.method,
                    "method": METHOD_LABELS[args.method],
                    "training_seed": args.seed,
                    "client_id": client_id,
                    "communication_round": 0,
                    "local_step": 0,
                    "protocol": "none",
                    "payload_bytes": 0,
                    "bytes_sent": 0,
                    "bytes_received": 0,
                    "messages_sent": 0,
                    "messages_received": 0,
                    "system_bytes": 0,
                    "communication_time_seconds": 0.0,
                }
                for client_id in range(size)
            ]
        local_output = flatten(gathered_local)
        heldout_output = flatten(gathered_heldout)
        shift_output = flatten(gathered_shift)
        write_csv(
            run_dir / "training_log.csv",
            training_output,
            [
                "environment",
                "method_key",
                "method",
                "training_seed",
                "client_id",
                "cluster",
                "local_step",
                "updates",
                "communication_rounds",
                "completed_episodes_in_window",
                "return_mean",
                "return_std",
                "episode_length_mean",
                "action_std",
                "personal_policy_loss",
                "value_loss",
                "bridge_policy_loss",
                "personal_to_bridge_regularizer",
                "bridge_to_personal_regularizer",
            ],
        )
        write_csv(
            run_dir / "communication_log.csv",
            communication_output,
            [
                "environment",
                "method_key",
                "method",
                "training_seed",
                "client_id",
                "communication_round",
                "local_step",
                "protocol",
                "payload_bytes",
                "bytes_sent",
                "bytes_received",
                "messages_sent",
                "messages_received",
                "system_bytes",
                "communication_time_seconds",
            ],
        )
        evaluation_fields = [
            "environment",
            "method_key",
            "method",
            "training_seed",
            "client_id",
            "source_cluster",
            "policy_type",
            "primary_eval_policy",
            "evaluation_seed",
            "repeat",
            "episode",
            "return",
            "episode_length",
            "evaluation_family",
            "target_cluster",
            "shift_factor",
            "shift_value",
            "mass",
            "friction",
            "gear",
            "incline_deg",
        ]
        write_csv(run_dir / "local_eval.csv", local_output, evaluation_fields)
        write_csv(run_dir / "heldout_eval.csv", heldout_output, evaluation_fields)
        write_csv(run_dir / "shift_eval.csv", shift_output, evaluation_fields)
        write_json(
            run_dir / "STATUS.json",
            {
                "status": "completed",
                "updated_at": utc_now(),
                "total_local_steps_per_client": total_local_steps,
                "updates_per_client": update_count,
                "communication_rounds": communication_round,
                "wall_time_seconds": time.time() - training_started,
                "row_counts": {
                    "training_log": len(training_output),
                    "communication_log": len(communication_output),
                    "local_eval": len(local_output),
                    "heldout_eval": len(heldout_output),
                    "shift_eval": len(shift_output),
                },
            },
        )
        print(f"[audited-run-completed] {run_dir}", flush=True)


def main() -> None:
    args = parse_args()
    configure_run(args)
    run_dir = Path(args.output_root) / run_slug(args)
    try:
        train(args)
    except Exception as exc:
        failure = {
            "status": "failed",
            "updated_at": utc_now(),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            write_json(run_dir / f"FAILURE_rank_{os.environ.get('OMPI_COMM_WORLD_RANK', 'unknown')}.json", failure)
        finally:
            raise


if __name__ == "__main__":
    main()
