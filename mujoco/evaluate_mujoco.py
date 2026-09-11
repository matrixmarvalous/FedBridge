#!/usr/bin/env python3
# Codex-added 2026-08-14: thin checkpoint-only D--F evaluator for the copied
# audited MuJoCo implementation. Intended caller: README evaluation command.
# Codex-modified 2026-08-18: load the public Bridge checkpoint schema.
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import torch

from experiment_spec import (
    ENVIRONMENTS,
    HELDOUT_CLUSTERS,
    HYPERPARAMETERS,
    METHOD_LABELS,
    N_CLIENTS,
    RESULT_ROOT,
    SINGLE_CRITIC_METHODS,
    task_hyperparameters,
)
from launcher import evaluate_episodes, run_slug
from runtime_support import build_agent, primary_policy_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate final client checkpoints on held-out clusters D--F."
    )
    parser.add_argument("--env-name", choices=ENVIRONMENTS, required=True)
    parser.add_argument("--method", choices=sorted(METHOD_LABELS), required=True)
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed associated with this user-generated training run.",
    )
    parser.add_argument("--n-clients", type=int, default=N_CLIENTS)
    parser.add_argument("--checkpoint-root", type=Path, default=RESULT_ROOT)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Explicit run directory; avoids any latest-checkpoint selection.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=1_000)
    return parser.parse_args()


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_agent_checkpoint(agent, method: str, path: Path) -> None:
    """Load the current public checkpoint schemas used by the copied algorithms."""
    payload = safe_torch_load(path)
    if method in SINGLE_CRITIC_METHODS:
        if payload.get("implementation") != "SingleCriticFedBridge":
            raise ValueError("Checkpoint does not use the single-critic schema")
        agent.personal_actor.load_state_dict(payload["personal_actor_state_dict"])
        agent.personal_actor_old.load_state_dict(payload["personal_actor_state_dict"])
        agent.bridge_actor.load_state_dict(payload["bridge_actor_state_dict"])
        agent.bridge_actor_old.load_state_dict(payload["bridge_actor_state_dict"])
        agent.critic.load_state_dict(payload["critic_state_dict"])
        agent.set_action_std(float(payload["action_std"]))
        return
    if method == "fedbridge_legacy_dualcritic":
        personal = payload["policy_priv_old_state_dict"]
        bridge = payload["policy_bridge_old_state_dict"]
        agent.policy_priv.load_state_dict(personal)
        agent.policy_priv_old.load_state_dict(personal)
        agent.policy_bridge.load_state_dict(bridge)
        agent.policy_bridge_old.load_state_dict(bridge)
        return
    if method in {"individual", "push_avg", "fedavg"}:
        agent.policy.load_state_dict(payload)
        agent.policy_old.load_state_dict(payload)
        return
    if method == "perfeddc":
        personal = payload["policy_priv_old_state_dict"]
        agent.policy_priv.load_state_dict(personal)
        agent.policy_priv_old.load_state_dict(personal)
        return
    if method == "pfedme":
        personal = payload["policy_priv_old_state_dict"]
        agent.policy_priv.load_state_dict(personal)
        agent.policy_priv_old.load_state_dict(personal)
        if "policy_global_state_dict" in payload:
            agent.policy_global.load_state_dict(payload["policy_global_state_dict"])
        return
    raise ValueError(f"Unsupported checkpoint method: {method}")


def write_csv(path: Path, rows: List[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No zero-shot rows were generated")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Mapping[str, object]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for cluster in ("D", "E", "F"):
        values = np.asarray(
            [float(row["return"]) for row in rows if row["target_cluster"] == cluster],
            dtype=float,
        )
        output.append(
            {
                "environment": rows[0]["environment"],
                "method_key": rows[0]["method_key"],
                "method": rows[0]["method"],
                "target_cluster": cluster,
                "mean_return": float(values.mean()),
                "standard_deviation": float(values.std(ddof=1))
                if values.size > 1
                else 0.0,
                "n_client_repeat_episodes": int(values.size),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    if args.seed < 0:
        raise ValueError("--seed must be a non-negative integer")
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != args.n_clients or size != N_CLIENTS:
        raise RuntimeError(f"Expected exactly {N_CLIENTS} MPI ranks, found {size}")
    if args.episodes < 1 or args.repeats < 1 or args.horizon < 1:
        raise ValueError("episodes, repeats and horizon must all be positive")

    HYPERPARAMETERS.update(task_hyperparameters(args.env_name, args.method))
    HYPERPARAMETERS["max_episode_length"] = int(args.horizon)
    run_dir = args.run_dir or (args.checkpoint_root / run_slug(args))
    checkpoint = run_dir / "checkpoints" / f"client_{rank}" / "final_policy.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    import gymnasium as gym

    probe = gym.make(args.env_name)
    state_dim = int(probe.observation_space.shape[0])
    action_dim = int(probe.action_space.shape[0])
    probe.close()
    agent = build_agent(args.method, state_dim, action_dim, torch.device("cpu"))
    load_agent_checkpoint(agent, args.method, checkpoint)

    rows: List[Dict[str, object]] = []
    policy_type = primary_policy_type(args.method)
    for cluster_index, (cluster, factors) in enumerate(
        HELDOUT_CLUSTERS[args.env_name].items()
    ):
        for repeat in range(args.repeats):
            rows.extend(
                evaluate_episodes(
                    agent=agent,
                    method=args.method,
                    env_name=args.env_name,
                    factors=factors,
                    policy_type=policy_type,
                    run_seed=args.seed,
                    client_id=rank,
                    family_id=2,
                    condition_id=cluster_index,
                    repeat=repeat,
                    episodes=args.episodes,
                    condition_fields={
                        "evaluation_family": "heldout_cluster",
                        "target_cluster": cluster,
                        "mass": factors["mass"],
                        "friction": factors["friction"],
                        "gear": factors["gear"],
                        "incline_deg": factors["incline_deg"],
                    },
                )
            )

    gathered = comm.gather(rows, root=0)
    if rank != 0:
        return
    combined = [row for part in gathered for row in part]
    output = args.output or (run_dir / "heldout_eval_recomputed.csv")
    write_csv(output, combined)
    write_csv(output.with_name("heldout_summary_recomputed.csv"), summarize(combined))
    protocol = {
        "environment": args.env_name,
        "method_key": args.method,
        "run_directory_name": run_dir.name,
        "clusters": HELDOUT_CLUSTERS[args.env_name],
        "episodes_per_client_cluster_repeat": args.episodes,
        "evaluation_repeats": args.repeats,
        "horizon": args.horizon,
        "deterministic_mean_action": True,
        "friction_component": "geom_friction[:, 0]",
        "fine_tuning": False,
    }
    output.with_name("heldout_protocol_recomputed.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[zero-shot-completed] {output}", flush=True)


if __name__ == "__main__":
    main()
