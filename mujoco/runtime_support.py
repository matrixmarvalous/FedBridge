#!/usr/bin/env python3
# Codex-added 2026-07-23: shared runtime adapters, environment construction and communication.
# Codex-modified 2026-08-14: import the packaged FedBridge algorithms from the
# publication directory instead of the private experiment directory.
# Codex-modified 2026-08-18: normalize dual-policy class, attribute and
# checkpoint metadata names; numerical behavior is retained.
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from experiment_spec import (
    BRIDGE_METHODS,
    CLUSTER_SEQUENCE,
    HYPERPARAMETERS,
    METHOD_LABELS,
    ALGORITHMS_DIR,
    SINGLE_CRITIC_METHODS,
    TRAIN_CLUSTERS,
    communication_protocol,
    policy_types,
)
from algorithms.single_critic_fedbridge import (
    GaussianActor,
    SingleCriticFedBridge,
    ValueCritic,
)


if str(ALGORITHMS_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHMS_DIR))


def client_cluster(client_id: int) -> str:
    return CLUSTER_SEQUENCE[int(client_id) % len(CLUSTER_SEQUENCE)]


def training_seed(run_seed: int, client_id: int) -> int:
    return 1_000_000 + 10_000 * int(run_seed) + int(client_id)


def initialization_seed(run_seed: int, client_id: int, component_offset: int) -> int:
    return 2_000_000 + 10_000 * int(run_seed) + 100 * int(client_id) + component_offset


def evaluation_seed(
    run_seed: int,
    client_id: int,
    family_id: int,
    condition_id: int,
    repeat: int,
    episode: int,
) -> int:
    return (
        100_000_000
        + 1_000_000 * int(run_seed)
        + 100_000 * int(family_id)
        + 10_000 * int(client_id)
        + 100 * int(condition_id)
        + 10 * int(repeat)
        + int(episode)
    )


def make_dynamics_env(
    env_name: str,
    factors: Mapping[str, float],
    random_seed: int,
):
    import gymnasium as gym

    env = gym.make(env_name)
    model = env.unwrapped.model
    model.body_mass[:] *= float(factors.get("mass", 1.0))
    model.geom_friction[:, 0] *= float(factors.get("friction", 1.0))
    if hasattr(model, "actuator_gear"):
        model.actuator_gear[:, 0] *= float(factors.get("gear", 1.0))
    incline_deg = float(factors.get("incline_deg", 0.0))
    if incline_deg:
        theta = math.radians(incline_deg)
        model.opt.gravity[:] = np.asarray(
            [9.81 * math.sin(theta), 0.0, -9.81 * math.cos(theta)],
            dtype=model.opt.gravity.dtype,
        )
    env.action_space.seed(int(random_seed))
    return env


def make_training_env(env_name: str, client_id: int, random_seed: int):
    cluster = client_cluster(client_id)
    return make_dynamics_env(env_name, TRAIN_CLUSTERS[cluster], random_seed)


def _common_agent_arguments(state_dim: int, action_dim: int) -> Dict[str, object]:
    return {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "lr_actor": HYPERPARAMETERS["actor_learning_rate"],
        "lr_critic": HYPERPARAMETERS["critic_learning_rate"],
        "gamma": HYPERPARAMETERS["gamma"],
        "K_epochs": HYPERPARAMETERS["ppo_epochs"],
        "eps_clip": HYPERPARAMETERS["eps_clip"],
        "has_continuous_action_space": True,
    }


def build_agent(
    method: str,
    state_dim: int,
    action_dim: int,
    device: torch.device,
):
    from PPO import PPO
    # Codex-modified 2026-08-19: import the final public module name.
    from BridgePPO import BridgePPO
    from perfeddc_ppo import perFedDC_PPO
    from pfedme_ppo import pFedMe_PPO

    common = _common_agent_arguments(state_dim, action_dim)
    if method in SINGLE_CRITIC_METHODS:
        return SingleCriticFedBridge(
            state_dim=state_dim,
            action_dim=action_dim,
            lr_actor=HYPERPARAMETERS["actor_learning_rate"],
            lr_critic=HYPERPARAMETERS["critic_learning_rate"],
            gamma=HYPERPARAMETERS["gamma"],
            k_epochs=HYPERPARAMETERS["ppo_epochs"],
            eps_clip=HYPERPARAMETERS["eps_clip"],
            lambda_kl=HYPERPARAMETERS["lambda_kl"],
            lambda_l2=HYPERPARAMETERS["lambda_l2_ablation"],
            device=device,
            variant_name=method,
            action_std_init=HYPERPARAMETERS["action_std_init"],
            minibatch_size=HYPERPARAMETERS["minibatch_size"],
            entropy_coefficient=HYPERPARAMETERS["entropy_coefficient"],
            value_coefficient=HYPERPARAMETERS["value_coefficient"],
        )
    if method == "fedbridge_legacy_dualcritic":
        return BridgePPO(
            **common,
            lambda_kl=HYPERPARAMETERS["lambda_kl"],
            device=device,
            action_std_init=HYPERPARAMETERS["action_std_init"],
        )
    if method in {"individual", "push_avg", "fedavg"}:
        return PPO(
            **common,
            device=device,
            action_std_init=HYPERPARAMETERS["action_std_init"],
        )
    if method == "perfeddc":
        return perFedDC_PPO(
            **common,
            lambda_l2=HYPERPARAMETERS["pfl_lambda_l2"],
            beta=HYPERPARAMETERS["pfl_beta"],
            device=device,
            action_std_init=HYPERPARAMETERS["action_std_init"],
        )
    if method == "pfedme":
        return pFedMe_PPO(
            **common,
            lambda_l2=HYPERPARAMETERS["pfl_lambda_l2"],
            beta=HYPERPARAMETERS["pfl_beta"],
            device=device,
            action_std_init=HYPERPARAMETERS["action_std_init"],
            local_steps=HYPERPARAMETERS["pfedme_local_steps"],
        )
    raise ValueError(f"Unsupported method: {method}")


def _seeded_module(factory, seed: int) -> nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return factory()


def _copy_parameters(destination: nn.Module, source: nn.Module) -> None:
    destination_parameters = list(destination.parameters())
    source_parameters = list(source.parameters())
    if len(destination_parameters) != len(source_parameters):
        raise RuntimeError("Canonical initialization parameter-count mismatch")
    with torch.no_grad():
        for destination_parameter, source_parameter in zip(
            destination_parameters, source_parameters
        ):
            if destination_parameter.shape != source_parameter.shape:
                raise RuntimeError(
                    f"Canonical initialization shape mismatch: "
                    f"{destination_parameter.shape} != {source_parameter.shape}"
                )
            destination_parameter.copy_(
                source_parameter.to(
                    device=destination_parameter.device,
                    dtype=destination_parameter.dtype,
                )
            )


def apply_canonical_initialization(
    agent,
    method: str,
    state_dim: int,
    action_dim: int,
    run_seed: int,
    client_id: int,
) -> None:
    personal_actor = _seeded_module(
        lambda: GaussianActor(state_dim, action_dim),
        initialization_seed(run_seed, client_id, 1),
    )
    bridge_actor = _seeded_module(
        lambda: GaussianActor(state_dim, action_dim),
        initialization_seed(run_seed, client_id, 2),
    )
    personal_critic = _seeded_module(
        lambda: ValueCritic(state_dim),
        initialization_seed(run_seed, client_id, 3),
    )
    legacy_bridge_critic = _seeded_module(
        lambda: ValueCritic(state_dim),
        initialization_seed(run_seed, client_id, 4),
    )

    if method in SINGLE_CRITIC_METHODS:
        _copy_parameters(agent.personal_actor, personal_actor)
        _copy_parameters(agent.personal_actor_old, personal_actor)
        _copy_parameters(agent.bridge_actor, bridge_actor)
        _copy_parameters(agent.bridge_actor_old, bridge_actor)
        _copy_parameters(agent.critic, personal_critic)
        return
    if method == "fedbridge_legacy_dualcritic":
        _copy_parameters(agent.policy_priv.actor, personal_actor)
        _copy_parameters(agent.policy_priv_old.actor, personal_actor)
        _copy_parameters(agent.policy_priv.critic, personal_critic)
        _copy_parameters(agent.policy_priv_old.critic, personal_critic)
        _copy_parameters(agent.policy_bridge.actor, bridge_actor)
        _copy_parameters(agent.policy_bridge_old.actor, bridge_actor)
        _copy_parameters(agent.policy_bridge.critic, legacy_bridge_critic)
        _copy_parameters(agent.policy_bridge_old.critic, legacy_bridge_critic)
        return
    if method in {"individual", "push_avg", "fedavg"}:
        _copy_parameters(agent.policy.actor, personal_actor)
        _copy_parameters(agent.policy_old.actor, personal_actor)
        _copy_parameters(agent.policy.critic, personal_critic)
        _copy_parameters(agent.policy_old.critic, personal_critic)
        return
    if method == "perfeddc":
        for policy in (agent.policy_priv, agent.policy_priv_old, agent.policy_hat):
            _copy_parameters(policy.actor, personal_actor)
            _copy_parameters(policy.critic, personal_critic)
        return
    if method == "pfedme":
        for policy in (agent.policy_priv, agent.policy_priv_old, agent.policy_global):
            _copy_parameters(policy.actor, personal_actor)
            _copy_parameters(policy.critic, personal_critic)
        return
    raise ValueError(method)


def update_agent(agent, method: str) -> Dict[str, float]:
    metrics = agent.update() if method in SINGLE_CRITIC_METHODS else (
        agent.update_dual() if method == "fedbridge_legacy_dualcritic" else agent.update()
    )
    return metrics or {}


def communicated_named_parameters(agent, method: str) -> Tuple[List[Tuple[str, nn.Parameter]], str]:
    if method in SINGLE_CRITIC_METHODS:
        return agent.communicated_named_parameters(), "bridge_actor_only"
    if method == "fedbridge_legacy_dualcritic":
        return [
            (f"bridge_policy.{name}", parameter)
            for name, parameter in agent.policy_bridge.named_parameters()
        ], "legacy_bridge_actor_and_critic"
    if method in {"individual", "push_avg", "fedavg"}:
        return [
            (f"actor.{name}", parameter)
            for name, parameter in agent.policy.actor.named_parameters()
        ], "actor_only"
    if method == "perfeddc":
        return [
            (f"personalized_policy.{name}", parameter)
            for name, parameter in agent.policy_priv.named_parameters()
        ], "personalized_actor_and_critic"
    if method == "pfedme":
        return [
            (f"global_policy.{name}", parameter)
            for name, parameter in agent.policy_global.named_parameters()
        ], "global_actor_and_critic"
    raise ValueError(method)


def payload_manifest(agent, method: str) -> Tuple[List[Dict[str, object]], int, str]:
    named_parameters, object_name = communicated_named_parameters(agent, method)
    rows: List[Dict[str, object]] = []
    total_bytes = 0
    enabled = communication_protocol(method) != "none"
    for name, parameter in named_parameters:
        num_scalars = int(parameter.numel())
        tensor_bytes = num_scalars * int(parameter.element_size())
        total_bytes += tensor_bytes
        rows.append(
            {
                "method_key": method,
                "communication_enabled": str(enabled).lower(),
                "object_name": object_name,
                "tensor_name": name,
                "num_scalars": num_scalars,
                "dtype": str(parameter.dtype),
                "bytes": tensor_bytes,
                "contains_critic": str("critic" in name.lower()).lower(),
            }
        )
    return rows, total_bytes, object_name


def architecture_manifest(agent, method: str) -> Dict[str, object]:
    if method in SINGLE_CRITIC_METHODS:
        return agent.architecture_manifest()
    if method == "fedbridge_legacy_dualcritic":
        return {
            "implementation": "legacy dual-critic BridgePPO",
            "diagnostic_only": True,
            "learned_personal_actor_count": 1,
            "learned_bridge_actor_count": 1,
            "learned_critic_count": 2,
            "bridge_critic_count": 1,
            "communicated_object": "legacy_bridge_actor_and_critic",
            "checkpoint_contains_bridge_critic": True,
        }
    if method in {"individual", "push_avg", "fedavg"}:
        return {
            "implementation": "original PPO.PPO",
            "learned_actor_count": 1,
            "learned_critic_count": 1,
            "communicated_object": (
                "none" if method == "individual" else "actor_only"
            ),
        }
    if method == "perfeddc":
        return {
            "implementation": "original perfeddc_ppo.perFedDC_PPO",
            "learned_personal_actor_count": 1,
            "learned_personal_critic_count": 1,
            "communicated_object": "personalized_actor_and_critic",
        }
    if method == "pfedme":
        return {
            "implementation": "original pfedme_ppo.pFedMe_PPO",
            "learned_personal_actor_count": 1,
            "learned_personal_critic_count": 1,
            "communicated_object": "global_actor_and_critic",
        }
    raise ValueError(method)


def flatten_parameters(named_parameters: Sequence[Tuple[str, nn.Parameter]]) -> np.ndarray:
    chunks = [parameter.detach().cpu().numpy().reshape(-1) for _, parameter in named_parameters]
    return np.concatenate(chunks).astype(np.float32) if chunks else np.empty(0, dtype=np.float32)


def load_flat_parameters(
    named_parameters: Sequence[Tuple[str, nn.Parameter]],
    flat_parameters: np.ndarray,
    device: torch.device,
) -> None:
    offset = 0
    with torch.no_grad():
        for _, parameter in named_parameters:
            count = parameter.numel()
            values = torch.from_numpy(flat_parameters[offset : offset + count]).to(
                device=device, dtype=parameter.dtype
            )
            parameter.copy_(values.view_as(parameter))
            offset += count
    if offset != int(flat_parameters.size):
        raise RuntimeError("Received communication payload has an unexpected size")


def sync_after_communication(agent, method: str) -> None:
    if method in SINGLE_CRITIC_METHODS:
        agent.sync_bridge_snapshot()
    elif method == "fedbridge_legacy_dualcritic":
        agent.policy_bridge_old.load_state_dict(agent.policy_bridge.state_dict())
    elif method in {"push_avg", "fedavg"}:
        agent.policy_old.actor.load_state_dict(agent.policy.actor.state_dict())
    elif method == "perfeddc":
        agent.policy_priv_old.load_state_dict(agent.policy_priv.state_dict())
    elif method in {"pfedme", "individual"}:
        return
    else:
        raise ValueError(method)


def communicate(
    agent,
    method: str,
    comm,
    rank: int,
    communication_round: int,
    device: torch.device,
) -> Dict[str, int]:
    protocol = communication_protocol(method)
    named_parameters, _ = communicated_named_parameters(agent, method)
    payload = flatten_parameters(named_parameters)
    payload_bytes = int(payload.nbytes)
    size = comm.Get_size()

    if protocol == "none":
        return {
            "bytes_sent": 0,
            "bytes_received": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "system_bytes": 0,
        }
    if protocol == "exponential_rotating_ring_replacement":
        offsets = (1, 2, 4)
        offset = offsets[int(communication_round) % len(offsets)]
        source = (int(rank) - offset) % size
        destination = (int(rank) + offset) % size
        received = np.empty_like(payload)
        comm.Sendrecv(payload, dest=destination, recvbuf=received, source=source)
        load_flat_parameters(named_parameters, received, device)
        sync_after_communication(agent, method)
        return {
            "bytes_sent": payload_bytes,
            "bytes_received": payload_bytes,
            "messages_sent": 1,
            "messages_received": 1,
            "system_bytes": 2 * size * payload_bytes,
        }
    if protocol == "centralized_server_average":
        gathered = comm.gather(payload, root=0)
        averaged = (
            np.mean(np.stack(gathered, axis=0), axis=0).astype(np.float32)
            if rank == 0
            else None
        )
        averaged = comm.bcast(averaged, root=0)
        load_flat_parameters(named_parameters, averaged, device)
        sync_after_communication(agent, method)
        return {
            "bytes_sent": payload_bytes,
            "bytes_received": payload_bytes,
            "messages_sent": 1,
            "messages_received": 1,
            "system_bytes": 2 * size * payload_bytes,
        }
    raise ValueError(protocol)


def deterministic_action(agent, method: str, state, policy_type: str):
    if method in SINGLE_CRITIC_METHODS:
        return agent.deterministic_action(state, policy_type)
    if method == "fedbridge_legacy_dualcritic":
        policy = (
            agent.policy_bridge_old if policy_type == "bridge" else agent.policy_priv_old
        )
    elif method in {"individual", "push_avg", "fedavg"}:
        policy = agent.policy_old
    elif method == "perfeddc":
        policy = agent.policy_priv_old
    elif method == "pfedme":
        policy = agent.policy_global if policy_type == "global" else agent.policy_priv_old
    else:
        raise ValueError(method)
    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=agent.device)
        return policy.actor(state_tensor).detach().cpu().numpy().flatten()


def primary_policy_type(method: str) -> str:
    if method in BRIDGE_METHODS:
        return "personal"
    if method == "pfedme":
        return "personal"
    if method == "fedavg":
        return "global"
    if method == "perfeddc":
        return "personalized"
    return "local"


def runtime_contract(agent, method: str) -> Dict[str, object]:
    architecture = architecture_manifest(agent, method)
    payload_rows, payload_bytes, object_name = payload_manifest(agent, method)
    errors: List[str] = []
    if method in SINGLE_CRITIC_METHODS:
        if architecture.get("learned_critic_count") != 1:
            errors.append("single-critic method does not report exactly one learned critic")
        if architecture.get("bridge_critic_count") != 0:
            errors.append("single-critic method reports a bridge critic")
        if any(row["contains_critic"] == "true" for row in payload_rows):
            errors.append("single-critic bridge payload contains critic parameters")
        if object_name != "bridge_actor_only":
            errors.append("single-critic communicated object is not bridge_actor_only")
    if method == "fedbridge_legacy_dualcritic":
        if architecture.get("learned_critic_count") != 2:
            errors.append("legacy diagnostic no longer exposes the expected two critics")
        if not any(row["contains_critic"] == "true" for row in payload_rows):
            errors.append("legacy diagnostic payload no longer contains its bridge critic")
    if errors:
        raise RuntimeError("; ".join(errors))
    return {
        "status": "passed",
        "method_key": method,
        "method_label": METHOD_LABELS[method],
        "architecture": architecture,
        "payload_bytes": payload_bytes,
        "communicated_object": object_name,
        "communication_protocol": communication_protocol(method),
        "checks": {
            "target_single_critic": method not in SINGLE_CRITIC_METHODS
            or architecture["learned_critic_count"] == 1,
            "target_has_no_bridge_critic": method not in SINGLE_CRITIC_METHODS
            or architecture["bridge_critic_count"] == 0,
            "target_payload_actor_only": method not in SINGLE_CRITIC_METHODS
            or all(row["contains_critic"] == "false" for row in payload_rows),
        },
    }
