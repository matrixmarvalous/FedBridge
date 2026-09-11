#!/usr/bin/env python3
# Codex-added 2026-07-23: auditable FedBridge with two actors and one learned critic.
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, kl_divergence


class RolloutBuffer:
    def __init__(self) -> None:
        self.actions: List[torch.Tensor] = []
        self.states: List[torch.Tensor] = []
        self.logprobs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.state_values: List[torch.Tensor] = []
        self.is_terminals: List[bool] = []

    def clear(self) -> None:
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.is_terminals.clear()


class GaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh(),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


class ValueCritic(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


@dataclass(frozen=True)
class Variant:
    regularizer: str
    personal_regularization: bool
    bridge_regularization: bool
    bridge_reward_aware: bool


VARIANTS: Dict[str, Variant] = {
    "fedbridge_singlecritic": Variant("kl", True, True, True),
    "no_exchange_singlecritic": Variant("kl", True, True, True),
    "one_way_kl_singlecritic": Variant("kl", True, False, True),
    "distill_only_singlecritic": Variant("kl", True, True, False),
    "mutual_l2_singlecritic": Variant("l2", True, True, True),
}


class SingleCriticFedBridge:
    """PPO with a personal actor, a bridge actor and exactly one learned critic."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float,
        lr_critic: float,
        gamma: float,
        k_epochs: int,
        eps_clip: float,
        lambda_kl: float,
        lambda_l2: float,
        device: torch.device,
        variant_name: str,
        action_std_init: float = 0.6,
        minibatch_size: int = 4000,
        entropy_coefficient: float = 0.01,
        value_coefficient: float = 0.5,
    ) -> None:
        if variant_name not in VARIANTS:
            raise ValueError(f"Unknown single-critic variant: {variant_name}")
        self.variant_name = variant_name
        self.variant = VARIANTS[variant_name]
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.device = device
        self.gamma = float(gamma)
        self.k_epochs = int(k_epochs)
        self.eps_clip = float(eps_clip)
        self.lambda_kl = float(lambda_kl)
        self.lambda_l2 = float(lambda_l2)
        self.minibatch_size = int(minibatch_size)
        self.entropy_coefficient = float(entropy_coefficient)
        self.value_coefficient = float(value_coefficient)
        self.init_action_std = float(action_std_init)
        self.action_std = float(action_std_init)

        self.personal_actor = GaussianActor(state_dim, action_dim).to(device)
        self.bridge_actor = GaussianActor(state_dim, action_dim).to(device)
        self.critic = ValueCritic(state_dim).to(device)

        # Old actors are frozen PPO sampling snapshots, not additional learned policies.
        self.personal_actor_old = GaussianActor(state_dim, action_dim).to(device)
        self.bridge_actor_old = GaussianActor(state_dim, action_dim).to(device)
        self.personal_actor_old.load_state_dict(self.personal_actor.state_dict())
        self.bridge_actor_old.load_state_dict(self.bridge_actor.state_dict())
        for parameter in self.personal_actor_old.parameters():
            parameter.requires_grad = False
        for parameter in self.bridge_actor_old.parameters():
            parameter.requires_grad = False

        self.personal_optimizer = torch.optim.Adam(
            [
                {"params": self.personal_actor.parameters(), "lr": lr_actor},
                {"params": self.critic.parameters(), "lr": lr_critic},
            ]
        )
        self.bridge_optimizer = torch.optim.Adam(self.bridge_actor.parameters(), lr=lr_actor)
        self.value_loss = nn.MSELoss()
        self.buffer = RolloutBuffer()

    def _action_var(self, batch_size: int) -> torch.Tensor:
        variance = torch.full(
            (batch_size, self.action_dim),
            self.action_std * self.action_std,
            dtype=torch.float32,
            device=self.device,
        )
        return torch.diag_embed(variance)

    def _distribution(self, actor: nn.Module, states: torch.Tensor) -> MultivariateNormal:
        if states.ndim == 1:
            states = states.unsqueeze(0)
        means = actor(states)
        return MultivariateNormal(means, self._action_var(means.shape[0]))

    def set_action_std(self, new_action_std: float) -> None:
        self.action_std = float(new_action_std)

    def decay_action_std(self, decay_rate: float, minimum: float) -> None:
        self.set_action_std(max(float(minimum), round(self.action_std - float(decay_rate), 4)))

    def select_action(self, state) -> torch.Tensor:
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            dist = self._distribution(self.personal_actor_old, state_tensor)
            action = dist.sample().squeeze(0)
            logprob = dist.log_prob(action.unsqueeze(0)).squeeze(0)
            value = self.critic(state_tensor).squeeze(-1)
        self.buffer.states.append(state_tensor)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(logprob)
        self.buffer.state_values.append(value)
        return action.detach().cpu().numpy().flatten()

    def deterministic_action(self, state, policy_type: str = "personal"):
        actor = self.personal_actor_old if policy_type == "personal" else self.bridge_actor_old
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            return actor(state_tensor).detach().cpu().numpy().flatten()

    @staticmethod
    def _l2_distance(source: nn.Module, target: nn.Module, device: torch.device) -> torch.Tensor:
        total = torch.zeros((), device=device)
        for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
            total = total + F.mse_loss(
                source_parameter,
                target_parameter.detach(),
                reduction="sum",
            )
        return total

    def _regularization(
        self,
        source: nn.Module,
        target: nn.Module,
        states: torch.Tensor,
    ) -> torch.Tensor:
        if self.variant.regularizer == "l2":
            return self._l2_distance(source, target, self.device)
        source_dist = self._distribution(source, states)
        with torch.no_grad():
            target_dist = self._distribution(target, states)
        return kl_divergence(source_dist, target_dist).mean()

    def _actor_terms(
        self,
        actor: nn.Module,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._distribution(actor, states)
        logprobs = dist.log_prob(actions)
        ratios = torch.exp(logprobs - old_logprobs)
        surrogate_1 = ratios * advantages
        surrogate_2 = torch.clamp(
            ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip
        ) * advantages
        policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
        entropy = dist.entropy().mean()
        return policy_loss, entropy, logprobs

    def update(self) -> Dict[str, float]:
        if not self.buffer.rewards:
            raise RuntimeError("Cannot update from an empty rollout buffer")

        discounted_returns: List[float] = []
        discounted = 0.0
        for reward, terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if terminal:
                discounted = 0.0
            discounted = float(reward) + self.gamma * discounted
            discounted_returns.insert(0, discounted)

        returns = torch.tensor(discounted_returns, dtype=torch.float32, device=self.device)
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-7)
        states = torch.stack(self.buffer.states).detach()
        actions = torch.stack(self.buffer.actions).detach()
        old_logprobs = torch.stack(self.buffer.logprobs).detach().reshape(-1)
        old_values = torch.stack(self.buffer.state_values).detach().reshape(-1)
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

        metric_totals = {
            "Loss/personal_total": 0.0,
            "Loss/personal_policy": 0.0,
            "Loss/value": 0.0,
            "Loss/bridge_total": 0.0,
            "Loss/bridge_policy": 0.0,
            "Entropy/personal": 0.0,
            "Entropy/bridge": 0.0,
            "Regularizer/personal_to_bridge": 0.0,
            "Regularizer/bridge_to_personal": 0.0,
            "Advantage/mean": float(advantages.mean().item()),
        }
        num_samples = states.shape[0]
        minibatch_size = min(max(1, self.minibatch_size), num_samples)
        num_updates = 0

        for _ in range(self.k_epochs):
            permutation = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, minibatch_size):
                indices = permutation[start : start + minibatch_size]
                states_b = states[indices]
                actions_b = actions[indices]
                old_logprobs_b = old_logprobs[indices]
                advantages_b = advantages[indices]
                returns_b = returns[indices]

                personal_policy_loss, personal_entropy, _ = self._actor_terms(
                    self.personal_actor,
                    states_b,
                    actions_b,
                    old_logprobs_b,
                    advantages_b,
                )
                predicted_values = self.critic(states_b).squeeze(-1)
                value_loss = self.value_coefficient * self.value_loss(
                    predicted_values, returns_b
                )
                personal_regularizer = torch.zeros((), device=self.device)
                if self.variant.personal_regularization:
                    personal_regularizer = self._regularization(
                        self.personal_actor, self.bridge_actor, states_b
                    )
                regularization_scale = (
                    self.lambda_kl * self.action_std / self.init_action_std
                    if self.variant.regularizer == "kl"
                    else self.lambda_l2
                )
                personal_total = (
                    personal_policy_loss
                    + value_loss
                    - self.entropy_coefficient * personal_entropy
                    + regularization_scale * personal_regularizer
                )
                self.personal_optimizer.zero_grad()
                personal_total.backward()
                self.personal_optimizer.step()

                bridge_regularizer = torch.zeros((), device=self.device)
                if self.variant.bridge_regularization:
                    bridge_regularizer = self._regularization(
                        self.bridge_actor, self.personal_actor, states_b
                    )
                if self.variant.bridge_reward_aware:
                    bridge_policy_loss, bridge_entropy, _ = self._actor_terms(
                        self.bridge_actor,
                        states_b,
                        actions_b,
                        old_logprobs_b,
                        advantages_b,
                    )
                    bridge_total = (
                        bridge_policy_loss
                        - self.entropy_coefficient * bridge_entropy
                        + regularization_scale * bridge_regularizer
                    )
                else:
                    bridge_policy_loss = torch.zeros((), device=self.device)
                    bridge_entropy = torch.zeros((), device=self.device)
                    bridge_total = regularization_scale * bridge_regularizer

                self.bridge_optimizer.zero_grad()
                bridge_total.backward()
                self.bridge_optimizer.step()

                metric_totals["Loss/personal_total"] += float(personal_total.item())
                metric_totals["Loss/personal_policy"] += float(personal_policy_loss.item())
                metric_totals["Loss/value"] += float(value_loss.item())
                metric_totals["Loss/bridge_total"] += float(bridge_total.item())
                metric_totals["Loss/bridge_policy"] += float(bridge_policy_loss.item())
                metric_totals["Entropy/personal"] += float(personal_entropy.item())
                metric_totals["Entropy/bridge"] += float(bridge_entropy.item())
                metric_totals["Regularizer/personal_to_bridge"] += float(
                    personal_regularizer.item()
                )
                metric_totals["Regularizer/bridge_to_personal"] += float(
                    bridge_regularizer.item()
                )
                num_updates += 1

        self.personal_actor_old.load_state_dict(self.personal_actor.state_dict())
        self.bridge_actor_old.load_state_dict(self.bridge_actor.state_dict())
        self.buffer.clear()
        for key in metric_totals:
            if key != "Advantage/mean":
                metric_totals[key] /= float(num_updates)
        return metric_totals

    def communicated_named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        return [
            (f"bridge_actor.{name}", parameter)
            for name, parameter in self.bridge_actor.named_parameters()
        ]

    def sync_bridge_snapshot(self) -> None:
        self.bridge_actor_old.load_state_dict(self.bridge_actor.state_dict())

    def architecture_manifest(self) -> Dict[str, object]:
        return {
            "implementation": "SingleCriticFedBridge",
            "variant_name": self.variant_name,
            "variant": asdict(self.variant),
            "learned_personal_actor_count": 1,
            "learned_bridge_actor_count": 1,
            "learned_critic_count": 1,
            "bridge_critic_count": 0,
            "personal_actor_parameters": sum(
                parameter.numel() for parameter in self.personal_actor.parameters()
            ),
            "bridge_actor_parameters": sum(
                parameter.numel() for parameter in self.bridge_actor.parameters()
            ),
            "critic_parameters": sum(
                parameter.numel() for parameter in self.critic.parameters()
            ),
            "communicated_object": "bridge_actor_only",
            "checkpoint_contains_bridge_critic": False,
        }

    def save(self, checkpoint_path: str | Path) -> None:
        torch.save(
            {
                "schema_version": 1,
                "implementation": "SingleCriticFedBridge",
                "variant_name": self.variant_name,
                "personal_actor_state_dict": self.personal_actor_old.state_dict(),
                "bridge_actor_state_dict": self.bridge_actor_old.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "action_std": self.action_std,
                "architecture_manifest": self.architecture_manifest(),
            },
            str(checkpoint_path),
        )

    def load(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        if checkpoint.get("implementation") != "SingleCriticFedBridge":
            raise ValueError("Checkpoint does not use the audited single-critic schema")
        self.personal_actor.load_state_dict(checkpoint["personal_actor_state_dict"])
        self.personal_actor_old.load_state_dict(checkpoint["personal_actor_state_dict"])
        self.bridge_actor.load_state_dict(checkpoint["bridge_actor_state_dict"])
        self.bridge_actor_old.load_state_dict(checkpoint["bridge_actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.set_action_std(float(checkpoint["action_std"]))

