from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F

from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.policy import A2CPolicy
from tianshou.utils.net.common import ActorCritic


class BridgePPOPolicy(A2CPolicy):
    r"""Implementation of Proximal Policy Optimization. arXiv:1707.06347.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.nn.Module critic: the critic network. (s -> V(s))
    :param torch.optim.Optimizer optim: the optimizer for actor and critic network.
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param float eps_clip: :math:`\epsilon` in :math:`L_{CLIP}` in the original
        paper. Default to 0.2.
    :param float dual_clip: a parameter c mentioned in arXiv:1912.09729 Equ. 5,
        where c > 1 is a constant indicating the lower bound.
        Default to 5.0 (set None if you do not want to use it).
    :param bool value_clip: a parameter mentioned in arXiv:1811.02553v3 Sec. 4.1.
        Default to True.
    :param bool advantage_normalization: whether to do per mini-batch advantage
        normalization. Default to True.
    :param bool recompute_advantage: whether to recompute advantage every update
        repeat according to https://arxiv.org/pdf/2006.05990.pdf Sec. 3.5.
        Default to False.
    :param float vf_coef: weight for value loss. Default to 0.5.
    :param float ent_coef: weight for entropy loss. Default to 0.01.
    :param float max_grad_norm: clipping gradients in back propagation. Default to
        None.
    :param float gae_lambda: in [0, 1], param for Generalized Advantage Estimation.
        Default to 0.95.
    :param bool reward_normalization: normalize estimated values to have std close
        to 1, also normalize the advantage to Normal(0, 1). Default to False.
    :param int max_batchsize: the maximum size of the batch when computing GAE,
        depends on the size of available memory and the memory cost of the model;
        should be as large as possible within the memory constraint. Default to 256.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    :param bool deterministic_eval: whether to use deterministic action instead of
        stochastic action sampled by the policy. Default to False.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        eps_clip: float = 0.2,
        dual_clip: Optional[float] = None,
        value_clip: bool = False,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        ####
        actor_bridge: Optional[torch.nn.Module] = None,
        optim_bridge: Optional[torch.optim.Optimizer] = None,
        lambda_kl: float = 0.05,
        ####
        **kwargs: Any,
    ) -> None:
        super().__init__(actor, critic, optim, dist_fn, **kwargs)
        ####
        self.actor_bridge = actor_bridge
        self.optim_bridge = optim_bridge
        self.lambda_kl = lambda_kl
        ####
        self._eps_clip = eps_clip
        assert dual_clip is None or dual_clip > 1.0, \
            "Dual-clip PPO parameter should greater than 1.0."
        self._dual_clip = dual_clip
        self._value_clip = value_clip
        self._norm_adv = advantage_normalization
        self._recompute_adv = recompute_advantage
        self._actor_critic: ActorCritic
        


    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        if self._recompute_adv:
            # buffer input `buffer` and `indices` to be used in `learn()`.
            self._buffer, self._indices = buffer, indices
        batch = self._compute_returns(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        with torch.no_grad():
            batch.logp_old = self(batch, self._buffer, indices=batch.indices, is_obs=True).dist.log_prob(batch.act)
        return batch

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        optim_RL, optim_state = self.optim
        optim_bridge = self.optim_bridge  ##
        # Codex-modified 2026-09-09: the historical Individual baseline reused
        # BridgePPO with lambda_kl=0.  Computing KL anyway can produce inf and
        # then 0 * inf == nan.  Skip the mathematically inactive branch so the
        # personal update is exactly the standard PPO update in ppo.py.
        bridge_coupling = float(self.lambda_kl) != 0.0
        if bridge_coupling and (self.actor_bridge is None or optim_bridge is None):
            raise RuntimeError(
                "BridgePPO requires actor_bridge and optim_bridge when lambda_kl is non-zero"
            )
        for step in range(repeat):
            if self._recompute_adv and step > 0:
                batch = self._compute_returns(batch, self._buffer, self._indices)
            for minibatch in batch.split(batch_size, merge_last=True):
                # calculate loss for actor
                dist = self(minibatch, self._buffer, indices=minibatch.indices, is_obs=True).dist
                ## Calculate log_prob_bridge
                if bridge_coupling:
                    with torch.no_grad():
                        obs_emb = self.state_tracker(self._buffer, minibatch.indices, is_obs=True)
                        out_bridge = self.actor_bridge(obs_emb)

                        if isinstance(out_bridge, tuple):
                            # 连续动作：直接传给 dist_fn
                            dist_bridge = self.dist_fn(*out_bridge)
                        else:
                            logits_bridge = out_bridge
                            if getattr(self, "action_type", "") == "discrete" and hasattr(minibatch, "mask"):
                                logits_bridge = logits_bridge * minibatch.mask
                            dist_bridge = self.dist_fn(logits_bridge)
                            
                ###### end
                if self._norm_adv:
                    mean, std = minibatch.adv.mean(), minibatch.adv.std()
                    minibatch.adv = (minibatch.adv -
                                     mean) / (std + self._eps)  # per-batch norm
                ratio = (dist.log_prob(minibatch.act) -
                         minibatch.logp_old).exp().float()
                ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                surr1 = ratio * minibatch.adv
                surr2 = ratio.clamp(
                    1.0 - self._eps_clip, 1.0 + self._eps_clip
                ) * minibatch.adv
                if self._dual_clip:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self._dual_clip * minibatch.adv)
                    clip_loss = -torch.where(minibatch.adv < 0, clip2, clip1).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()
                # calculate loss for critic
                obs_emb = self.state_tracker(self._buffer, minibatch.indices, is_obs=True)
                value = self.critic(obs_emb).flatten()
                if self._value_clip:
                    v_clip = minibatch.v_s + \
                        (value - minibatch.v_s).clamp(-self._eps_clip, self._eps_clip)
                    vf1 = (minibatch.returns - value).pow(2)
                    vf2 = (minibatch.returns - v_clip).pow(2)
                    vf_loss = torch.max(vf1, vf2).mean()
                else:
                    vf_loss = (minibatch.returns - value).pow(2).mean()
                # calculate regularization and overall loss
                ent_loss = dist.entropy().mean()
                ##### calculate kl_divergence for private actor
                # kl_priv = F.kl_div(dist.log_prob(minibatch.act), log_prob_bridge, reduction='batchmean', log_target=True)
                loss = clip_loss + self._weight_vf * vf_loss \
                    - self._weight_ent * ent_loss
                if bridge_coupling:
                    kl_priv = kl_divergence(dist, dist_bridge).mean()
                    loss = loss + self.lambda_kl * kl_priv
                ####### end
                optim_RL.zero_grad()
                optim_state.zero_grad()
                loss.backward()
                if self._grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        self._actor_critic.parameters(), max_norm=self._grad_norm
                    )
                optim_RL.step()
                optim_state.step()

                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())

                if bridge_coupling:
                    ### update for actor_bridge
                    with torch.no_grad():
                        dist_priv_teacher = self(minibatch, self._buffer, indices=minibatch.indices, is_obs=True).dist

                    obs_emb = self.state_tracker(self._buffer, minibatch.indices, is_obs=True)
                    out_bridge = self.actor_bridge(obs_emb, state=None, info=minibatch.info)
                    if isinstance(out_bridge, tuple):
                        dist_bridge = self.dist_fn(*out_bridge)
                    else:
                        logits_bridge = out_bridge
                        if getattr(self, "action_type", "") == "discrete" and hasattr(minibatch, "mask"):
                            logits_bridge = logits_bridge * minibatch.mask
                        dist_bridge = self.dist_fn(logits_bridge)
                    log_prob_bridge = dist_bridge.log_prob(minibatch.act)

                    ent_loss_bridge = dist_bridge.entropy().mean()

                    ratio_bridge = (log_prob_bridge -
                             minibatch.logp_old).exp().float()
                    ratio_bridge = ratio_bridge.reshape(ratio_bridge.size(0), -1).transpose(0, 1)
                    surr1_bridge = ratio_bridge * minibatch.adv
                    surr2_bridge = ratio_bridge.clamp(
                        1.0 - self._eps_clip, 1.0 + self._eps_clip
                    ) * minibatch.adv
                    if self._dual_clip:
                        clip1_bridge = torch.min(surr1_bridge, surr2_bridge)
                        clip2_bridge = torch.max(clip1_bridge, self._dual_clip * minibatch.adv)
                        clip_loss_bridge = -torch.where(minibatch.adv < 0, clip2_bridge, clip1_bridge).mean()
                    else:
                        clip_loss_bridge = -torch.min(surr1_bridge, surr2_bridge).mean()

                    kl_bridge = kl_divergence(dist_bridge, dist_priv_teacher).mean()

                    loss_bridge = clip_loss_bridge - self._weight_ent * ent_loss_bridge + self.lambda_kl * kl_bridge

                    optim_bridge.zero_grad()
                    loss_bridge.backward()
                    optim_bridge.step()

                #### end
        return {
            "loss": losses,
            "loss/clip": clip_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
        }
