from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer
from tianshou.exploration import BaseNoise, GaussianNoise
from tianshou.policy import DDPGPolicy
import torch.nn.functional as F


class BridgeTD3Policy(DDPGPolicy):
    """Implementation of TD3, arXiv:1802.09477.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic1: the first critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic1_optim: the optimizer for the first
        critic network.
    :param torch.nn.Module critic2: the second critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic2_optim: the optimizer for the second
        critic network.
    :param float tau: param for soft update of the target network. Default to 0.005.
    :param float gamma: discount factor, in [0, 1]. Default to 0.99.
    :param float exploration_noise: the exploration noise, add to the action.
        Default to ``GaussianNoise(sigma=0.1)``
    :param float policy_noise: the noise used in updating policy network.
        Default to 0.2.
    :param int update_actor_freq: the update frequency of actor network.
        Default to 2.
    :param float noise_clip: the clipping range used in updating policy network.
        Default to 0.5.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action) or empty string for no bounding.
        Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1: torch.nn.Module,
        critic1_optim: torch.optim.Optimizer,
        critic2: torch.nn.Module,
        critic2_optim: torch.optim.Optimizer,
        optim_state: Optional[torch.optim.Optimizer],
        tau: float = 0.005,
        gamma: float = 0.99,
        exploration_noise: Optional[BaseNoise] = GaussianNoise(sigma=0.1),
        policy_noise: float = 0.2,
        update_actor_freq: int = 2,
        noise_clip: float = 0.5,
        reward_normalization: bool = False,
        state_tracker = None,
        estimation_step: int = 1,
        ####
        actor_bridge: Optional[torch.nn.Module] = None,
        optim_bridge: Optional[torch.optim.Optimizer] = None,
        lambda_kl: float = 0.05,
        ####
        **kwargs: Any,
    ) -> None:
        super().__init__(
            actor, actor_optim, None, None, optim_state, tau, gamma, exploration_noise,
            reward_normalization, state_tracker, estimation_step, **kwargs
        )
        ####
        self.actor_bridge = actor_bridge
        self.optim_bridge = optim_bridge
        self.lambda_kl = lambda_kl
        ####
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic1_optim = critic1_optim
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()
        self.critic2_optim = critic2_optim
        self._policy_noise = policy_noise
        self._freq = update_actor_freq
        self._noise_clip = noise_clip
        self._cnt = 0
        self._last = 0

    def train(self, mode: bool = True) -> "TD3Policy":
        self.training = mode
        self.actor.train(mode)
        self.critic1.train(mode)
        self.critic2.train(mode)
        return self

    def sync_weight(self) -> None:
        self.soft_update(self.critic1_old, self.critic1, self.tau)
        self.soft_update(self.critic2_old, self.critic2, self.tau)
        self.soft_update(self.actor_old, self.actor, self.tau)

    def _target_q(self, batch, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        # batch = buffer[indices]  # batch.obs: s_{t+n}
        act_ = self(batch, buffer, indices, model='actor_old', input='obs_next').act
        noise = torch.randn(size=act_.shape, device=act_.device) * self._policy_noise
        if self._noise_clip > 0.0:
            noise = noise.clamp(-self._noise_clip, self._noise_clip)
        act_ += noise
        obs_next_emb = self.state_tracker(buffer=buffer, indices=indices, is_obs=False)
        target_q = torch.min(
            self.critic1_old(obs_next_emb, act_),
            self.critic2_old(obs_next_emb, act_),
        )
        return target_q

#     def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
#         self.optim_state.zero_grad()
#         # critic 1&2
#         td1, critic1_loss = self._mse_optimizer(
#             batch, self.critic1, self.critic1_optim
#         )
#         td2, critic2_loss = self._mse_optimizer(
#             batch, self.critic2, self.critic2_optim
#         )
#         batch.weight = (td1 + td2) / 2.0  # prio-buffer

#         # # actor
#         # if self._cnt % self._freq == 0:
#         #     obs_emb = self.state_tracker(self._buffer, indices=batch.indices, is_obs=True)
#         #     actor_loss = -self.critic1(obs_emb, self(batch, self._buffer, batch.indices, eps=0.0).act).mean()
#         #     self.actor_optim.zero_grad()
#         #     actor_loss.backward()
#         #     self._last = actor_loss.item()
#         #     self.actor_optim.step()
#         #     self.optim_state.step()
#         #     self.sync_weight()
#         # self._cnt += 1
        
#         if self._cnt % self._freq == 0:
#             obs_emb = self.state_tracker(self._buffer, indices=batch.indices, is_obs=True)

#             # === 主 actor（personal） ===
#             a_personal = self(batch, self._buffer, batch.indices, eps=0.0).act  # 确定性动作

#             # 老师：bridge 的当前动作（不回传梯度）
#             if self.actor_bridge is None or self.optim_bridge is None:
#                 raise RuntimeError("actor_bridge/optim_bridge is required for Bridge-TD3.")
#             with torch.no_grad():
#                 out = self.actor_bridge(obs_emb, state=None, info=batch.info)
#                 # ★ 统一转成动作张量
#                 if isinstance(out, tuple):
#                     a_bridge_teacher = out[0]
#                 else:
#                     a_bridge_teacher = out
#                 # 若 actor_bridge 输出的是 pre-tanh logits，这里需要和主 actor 保持一致（如果主 actor 是 tanh 后的）
#                 # a_bridge_teacher = torch.tanh(a_bridge_teacher)  # 视你的 actor_bridge 实现而定

#                 # ★ 保证 dtype/device 一致
#                 a_bridge_teacher = a_bridge_teacher.to(dtype=a_personal.dtype, device=a_personal.device)


#             # TD3 主项：-Q1(s, a_personal)
#             q1_personal = self.critic1(obs_emb, a_personal).mean()

#             # 蒸馏对齐（纯 MSE）：||a_personal - a_bridge||^2
#             align_personal = F.mse_loss(a_personal, a_bridge_teacher, reduction="mean")

#             actor_loss = - q1_personal + self.lambda_kl * align_personal

#             self.actor_optim.zero_grad()
#             actor_loss.backward()
#             self._last = float(actor_loss.item())
#             self.actor_optim.step()

#             with torch.no_grad():
#                 out_p = self.actor(obs_emb, state=None, info=batch.info)
#                 a_personal_teacher = out_p[0] if isinstance(out_p, tuple) else out_p
#                 # 若需要：a_personal_teacher = torch.tanh(a_personal_teacher)
#                 a_personal_teacher = a_personal_teacher.to(dtype=a_bridge.dtype, device=a_bridge.device)




#             out_b = self.actor_bridge(obs_emb, state=None, info=batch.info)
#             a_bridge = out_b[0] if isinstance(out_b, tuple) else out_b
#             # 若需要：a_bridge = torch.tanh(a_bridge)
#             a_bridge = a_bridge.to(dtype=a_personal.dtype, device=a_personal.device)
#             q1_bridge = self.critic1(obs_emb, a_bridge).mean()

#             align_bridge = F.mse_loss(a_bridge, a_personal_teacher.detach(), reduction="mean")
#             bridge_loss = - q1_bridge + self.lambda_kl * align_bridge

#             self.optim_bridge.zero_grad()
#             bridge_loss.backward()
#             self.optim_bridge.step()

#             # 维持你原来的状态优化 & 软更新顺序
#             self.optim_state.step()
#             self.sync_weight()
#         self._cnt += 1

#         return {
#             "loss/actor": self._last,
#             "loss/critic1": critic1_loss.item(),
#             "loss/critic2": critic2_loss.item(),
#         }


    import torch.nn.functional as F  # 确保在文件顶部已导入

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        self.optim_state.zero_grad()

        # ====== critics ======
        td1, critic1_loss = self._mse_optimizer(batch, self.critic1, self.critic1_optim)
        td2, critic2_loss = self._mse_optimizer(batch, self.critic2, self.critic2_optim)
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # 小工具：把网络输出统一成“动作张量”
        def _to_action_tensor(net_out):
            return net_out[0] if isinstance(net_out, tuple) else net_out

        # ====== delayed policy update (TD3) ======
        if self._cnt % self._freq == 0:
            # -------- 主 actor（personal）更新 --------
            obs_emb = self.state_tracker(self._buffer, indices=batch.indices, is_obs=True)

            # 确定性动作（你的 policy 前向里已处理 eps/tanh/bound）
            a_personal = self(batch, self._buffer, batch.indices, eps=0.0).act

            if self.actor_bridge is None or self.optim_bridge is None:
                raise RuntimeError("actor_bridge/optim_bridge is required for Bridge-TD3.")

            # 老师：bridge 的当前动作（stop-grad）
            with torch.no_grad():
                out_b_teacher = self.actor_bridge(obs_emb, state=None, info=batch.info)
                a_bridge_teacher = _to_action_tensor(out_b_teacher)
                # 若 bridge 输出是 pre-tanh，请与 a_personal 的尺度保持一致：
                # a_bridge_teacher = torch.tanh(a_bridge_teacher)
                a_bridge_teacher = a_bridge_teacher.to(dtype=a_personal.dtype, device=a_personal.device)

            # TD3 主项：-Q1(s, a_personal)
            q1_personal = self.critic1(obs_emb, a_personal).mean()

            # 纯 MSE 蒸馏：||a_personal - a_bridge_teacher||^2
            align_personal = F.mse_loss(a_personal, a_bridge_teacher, reduction="mean")

            actor_loss = - q1_personal + self.lambda_kl * align_personal

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self._last = float(actor_loss.item())
            self.actor_optim.step()

            # -------- Bridge 更新（重算 obs_emb，避免二次反传同一图）--------
            obs_emb_b = self.state_tracker(self._buffer, indices=batch.indices, is_obs=True)

            # 老师：最新主 actor 的动作（stop-grad）
            with torch.no_grad():
                out_p_teacher = self.actor(obs_emb_b, state=None, info=batch.info)
                a_personal_teacher = _to_action_tensor(out_p_teacher)
                # 若主 actor 输出是 pre-tanh，请与 a_personal 的尺度保持一致：
                # a_personal_teacher = torch.tanh(a_personal_teacher)
                a_personal_teacher = a_personal_teacher.to(dtype=a_personal.dtype, device=a_personal.device)

            # 学生：bridge 的动作
            out_b = self.actor_bridge(obs_emb_b, state=None, info=batch.info)
            a_bridge = _to_action_tensor(out_b)
            # 若 bridge 输出是 pre-tanh，请与 a_personal 的尺度保持一致：
            # a_bridge = torch.tanh(a_bridge)
            a_bridge = a_bridge.to(dtype=a_personal.dtype, device=a_personal.device)

            q1_bridge = self.critic1(obs_emb_b, a_bridge).mean()
            align_bridge = F.mse_loss(a_bridge, a_personal_teacher.detach(), reduction="mean")
            bridge_loss = - q1_bridge + self.lambda_kl * align_bridge

            self.optim_bridge.zero_grad()
            bridge_loss.backward()
            self.optim_bridge.step()

            # 状态跟踪器参数一次 step（累积了两路的梯度）
            self.optim_state.step()
            self.sync_weight()

        self._cnt += 1

        return {
            "loss/actor": self._last,
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }


    
    
    
    from copy import deepcopy
# from typing import Any, Callable, Dict, Optional, Tuple

# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch.distributions import Categorical

# from tianshou.data import Batch, ReplayBuffer
# from tianshou.exploration import BaseNoise, GaussianNoise
# from tianshou.policy import DDPGPolicy


# class BridgeTD3Policy(DDPGPolicy):
#     """TD3 + discrete bridge head (compatible with A2C/PPO bridge).

#     Main actor: deterministic continuous action.
#     Bridge head: Categorical logits over K candidates (parameter-shape compatible
#     with A2C/PPO Actor). For TD3 side we map the discrete π(a|s) to an expected
#     action vector via candidate embeddings, and align it with the continuous actor.
#     """

#     def __init__(
#         self,
#         actor: torch.nn.Module,
#         actor_optim: torch.optim.Optimizer,
#         critic1: torch.nn.Module,
#         critic1_optim: torch.optim.Optimizer,
#         critic2: torch.nn.Module,
#         critic2_optim: torch.optim.Optimizer,
#         optim_state: Optional[torch.optim.Optimizer],
#         tau: float = 0.005,
#         gamma: float = 0.99,
#         exploration_noise: Optional[BaseNoise] = GaussianNoise(sigma=0.1),
#         policy_noise: float = 0.2,
#         update_actor_freq: int = 2,
#         noise_clip: float = 0.5,
#         reward_normalization: bool = False,
#         state_tracker=None,
#         estimation_step: int = 1,
#         # === Bridge-specific ===
#         actor_bridge: Optional[torch.nn.Module] = None,          # outputs logits ∈ R^K
#         optim_bridge: Optional[torch.optim.Optimizer] = None,
#         action_embedder: Optional[
#             Callable[[ReplayBuffer, np.ndarray, Batch], Tuple[torch.Tensor, Optional[torch.Tensor]]]
#         ] = None,  # returns (E: [B,K,d], mask: [B,K] or None)
#         temperature: float = 1.0,        # τ for Q-softening
#         w_teach: float = 1.0,            # weight for KL(softQ || π_bridge)
#         w_align_actor: float = 1e-2,     # weight for ||a_det - E[π_bridge]||^2 when updating main actor
#         w_align_bridge: float = 1e-2,    # weight for ||E[π_bridge] - a_det||^2 when updating bridge
#         w_entropy: float = 0.0,          # entropy bonus for bridge
#         **kwargs: Any,
#     ) -> None:
#         super().__init__(
#             actor, actor_optim, None, None, optim_state, tau, gamma,
#             exploration_noise, reward_normalization, state_tracker, estimation_step, **kwargs
#         )
#         # critics
#         self.critic1, self.critic1_old = critic1, deepcopy(critic1); self.critic1_old.eval()
#         self.critic1_optim = critic1_optim
#         self.critic2, self.critic2_old = critic2, deepcopy(critic2); self.critic2_old.eval()
#         self.critic2_optim = critic2_optim

#         # TD3 hyper-params
#         self._policy_noise = policy_noise
#         self._freq = update_actor_freq
#         self._noise_clip = noise_clip
#         self._cnt = 0
#         self._last = 0.0

#         # bridge bits
#         self.actor_bridge = actor_bridge
#         self.optim_bridge = optim_bridge
#         self.action_embedder = action_embedder
#         self.temperature = float(temperature)
#         self.w_teach = float(w_teach)
#         self.w_align_actor = float(w_align_actor)
#         self.w_align_bridge = float(w_align_bridge)
#         self.w_entropy = float(w_entropy)

#     # -------- utilities --------
#     def _apply_mask(self, logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
#         if mask is None:
#             return logits
#         # ensure boolean
#         m = mask.bool()
#         # -inf on invalid positions for proper categorical
#         neg_inf = torch.finfo(logits.dtype).min
#         return logits.masked_fill(~m, neg_inf)

#     def _get_candidates(
#         self, buffer: ReplayBuffer, indices: np.ndarray, batch: Batch
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
#         """Return (E, mask): E [B,K,d], mask [B,K] or None."""
#         if hasattr(batch, "cand_emb"):
#             E = batch.cand_emb  # expected to be torch.Tensor
#             mask = getattr(batch, "mask", None)
#             return E, mask
#         if self.action_embedder is not None:
#             E, mask = self.action_embedder(buffer, indices, batch)
#             return E, mask
#         raise RuntimeError(
#             "BridgeTD3Policy: need candidate embeddings. "
#             "Provide batch.cand_emb or pass action_embedder=..."
#         )

#     def _expected_action(self, probs: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
#         # probs: [B,K], E: [B,K,d] -> [B,d]
#         return (probs.unsqueeze(-1) * E).sum(dim=1)

#     def train(self, mode: bool = True) -> "BridgeTD3Policy":
#         self.training = mode
#         self.actor.train(mode)
#         self.critic1.train(mode)
#         self.critic2.train(mode)
#         if self.actor_bridge is not None:
#             self.actor_bridge.train(mode)
#         return self

#     def sync_weight(self) -> None:
#         self.soft_update(self.critic1_old, self.critic1, self.tau)
#         self.soft_update(self.critic2_old, self.critic2, self.tau)
#         self.soft_update(self.actor_old, self.actor, self.tau)

#     def _target_q(self, batch, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
#         act_ = self(batch, buffer, indices, model="actor_old", input="obs_next").act
#         noise = torch.randn(size=act_.shape, device=act_.device) * self._policy_noise
#         if self._noise_clip > 0.0:
#             noise = noise.clamp(-self._noise_clip, self._noise_clip)
#         act_ = act_ + noise
#         obs_next_emb = self.state_tracker(buffer=buffer, indices=indices, is_obs=False)
#         target_q = torch.min(
#             self.critic1_old(obs_next_emb, act_),
#             self.critic2_old(obs_next_emb, act_),
#         )
#         return target_q

#     # -------- main learning --------
#     def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
#         self.optim_state.zero_grad()

#         # === 1) Critics update ===
#         td1, critic1_loss = self._mse_optimizer(batch, self.critic1, self.critic1_optim)
#         td2, critic2_loss = self._mse_optimizer(batch, self.critic2, self.critic2_optim)
#         batch.weight = (td1 + td2) / 2.0

#         # === 2) Delayed actor/bridge update (TD3) ===
#         if self._cnt % self._freq == 0:
#             # ------- 2a) Main actor update -------
#             obs_emb = self.state_tracker(self._buffer, indices=batch.indices, is_obs=True)
#             # deterministic action (your BasePolicy handles eps/tanh/bounds)
#             a_det = self(batch, self._buffer, batch.indices, eps=0.0).act

#             if self.actor_bridge is None or self.optim_bridge is None:
#                 raise RuntimeError("actor_bridge/optim_bridge is required for Bridge-TD3.")

#             # candidates for expected action of bridge (teacher for actor)
#             with torch.no_grad():
#                 E, mask = self._get_candidates(self._buffer, batch.indices, batch)  # E:[B,K,d]
#                 out_bridge = self.actor_bridge(obs_emb, state=None, info=batch.info)
#                 logits_bridge = out_bridge[0] if isinstance(out_bridge, tuple) else out_bridge  # [B,K]
#                 logits_bridge = self._apply_mask(logits_bridge, mask)
#                 dist_bridge = Categorical(logits=logits_bridge)
#                 probs_bridge = dist_bridge.probs  # [B,K]
#                 a_bar = self._expected_action(probs_bridge, E)  # [B,d]

#             q1 = self.critic1(obs_emb, a_det).mean()
#             loss_actor = -q1 + self.w_align_actor * F.mse_loss(a_det, a_bar, reduction="mean")

#             self.actor_optim.zero_grad()
#             loss_actor.backward()
#             self._last = float(loss_actor.item())
#             self.actor_optim.step()

#             # ------- 2b) Bridge head update (discrete) -------
#             # fresh graph for state-tracker to avoid double-backward on same graph
#             obs_emb_b = self.state_tracker(self._buffer, indices=batch.indices, is_obs=True)

#             # student: current bridge distribution
#             E_b, mask_b = self._get_candidates(self._buffer, batch.indices, batch)
#             out_b = self.actor_bridge(obs_emb_b, state=None, info=batch.info)
#             logits_b = out_b[0] if isinstance(out_b, tuple) else out_b  # [B,K]
#             logits_b = self._apply_mask(logits_b, mask_b)
#             dist_b = Categorical(logits=logits_b)
#             probs_b = dist_b.probs  # [B,K]
#             a_bar_b = self._expected_action(probs_b, E_b)  # [B,d]

#             # teacher 1: Q-soft distribution y(i|s) ∝ exp(min(Q1,Q2)/τ)  (no grad to critics)
#             with torch.no_grad():
#                 B, K, d = E_b.shape
#                 obs_rep = obs_emb_b.unsqueeze(1).expand(-1, K, -1).reshape(B * K, obs_emb_b.shape[-1])
#                 act_flat = E_b.reshape(B * K, d)
#                 q1_all = self.critic1(obs_rep, act_flat).view(B, K)
#                 q2_all = self.critic2(obs_rep, act_flat).view(B, K)
#                 q_min = torch.min(q1_all, q2_all) / max(self.temperature, 1e-6)
#                 if mask_b is not None:
#                     # -inf for invalid to keep softmax clean
#                     q_min = q_min.masked_fill(~mask_b.bool(), torch.finfo(q_min.dtype).min)
#                 y = torch.softmax(q_min, dim=1).detach()  # [B,K], teacher probs

#             # teacher 2: main deterministic action for semantic alignment (no grad)
#             with torch.no_grad():
#                 out_p = self.actor(obs_emb_b, state=None, info=batch.info)
#                 a_det_teacher = out_p[0] if isinstance(out_p, tuple) else out_p
#                 a_det_teacher = a_det_teacher.to(dtype=a_bar_b.dtype, device=a_bar_b.device)

#             # losses for bridge
#             # KL(y || π_bridge)
#             teacher_dist = Categorical(probs=y)
#             kl_teach = torch.distributions.kl_divergence(teacher_dist, dist_b).mean()

#             align_bridge = F.mse_loss(a_bar_b, a_det_teacher, reduction="mean")
#             ent = dist_b.entropy().mean()

#             loss_bridge = self.w_teach * kl_teach + self.w_align_bridge * align_bridge - self.w_entropy * ent

#             self.optim_bridge.zero_grad()
#             loss_bridge.backward()
#             self.optim_bridge.step()

#             # one step for state-tracker (it has received grads from both passes)
#             self.optim_state.step()
#             self.sync_weight()

#         self._cnt += 1

#         return {
#             "loss/actor": self._last,
#             "loss/critic1": float(critic1_loss.item()),
#             "loss/critic2": float(critic2_loss.item()),
#         }