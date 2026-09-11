# Codex-modified 2026-08-19: publish this implementation as BridgePPO.py;
# numerical code is unchanged.
# Codex-modified 2026-08-18: rename the pre-publication API to BridgePPO
# without changing optimization or communication behavior. The full
# pre-rename release is preserved outside this bundle in release_backups/.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.distributions import kl_divergence

################################## set device ##################################
# print("============================================================================================")
# # set device to cpu or cuda
# device = torch.device('cpu')
# if(torch.cuda.is_available()): 
#     device = torch.device('cuda:0') 
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
#     print("Device set to : cpu")
# print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.state_values[:]



class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init, device):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.device = device
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class BridgePPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, lambda_kl, device, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init
            self.init_action_std = action_std_init    # for scaling λ_kl

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        
        self.lambda_kl = lambda_kl
        
        self.buffer = RolloutBuffer()
        
        # create policy_priv
        self.policy_priv = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, device).to(device)
        self.optimizer_priv = torch.optim.Adam([
                        {'params': self.policy_priv.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy_priv.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_priv_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, device).to(device)
        self.policy_priv_old.load_state_dict(self.policy_priv.state_dict())
        
        # create policy_bridge
        self.policy_bridge = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, device).to(device)
        self.optimizer_bridge = torch.optim.Adam([
                        {'params': self.policy_bridge.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy_bridge.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_bridge_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, device).to(device)
        self.policy_bridge_old.load_state_dict(self.policy_bridge.state_dict())
        
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy_priv.set_action_std(new_action_std)
            self.policy_priv_old.set_action_std(new_action_std)
            self.policy_bridge.set_action_std(new_action_std)
            self.policy_bridge_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy_priv")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob, state_val = self.policy_priv_old.act(state)
                
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)


            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob, state_val = self.policy_priv_old.act(state)
                
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)
            
            return action.item()

#     def update_dual(self):
#         # Monte Carlo estimate of returns
#         metrics = {                     # 用来累计后返回给主循环
#         "Loss/priv_total": 0.0, "Loss/bridge_total": 0.0,
#         "Loss/priv_value": 0.0, "Loss/bridge_value": 0.0,
#         "Loss/priv_policy": 0.0, "Loss/bridge_policy": 0.0,
#         "Entropy/priv": 0.0, "Entropy/bridge": 0.0,
#         "KL/priv_vs_bridge": 0.0, "KL/bridge_vs_priv": 0.0,
#         "Advantage/mean": 0.0
#         }
        
#         rewards = []
#         discounted_reward = 0
#         for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
#             if is_terminal:
#                 discounted_reward = 0
#             discounted_reward = reward + (self.gamma * discounted_reward)
#             rewards.insert(0, discounted_reward)
            
#         # Normalizing the rewards
#         rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
#         rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

#         # convert list to tensor
#         old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
#         old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
#         old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
#         old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

#         # calculate advantages
#         advantages = rewards.detach() - old_state_values.detach()

#         # Optimize private policy for K epochs
#         for _ in range(self.K_epochs):
#             # === 动态缩放 KL 权重（仅连续动作生效） ===
#             if self.has_continuous_action_space:
#                 lambda_kl_scaled = self.lambda_kl * (self.action_std / self.init_action_std)
#             else:
#                 lambda_kl_scaled = self.lambda_kl

#             # ====== 更新私有策略 ======
#             # Evaluating old actions and values
#             logprobs_priv, state_values, dist_entropy_priv = self.policy_priv.evaluate(old_states, old_actions)

#             # match state_values tensor dimensions with rewards tensor
#             state_values = torch.squeeze(state_values)
            
#             with torch.no_grad():
#                 logprobs_bridge, _, _ = self.policy_bridge.evaluate(old_states, old_actions)
#                 logprobs_bridge = torch.squeeze(logprobs_bridge)
            
#             # Finding the ratio (pi_theta / pi_theta__old)
#             ratios = torch.exp(logprobs_priv - old_logprobs.detach())

#             # Finding Surrogate Loss  
#             surr1 = ratios * advantages
#             surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
#             # compute kl-divergence
#             # === KL(priv vs bridge) ===
#             if self.has_continuous_action_space:
#                 with torch.no_grad():
#                     mean_bridge = self.policy_bridge.actor(old_states)
#                     cov_bridge  = torch.diag(self.policy_bridge.action_var).unsqueeze(0).expand(mean_bridge.shape[0], -1, -1)

#                 mean_priv = self.policy_priv.actor(old_states)
#                 cov_priv  = torch.diag(self.policy_priv.action_var).unsqueeze(0).expand(mean_priv.shape[0], -1, -1)

#                 dist_priv  = MultivariateNormal(mean_priv,  cov_priv)
#                 dist_bridge = MultivariateNormal(mean_bridge, cov_bridge)

#                 kl_priv = kl_divergence(dist_priv, dist_bridge)
#             else:
#                 kl_priv = F.kl_div(logprobs_priv, logprobs_bridge, reduction='batchmean', log_target=True)
            
#             # final loss of clipped objective PPO
#             policy_loss_priv = -torch.min(surr1, surr2).mean()
#             value_loss_priv  = 0.5 * self.MseLoss(state_values, rewards)
#             entropy_priv     = dist_entropy_priv.mean()
            

#             # final loss of clipped objective PPO
#             loss_priv = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy_priv + lambda_kl_scaled*kl_priv
            
#             # take gradient step
#             self.optimizer_priv.zero_grad()
#             loss_priv.mean().backward()
#             self.optimizer_priv.step()
            
#             # ---------- 累计日志 ----------
#             loss_priv_mean = loss_priv.mean()
#             kl_priv_mean = kl_priv.mean()
#             metrics["Loss/priv_total"]   += loss_priv_mean.item()
#             metrics["Loss/priv_policy"]  += policy_loss_priv.item()
#             metrics["Loss/priv_value"]   += value_loss_priv.item()
#             metrics["Entropy/priv"]      += entropy_priv.item()
#             metrics["KL/priv_vs_bridge"]  += kl_priv_mean.item()
            
            
#             # ====== 更新代理策略 ======
#             # Evaluating old actions and values
#             logprobs_bridge, state_values_bridge, dist_entropy_bridge = self.policy_bridge.evaluate(old_states, old_actions)

#             # match state_values tensor dimensions with rewards tensor
#             state_values_bridge = torch.squeeze(state_values_bridge)
            
#             with torch.no_grad():
#                 logprobs_priv, _, _ = self.policy_priv.evaluate(old_states, old_actions)
#                 logprobs_priv = torch.squeeze(logprobs_priv)
            
#             # Finding the ratio (pi_theta / pi_theta__old)
#             ratios_bridge = torch.exp(logprobs_bridge - old_logprobs.detach())

#             # Finding Surrogate Loss  
#             surr3 = ratios_bridge * advantages
#             surr4 = torch.clamp(ratios_bridge, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
#             # compute kl-divergence
#             if self.has_continuous_action_space:
#                 with torch.no_grad():
#                     mean_priv = self.policy_priv.actor(old_states)
#                     cov_priv  = torch.diag(self.policy_priv.action_var).unsqueeze(0).expand(mean_priv.shape[0], -1, -1)

#                 mean_bridge = self.policy_bridge.actor(old_states)
#                 cov_bridge  = torch.diag(self.policy_bridge.action_var).unsqueeze(0).expand(mean_bridge.shape[0], -1, -1)

#                 dist_priv  = MultivariateNormal(mean_priv,  cov_priv)
#                 dist_bridge = MultivariateNormal(mean_bridge, cov_bridge)

#                 kl_bridge = kl_divergence(dist_bridge, dist_priv)
#             else:
#                 kl_bridge = F.kl_div(logprobs_bridge, logprobs_priv, reduction='batchmean', log_target=True)
           
            
#             # final loss of clipped objective PPO
#             policy_loss_bridge = -torch.min(surr3, surr4).mean()
#             entropy_bridge     = dist_entropy_bridge.mean()
            

#             # final loss of clipped objective PPO
#             loss_bridge = -torch.min(surr3, surr4)  - 0.01 * dist_entropy_bridge + lambda_kl_scaled*kl_bridge
            
#             # take gradient step
#             self.optimizer_bridge.zero_grad()
#             loss_bridge.mean().backward()
#             self.optimizer_bridge.step()
            
#             # ---------- 累计日志 ----------
#             loss_bridge_mean = loss_bridge.mean()
#             kl_bridge_mean = kl_bridge.mean()
#             metrics["Loss/bridge_total"]   += loss_bridge_mean.item()
#             metrics["Loss/bridge_policy"]  += policy_loss_bridge.item()
#             metrics["Entropy/bridge"]      += entropy_bridge.item()
#             metrics["KL/bridge_vs_priv"]   += kl_bridge_mean.item()
            
            
#         # Copy new weights into old policy
#         self.policy_priv_old.load_state_dict(self.policy_priv.state_dict())
#         self.policy_bridge_old.load_state_dict(self.policy_bridge.state_dict())
        
#         # 把累计值换成「平均到每个 epoch」的标量
#         for k in metrics.keys():
#             metrics[k] /= self.K_epochs

#         # clear buffer
#         self.buffer.clear()
        
#         return metrics
    def update_dual(self):
        """
        PPO 双策略（priv/bridge）的小批次(minibatch)更新版本
        关键变化：
          - 按照 self.minibatch_size 打乱切分经验，做 SGD 式更新
          - 日志指标按 (K_epochs * num_minibatches) 归一
        """
        # ----------------- 统计指标容器 -----------------
        metrics = {
            "Loss/priv_total": 0.0, "Loss/bridge_total": 0.0,
            "Loss/priv_value": 0.0, "Loss/bridge_value": 0.0,
            "Loss/priv_policy": 0.0, "Loss/bridge_policy": 0.0,
            "Entropy/priv": 0.0, "Entropy/bridge": 0.0,
            "KL/priv_vs_bridge": 0.0, "KL/bridge_vs_priv": 0.0,
            "Advantage/mean": 0.0
        }

        # ----------------- 1) 计算折扣回报 & 优势 -----------------
        rewards = []
        discounted_reward = 0.0
        for reward, is_terminal in zip(reversed(self.buffer.rewards),
                                       reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0.0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 从 buffer 拿出旧轨迹（这些值在采样时就由 old_policy 评估好了）
        old_states       = torch.squeeze(torch.stack(self.buffer.states,  dim=0)).detach().to(self.device)
        old_actions      = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs     = torch.squeeze(torch.stack(self.buffer.logprobs,dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        advantages = (rewards - old_state_values).detach()
        # 建议对 advantages 做标准化（全 buffer 级别，而不是每个批次单独标准化）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        metrics["Advantage/mean"] = advantages.mean().item()

        # ----------------- 2) 小批次配置 -----------------
        num_samples = old_states.size(0)
        mb_size = getattr(self, "minibatch_size", 4000)
        if (mb_size is None) or (mb_size <= 0) or (mb_size > num_samples):
            mb_size = num_samples  # 退化为全量 batch（与原实现一致）

        num_minibatches = (num_samples + mb_size - 1) // mb_size
        total_updates = self.K_epochs * num_minibatches

        # ----------------- 3) 训练循环（K epochs × minibatch） -----------------
        for _ in range(self.K_epochs):
            # 打乱索引
            perm = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, mb_size):
                idx = perm[start:start + mb_size]

                # ==== 动态 KL 系数（连续动作才生效）====
                if self.has_continuous_action_space:
                    lambda_kl_scaled = self.lambda_kl * (self.action_std / self.init_action_std)
                else:
                    lambda_kl_scaled = self.lambda_kl

                # ----------------- A) 更新私有策略 priv -----------------
                # 取小批次切片
                states_b       = old_states[idx]
                actions_b      = old_actions[idx]
                old_logprobs_b = old_logprobs[idx]
                adv_b          = advantages[idx]
                rewards_b      = rewards[idx]

                # 新策略在小批次上的评估
                logprobs_priv, state_values_priv, dist_entropy_priv = self.policy_priv.evaluate(states_b, actions_b)
                state_values_priv = torch.squeeze(state_values_priv)

                with torch.no_grad():
                    logprobs_bridge_b, _, _ = self.policy_bridge.evaluate(states_b, actions_b)
                    logprobs_bridge_b = torch.squeeze(logprobs_bridge_b)

                # PPO 比率
                ratios = torch.exp(logprobs_priv - old_logprobs_b)

                # PPO surrogate
                surr1 = ratios * adv_b
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv_b

                # KL(priv || bridge)
                if self.has_continuous_action_space:
                    with torch.no_grad():
                        mean_bridge = self.policy_bridge.actor(states_b)
                        cov_bridge  = torch.diag(self.policy_bridge.action_var).unsqueeze(0).expand(mean_bridge.shape[0], -1, -1)
                    mean_priv = self.policy_priv.actor(states_b)
                    cov_priv  = torch.diag(self.policy_priv.action_var).unsqueeze(0).expand(mean_priv.shape[0], -1, -1)
                    dist_priv  = MultivariateNormal(mean_priv,  cov_priv)
                    dist_bridge = MultivariateNormal(mean_bridge, cov_bridge)
                    kl_priv = kl_divergence(dist_priv, dist_bridge)  # [B]
                    kl_priv = kl_priv.mean()
                else:
                    # 输入是 log probs（log_target=True）
                    kl_priv = F.kl_div(logprobs_priv, logprobs_bridge_b, reduction='batchmean', log_target=True)

                # 分量 loss（仅用于记录）
                policy_loss_priv = -torch.min(surr1, surr2).mean()
                value_loss_priv  = 0.5 * self.MseLoss(state_values_priv, rewards_b)
                entropy_priv     = dist_entropy_priv.mean()

                # 总 priv loss
                loss_priv = (
                    -torch.min(surr1, surr2).mean()
                    + 0.5 * self.MseLoss(state_values_priv, rewards_b)
                    - 0.01 * dist_entropy_priv.mean()
                    + lambda_kl_scaled * kl_priv
                )

                self.optimizer_priv.zero_grad()
                loss_priv.backward()
                self.optimizer_priv.step()

                # 记录
                metrics["Loss/priv_total"]  += loss_priv.item()
                metrics["Loss/priv_policy"] += policy_loss_priv.item()
                metrics["Loss/priv_value"]  += value_loss_priv.item()
                metrics["Entropy/priv"]     += entropy_priv.item()
                metrics["KL/priv_vs_bridge"] += kl_priv.item()

                # ----------------- B) 更新代理策略 bridge -----------------
                logprobs_bridge, state_values_bridge, dist_entropy_bridge = self.policy_bridge.evaluate(states_b, actions_b)
                state_values_bridge = torch.squeeze(state_values_bridge)

                with torch.no_grad():
                    logprobs_priv_b, _, _ = self.policy_priv.evaluate(states_b, actions_b)
                    logprobs_priv_b = torch.squeeze(logprobs_priv_b)

                ratios_bridge = torch.exp(logprobs_bridge - old_logprobs_b)
                surr3 = ratios_bridge * adv_b
                surr4 = torch.clamp(ratios_bridge, 1 - self.eps_clip, 1 + self.eps_clip) * adv_b

                # KL(bridge || priv)
                if self.has_continuous_action_space:
                    with torch.no_grad():
                        mean_priv2 = self.policy_priv.actor(states_b)
                        cov_priv2  = torch.diag(self.policy_priv.action_var).unsqueeze(0).expand(mean_priv2.shape[0], -1, -1)
                    mean_bridge2 = self.policy_bridge.actor(states_b)
                    cov_bridge2  = torch.diag(self.policy_bridge.action_var).unsqueeze(0).expand(mean_bridge2.shape[0], -1, -1)
                    dist_priv2  = MultivariateNormal(mean_priv2,  cov_priv2)
                    dist_bridge2 = MultivariateNormal(mean_bridge2, cov_bridge2)
                    kl_bridge = kl_divergence(dist_bridge2, dist_priv2).mean()
                else:
                    kl_bridge = F.kl_div(logprobs_bridge, logprobs_priv_b, reduction='batchmean', log_target=True)

                policy_loss_bridge = -torch.min(surr3, surr4).mean()
                entropy_bridge     = dist_entropy_bridge.mean()

                loss_bridge = (
                    -torch.min(surr3, surr4).mean()
                    # bridge 分支你原来没有加 value loss，这里保持一致
                    - 0.01 * dist_entropy_bridge.mean()
                    + lambda_kl_scaled * kl_bridge
                )

                self.optimizer_bridge.zero_grad()
                loss_bridge.backward()
                self.optimizer_bridge.step()

                metrics["Loss/bridge_total"]  += loss_bridge.item()
                metrics["Loss/bridge_policy"] += policy_loss_bridge.item()
                # 若以后给 bridge 加上 value head 再记录 "Loss/bridge_value"
                metrics["Entropy/bridge"]     += entropy_bridge.item()
                metrics["KL/bridge_vs_priv"]  += kl_bridge.item()

        # ----------------- 4) 同步 old policy -----------------
        self.policy_priv_old.load_state_dict(self.policy_priv.state_dict())
        self.policy_bridge_old.load_state_dict(self.policy_bridge.state_dict())

        # ----------------- 5) 日志归一化 -----------------
        for k in metrics.keys():
            metrics[k] /= float(total_updates)

        # ----------------- 6) 清空缓冲区 -----------------
        self.buffer.clear()

        return metrics

    
    # def save(self, checkpoint_path):
    #     torch.save(self.policy_priv_old.state_dict(), checkpoint_path)
    def save(self, checkpoint_path):
        torch.save({
            'policy_priv_old_state_dict': self.policy_priv_old.state_dict(),
            'policy_bridge_old_state_dict': self.policy_bridge_old.state_dict()
        }, checkpoint_path)

    def load(self, checkpoint_path):
        # Codex-modified 2026-08-19: the former implementation passed the
        # complete two-key checkpoint payload to each ActorCritic and therefore
        # could not read files produced by save(). The previous calls were:
        # self.policy_priv_old.load_state_dict(torch.load(checkpoint_path, ...))
        # self.policy_priv.load_state_dict(torch.load(checkpoint_path, ...))
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch versions before weights_only support.
            payload = torch.load(checkpoint_path, map_location="cpu")
        personal = payload['policy_priv_old_state_dict']
        bridge = payload['policy_bridge_old_state_dict']
        self.policy_priv_old.load_state_dict(personal)
        self.policy_priv.load_state_dict(personal)
        self.policy_bridge_old.load_state_dict(bridge)
        self.policy_bridge.load_state_dict(bridge)
        
        
       
