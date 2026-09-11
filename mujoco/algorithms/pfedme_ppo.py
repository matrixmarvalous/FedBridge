import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical


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
        if has_continuous_action_space:
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


class pFedMe_PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
        lambda_l2,
        beta,
        device,
        action_std_init=0.6,
        local_steps=1,
    ):
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init
            self.init_action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.lambda_l2 = lambda_l2
        self.beta = beta  # pFedMe eta step for updating w toward theta
        self.local_steps = local_steps

        self.buffer = RolloutBuffer()

        # personalized (theta)
        self.policy_priv = ActorCritic(
            state_dim, action_dim, has_continuous_action_space, action_std_init, device
        ).to(device)
        self.optimizer_priv = torch.optim.Adam([
            {'params': self.policy_priv.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy_priv.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_priv_old = ActorCritic(
            state_dim, action_dim, has_continuous_action_space, action_std_init, device
        ).to(device)
        self.policy_priv_old.load_state_dict(self.policy_priv.state_dict())

        # global (w), updated by server aggregation
        self.policy_global = ActorCritic(
            state_dim, action_dim, has_continuous_action_space, action_std_init, device
        ).to(device)
        self.policy_global.load_state_dict(self.policy_priv.state_dict())
        for p in self.policy_global.parameters():
            p.requires_grad = False

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy_priv.set_action_std(new_action_std)
            self.policy_priv_old.set_action_std(new_action_std)
            self.policy_global.set_action_std(new_action_std)
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

    def update(self):
        metrics = {
            "Loss/priv_total": 0.0,
            "Loss/priv_value": 0.0,
            "Loss/priv_policy": 0.0,
            "Entropy/priv": 0.0,
            "Advantage/mean": 0.0
        }

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        advantages = rewards.detach() - old_state_values.detach()

        # local update rounds (R)
        for _ in range(self.local_steps):
            for _ in range(self.K_epochs):
                logprobs_priv, state_values, dist_entropy_priv = self.policy_priv.evaluate(
                    old_states, old_actions
                )
                state_values = torch.squeeze(state_values)

                ratios = torch.exp(logprobs_priv - old_logprobs.detach())
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                loss_ppo = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy_priv
                l2 = sum((p_theta - p_w).pow(2).sum()
                         for p_theta, p_w in zip(self.policy_priv.parameters(), self.policy_global.parameters()))
                loss = loss_ppo + self.lambda_l2 * l2

                self.optimizer_priv.zero_grad()
                loss.mean().backward()
                self.optimizer_priv.step()

            # w <- w - eta * (w - theta)
            with torch.no_grad():
                for p_w, p_theta in zip(self.policy_global.parameters(), self.policy_priv.parameters()):
                    p_w.sub_(self.beta * (p_w - p_theta))

        self.policy_priv_old.load_state_dict(self.policy_priv.state_dict())
        self.buffer.clear()
        return metrics

    def save(self, checkpoint_path):
        torch.save({
            'policy_priv_old_state_dict': self.policy_priv_old.state_dict(),
            'policy_global_state_dict': self.policy_global.state_dict(),
        }, checkpoint_path)

    def load(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.policy_priv_old.load_state_dict(ckpt['policy_priv_old_state_dict'])
        self.policy_priv.load_state_dict(ckpt['policy_priv_old_state_dict'])
        if 'policy_global_state_dict' in ckpt:
            self.policy_global.load_state_dict(ckpt['policy_global_state_dict'])
