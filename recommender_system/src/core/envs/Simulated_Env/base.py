import gymnasium as gym
import numpy as np
import torch

from torch import FloatTensor


class BaseSimulatedEnv(gym.Env):
    def __init__(self, ensemble_models,
                 env_task_class, task_env_param: dict, task_name: str,
                 predicted_mat=None, 
                 ):

        self.ensemble_models = ensemble_models.eval()
        self.env_task = env_task_class(**task_env_param)
        self.observation_space = self.env_task.observation_space
        self.action_space = self.env_task.action_space
        self.cum_reward = 0  # total_a in virtualtaobao
        self.total_turn = 0  # total_c in virtualtaobao
        self.env_name = task_name
        self.predicted_mat = predicted_mat

        self._reset_history()
        self.MIN_R = predicted_mat.min()
        self.MAX_R = predicted_mat.max()

        self.reset()


    def _construct_state(self, reward):
        res = self.env_task.state
        return res

    def seed(self, sd=0):
        torch.manual_seed(sd)

    def reset(self):
        self.cum_reward = 0
        self.total_turn = 0
        self.reward = 0
        self.action = None
        self.env_task.action = None
        self.state, self.info = self.env_task.reset()

        self._reset_history()
        if self.env_name == "VirtualTB-v0":
            self.cur_user = self.state[:-3]
        else:  # elif self.env_name == "KuaiEnv-v0":
            self.cur_user = self.state[0]
        return self.state, {'cum_reward': 0.0}

    def render(self, mode='human', close=False):
        self.env_task.render(mode)

    def _compute_pred_reward(self, action):
        if self.env_name == "VirtualTB-v0":
            feature = np.concatenate((self.cur_user, np.array([self.reward, 0, self.total_turn]), action), axis=-1)
            feature_tensor = torch.unsqueeze(torch.tensor(feature, device=self.user_model.device, dtype=torch.float), 0)
            pred_reward = self.user_model.forward(feature_tensor).detach().cpu().numpy().squeeze()
            if pred_reward < 0:
                pred_reward = 0
            if pred_reward > 10:
                pred_reward = 10
        else:  # elif self.env_name == "KuaiEnv-v0":
            # get prediction
            pred_reward = self.predicted_mat[self.cur_user, action] - self.MIN_R

        return pred_reward

    def step(self, action: FloatTensor):
        self.action = action
        # real_state, real_reward, real_done, real_info = self.env_task.step(action)
        real_state, real_reward, real_terminated, real_truncated, real_info = self.env_task.step(action)

        t = int(self.total_turn)

        if t < self.env_task.max_turn:
            self._add_action_to_history(t, action)

        pred_reward = self._compute_pred_reward(action)

        self.cum_reward += pred_reward
        self.total_turn = self.env_task.total_turn

        terminated = real_terminated
        # Rethink commented, do not use new user as new state
        #     self.state, self.info = self.env_task.reset()

        self.state = self._construct_state(pred_reward)
        
        info =  {'cum_reward': self.cum_reward}
        truncated = False

        return self.state, pred_reward, terminated, truncated, info

    def _reset_history(self):
        if self.env_name == "VirtualTB-v0":
            self.history_action = np.zeros([self.env_task.max_turn, self.env_task.action_space.shape[0]])
        else:  # elif self.env_name == "KuaiEnv-v0":
            self.history_action = np.zeros(self.env_task.max_turn, dtype=int)
        self.max_history = 0

    def _add_action_to_history(self, t, action):
        if self.env_name == "VirtualTB-v0":
            action2 = np.expand_dims(action, 0)
            self.history_action[t] = action2
        else:  # elif self.env_name == "KuaiEnv-v0":
            self.history_action[t] = action

        assert self.max_history == t
        self.max_history += 1
