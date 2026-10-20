import os
import sys
import numpy as np

sys.path.extend([".", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])
from src.core.envs.BaseEnv import BaseEnv
from src.core.envs.Coat.CoatData import CoatData

ROOTPATH = "data/Coat"
DATAPATH = os.path.join(ROOTPATH, "data_raw")
PRODATAPATH = os.path.join(ROOTPATH, "data_processed")


class CoatEnv(BaseEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, mat=None, df_item=None, mat_distance=None, 
                 num_leave_compute=5, leave_threshold=1, max_turn=100, random_init=False):
        if mat is not None:
            self.mat = mat
            self.df_item = df_item
            self.mat_distance = mat_distance
        else:
            self.mat, self.df_item, self.mat_distance = self.load_env_data()

        super(CoatEnv, self).__init__(num_leave_compute, leave_threshold, max_turn, random_init)

    @staticmethod
    def load_env_data():
        mat = CoatData.load_mat()
        df_item = CoatData.load_item_feat()
        mat_distance = CoatData.get_saved_distance_mat(mat, PRODATAPATH)
        return mat, df_item, mat_distance

    def _determine_whether_to_leave(self, t, action):
        # self.list_feat[action]
        if t == 0:
            return False
        window_actions = self.sequence_action[t - self.num_leave_compute:t]

        dist_list = np.array([self.mat_distance[action, x] for x in window_actions])


        if any(dist_list < self.leave_threshold):
            return True


        return False
