
import os
import sys
import numpy as np

from src.core.envs.BaseEnv import BaseEnv
sys.path.extend(["./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from src.core.envs.MovieLens.MovieLensData import MovieLensData

ROOTPATH = "data/MovieLens"
DATAPATH = os.path.join(ROOTPATH, "data_raw")
PRODATAPATH = os.path.join(ROOTPATH, "data_processed")


class MovieLensEnv(BaseEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, datapath, prodatapath, mat=None, lbe_user=None, lbe_item=None, mat_distance=None,
                 num_leave_compute=5, leave_threshold=80, max_turn=100, random_init=False):

        self.datapath = datapath
        self.prodatapath = prodatapath

        if mat is not None:
            self.mat = mat
            self.lbe_user = lbe_user
            self.lbe_item = lbe_item
            self.mat_distance = mat_distance
        else:
            self.mat, self.lbe_user, self.lbe_item, self.mat_distance = self.load_env_data(self.datapath, self.prodatapath)

        super(MovieLensEnv, self).__init__(num_leave_compute, leave_threshold, max_turn, random_init)

    @staticmethod
    def load_env_data(datapath: str = None, prodatapath: str = None):
        dataset = MovieLensData(datapath=datapath, prodatapath=prodatapath)
        mat = dataset.load_mat(datapath)
        lbe_user, lbe_item = dataset.get_lbe(datapath)
        mat_distance = dataset.get_saved_distance_mat(mat, prodatapath)
        
        
        return mat, lbe_user, lbe_item, mat_distance
    
    def _determine_whether_to_leave(self, t, action):
        if t == 0:
            return False


        window_actions = self.sequence_action[t - self.num_leave_compute:t]
        dist_list = np.array([self.mat_distance[action, x] for x in window_actions])
        if any(dist_list < self.leave_threshold):
            return True

        return False
