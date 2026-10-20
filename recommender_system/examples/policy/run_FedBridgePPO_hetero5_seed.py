import argparse
import sys
import traceback
from gymnasium.spaces import Discrete
import numpy as np


import torch

sys.path.extend([".", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from policy_utils import get_args_all, learn_policy, prepare_dir_log, prepare_user_model, setup_state_tracker, prepare_train_test_envs


from src.core.collector.collector_set import CollectorSet
from src.core.util.data import get_env_args
from src.core.collector.collector import Collector
from src.core.policy.RecPolicy import RecPolicy

from src.tianshou.tianshou.data import VectorReplayBuffer

from src.tianshou.tianshou.utils.net.common import ActorCritic, Net
from src.tianshou.tianshou.utils.net.discrete import Actor, Critic
from src.tianshou.tianshou.policy import BridgePPOPolicy


from mpi4py import MPI
import os
import time
from datetime import datetime


import logzero

try:
    import envpool
except ImportError:
    envpool = None


def get_args_PPO():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="PPO")
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', action="store_true", default=False)
    parser.add_argument('--norm-adv',action="store_true", default=True)
    parser.add_argument('--recompute-adv', action="store_true", default=False)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', action="store_true", default=False)
    
    parser.add_argument('--comm_times_per_epoch', type=int, default=25,
                        help='每个 epoch 期望通信次数（与 updates_per_epoch 换算 comm_interval）')
    parser.add_argument('--lambda_kl', type=float, default=0.005,
                        help='Bridge/Private 双向 KL 系数')

    
    parser.add_argument("--message", type=str, default="PPO")
    parser.add_argument("--hetero_degree", type=int, default=2,
    help="0=IID, larger -> more heterogeneous; affects Dirichlet α etc.")
    

    args = parser.parse_known_args()[0]
    return args


def setup_policy_model(args, state_tracker, train_envs, test_envs_dict):
    if args.cpu:
        args.device = "cpu"
    else:
        args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    # model
    net = Net(args.state_dim, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = Actor(net, args.action_shape, device=args.device).to(args.device)
    critic = Critic(net, device=args.device).to(args.device)
    actor_critic = ActorCritic(actor, critic)

    net_bridge = Net(args.state_dim, hidden_sizes=args.hidden_sizes, device=args.device)
    actor_bridge = Actor(net_bridge, args.action_shape, device=args.device).to(args.device)
    
    optim_RL = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)
    optim_state = torch.optim.Adam(state_tracker.parameters(), lr=args.lr)
    optim = [optim_RL, optim_state]

    optim_bridge =  torch.optim.Adam(actor_bridge.parameters(), lr=args.lr)

    dist = torch.distributions.Categorical
    policy = BridgePPOPolicy(
        actor,
        critic,
        optim,
        dist,
        state_tracker=state_tracker,
        discount_factor=args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        gae_lambda=args.gae_lambda,
        reward_normalization=args.rew_norm,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        action_space=Discrete(args.action_shape),
        deterministic_eval=True,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
        actor_bridge = actor_bridge,
        optim_bridge = optim_bridge,
        lambda_kl = args.lambda_kl,
        action_bound_method="",  # not clip  # follow A2C?
        action_scaling=False
    )
    policy.set_eps(args.explore_eps)

    rec_policy = RecPolicy(args, policy, state_tracker)

    # Prepare the collectors and logs
    train_collector = Collector(
        rec_policy, train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=args.exploration_noise,
        remove_recommended_ids = args.remove_recommended_ids
    )

    test_collector_set = CollectorSet(rec_policy, test_envs_dict, args.buffer_size, args.test_num,
                                      exploration_noise=args.exploration_noise,
                                      force_length=args.force_length)

    return rec_policy, train_collector, test_collector_set, optim


def main(args):

    comm  = MPI.COMM_WORLD
    rank  = comm.Get_rank()
    size  = comm.Get_size()


    args.rank          = rank
    args.world_size    = size
    args.rota          = [2 ** i for i in range(int(np.log2(size - 1)) + 1)] if size > 1 else [1]


    pattern = list("ABCABCAC")
    

    base_msg = 'hetero5_pointneg'
    
    args.read_message = base_msg + pattern[rank % len(pattern)]

    letter = pattern[rank % len(pattern)]


    subset_map = {'A': 'sub1', 'B': 'sub2', 'C': 'sub3'}
    subset = subset_map[letter]
    
    args.DATAPATH = f"hetero5_data_raw_{subset}"
    args.PRODATAPATH = f"hetero5_data_processed_{subset}"


    num_gpus = torch.cuda.device_count()
    local_rank_env = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
    
    if num_gpus > 0:

        if local_rank_env is not None:
            local_rank = min(int(local_rank_env), num_gpus - 1)
        else:
            local_rank = min(rank % num_gpus, num_gpus - 1)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"[Rank {rank}] Using GPU {torch.cuda.get_device_name(local_rank)}", flush=True)
    else:

        device = torch.device("cpu")
        print(f"[Rank {rank}] Using CPU", flush=True)

    
    args.rota = [2 ** i for i in range(int(np.log2(size - 1)) + 1)] if size > 1 else [1]
    
    args.verbose_sync = not getattr(args, "quiet_progress", False)

    args.seed = int(args.seed) + 10*args.rank
    

    # %% 1. Prepare the saved path.
    MODEL_SAVE_PATH, logger_path = prepare_dir_log(args)  ##20250914

    # %% 2. Prepare user model and environment
    ensemble_models = prepare_user_model(args)
    env, dataset, train_envs, test_envs_dict = prepare_train_test_envs(args, ensemble_models)

    # %% 3. Setup policy
    state_tracker = setup_state_tracker(args, ensemble_models, env, train_envs, test_envs_dict)
    policy, train_collector, test_collector_set, optim = setup_policy_model(args, state_tracker, train_envs, test_envs_dict)

    # %% 4. Learn policy
    learn_policy(args, env, dataset, policy, train_collector, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH,
                 logger_path, comm, trainer="fedbridgeonp")


if __name__ == '__main__':
    trainer = "onpolicy"
    args_all = get_args_all(trainer)
    args = get_env_args(args_all)
    args_PPO = get_args_PPO()
    args_all.__dict__.update(args.__dict__)
    args_all.__dict__.update(args_PPO.__dict__)
    try:
        main(args_all)
    except Exception as e:
        var = traceback.format_exc()
        print(var)
        logzero.logger.error(var)
        raise
