# Codex-added 2026-05-21: seed-aware experiment entrypoint for EasyRL4Rec confidence-interval sweeps; original runner kept unchanged.
import argparse
import sys
import traceback
from gymnasium.spaces import Discrete
import numpy as np

import torch

sys.path.extend([".", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])

from policy_utils import get_args_all, learn_policy, prepare_dir_log, prepare_user_model, setup_state_tracker, prepare_train_test_envs

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from src.core.collector.collector_set import CollectorSet
from src.core.util.data import get_env_args
from src.core.collector.collector import Collector
from src.core.policy.RecPolicy import RecPolicy

from src.tianshou.tianshou.data import VectorReplayBuffer

from src.tianshou.tianshou.utils.net.common import ActorCritic, Net
from src.tianshou.tianshou.utils.net.discrete import Actor, Critic
from src.tianshou.tianshou.policy import PPOPolicy


#新加的-----------------
from mpi4py import MPI
import os
import time
from datetime import datetime
#新加的-------------


# from util.upload import my_upload
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
    # if torch.cuda.is_available():
    #     actor = DataParallelNet(
    #         Actor(net, args.action_shape, device=None).to(args.device)
    #     )
    #     critic = DataParallelNet(Critic(net, device=None).to(args.device))
    # else:
    #     actor = Actor(net, args.action_shape, device=args.device).to(args.device)
    #     critic = Critic(net, device=args.device).to(args.device)
    optim_RL = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)
    optim_state = torch.optim.Adam(state_tracker.parameters(), lr=args.lr)
    optim = [optim_RL, optim_state]

    dist = torch.distributions.Categorical
    policy = PPOPolicy(
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
        action_bound_method="",  # not clip  # follow A2C?
        action_scaling=False
    )
    policy.set_eps(args.explore_eps)

    rec_policy = RecPolicy(args, policy, state_tracker)

    # Prepare the collectors and logs
    train_collector = Collector(
        rec_policy, train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        # preprocess_fn=state_tracker.build_state,
        exploration_noise=args.exploration_noise,
        remove_recommended_ids = args.remove_recommended_ids
    )

    test_collector_set = CollectorSet(rec_policy, test_envs_dict, args.buffer_size, args.test_num,
                                    #   preprocess_fn=state_tracker.build_state,
                                      exploration_noise=args.exploration_noise,
                                      force_length=args.force_length)

    return rec_policy, train_collector, test_collector_set, optim





def main(args):
    # ---------- 1. MPI 初始化 ----------
    comm  = MPI.COMM_WORLD
    rank  = comm.Get_rank()
    size  = comm.Get_size()

    # ---------- 2. 把 MPI 信息塞进 args ----------
    args.rank          = rank
    args.world_size    = size
    args.rota          = [2 ** i for i in range(int(np.log2(size - 1)) + 1)] if size > 1 else [1]
    # 若想在 Trainer 里用，保持同样字段名即可
    # 通信步长可以来自命令行，也可以在这里硬编码
    # args.comm_times_per_epoch = getattr(args, "comm_times_per_epoch", 25)

        # 下面分配user_model
    pattern = list("ABCABCAC")
    
    # 假设命令行传入的是类似 "pointneg"，或者 "pointnegC"
    # 我们要去掉尾部已有的 A/B/C 再拼接新的
    base_msg = 'homo3_pointneg'  # 去掉末尾的 A/B/C
    
    args.read_message = base_msg + pattern[rank % len(pattern)]

    letter = pattern[rank % len(pattern)]  # 当前进程对应的 A/B/C

    # A→sub1, B→sub2, C→sub3 映射并设置路径
    subset_map = {'A': 'sub1', 'B': 'sub2', 'C': 'sub3'}
    subset = subset_map[letter]
    
    # root = os.path.join("data", "MovieLens")
    args.DATAPATH = f"homo3_data_raw_{subset}"
    args.PRODATAPATH = f"homo3_data_processed_{subset}"

    
    # # 设备选择
    # num_gpus = torch.cuda.device_count()
    # local_rank_env = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
    # local_rank = int(local_rank_env) if local_rank_env is not None else rank % max(num_gpus, 1)
    # torch.cuda.set_device(local_rank if num_gpus else -1)
    # device = torch.device(f"cuda:{local_rank}" if num_gpus else "cpu")
    
    # if device.type == "cuda":
    #     print(f"[Rank {rank}] Using GPU {torch.cuda.get_device_name(local_rank)}", flush=True)
    # else:
    #     print(f"[Rank {rank}] Using CPU", flush=True)

    num_gpus = torch.cuda.device_count()
    local_rank_env = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
    
    if num_gpus > 0:
        # 如果有 GPU，确保 local_rank 在有效范围
        if local_rank_env is not None:
            local_rank = min(int(local_rank_env), num_gpus - 1)
        else:
            local_rank = min(rank % num_gpus, num_gpus - 1)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"[Rank {rank}] Using GPU {torch.cuda.get_device_name(local_rank)}", flush=True)
    else:
        # 无 GPU 直接使用 CPU
        device = torch.device("cpu")
        print(f"[Rank {rank}] Using CPU", flush=True)

    
    # 旋转拓扑用于 bridge 网络同步
    args.rota = [2 ** i for i in range(int(np.log2(size - 1)) + 1)] if size > 1 else [1]
    
    # Codex-modified 2026-05-21: original seed runner used args.verbose_sync = True.
    # args.verbose_sync = True
    args.verbose_sync = not getattr(args, "quiet_progress", False)

    # Codex-modified 2026-08-14: derive rank-local seeds only from the required
    # user-supplied base; no paper seed value is embedded.
    args.seed = int(args.seed) + 10*args.rank
    

    #新加的-------------

        
    # %% 1. Prepare the saved path.
    MODEL_SAVE_PATH, logger_path = prepare_dir_log(args)

    # %% 2. Prepare user model and environment
    ensemble_models = prepare_user_model(args)
    env, dataset, train_envs, test_envs_dict = prepare_train_test_envs(args, ensemble_models)

    # %% 3. Setup policy
    state_tracker = setup_state_tracker(args, ensemble_models, env, train_envs, test_envs_dict)
    policy, train_collector, test_collector_set, optim = setup_policy_model(args, state_tracker, train_envs, test_envs_dict)

    # %% 4. Learn policy
    learn_policy(args, env, dataset, policy, train_collector, test_collector_set, state_tracker, optim, MODEL_SAVE_PATH,
                 logger_path, comm, trainer="fedonpolicy_push_avg")


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
        # Codex-modified 2026-08-14: make MPI/Slurm failure status reliable.
        raise
