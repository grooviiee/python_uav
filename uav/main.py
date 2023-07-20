# Main function of system
# Choose environment, algorithm and other settings

import numpy as np
import time
import pickle
import torch
import os
from utils.config import parse_args
from pathlib import Path
from envs.UavEnvMain import UAVEnvMain

# import maddpg.common.tf_util as U
from algorithms.mappo import MAPPOAgentTrainer
from envs.UavEnvMain import UAVEnvMain

def make_train_env(arglist, benchmark=False):
    if arglist.env_name == "uavnet":
        print("You choose " + arglist.scenario_name + " environment.")
        env = UAVEnvMain(arglist)
    else:
        print("Can not support the " + arglist.scenario_name + " environment.")
        raise NotImplementedError
    
    env.seed(arglist.seed + arglist.rank * 1000)
    return env

def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "uavnet":
                env = UAVEnvMain(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env
    
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def main(arglist):
    # select device
    print("choose device...", arglist.n_training_threads)
    if torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("gpu")
        torch.set_num_threads(arglist.n_training_threads)
        if arglist.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(arglist.n_training_threads)

    # select algorithm 
    if arglist.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy to be True")
    elif arglist.algorithm_name == "ddpg":
        print("u are choosing to use ddpg, we set use_recurrent_policy to be True")
        print(f"Not implemented yet")
        raise NotImplementedError
    elif arglist.algorithm_name == "attention_mappo":
        print("u are choosing to use attention_mappo, we set use_recurrent_policy to be True")
        print(f"Not implemented yet")
        raise NotImplementedError
    else:
        raise NotImplementedError

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / arglist.env_name / arglist.scenario_name / arglist.algorithm_name / arglist.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if arglist.use_wandb:
        run = wandb.init(config=arglist,
                         project=arglist.env_name,
                         entity=arglist.user_name,
                         notes=socket.gethostname(),
                         name=str(arglist.algorithm_name) + "_" +
                         str(arglist.experiment_name) +
                         "_seed" + str(arglist.seed),
                         group=arglist.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    # env init
    print("Load Environement...")
    envs = make_train_env(arglist)
    eval_envs = make_eval_env(arglist) if arglist.use_eval else None

    config = {
        "args": arglist,
        "envs": envs,
        "eval_envs": eval_envs,
        "device": device,
        "num_uavs": arglist.num_uavs,
        "num_mbs": arglist.num_mbs,
        "num_users": arglist.num_users,
        "run_dir": run_dir,
    }

    # run experiment
    if arglist.runner_name == "singleBS":
        from runner.singleBS_runner import SingleBS_runner as Runner
    elif arglist.runner_name == "multipleBS":
        from runner.multipleBS_runner import MultipleBS_runner as Runner
    else:
        NotImplemented

    print(f'Load runner as {arglist.runner_name}')
    runner = Runner(config)
    runner.run()

    envs.close()

if __name__ == "__main__":
    print("Main code starts")
    arglist = parse_args()
    main(arglist)


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Reinforcement Learning experiments for multiagent environments"
#     )
    
#     # prepare parameters
#     parser.add_argument("--log_level", type=int, default=2, help="Choose printing logs(1:LOW, 2:MID, 3:HIGH)")
#     parser.add_argument("--device", default="gpu", help="Choose device. cpu or gpu?")
#     parser.add_argument("--num_env_steps", type=int, default=10e6,
#                         help='Number of environment steps to train (default: 10e6)')
    
#     parser.add_argument("--env_name", type=str, default="uavnet", choices=["uavnet"])
#     parser.add_argument("--scenario_name", type=str, default="uavenv", choices=["uavenv"])
#     parser.add_argument("--runner_name", type=str, default="singleBS", choices=["singleBS", "multipleBS"]) #
#     parser.add_argument("--algorithm_name", type=str, default="mappo", choices=["ddpg", "mappo", "attention_mappo"])
#     parser.add_argument("--experiment_name",type=str,default="check",help="an identifier to distinguish different experiment.",)
    
#     parser.add_argument(
#         "--seed", type=int, default=1, help="Random seed for numpy/torch"
#     )
#     parser.add_argument(
#         "--cuda",
#         action="store_false",
#         default=True,
#         help="by default True, will use GPU to train; or else will use CPU;",
#     )
#     parser.add_argument(
#         "--cuda_deterministic",
#         action="store_false",
#         default=True,
#         help="by default, make sure random seed effective. if set, bypass such function.",
#     )
#     parser.add_argument(
#         "--n_training_threads",
#         type=int,
#         default=1,
#         help="Number of torch threads for training",
#     )
#     parser.add_argument(
#         "--n_rollout_threads",
#         type=int,
#         default=1,
#         #default=32,
#         help="Number of parallel envs for training rollouts",
#     )
#     parser.add_argument(
#         "--n_eval_rollout_threads",
#         type=int,
#         default=1,
#         help="Number of parallel envs for evaluating rollouts",
#     )
#     parser.add_argument(
#         "--n_render_rollout_threads",
#         type=int,
#         default=1,
#         help="Number of parallel envs for rendering rollouts",
#     )
#     parser.add_argument(
#         "--user_name",
#         type=str,
#         default="marl",
#         help="[for wandb usage], to specify user's name for simply collecting training data.",
#     )
#     parser.add_argument("--use_wandb", action='store_false', default=False, help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.")

    
#     # replay buffer parameters
#     parser.add_argument("--episode_length", type=int, default=3, help="Max length for any episode")
    
#     # Environment settings
#     parser.add_argument(
#         "--num_mbs", type=int, default=1
#     )
#     parser.add_argument(
#         "--num_uavs", type=int, default=4
#     )
#     parser.add_argument(
#         "--cache_capa", type=int, default=3, help="Max Number of File stored in UAV"
#     )
#     parser.add_argument(
#         "--num_users", type=int, default=20, help="Number of User Equipment"
#     )
#     parser.add_argument(
#         "--num_files", type=int, default=30
#     )
#     parser.add_argument(
#         "--file_size", type=int, default=10*10^6, help="Constant File size."
#     )
#     parser.add_argument(
#         "--map_size", type=int, default=1800
#     )
#     parser.add_argument(
#         "--zipf_parameter", type=float, default=0.8
#     )
#     parser.add_argument(
#         "--rank", type=int, default=5
#     )

#     # network paramters
#     parser.add_argument("--share_policy", action='store_false',
#                         default=True, help='Whether agent share the same policy')
#     parser.add_argument("--use_centralized_V", action='store_false',
#                         default=True, help="Whether to use centralized V function")
#     parser.add_argument("--stacked_frames", type=int, default=1,
#                         help="Dimension of hidden layers for actor/critic networks")
#     parser.add_argument("--use_stacked_frames", action='store_true',
#                         default=False, help="Whether to use stacked_frames")
#     parser.add_argument("--hidden_size", type=int, default=64,
#                         help="Dimension of hidden layers for actor/critic networks") 
#     parser.add_argument("--layer_N", type=int, default=1,
#                         help="Number of layers for actor/critic networks")
#     parser.add_argument("--use_ReLU", action='store_false',
#                         default=True, help="Whether to use ReLU")
#     parser.add_argument("--use_popart", action='store_true', default=False, help="by default False, use PopArt to normalize rewards.")
#     parser.add_argument("--use_valuenorm", action='store_false', default=True, help="by default True, use running mean and std to normalize rewards.")
#     parser.add_argument("--use_feature_normalization", action='store_false',
#                         default=True, help="Whether to apply layernorm to the inputs")
#     parser.add_argument("--use_orthogonal", action='store_false', default=True,
#                         help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
#     parser.add_argument("--gain", type=float, default=0.01,
#                         help="The gain # of last action layer")

#     # recurrent parameters
#     parser.add_argument("--use_naive_recurrent_policy", action='store_true',
#                         default=False, help='Whether to use a naive recurrent policy')
#     parser.add_argument("--use_recurrent_policy", action='store_false',
#                         default=False, help='use a recurrent policy')
#     parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
#     parser.add_argument("--data_chunk_length", type=int, default=10,
#                         help="Time length of chunks used to train a recurrent_policy")

#     # optimizer parameters
#     parser.add_argument("--lr", type=float, default=5e-4,
#                         help='learning rate (default: 5e-4)')
#     parser.add_argument("--critic_lr", type=float, default=5e-4,
#                         help='critic learning rate (default: 5e-4)')
#     parser.add_argument("--opti_eps", type=float, default=1e-5,
#                         help='RMSprop optimizer epsilon (default: 1e-5)')
#     parser.add_argument("--weight_decay", type=float, default=0)

#     # ppo parameters
#     parser.add_argument("--ppo_epoch", type=int, default=15,
#                         help='number of ppo epochs (default: 15)')
#     parser.add_argument("--use_clipped_value_loss",
#                         action='store_false', default=True, help="by default, clip loss value. If set, do not clip loss value.")
#     parser.add_argument("--clip_param", type=float, default=0.2,
#                         help='ppo clip parameter (default: 0.2)')
#     parser.add_argument("--num_mini_batch", type=int, default=1,
#                         help='number of batches for ppo (default: 1)')
#     parser.add_argument("--entropy_coef", type=float, default=0.01,
#                         help='entropy term coefficient (default: 0.01)')
#     parser.add_argument("--value_loss_coef", type=float,
#                         default=1, help='value loss coefficient (default: 0.5)')
#     parser.add_argument("--use_max_grad_norm",
#                         action='store_false', default=False, help="by default, use max norm of gradients. If set, do not use.")
#     parser.add_argument("--max_grad_norm", type=float, default=10.0,
#                         help='max norm of gradients (default: 0.5)')
#     parser.add_argument("--use_gae", action='store_false',
#                         default=True, help='use generalized advantage estimation')
#     parser.add_argument("--gamma", type=float, default=0.99,
#                         help='discount factor for rewards (default: 0.99)')
#     parser.add_argument("--gae_lambda", type=float, default=0.95,
#                         help='gae lambda parameter (default: 0.95)')
#     parser.add_argument("--use_proper_time_limits", action='store_true',
#                         default=False, help='compute returns taking into account time limits')
#     parser.add_argument("--use_huber_loss", action='store_false', default=True, help="by default, use huber loss. If set, do not use huber loss.")
#     parser.add_argument("--use_value_active_masks",
#                         action='store_false', default=True, help="by default True, whether to mask useless data in value loss.")
#     parser.add_argument("--use_policy_active_masks",
#                         action='store_false', default=True, help="by default True, whether to mask useless data in policy loss.")
#     parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

#     # run parameters
#     parser.add_argument("--use_linear_lr_decay", action='store_true',
#                         default=False, help='use a linear schedule on the learning rate')

#     # save parameters
#     parser.add_argument("--save_interval", type=int, default=1, help="time duration between contiunous twice models saving.")

#     # evaluation parameters
#     parser.add_argument("--use_eval", action='store_true', default=False, help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
#     parser.add_argument("--eval_interval", type=int, default=25, help="time duration between contiunous twice evaluation progress.")
#     parser.add_argument("--eval_episodes", type=int, default=32, help="number of episodes of a single evaluation.")

#     # render parameters
#     parser.add_argument("--save_gifs", action='store_true', default=False, help="by default, do not save render video. If set, save video.")
#     parser.add_argument("--use_render", action='store_true', default=False, help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")


#     arglist = parser.parse_args()
#     return arglist