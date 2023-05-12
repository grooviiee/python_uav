import argparse
import numpy as np
import time
import pickle
import torch

# import maddpg.common.tf_util as U
from algorithms.mappo import MAPPOAgentTrainer
from envs.uavenv import UAV_ENV

# def make_train_env(arglist, benchmark=False):
    # from multiagent.environment import MultiAgentEnv
    # import multiagent.scenarios as scenarios

    # create world
        #runner = SingleBS_runner(arglist)
        #world = SingleBS_runner.make_world(runner, arglist)
    # create multiagent environment
    # if arglist. :
    #     env = MultiAgentEnv(
    #         world,
    #         scenario.reset_world,
    #         scenario.reward,
    #         scenario.observation,
    #         scenario.benchmark_data,
    #     )
    # else:
    #     env = MultiAgentEnv(
    #         world, scenario.reset_world, scenario.reward, scenario.observation
    #     )
    # return env

def make_train_env(arglist, benchmark=False):
    if arglist.scenario_name == "uavenv":
        env = UAV_ENV(arglist)
    else:
        print("Can not support the " + arglist.scenario_name + " environment.")
        raise NotImplementedError
    
    env.seed(arglist.seed + arglist.rank * 1000)
    return env

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
    elif arglist.algorithm_name == "attention_mappo":
        print("u are choosing to use attention_mappo, we set use_recurrent_policy to be True")
    else:
        raise NotImplementedError

    # env init
    envs = make_train_env(arglist)
    #eval_envs = make_eval_env(arglist) if arglist.use_eval else None

    config = {
        "args": arglist,
        "envs": envs,
        "device": device,
        "num_uavs": arglist.num_uavs,
        "num_mbs": arglist.num_mbs,
        #"run_dir": run_dir,  -> used in wandb???
    }

    # run experiment
    if arglist.runner_name == "singleBS":
        from runner.singleBS_runner import SingleBS_runner as Runner
    elif arglist.runner_name == "multipleBS":
        from runner.multipleBS_runner import MultipleBS_runner as Runner
    else:
        NotImplemented

    runner = Runner(config)
    runner.run()

    envs.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reinforcement Learning experiments for multiagent environments"
    )

    parser.add_argument("--device", default="gpu", help="Choose device. cpu or gpu?")

    parser.add_argument(
        "--scenario_name", type=str, default="uavenv", choices=["uavenv"]
    )

    parser.add_argument(
        "--runner_name", type=str, default="singleBS", choices=["singleBS", "multipleBS"]
    )

    parser.add_argument(
        "--algorithm_name", type=str, default="mappo", choices=["ddpg", "mappo", "attention_mappo"]
    )
    
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="check",
        help="an identifier to distinguish different experiment.",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for numpy/torch"
    )
    parser.add_argument(
        "--cuda",
        action="store_false",
        default=True,
        help="by default True, will use GPU to train; or else will use CPU;",
    )
    parser.add_argument(
        "--cuda_deterministic",
        action="store_false",
        default=True,
        help="by default, make sure random seed effective. if set, bypass such function.",
    )
    parser.add_argument(
        "--n_training_threads",
        type=int,
        default=1,
        help="Number of torch threads for training",
    )
    parser.add_argument(
        "--n_rollout_threads",
        type=int,
        default=32,
        help="Number of parallel envs for training rollouts",
    )
    parser.add_argument(
        "--n_eval_rollout_threads",
        type=int,
        default=1,
        help="Number of parallel envs for evaluating rollouts",
    )
    parser.add_argument(
        "--n_render_rollout_threads",
        type=int,
        default=1,
        help="Number of parallel envs for rendering rollouts",
    )
    parser.add_argument(
        "--num_env_steps",
        type=int,
        default=10e6,
        help="Number of environment steps to train (default: 10e6)",
    )
    parser.add_argument(
        "--user_name",
        type=str,
        default="marl",
        help="[for wandb usage], to specify user's name for simply collecting training data.",
    )
    
    # Environment settings
    parser.add_argument(
        "--num_mbs", type=int, default=1
    )
    parser.add_argument(
        "--num_uavs", type=int, default=4
    )
    parser.add_argument(
        "--num_users", type=int, default=20
    )
    parser.add_argument(
        "--num_files", type=int, default=10
    )
    parser.add_argument(
        "--map_size", type=int, default=1800
    )
    
    parser.add_argument(
        "--rank", type=int, default=5
    )
    
    parser.add_argument(
        "--use_wandb",
        action="store_false",
        default=True,
        help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.",
    )

    arglist = parser.parse_args()
    return arglist


if __name__ == "__main__":
    print("Main code starts")
    arglist = parse_args()
    main(arglist)
