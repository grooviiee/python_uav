import argparse
import numpy as np
import time
import pickle
import torch

# import maddpg.common.tf_util as U
from trainer.mappo import MAPPOAgentTrainer
from scenario.singleBS_runner import SingleBS_runner


def make_env(arglist, benchmark=False):
    # from multiagent.environment import MultiAgentEnv
    # import multiagent.scenarios as scenarios

    # create world
    runner = SingleBS_runner(arglist)
    world = SingleBS_runner.make_world(runner, arglist)
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(
            world,
            scenario.reset_world,
            scenario.reward,
            scenario.observation,
            scenario.benchmark_data,
        )
    else:
        env = MultiAgentEnv(
            world, scenario.reset_world, scenario.reward, scenario.observation
        )
    return env


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    traubers = []
    # model = mlp_model
    trainer = MAPPOAgentTrainer  # AttentionMAPPOAgentTrainer


def main(arglist):
    # device selection

    # cuda case
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

    # env init
    envs = make_env(arglist)
    eval_envs = make_eval_env(arglist) if arglist.use_eval else None
    num_agents = arglist.num_agents

    config = {
        "args": arglist,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # run experiment
    if all_args.numBS == 0:
        from UAV.runner.singleBS_runner import singleBS_runner as Runner
    else:
        from UAV.runner.multipleBS_runner import multipleBS_runner as Runner

    runner = Runner(config)
    runner.run()

    envs.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reinforcement Learning experiments for multiagent environments"
    )

    parser.add_argument("--device", default="gpu", help="Choose device. cpu or gpu?")

    parser.add_argument(
        "--scenario", type=str, default="mappo", choices=["rmappo", "mappo"]
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
