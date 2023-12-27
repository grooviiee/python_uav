# Main function of system
# Choose environment, algorithm and other settings

import numpy as np
import sys
import time
import pickle
import torch
import os
import wandb
import socket
import logging
from utils.logger import SetLogger
from utils.config import parse_args
from pathlib import Path
from envs.UavEnvMain import UAVEnvMain
from runner.singleBS_runner import SingleBS_runner
from runner.multipleBS_runner import MultipleBS_runner


def main(arglist):
    # set logging system
    LOGGER = SetLogger("python_sim.log")
    formatter = logging.Formatter("%(asctime)s [%(levelname)8s] %(message)s")
    LOGGER.debug("Log system just set up...")
    arglist.logger = LOGGER

    # select device
    print(f"[MAIN] Training device...")
    if torch.cuda.is_available():
        print(f"[MAIN] choose to use [GPU]...")
        device = torch.device("cuda")
        torch.set_num_threads(arglist.n_training_threads)
        if arglist.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print(f"[MAIN] choose to use [CPU]...")
        device = torch.device("cpu")
        torch.set_num_threads(arglist.n_training_threads)

    # select algorithm settings
    print(
        f"[MAIN] choose algorithm settings. n_training_threads ({arglist.n_training_threads})"
    )

    # select algorithm
    if arglist.algorithm_name == "random":
        print("[MAIN] Choose Learning Algorithm.. Random walk.")
    elif arglist.algorithm_name == "mappo":
        print(
            "[MAIN] Choose Learning Algorithm.. MAPPO, we set use_recurrent_policy to be False"
        )
    elif arglist.algorithm_name == "attention":
        print(
            "[MAIN] Choose Learning Algorithm.. Attention_based_MAPPO, we set use_recurrent_policy to be False"
        )
    elif arglist.algorithm_name == "ddpg":
        print(
            "[MAIN] Choose Learning Algorithm.. DDPG, we set use_recurrent_policy to be False"
        )
        print(f"This Algorithm has Not implemented yet")
        raise NotImplementedError
    else:
        raise NotImplementedError

    # run dir
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / arglist.env_name
        / arglist.scenario_name
        / arglist.algorithm_name
        / arglist.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # Experiment_tracking_tool
    if arglist.experiment_tracking_tool == "use_wandb":
        run = wandb.init(
            config=arglist,
            project="python_uav",
            entity=arglist.user_name,
            notes=socket.gethostname(),
            name=str(arglist.algorithm_name)
            + "_"
            + str(arglist.experiment_name)
            + "_seed"
            + str(arglist.seed),
            group=arglist.scenario_name,
            dir=str(run_dir),
            job_type="training",
        )
    elif arglist.experiment_tracking_tool == "manual":
        if not run_dir.exists():
            curr_run = "run1"
        else:
            exst_run_nums = [
                int(str(folder.name).split("run")[1])
                for folder in run_dir.iterdir()
                if str(folder.name).startswith("run")
            ]
            if len(exst_run_nums) == 0:
                curr_run = "run1"
            else:
                curr_run = "run%i" % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))
    else:
        print("[MAIN] Do not store experiment results")

    # env init
    print("[MAIN] Load Environement...")
    envs = make_train_env(arglist)

    print("[MAIN] Set Evaluation: ", arglist.use_eval)
    if arglist.use_eval:
        eval_envs = make_eval_env(arglist)
    else:
        eval_envs = None

    config = {
        "args": arglist,
        "envs": envs,
        "eval_envs": eval_envs,
        "device": device,
        "algorithm": arglist.algorithm_name,
        "num_uavs": arglist.num_uavs,
        "num_mbs": arglist.num_mbs,
        "num_users": arglist.num_users,
        "run_dir": run_dir,
    }

    print(f"[MAIN] Load runner as {arglist.runner_name}")
    # run experiment
    if arglist.runner_name == "singleBS":
        runner = SingleBS_runner(config)
    elif arglist.runner_name == "multipleBS":
        runner = MultipleBS_runner(config)
    else:
        raise NotImplementedError

    runner.run()
    print("[MAIN] Running alorithm finished.")

    envs.close()


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
    # TODO: Need Implementation
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "uavnet":
                env = UAVEnvMain(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv(
            [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)]
        )


if __name__ == "__main__":
    print("[MAIN] Main code starts")
    arglist = parse_args()

    if len(sys.argv) == 2:
        arglist.algorithm_name = sys.argv[1]

    # Usage: input algorithm_name python main.pu --algorithm_name {random, ddpg, mappo, attention)
    print(f'[MAIN] You choose "{arglist.algorithm_name}"')

    main(arglist)
