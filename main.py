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
from utils.logger import Logger
from utils.config import parse_args
from pathlib import Path
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
    # set logging system
    logger = Logger("python_sim.log")
    logger.debug("Log system just set up...")
    
    formatter = logging.Formatter(u'%(asctime)s [%(levelname)8s] %(message)s')

    # select device
    print("choose device...", arglist.n_training_threads)
    if torch.cuda.is_available():
        print("choose to use [GPU]...")
        device = torch.device("cuda")
        torch.set_num_threads(arglist.n_training_threads)
        if arglist.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use [CPU]...")
        device = torch.device("cpu")
        torch.set_num_threads(arglist.n_training_threads)

    # select algorithm 
    if arglist.algorithm_name =="random":
        print("Choose to use random walker. RL not work")
    elif arglist.algorithm_name == "mappo":
        print("Choose to use mappo, we set use_recurrent_policy to be True")
    elif arglist.algorithm_name == "attention":
        print("Choose to use attention_based_mappo, we set use_recurrent_policy to be True")
    elif arglist.algorithm_name == "ddpg":
        print("Choose to use ddpg, we set use_recurrent_policy to be True")
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
                         project="python_uav",
                         entity=arglist.user_name,
                         notes=socket.gethostname(),
                         name=str(arglist.algorithm_name) + "_" + str(arglist.experiment_name) + "_seed" + str(arglist.seed),
                         group=arglist.scenario_name,
                         dir=str(run_dir),
                         job_type="training")
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
        "algorithm": arglist.algorithm_name,
        "num_uavs": arglist.num_uavs,
        "num_mbs": arglist.num_mbs,
        "num_users": arglist.num_users,
        "run_dir": run_dir,
        "logger": logger,
    }

    # run experiment
    if arglist.runner_name == "singleBS":
        from runner.singleBS_runner import SingleBS_runner as Runner
    elif arglist.runner_name == "multipleBS":
        from runner.multipleBS_runner import MultipleBS_runner as Runner
    else:
        NotImplemented

    print(f"Load runner as {arglist.runner_name}")
    runner = Runner(config)
    runner.run()
    print(f"Running alorithm finished.")


    envs.close()

if __name__ == "__main__":
    print("Main code starts")
    arglist = parse_args()
    
    if len(sys.argv) == 2:
        arglist.algorithm_name = sys.argv[1]

    # Usage: input algorithm_name python main.pu --algorithm_name {random, ddpg, mappo, attention)
    print(f"You choose \"{arglist.algorithm_name}\"")

    main(arglist)