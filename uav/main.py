import argparse
import numpy as np
import time
import pickle

import maddpg.common.tf_util as U
from uav_env.trainer.mappo import MAPPOAgentTrainer


def make_env(arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(arglist.scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
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
    model = mlp_model
    trainer = MAPPOAgentTrainer  # AttentionMAPPOAgentTrainer


def main(arglist):
    # device selection
    
    # cuda case
    if arglist.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
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
        "run_dir": run_dir
    }  
    
    #run experiment
    if all_args.numBS == 0:
        from UAV.runner.singleBS_runner import singleBS_runner as Runner
    else:
        from UAV.runner.multipleBS_runner import multipleBS_runner as Runner
        
    runner = Runner(config)
    runner.run()
    
    envs.close()


if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)
