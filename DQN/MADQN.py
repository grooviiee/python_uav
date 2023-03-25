import time

import numpy as np
import torch

from DQN.DQNAgent import DQNAgent
# from DQN.VDNet import VDNet
from DQN.parameters import agent_params, vdn_params


class MADQN:
    def __init__(self, env):
        print("MADDQN initialized")
        # environment
        self.env = env
        self.obs_dim = env.observation_space[0].shape[0] + 1
        self.n_actions = env.action_space[0].n

        # training parameters
        self.batch_size = agent_params["batch_size"]
        self.steps_per_episode = vdn_params["steps_per_episode"]

        # initialize agents
        self.agents = [
            DQNAgent(self.obs_dim, self.n_actions, **agent_params)
            for _ in range(env.n_agents)
        ]
        self.burnin_steps = agent_params["burnin_steps"]
