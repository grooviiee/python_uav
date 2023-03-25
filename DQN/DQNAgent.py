import numpy as np
import torch

from DQN.DQNet import DQN
from DQN.ReplayBuffer import *


class DQAgent:
    def __init__(self, obs_dim: int, act_dim: int, **hyperparams):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self._init_hyperparams(hyperparams)
        self.epsilon = self.epsilon_start

        # initialize replay buffer
        self.memory = ReplayBuffer(capacity=self.replay_buffer_capacity)

        # policy and target approximation neural nets
        self.policy_net = DQN(obs_dim, act_dim, self.hidden_layer_dim)
        self.target_net = DQN(obs_dim, act_dim, self.hidden_layer_dim)

        self.t = 0
