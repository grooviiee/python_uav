import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import random
import gym
from gym import wrappers

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt


class REINFORCE:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = 0.001
        self.gamma = 0.99
        self.pi = nn.Sequential(
            nn.Linear(self.num_states, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
            nn.ReLU(),
            nn.Linear(32, self.num_actions),
            nn.Softmax()
        )
        self.optimizer = optim.Adam(self.pi.parameters(),
                                    lr=self.alpha)
        self.memory = []
        
    def act(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            policy_probs = torch.distributions.Categorical(self.pi(state))
            
        return policy_probs.sample()
        
    def append_sample
        
