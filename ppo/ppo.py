# reference: https://github.com/rlsotlr01/PPO_practice/blob/master/main.py 

import gym
import torch
from torch.distribution import Categorical
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# PPO Class
class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init()
        self.data = []
        self.gamma = 0.98
        self.lmbda = 0.95
        self.eps = 0.1
        self.K = 3
        
        self.fc1 = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0005)
        
    # PPO - Policy, Value Network 코드
    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        
        x = self.fc_pi(x)
        
        prob = F.softmax(x, dim=softmax_dim)
        
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        
        return v
    
    