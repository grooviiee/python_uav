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
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_actions),
            nn.Softmax()
        )
        self.optimizer = optim.Adam(self.pi.parameters(),
                                    lr=self.alpha)
        self.memory = []
        
    # pi에서 softmax 함수에 의해 각 행동을 취할 확률이 나오고, 그에 따라 행동이 return된다.
    def act(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            policy_probs = torch.distributions.Categorical(self.pi(state))

        return policy_probs.sample()
        
    def append_sample(self, state, action, reward):
        state = torch.FloatTensor(state)
        reward = torch.FloatTensor([reward])
        self.memory.append((state, action, reward))
        
    def calcluate_returns(self, rewards):
        returns = torch.zeros(rewards.shape)
        g_t = 0
        for t in reversed(range(0, len(rewards))):
            g_t = g_t * .99 + rewards[t].item()
            returns[t] = g_t
        return returns.detach()
    
    def update(self):
        states = torch.stack([m[0] for m in self.memory])
        actions = torch.stack([m[1] for m in self.memory]) 
        rewards = torch.stack([m[2] for m in self.memory])
        
        returns = self.calculate_returns(rewards)
        returns = (returns - returns.mean()) / returns.std()
        self.optimizer.zero_grad()
        policy_log_probs = self.pi(torch.FloatTensor(states)).log()
        policy_loss = torch.cat([-lp[a].unsqueeze(0) * g for a, lp, g in zip(actions, policy_log_probs, returns)])
        policy_loss = policy_loss.sum()
        policy_loss.backward()
        
        self.optimizer.step()
            
        self.memory = []
        return policy_loss.item()
    
    
env = gym.make('CartPole-v1')
env = wrappers.Monitor(env, "./video", force=True)
observation = env.reset()
agent = REINFORCE(observation.shape[0], 2)

observation

rewards = []
for ep in range(500):
    done = False
    obs = env.reset()
    action = agent.act(obs)

    ep_rewards = 0
    
    #일단 실행을 하고
    while not done:
        next_obs, reward, done, info = env.step(action.item())
        ep_rewards += reward
        next_action = agent.act(next_obs)

        agent.append_sample(obs, action, reward)
        obs = next_obs
        action = next_action

    # 실행값을 REINFORCE 알고리즘을 통해 update
    pi_loss = agent.update()
    rewards.append(ep_rewards)

    if (ep+1) % 10 == 0:
        print("episode: {}, loss: {:.3f}, rewards: {:.3f}".format(ep+1, pi_loss, ep_rewards))

env.close()


from utils import show_video

show_video()