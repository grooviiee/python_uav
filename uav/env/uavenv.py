import numpy as np
from gym import spaces
from uav.env.util import util
import math

# Refernece : C:\Users\June\Desktop\git\rl\maac\MAAC\envs\mpe_scenarios\fullobs_collect_treasure.py
# UAV Environment scenario

S = 10 * 1024 * 1024 # 10 Mbits
B = 20*10^6
W = 10*10^6
MBS_POWER = 2 #Watt
SPEED_OF_LIGHT = 3 * 10^8
CARRIER_FREQEUENCY = 2*10^9
QUOTA_UAV = 4
QUOTA_MBS = 20
PATHLOSS_EXP = 2
NOISE_POWER = -100 #dB/Hz
class Scenario(object):
    def __init__(self, args):
        # parameter setting
        self.num_agents = args.num_agents
        self.num_users = args.num_users
        self.num_files = args.num_files
        self.scenario_name = args.scenario_name
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        
        if self.num_agents == 1:
            self.act_space.append(self.env.action_space)
            self.obs_space.append(self.env.observation_space)
            self.share_obs_space.append(self.env.observation_space)
        else:
            for agentIdx in range(self.num_agents):
                self.action_space.append(spaces.Discrete(
                    n=self.env.action_space[agentIdx].n
                ))
                self.observation_space.append(spaces.Box(
                    low=self.env.observation_space.low[agentIdx],
                    high=self.env.observation_space.high[agentIdx],
                    shape=self.env.observation_space.shape[1:],
                    dtype=self.env.observation_space.dtype
                ))
                self.share_observation_space.append(spaces.Box(
                    low=self.env.observation_space.low[agentIdx],
                    high=self.env.observation_space.high[agentIdx],
                    shape=self.env.observation_space.shape[1:],
                    dtype=self.env.observation_space.dtype
                ))

    def reset(self):
        obs = self.env.reset()
        obs = self._obs_wrapper(obs)
        return obs

    # returns: next state, reward, done, etc.
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._obs_wrapper(obs)
        reward = reward.reshape(self.num_agents, 1)
        if self.share_reward:
            global_reward = np.sum(reward)
            reward = [[global_reward]] * self.num_agents

        done = np.array([done] * self.num_agents)
        info = self._info_wrapper(info)
        return obs, reward, done, info
    
    # "UAV"
    #     state = {x, y} for UAV , {x, y} for all users , {x_u,f} for all user and files
    #     action = {y_m,f} for all files, {d_m}, {theta_m}, {power_m}
    # "Master MBS"
    #     state = {x, y} for BS , {x, y} for all users , {x, y} for UAVs
    #     action = {z_u,i} which node connected with user
    def reward(self, state, action):
        # Receiving state and action as list 
        epsilon = 0.2
        # step 1. Get extr reward (Action에 대해서 state를 모두 바꾸었을 때, reward를 계산)
        extr_reward = CalaExtrReward(self, state, action)
        
        # step 2. Get intr reward
        intr_reward = CalcIIntrReward(self)

        return (extr_reward + epsilon * intr_reward)
    
    def CalcExtrReward(self, status, action):
        num_agent = len(status)
        # step 1. action에 대한 status 변경 
        for idx in num_agent:
            if (idx == 0):
                print("Process for MBS")
                
            else:
                print("Process for UAV")
            
        # step 2. 변경된 status로 reward 계산
        L = 100000000 # very large number
        Delay = 0
        for node in self.num_agent:
            for user in self.num_user:
                if node == 0:
                    isMbs = False
                else:
                    isMbs = True
                    
                Delay += GetDelay(self, node, user, isMbs)
                
        return (L - Delay)
            
    def GetDelay(self, node, user, isMbs):
        if isMbs == True: # for MBS
            for file in self.num_files:
                delay = self.x(user, file) * self.z(user, node) * self.T_down(node, user)
        else: # for User
            for file in self.num_files:
                delay = self.x(user, file) * self.z(user, node) * {self.T_down(node, user) + (1 - self.y(node, file)) * self.T_back(node, user)}
        
        return delay

    def CalcT_down(self):
        return S / R_2(i,u)
        
    def CalcT_back(self):
        return S / R_3(b,m,u)

    def R_2(self, i, u):
        numfile = 0
        for file in self.num_files:
            numfile += x(u, file) * z(u, i)
            
        upper = numfile * W
        lower = 0
        for user in self.num_users:
            for file in self.num_files:
                lower += x(u, file) * z(u, i)
                
        return (upper / lower * math.log2(1 + r(i, u)))
    
    def R_3(self, b, m, u):
        left = math.log2(1 + r(b,m))
        upper, lower = 0
        for file in self.num_files:
            upper += x(u, file) * z(u, m) * (1 - y(m, file))
        upper *= B
        
        for user in self.num_users:
            for file in self.num_files:
                for node in self.num_agents:
                    lower += x(user, file) * z(user, file) * (1 - y(node, file))
                    
        return (left * upper / lower)
                    
    def r(self, i , u):
        NotImplemented