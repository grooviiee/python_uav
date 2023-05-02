import numpy as np
from gym import spaces


# Refernece : C:\Users\June\Desktop\git\rl\maac\MAAC\envs\mpe_scenarios\fullobs_collect_treasure.py
# UAV Environment scenario
class Scenario(object):
    def __init__(self, args):
        self.num_agents = args.num_agents
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
                Delay += GetDelay()
                
        return L - Delay
            
    def GetDelay(self):
        
        
        
        