import numpy as np
from gym import spaces

# from util import util
import math
import random

# Refernece : C:\Users\June\Desktop\git\rl\maac\MAAC\envs\mpe_scenario\environment.py

# UAV Environment scenario
# Rate calculation type
TYPE_MBS_USER = 0
TYPE_UAV_USER = 1
TYPE_MBS_UAV = 2

S = 10 * 1024 * 1024  # 10 Mbits
B = 20 * 10 ^ 6
W = 10 * 10 ^ 6
MBS_POWER = 2  # Watt
SPEED_OF_LIGHT = 3 * 10 ^ 8
CARRIER_FREQEUENCY = 2 * 10 ^ 9
QUOTA_UAV = 4
QUOTA_MBS = 20
PATHLOSS_EXP = 2
NOISE_POWER = -100  # dB/Hz


class UAV_ENV(object):
    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, post_step_callback=None,
                 shared_viewer=True, discrete_action=True):

        # parameter setting from args
        self.world = world
        self.num_mbs = args.num_mbs
        self.num_uavs = args.num_uavs
        self.num_users = args.num_users
        self.num_nodes = self.num_uavs + self.num_mbs
        self.agents = self.world.agents
        self.num_files = args.num_files
        self.scenario_name = args.scenario_name
        self.map_x_len = args.world.map_size
        self.map_y_len = args.world.map_size
        
        # for debugging
        self.uav_obs_size = [ [1, [self.map_x_len, self.map_y_len]], [self.num_users, [self.map_x_len, self.map_y_len]], [[self.num_users], [self.num_files]] ]
        self.mbs_obs_size = [ [1, [self.map_x_len, self.map_y_len]], [self.num_users, [self.map_x_len, self.map_y_len]], [self.num_uavs, [self.map_x_len, self.map_y_len]] ]
        
        # init parameters
        self.n_uav_observation_space = len(self.uav_obs_size)
        self.n_mbs_observation_space = len(self.mbs_obs_size)
        
        # Number of action space
        self.n_mbs_action = 1
        self.n_uav_action = 4
        self.n_actions = self.n_mbs_action + self.n_uav_action
        
        
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0

        for agent in self.agents:
            total_action_space = []
            
            
            self.action_space.append(
                spaces.Discrete(n=self.action_space[nodeIdx].n)
            )

            self.observation_space.append(
                spaces.Box(
                    low=self.observation_space.low[nodeIdx],
                    high=self.observation_space.high[nodeIdx],
                    shape=self.observation_space.shape[1:],
                    dtype=self.observation_space.dtype,
                )
            )
        # setting state, action 
        # if self.num_mbs == 1:
        #     self.action_space.append(spaces.Discrete(self.n_actions))
        #     # TODO: choose num observations
        #     self.observation_space.append(self.get_obs_size())
        #     self.share_observation_space.append(self.observation_space)
        # else:
           

    def get_obs_size(self):
        """Returns the size of the observation."""
        #     state = {x, y} for UAV itself, {x, y} for all users , file request dist. {x_u,f} for all user and files
        #     state = {x, y} for MBS itself, {x, y} for all users , {x, y} for UAVs

        # uav_obs_size = [[1, [self.x_len, self.y_len]], [self.num_users, [self.x_len, self.y_len]], [[self.num_users], [self.num_files]] ]
        # mbs_obs_size = [[1, [self.x_len, self.y_len]], [self.num_users, [self.x_len, self.y_len]], [self.num_uavs, [self.x_len, self.y_len]]] 
        #print("uav_obs_size: %d, mbs_obs_size: %d", %self.n_uav_observation_space, %self.n_mbs_observation_space)
        return self.n_uav_observation_space + self.n_mbs_observation_space

                
    def seed(self, seed=None):
        if seed is None:
            random.seed(1)
        else:
            random.seed(seed)

    def reset(self):
        # TODO: Need to clearence
        obs_n = []
        
        # for agent in self.num_mbs:
        #     obs_n.append(np.zeros(0))
        print("Reset environment")
        return obs_n

    # returns: next state, reward, done, etc.
    def step(self, action):
        obs, reward, done, info = self.step(action)
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
        extr_reward = CalcExtrReward(self, state, action)

        # step 2. Get intr reward
        intr_reward = CalcIIntrReward(self)

        return extr_reward + epsilon * intr_reward

    def CalcExtrReward(self, state, action):
        num_agent = len(state)
        # step 1. action에 대한 status 변경
        for idx in num_agent:
            if idx == 0:
                print("Process for MBS")

            else:
                print("Process for UAV")

        # step 2. 변경된 status로 reward 계산
        L = 100000000  # very large number
        Delay = 0
        for node in self.num_agent:
            for user in self.num_user:
                if node == 0:
                    isMbs = False
                else:
                    isMbs = True

                Delay += GetDelay(self, node, user, isMbs)

        return L - Delay

    def GetDelay(self, node, user, isMbs):
        if isMbs == True:  # for MBS
            for file in self.num_files:
                delay = (
                    self.x(user, file) * self.z(user, node) * self.T_down(node, user)
                )
        else:  # for User
            for file in self.num_files:
                delay = (
                    self.x(user, file)
                    * self.z(user, node)
                    * {
                        self.T_down(node, user)
                        + (1 - self.y(node, file)) * self.T_back(node, user)
                    }
                )

        return delay

    def Calc_T_down(self):
        return S / R_T_down(self, i, u)

    def Calc_T_back(self):
        return S / R_T_back(self, b, m, u)

    def R_T_down(self, i, u):
        numfile = 0
        for file in self.num_files:
            numfile += x(u, file) * z(u, i)

        upper = numfile * W
        lower = 0
        for user in self.num_users:
            for file in self.num_files:
                lower += x(u, file) * z(u, i)

        return upper / lower * math.log2(1 + r(i, u))

    def R_T_back(self, b, m, u):
        left = math.log2(1 + r(b, m))
        upper, lower = 0
        for file in self.num_files:
            upper += x(u, file) * z(u, m) * (1 - y(m, file))
        upper *= B

        for user in self.num_users:
            for file in self.num_files:
                for node in self.num_agents:
                    lower += x(user, file) * z(user, file) * (1 - y(node, file))

        return left * upper / lower

    def r(self, i, u, type):
        if type == TYPE_MBS_USER:
            res = MBS_POWER / (NOISE_POWER * math.pow(10, h_MbsUser(self, i, u) / 10))

        elif type == TYPE_UAV_USER:  # follows UAV Power
            lower = 0
            for uavIdx in self.num_uavs:
                if uavIdx == i:
                    continue
                lower += P(uavIdx) * math.pow(10, -h_UavUser(i, u) / 10)

            res = P(i) * math.pow(10, -h_UavUser(i, u) / 10) / (NOISE_POWER * lower)

        elif type == TYPE_MBS_UAV:
            res = MBS_POWER / (NOISE_POWER * math.pow(10, h_MbsUav(self, i, u) / 10))

        else:
            res = 0

        return res

    # Calculate pathloss
    def h_UavUser(self, m, u):
        return PLos(m, u) * hLos(m, u) + (1 - PLos(m, u)) * hNLos(m, u)

    def h_MbsUav(self, b, m):
        return PLos(b, m) * hLos(b, m) + (1 - PLos(b, m)) * hNLos(b, m)

    def h_MbsUser(self, b, u):
        return 15.3 + 37.6 * math.log10(d(self, b, u))

    def PLos(self, m, u):
        return 1 / (1 + c_1 * math.exp(-c_2 * (theta(m, u) - c_1)))

    def hLos(self, m, u):
        return 20 * math.log(4 * math.pi * d(self, m, u) / v_c) + X_Los

    def hNLos(self, m, u):
        return 20 * math.log(4 * math.pi * d(self, m, u) / v_c) + X_NLos

    # Calculate Distance
