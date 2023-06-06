import numpy as np
from gym import spaces
import gym
from gym.spaces.space import Space
from gym.spaces.utils import flatdim, flatten

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


class UAV_ENV(gym.Env):
    def __init__(
        self,
        world,
        reset_callback=None,
        reward_callback=None,
        observation_callback=None,
        info_callback=None,
        done_callback=None,
        post_step_callback=None,
        shared_viewer=True,
        discrete_action=True,
    ):

        # parameter setting from args
        self.world = world
        self.current_step = 0
        self.world_length = world.world_length
        self.num_uavs = world.num_uavs
        self.num_mbs = world.num_mbs
        self.num_nodes = self.num_uavs + self.num_mbs
        self.agents = self.world.agents
        self.num_files = world.num_files
        self.map_x_len = world.map_size
        self.map_y_len = world.map_size

        self.num_users = world.num_users
        self.users = self.world.users

        # scenario callback
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback

        self.post_step_callback = post_step_callback

        # for debugging
        self.uav_obs_size = [
            [1, [self.map_x_len, self.map_y_len]],
            [self.num_users, [self.map_x_len, self.map_y_len]],
            [[self.num_users], [self.num_files]],
        ]
        self.mbs_obs_size = [
            [1, [self.map_x_len, self.map_y_len]],
            [self.num_users, [self.map_x_len, self.map_y_len]],
            [self.num_uavs, [self.map_x_len, self.map_y_len]],
        ]

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

        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False

        # "Master MBS"
        print(f"[INIT_ENV_AGENT] Set MBS state and action space. NUM_UAV: {world.num_uavs}, NUM_MBS: {world.num_mbs}, NUM_USER: {world.num_users}")
        for agent in self.agents:
            if agent.isMBS == False:
                continue

            # Action space Definition
            # Action = {z_u,i} which node connected with user(discrete)
            total_action_space = []

            u_action_space = spaces.Box(
                low=0, high=1, shape=(world.num_uavs + world.num_mbs, world.num_users),  dtype=np.bool8,
            )  # [0,1][Association]
            total_action_space.append(u_action_space)

            act_space = u_action_space
            self.action_space.append(act_space)

            # Observation space Definition (n*n 모양의 배열로 만들어준다)
            # Observation = location {x, y} for BS(discrete) , {x, y} for all users(discrete) , {x, y} for UAVs(discrete)
            total_observation_space = []

            u_observation_space = spaces.Box(
                low=0, high=world.map_size, shape=(2, world.num_mbs)
            )  # [location][mbs]
            total_observation_space.append(u_observation_space)

            u_observation_space = spaces.Box(
                low=0,
                high=world.map_size,
                shape=(2, world.num_uavs),
                dtype=np.float32,
            )  # [location][uav]
            total_observation_space.append(u_observation_space)

            u_observation_space = spaces.Box(
                low=0,
                high=world.map_size,
                shape=(2, world.num_users),
                dtype=np.float32,
            )  # [location][user]
            total_observation_space.append(u_observation_space)

            obs_space = spaces.Tuple(total_observation_space)
            self.observation_space.append(obs_space)
        print(f"[INIT_ENV_AGENT] Set MBS state and action space Finished")

        # "UAV"
        print(f"[INIT_ENV_AGENT] Set UAV state and action space")
        for agent in self.agents:
            if agent.isMBS == True:
                continue
            # Action space Definition
            # Action = {y_m,f} for all files, {d_m}, {theta_m}, {power_m}
            total_action_space = []

            u_action_space = spaces.Box(
                low=0, high=1, shape=(world.num_files, ), dtype=np.bool8
            )  # [0,1][Num_files]
            total_action_space.append(u_action_space)

            u_action_space = spaces.Box(
                low=0, high=23, shape=(1, ), dtype=np.float32
            )  # [max_power]
            total_action_space.append(u_action_space)

            u_action_space = spaces.Box(
                low=0, high=20, shape=(1, ), dtype=np.float32
            )  # [d_m]
            total_action_space.append(u_action_space)

            u_action_space = spaces.Box(
                low=0, high=360, shape=(1, ), dtype=np.float32
            )  # [theta_m]
            total_action_space.append(u_action_space)

            act_space = spaces.Tuple(total_action_space)
            self.action_space.append(act_space)

            # Observation space Definition (n*n 모양의 배열로 만들어준다)
            # Observation = location {x, y} for UAV , location {x, y} for all users , {x_u,f} for all user and files
            total_observation_space = []

            u_observation_space = spaces.Box(
                low=0,
                high=world.map_size,
                shape=(2, world.num_uavs),
                dtype=np.float32,
            )  # [location][uav]
            total_observation_space.append(u_observation_space)

            u_observation_space = spaces.Box(
                low=0,
                high=world.map_size,
                shape=(2, world.num_users),
                dtype=np.float32,
            )  # [location][user]
            total_observation_space.append(u_observation_space)

            u_observation_space = spaces.Box(
                low=0, high=1, shape=(world.num_users, world.num_files), dtype=np.bool8
            )  # [user][files]
            total_observation_space.append(u_observation_space)

            obs_space = spaces.Tuple(total_observation_space)
            self.observation_space.append(obs_space)

        # "User"
        print(f"[INIT_ENV_AGENT] Set UAV state and action space")
        for user in self.users:
            NotImplemented

        print(f"[INIT_ENV_AGENT] Set UAV state and action space Finished")

    def get_obs_size(self):
        """Returns the size of the observation."""
        #     state = {x, y} for UAV itself, {x, y} for all users , file request dist. {x_u,f} for all user and files
        #     state = {x, y} for MBS itself, {x, y} for all users , {x, y} for UAVs

        # uav_obs_size = [[1, [self.x_len, self.y_len]], [self.num_users, [self.x_len, self.y_len]], [[self.num_users], [self.num_files]] ]
        # mbs_obs_size = [[1, [self.x_len, self.y_len]], [self.num_users, [self.x_len, self.y_len]], [self.num_uavs, [self.x_len, self.y_len]]]
        # print("uav_obs_size: %d, mbs_obs_size: %d", %self.n_uav_observation_space, %self.n_mbs_observation_space)
        return self.n_uav_observation_space + self.n_mbs_observation_space

    def seed(self, seed=None):
        if seed is None:
            random.seed(1)
        else:
            random.seed(seed)

    def reset(self):
        # TODO: Need to be clear
        self.current_step = 0
        self.reset_callback(self.world)
        obs_n = []
        self.agents = self.world.agents

        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        print(f"[ENV] Reset Environment.. (Obs) dType: {type(obs_n)}, {obs_n}")
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            if self.current_step >= self.world_length:
                return True
            else:
                return False
        return self.done_callback(agent, self.world)
        
    # set env action for the agent.. Just setting. Change states at core.py 
    def _set_action(self, agent_id, action, agent, action_space, time=None):
        print(f'Set action for agent_id: {agent_id}, isUAV: {agent.isUAV}, actionType: {type(action)}, len: {len(action)}')
        action_set = action[agent_id]
        if agent.isUAV == True:
            # Do UAV Action  (Set caching, trajectory, power)
            # for i in len(action_set):
            #     array = np.prod(action_set[i].shape)
            print(f"UAV Agent {agent_id}-th Action({type(action_set)})\n{action_set})..")
            # action = flatten(action_set[agent], 1)
            # agent.action = flatten(action_set[agent_id], 1)
            agent.action = list(action_set)

        elif agent.isUAV == False:
            # Do MBS Action (Set associateion)
            print(f"MBS Action({action_set})..")
            agent.action = action_set

        else:
            NotImplementedError

    # desc. Take step in environments
    # returns: next state, reward, done, etc.
    def step(self, action):
        # action is coming with n_threads
        print(f"[ENV_STEP] current_step: {self.current_step}, STEP: {action}, length: {len(action)}/{len(self.action_space)}")
        self.current_step = self.current_step + 1
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []

        self.agents = self.world.agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(i, action[0], agent, self.action_space[i])

        # advance world state
        self.world.world_take_step()  # core.step()

        # record observation for each agent
        for i, agent in enumerate(self.agents):
            obs_n.append(self._get_obs(agent))
            reward_n.append([self._get_reward(agent)])
            done_n.append(self._get_done(agent))
            info = {"individual_reward": self._get_reward(agent)}
            env_info = self._get_info(agent)
            if "fail" in env_info.keys():
                info["fail"] = env_info["fail"]
            info_n.append(info)

        # all agents get total reward in cooperative case, if shared reward, all agents have the same reward, and reward is sum
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [[reward]] * self.n

        if self.post_step_callback is not None:
            self.post_step_callback(self.world)

        return obs_n, reward_n, done_n, info_n