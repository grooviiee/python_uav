import numpy as np
from gym import spaces
import gym
from gym.spaces.space import Space
from gym.spaces.utils import flatdim, flatten

# from util import util
import math
import random

# UAV Environment scenario
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
        self.log_level = world.log_level
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
        print(f"[INIT_ENV] NUM_UAV: {world.num_uavs}, NUM_MBS: {world.num_mbs}, NUM_USER: {world.num_users}")
        for agent in self.agents:
            if agent.isMBS == False:
                continue

            # Action space Definition
            # Action = {z_u,i} which node connected with user(discrete)
            total_action_space = []

            u_action_space = spaces.Box(low=0, high=1, shape=((world.num_uavs + world.num_mbs) * world.num_users, ),  dtype=np.bool8,)  # [0,1][Association]
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
            print(f"[INIT_ENV_MBS] obs_space: {obs_space}, act_space: {act_space}")

        # "UAV"
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

            u_observation_space = spaces.Box(low=0, high=world.map_size, shape=(2, ), dtype=np.int32)  # [location][uav]
            total_observation_space.append(u_observation_space)

            u_observation_space = spaces.Box(low=0, high=world.map_size, shape=(2*world.num_users, ), dtype=np.int32)  # [location][user]
            total_observation_space.append(u_observation_space)

            u_observation_space = spaces.Box(low=0, high=world.num_files, shape=(world.num_users, ), dtype=np.int16)  # [user][files]
            total_observation_space.append(u_observation_space)

            obs_space = spaces.Tuple(total_observation_space)
            self.observation_space.append(obs_space)

            print(f"[INIT_ENV_UAV] agent_id {agent.agent_id} Finished, obs_space: {obs_space}, act_space: {act_space}")

        # "User"
        print(f"[INIT_ENV_AGENT] Set UAV state and action space")
        for user in self.users:
            NotImplemented



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
    def _get_obs(self, agent, world):
        obs = []
        uav_location = []
        for id, uav in enumerate(world.agents):
            if uav.isUAV == False:
                continue
            uav_location.append(uav.state.x)
            uav_location.append(uav.state.y)

        user_location = []            
        file_request = []
        for id, user in enumerate(world.users):
            user_location.append(user.state.x)
            user_location.append(user.state.y)
            file_request.append(user.state.file_request)
        
        if agent.isUAV == False:
            # Every entity location
            my_location = []
            my_location.append(agent.state.x)
            my_location.append(agent.state.y)
            obs.append(my_location)
            obs.append(uav_location)
            obs.append(user_location)
        else:
            # UAV itself location + User location + User file request
            my_location = []
            my_location.append(agent.state.x)
            my_location.append(agent.state.y)
            obs.append(my_location)
            obs.append(user_location)
            obs.append(file_request)

        return obs

    # get reward for a particular agent
    def _get_reward(self, agent):
        # if self.reward_callback is None:
        #     return 0.0
        return agent.reward

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        is_done = False
        if self.done_callback is None:
            if self.current_step >= self.world_length:
                is_done = True
        print(f'is_done: {is_done}, self.current_step: {self.current_step}, self.world_length: {self.world_length}')
        return is_done

    # set env action for the agent.. Just setting. Change states at core.py 
    def _set_action(self, agent_id, action, agent, action_space, time=None):

        action_set = action[agent_id]
        if agent.isUAV == True:
            # Do UAV Action  (Set caching, trajectory, power)
            # for i in len(action_set):
            #     array = np.prod(action_set[i].shape)
            # action = flatten(action_set[agent], 1)
            # agent.action = flatten(action_set[agent_id], 1)
            agent.action = list(action_set)
            if self.log_level >= 3:
                print(f"[UAVENV] (_set_action) agent_id: {agent_id}, action_space: {action_space}, action: {action[agent_id]}")
                print(f"[UAVENV] (_set_action) action: {agent.action}")
        elif agent.isUAV == False:
            # Do MBS Action (Set associateion)
            agent.action = action_space.sample()
            if self.log_level >= 3:
                print(f"[UAVENV] (_set_action) agent_id: {agent_id}, action_space: {action_space}, action: {action[agent_id]}")
                print(f"[UAVENV] (_set_action) action: {agent.action}")

        else:
            NotImplementedError

    # desc. Take step in environments
    # returns: next state, reward, done, etc.
    def step(self, action):
        # action is coming with n_threads
        print(f"[ENV_STEP] Current_step: {self.current_step}, length: {len(action)}/{len(self.action_space)}")
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

        # record observation for each agent (return type is "list")
        for i, agent in enumerate(self.agents):
            obs_n.append(self._get_obs(agent, self.world))
            reward_n.append([agent.reward])
            done_n.append(self._get_done(agent))
            info = {"individual_reward": self._get_reward(agent)}
            env_info = self._get_info(agent)
            if "fail" in env_info.keys():
                info["fail"] = env_info["fail"]
            info_n.append(info)

        # all agents get total reward in cooperative case, if shared reward, all agents have the same reward, and reward is sum
        origin_reward_n = reward_n
        reward = np.sum(reward_n)
        print(f"[ENV_STEP] get reward_n: {reward_n}, self.shared_reward: {self.shared_reward}, reward: {reward}")

        if self.shared_reward:
            reward_n = [[reward]] * len(self.agents)

        if self.post_step_callback is not None:
            self.post_step_callback(self.world)

        return obs_n, reward_n, origin_reward_n, done_n, info_n