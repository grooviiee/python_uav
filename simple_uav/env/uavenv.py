import numpy as np
from gym import spaces
# import env
from itertools import chain
from gym.spaces.space import Space
from gym.spaces.utils import flatdim, flatten

# from util import util
import math
import random


# UAV Environment scenario
# class UAV_ENV(env.Env):
class UAV_ENV():
    def __init__(
        self,
        world,
        logger,
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
        self.logger = logger
        self.current_step = 0
        self.world_length = world.world_length
        self.num_uavs = world.num_uavs  # number of uavs
        self.num_mbs = world.num_mbs  # number of base stations
        # self.num_nodes = self.num_uavs + self.num_mbs    # Not used currently
        self.agents = (
            self.world.agents
        )  # number of agents which has Deep Neural Network
        self.num_contents = world.num_contents
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
            [self.num_users, self.num_contents],
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
        self.shared_reward = (
            world.collaborative if hasattr(world, "collaborative") else False
        )

        print(
            f"[INIT_ENV] NUM_UAV: {world.num_uavs}, NUM_MBS: {world.num_mbs}, NUM_USER: {world.num_users}"
        )
        # "MBS"
        for agent in self.agents:
            if agent.isMBS == False:
                continue

            # Action space Definition
            # Action = {z_u,i} which node connected with user(discrete)
            total_action_space = []

            u_action_space = spaces.Box(
                low=0,
                high=1,
                shape=((world.num_uavs + world.num_mbs) * world.num_users,),
                dtype=np.bool8,
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
            print(f"[INIT_ENV_MBS] obs_space: {obs_space}, act_space: {act_space}")

        # "UAV"
        for agent in self.agents:
            if agent.isMBS == True:
                continue
            # Action space Definition
            # Action = {y_m,f} for all files, {d_m}, {theta_m}, {power_m}
            total_action_space = []

            # Caching
            u_action_space = spaces.Box(
                low=0, high=1, shape=(world.cache_capa,), dtype=np.bool8
            )  # [0,1][num_contents]
            total_action_space.append(u_action_space)

            # Power
            u_action_space = spaces.Box(
                low=0, high=23, shape=(1,), dtype=np.float32
            )  # [max_power]
            total_action_space.append(u_action_space)

            # Trajectory 1
            u_action_space = spaces.Box(
                low=0, high=20, shape=(1,), dtype=np.float32
            )  # [d_m]
            total_action_space.append(u_action_space)

            # Trajectory 2
            u_action_space = spaces.Box(
                low=0, high=360, shape=(1,), dtype=np.float32
            )  # [theta_m]
            total_action_space.append(u_action_space)

            act_space = spaces.Tuple(total_action_space)
            self.action_space.append(act_space)

            # Observation space Definition (n*n 모양의 배열로 만들어준다)
            # Observation = location {x, y} for UAV , location {x, y} for all users , {x_u,f} for all user and files
            total_observation_space = []

            u_observation_space = spaces.Box(
                low=0, high=world.map_size, shape=(2,), dtype=np.int32
            )  # [location][uav]
            total_observation_space.append(u_observation_space)

            u_observation_space = spaces.Box(
                low=0, high=world.map_size, shape=(2 * world.num_users,), dtype=np.int32
            )  # [location][user]
            total_observation_space.append(u_observation_space)

            u_observation_space = spaces.Box(
                low=0, high=world.num_contents, shape=(world.num_users,), dtype=np.int16
            )  # [user][files]
            total_observation_space.append(u_observation_space)

            obs_space = spaces.Tuple(total_observation_space)
            self.observation_space.append(obs_space)

            print(
                f"[INIT_ENV_UAV] agent_id {agent.agent_id} Finished, obs_space: {obs_space}, act_space: {act_space}"
            )

        # "User"
        print(f"[INIT_ENV_AGENT] Set UAV state and action space")
        for user in self.users:
            NotImplemented

    def get_obs_size(self):
        """Returns the size of the observation."""
        #     state = {x, y} for UAV itself, {x, y} for all users , file request dist. {x_u,f} for all user and files
        #     state = {x, y} for MBS itself, {x, y} for all users , {x, y} for UAVs

        # uav_obs_size = [[1, [self.x_len, self.y_len]], [self.num_users, [self.x_len, self.y_len]], [[self.num_users], [self.num_contents]] ]
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
            file_request = np.squeeze(file_request)
            obs.append(my_location)
            obs.append(user_location)
            obs.append(file_request)

        print(f"[ENV] [get_obs] agent_id: {agent.agent_id}, obs: {obs}")
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
        print(
            f"is_done: {is_done}, self.current_step: {self.current_step}, self.world_length: {self.world_length}"
        )
        return is_done

    # change action values from probability into valid number
    def _set_action(self, agent_id, action, agent, action_space, time=None):
        agent_action_set = action[agent_id]
        if agent.isUAV == True:
            # Do UAV Action  (Set caching, trajectory, power)
            print(
                f"[UAV_ACTION] agent_action_set length:({len(agent_action_set)}), agent_action_set[0]({len(agent_action_set[0])}): {agent_action_set[0]}"
            )
            agent.action = list(agent_action_set)
            agent.action = self.refine_uav_action(
                agent_action_set, agent.state.cache_size
            )
            if self.log_level >= 1:
                print(
                    f"[UAVENV] (_set_action) agent_id: {agent_id}, action_space: {action_space}, action: {action[agent_id]}"
                )
                print(f"[UAVENV] (_set_action) action: {agent.action}")
        elif agent.isUAV == False:
            # Do MBS Action (Decide associateion)
            print(
                f"[MBS_ACTION] action_set: {len(agent_action_set)}, {len(agent_action_set[0])}: agent_action_set[0]: {agent_action_set[0]}, agent_action_set: {agent_action_set}"
            )
            agent.action = self.refine_mbs_action(agent_action_set)
            if self.log_level >= 1:
                print(
                    f"[UAVENV] (_set_action) agent_id: {agent_id}, action_space: {action_space}, action: {action[agent_id]}"
                )
                print(f"[UAVENV] (_set_action) action: {agent.action}")

        else:
            raise NotImplementedError

    def refine_uav_action(self, action_space, cache_capa):
        print(
            f"[UAVENV] (refine_uav_action) cache_logit ({len(action_space[0][0])}): {action_space[0][0][0]}, power ({len(action_space[1])}): {action_space[1]}, location1 ({len(action_space[2])}), location2 ({len(action_space[3])})"
        )
        cache_logit = action_space[0][0][0].tolist()
        cache_logit = list(chain(*cache_logit))
        power = action_space[1]
        location1 = action_space[2]
        location2 = action_space[3]

        ranks = [sorted(cache_logit).index(ele) for ele in cache_logit]
        cache_list = ranks[0:cache_capa]
        print(
            f"[UAVENV] (refine_uav_action) {cache_logit} ranks: {ranks}/{cache_capa}, power: {power}, location1: {location1}, location2: {location2}"
        )

        action_results = [cache_list, power.item(), location1.item(), location2.item()]
        return action_results

    def refine_mbs_action(self, action_space):
        # action_space.shape = ((world.num_uavs + world.num_mbs) * world.num_users, )
        action_space = np.squeeze(action_space, axis=(0, 2))
        print(
            f"[refine_mbs_action] {type(action_space)} action_space: {action_space}, {action_space.shape}"
        )

        action_results = []
        # TODO: why we choose action_space[0]???
        for idx, val in enumerate(action_space[0]):
            agent_id = idx / self.num_users
            user = idx % self.num_users

            if agent_id == 0:
                if val >= 0:
                    action_results.append(True)
                else:
                    action_results.append(False)
            else:
                if val >= 0:
                    action_results.append(True)
                else:
                    action_results.append(False)

        print(f"[MBS_ACTION] refined_actions: {action_results}")
        return action_results

    def _set_random_action(self, agent_id, agent, action_space, time=None):
        if agent.isUAV == False:
            # association dimension: Array[nodes][users], value: {0,1}   shape=((world.num_uavs + world.num_mbs) * world.num_users, )
            # constraint Assocation MBS-USER = unlimit, UAV-USER = 4
            random_action = [[]]  # define
            random_action = [
                [False for i in range(self.world.num_uavs + self.world.num_mbs)]
                for j in range(self.world.num_users)
            ]  # initialization
            print(
                f"{self.world.num_uavs}, {self.world.num_mbs}, {self.world.num_users}, {len(random_action)}"
            )
            for i in range(self.world.num_uavs + self.world.num_mbs):
                for j in range(self.world.num_users):
                    random_action[j][i] = random.randrange(
                        0, 2
                    )  # association: true, false

            agent.action = random_action
            self.logger.debug(
                f"MBS agein_id: {agent_id}, random_action: {agent.action}"
            )

        elif agent.isUAV == True:
            # location, power, caching
            power = random.randrange(0, 23)  # power: 0~23
            cache = []
            for i in range(self.world.cache_capa):
                cache.append(
                    random.randrange(0, self.num_contents - 1)
                )  # caching: files capacity (3~7)

            location1 = random.randrange(0, 21)
            location2 = random.randrange(0, 361)

            # 1st dim : threads, 2nd dim: action space, order: cache -> power -> traj
            action_result = [[]]
            action_result[0].append(cache)
            action_result[0].append(power)
            action_result[0].append(location1)
            action_result[0].append(location2)

            agent.action = action_result
            self.logger.debug(
                f"UAV agein_id: {agent_id}, random_action: {agent.action}"
            )

        else:
            NotImplementedError

    # desc. Take step in environments
    # returns: next state, reward, done, etc.
    def step(self, action, is_random_mode):
        # action is coming with n_threads
        if is_random_mode is False:
            print(
                f"[ENV_STEP] NUM_MBS({self.num_mbs}) NUM_UAV({self.num_uavs}), Current_step: {self.current_step}, length: {len(action)}/{len(self.action_space)}"
            )
        else:
            print(
                f"[ENV_STEP] NUM_MBS({self.num_mbs}) NUM_UAV({self.num_uavs}), Current_step: {self.current_step}, length: {len(action)}, is_random_mode: {is_random_mode}"
            )

        self.current_step = self.current_step + 1
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []
        action_n = []

        self.agents = self.world.agents

        self.displayAgentState()
        self.displayUserState()

        # set action for each agent
        for i, agent in enumerate(self.agents):
            if is_random_mode is False:
                self._set_action(i, action, agent, self.action_space[i])
            else:
                self._set_random_action(i, agent, self.action_space[i])

            action_n.append(agent.action)

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
        print(
            f"[ENV_STEP] get reward_n: {reward_n}, self.shared_reward: {self.shared_reward}, reward: {reward}"
        )

        if self.shared_reward:
            reward_n = [[reward]] * len(self.agents)

        if self.post_step_callback is not None:
            self.post_step_callback(self.world)

        return obs_n, reward_n, origin_reward_n, done_n, info_n, action_n

    def displayAgentState(self):
        print(f"[STATE(AGENT)] displayAgentState")
        for idx, agent in enumerate(self.agents):
            if agent.isMBS:
                print(
                    f"[STATE(AGENT)] agent_id({idx}), is_uav({agent.isUAV}), state(x,y): ({agent.state.x}, {agent.state.y}) state(association): {agent.state.association}"
                )
            else:
                print(
                    f"[STATE(AGENT)] agent_id({idx}), is_uav({agent.isUAV}), state(x,y): ({agent.state.x}, {agent.state.y}) state(has_file): {agent.state.has_file}, state(cache_size): {agent.state.cache_size}, state(conn_user_file_req): {agent.state.file_request}"
                )

    def displayUserState(self):
        print(f"[STATE(USER)] displayUserState")
        for idx, user in enumerate(self.users):
            print(
                f"[STATE(USER)] user_id({idx}), state(x,y): ({user.state.x}, {user.state.y}) state(file_request): {user.state.file_request}"
            )
