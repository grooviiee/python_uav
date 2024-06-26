import numpy as np
import random
from envs.core import World, Agent, User
from envs.scenarios.scenario_base import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        world.logger = args.logger
        world.log_level = args.log_level
        # set any world properties first
        world.world_length = args.episode_length
        world.collaborative = True  # whether agents share rewards
        world.num_uavs = args.num_uavs
        world.num_mbs = args.num_mbs
        world.num_users = args.num_users  # dtype: int
        world.num_agents = args.num_uavs + args.num_mbs  # dtype: int
        world.map_size = args.map_size
        world.num_contents = args.num_contents
        world.cache_capa = args.cache_capa
        # world.users = [User(args.file_size, args.zipf_parameter) for i in range(world.num_users)]
        world.file_size = args.file_size
        world.zipf_parameter = args.zipf_parameter

        # Add agent as mbs
        for i in range(world.num_mbs):
            world.agents.append(Agent(True, 0))
        # Add agent as uav
        for i in range(world.num_uavs):
            world.agents.append(Agent(False, world.cache_capa))

        for i in range(world.num_agents):
            world.agents[i].agent_id = i

        # Add user
        for i in range(world.num_users):
            world.users.append(
                User(args.file_size, args.num_contents, args.zipf_parameter)
            )
            world.users[i].user_id = i

        for i, agent in enumerate(world.agents):
            agent.name = "agent %d" % i

        for i, user in enumerate(world.users):
            user.name = "user %d" % i

        # reset the world
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.assign_agent_colors()
        # random properties for landmarks
        world.assign_user_colors()

        # 위치 조정 (randomly)
        for agent in world.agents:
            if agent.isMBS:
                agent.state.x = world.map_size / 2
                agent.state.y = world.map_size / 2
            else:
                agent.state.x = random.randint(0, world.map_size)  # agent.agent_id%4
                agent.state.y = random.randint(0, world.map_size)  # agent.agent_id%4

        for user in world.users:
            user.state.x = random.randint(0, world.map_size)  # user.user_id%5
            user.state.y = random.randint(0, world.map_size)  # user.user_id%5
            user.state.file_request = random.randint(0, world.num_contents)
            user.remaining_file_size = user.file_size

    def reward(self, agent, world):
        # It is used at uavenv.py
        NotImplemented

    def observation(self, agent, world):
        # It is used at uavenv.py
        NotImplemented
