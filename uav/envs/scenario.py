import numpy as np
from envs.core import World, Agent, User

class BaseScenario(object):
    # Create Elements of the world.. It will be used as common settings
    def make_workd(self):
        raise NotImplementedError()
    
    # create initial conditions of the world
    def reset_world(self, world):
        raise NotImplementedError()

    def info(self, agent, world):
        return {}
    
    
class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        # set any world properties first
        world.world_length = args.map_size
        world.num_uavs = args.num_uavs
        world.num_mbs = args.num_mbs
        world.num_users = args.num_users
        world.map_size = args.map_size
        world.num_files = args.num_files
        world.users = [User() for i in range(world.num_users)]
        # Add agent as mbs
        for i in range(world.num_mbs):
            world.agents.append(Agent(True))
        # Add agent as uav
        for i in range(world.num_uavs):
            world.agents.append(Agent(False))
        # Add user
        for i in range(world.num_users):
            world.users.append(User())
        
        
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i

        for i, user in enumerate(world.users):
            user.name = 'user %d' % i
        
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
            agent.x_loc = None
            agent.y_loc = None
            
        for user in world.users:
            user.x_loc = None
            user.y_loc = None
        
    def reward(self, agent, world):
        # It is used at uavenv.py
        NotImplemented
        
    def observation(self, agent, world):
        # It is used at uavenv.py
        NotImplemented