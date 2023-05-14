import numpy as np
from core import World, Agent, User

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
        world.world_length = args.map_size
        world.num_uavs = args.num_uavs
        world.num_mbs = args.num_mbs
        world.num_users = args.num_users
        world.users = [User() for i in range(world.num_users)]
        world.agents = [Agent(True) for i in range(world.num_mbs)]
        world.agents.append([Agent(False) for i in range(world.num_uavs)])
        
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i

        for i, user in enumerate(world.users):
            user.name = 'user %d' % i
        
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # 위치 조정
        for agent in world.agents:
            agent.x_loc = None
            agent.y_loc = None
            
        for user in world.users:
            user.x_loc = None
            user.y_loc = None
        
    def reward():
        # It is used at uavenv.py
        NotImplemented
        
    def observation(self, agent, world):
        # It is used at uavenv.py
        NotImplemented