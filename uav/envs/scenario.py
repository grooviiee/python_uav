import numpy as np
from core import World, Agent

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
        world.agents = [Agent(True) for i in range(world.num_mbs)]
        world.agents.append([Agent(False) for i in range(world.num_uavs)])
        
        for i, agents in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            
        self.reset_world(world)
        return world

    def reset_world(self, world):
        #TODO
        # 위치 조정
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        
    def reward():
        # It is used at uavenv.py
        
    def observation(self, agent, world):
        # It is used at uavenv.py