import numpy as np
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario


# Refernece : C:\Users\June\Desktop\git\rl\maac\MAAC\envs\mpe_scenarios\fullobs_collect_treasure.py


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        num_collectors = 6
        num_agents = 8
        world.cache_dists = True
        
        # add agents
        world.agents = [Agent() for idx in range(num_agents)]        
        for idx, agent in enumerate(world.agents):
            agent.idx = idx
            agent.name = 'agent %d' % idx
            agent.collector = True if idx < num_collector else False
            # decide collector's display color
            if agent.collector == True:
                agent.color = np.array([0.85, 0.85, 0.85])
            else:
                agent.d_i = idx - num_collectors
                agent.colot = world.treasure_colors[agent.d_i] * 0.35
            
            agent.collide = True
            agent.silent = True
            agent.ghost = True
            agent.holding = None
            agent.size = 0.05 if agent.collector else 0.075
            agent.accel = 1.5
            agent.initial_mass = 1.0 if agent.collector else 2.25
            agent.max_speed = 1.0
        
    def reward(self, agent, world):
    
    def observation(self, agent, world):
    
    