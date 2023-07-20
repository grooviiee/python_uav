import numpy as np
import random
from envs.core import World, Agent, User
from envs.scenarios.scenario_base import BaseScenario

# obs_space and action_space need to be changed by adding RIC

class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()