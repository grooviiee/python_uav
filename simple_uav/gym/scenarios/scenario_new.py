import numpy as np
import random


class Scenario(BaseScenario):
	def make_world(self, args):
		world = World()
		world.logger = args.logger
		