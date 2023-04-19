import numpy as np
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario


# Refernece : C:\Users\June\Desktop\git\rl\maac\MAAC\envs\mpe_scenarios\fullobs_collect_treasure.py
# UAV Environment scenario
class Scenario(BaseScenario):
    def make_world(self):
        """
        Creates a MultiAgentEnv object as env. This can be used similar to a gym
        environment by calling env.reset() and env.step().
        Use env.render() to view the environment on the screen.

        Input:
            scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                                (without the .py extension)
            benchmark       :   whether you want to produce benchmarking data
                                (usually only done during evaluation)

        Some useful env properties (see environment.py):
            .observation_space  :   Returns the observation space for each agent
            .action_space       :   Returns the action space for each agent
            .n                  :   Returns the number of Agents
        """

        world = World()
        # add agents
        world.agents = [Agent() for i in range(1)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent %d" % i
            agent.collide = False
            agent.silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(1)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75, 0.75, 0.75])
        world.landmarks[0].color = np.array([0.75, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
        return -dist2

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)

    def step(self, actions):
        status = 1
        reward = 2
        done = False
        info = "Donno what"

        return status, reward, done, info

    def reset(self, num_gnb, num_uav, num_ue):
        return status
