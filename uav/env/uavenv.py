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
            landmark.state.p_vel = np.zeros(world.m_p)

    def reward(self, agent, world):
        dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
        return -dist2

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)

    # def step(self, action):
    #     """Take one step in the game.

    #     Args:
    #       action: dict, mapping to a legal action taken by an agent. The following
    #         actions are supported:
    #           - { 'action_type': 'PLAY', 'card_index': int }
    #           - { 'action_type': 'DISCARD', 'card_index': int }
    #           - {
    #               'action_type': 'REVEAL_COLOR',
    #               'color': str,
    #               'target_offset': int >=0
    #             }
    #           - {
    #               'action_type': 'REVEAL_RANK',
    #               'rank': str,
    #               'target_offset': int >=0
    #             }
    #         Alternatively, action may be an int in range [0, num_moves()).

    #     Returns:
    #       observation: dict, containing the full observation about the game at the
    #         current step. *WARNING* This observation contains all the hands of the
    #         players and should not be passed to the agents.
    #         An example observation:
    #         {'current_player': 0,
    #          'player_observations': [{'current_player': 0,
    #                             'current_player_offset': 0,
    #                             'deck_size': 40,
    #                             'discard_pile': [],
    #                             'fireworks': {'B': 0,
    #                                       'G': 0,
    #                                       'R': 0,
    #                                       'W': 0,
    #                                       'Y': 0},
    #                             'information_tokens': 8,
    #                             'legal_moves': [{'action_type': 'PLAY',
    #                                          'card_index': 0},
    #                                         {'action_type': 'PLAY',
    #                                          'card_index': 1},
    #                                         {'action_type': 'PLAY',
    #                                          'card_index': 2},
    #                                         {'action_type': 'PLAY',
    #                                          'card_index': 3},
    #                                         {'action_type': 'PLAY',
    #                                          'card_index': 4},
    #                                         {'action_type': 'REVEAL_COLOR',
    #                                          'color': 'R',
    #                                          'target_offset': 1},
    #                                         {'action_type': 'REVEAL_COLOR',
    #                                          'color': 'G',
    #                                          'target_offset': 1},
    #                                         {'action_type': 'REVEAL_COLOR',
    #                                          'color': 'B',
    #                                          'target_offset': 1},
    #                                         {'action_type': 'REVEAL_RANK',
    #                                          'rank': 0,
    #                                          'target_offset': 1},
    #                                         {'action_type': 'REVEAL_RANK',
    #                                          'rank': 1,
    #                                          'target_offset': 1},
    #                                         {'action_type': 'REVEAL_RANK',
    #                                          'rank': 2,
    #                                          'target_offset': 1}],
    #                             'life_tokens': 3,
    #                             'observed_hands': [[{'color': None, 'rank': -1},
    #                                             {'color': None, 'rank': -1},
    #                                             {'color': None, 'rank': -1},
    #                                             {'color': None, 'rank': -1},
    #                                             {'color': None, 'rank': -1}],
    #                                            [{'color': 'G', 'rank': 2},
    #                                             {'color': 'R', 'rank': 0},
    #                                             {'color': 'R', 'rank': 1},
    #                                             {'color': 'B', 'rank': 0},
    #                                             {'color': 'R', 'rank': 1}]],
    #                             'num_players': 2,
    #                             'vectorized': [ 0, 0, 1, ... ]},
    #                            {'current_player': 0,
    #                             'current_player_offset': 1,
    #                             'deck_size': 40,
    #                             'discard_pile': [],
    #                             'fireworks': {'B': 0,
    #                                       'G': 0,
    #                                       'R': 0,
    #                                       'W': 0,
    #                                       'Y': 0},
    #                             'information_tokens': 8,
    #                             'legal_moves': [],
    #                             'life_tokens': 3,
    #                             'observed_hands': [[{'color': None, 'rank': -1},
    #                                             {'color': None, 'rank': -1},
    #                                             {'color': None, 'rank': -1},
    #                                             {'color': None, 'rank': -1},
    #                                             {'color': None, 'rank': -1}],
    #                                            [{'color': 'W', 'rank': 2},
    #                                             {'color': 'Y', 'rank': 4},
    #                                             {'color': 'Y', 'rank': 2},
    #                                             {'color': 'G', 'rank': 0},
    #                                             {'color': 'W', 'rank': 1}]],
    #                             'num_players': 2,
    #                             'vectorized': [ 0, 0, 1, ... ]}]}
    #       reward: float, Reward obtained from taking the action.
    #       done: bool, Whether the game is done.
    #       info: dict, Optional debugging information.

    #     Raises:
    #       AssertionError: When an illegal action is provided.
    #     """
    #     action = int(action[0])
    #     if isinstance(action, dict):
    #         # Convert dict action HanabiMove
    #         action = self._build_move(action)
    #     elif isinstance(action, int):
    #         if action == -1:  # invalid action
    #             obs = np.zeros(self.vectorized_observation_shape()[0]+self.players)
    #             share_obs = np.zeros(self.vectorized_share_observation_shape()[0]+self.players)
    #             rewards = np.zeros((self.players, 1))
    #             done = None
    #             infos = {'score': self.state.score()}
    #             available_actions = np.zeros(self.num_moves())
    #             return obs, share_obs, rewards, done, infos, available_actions
    #         # Convert int action into a Hanabi move.
    #         action = self.game.get_move(action)
    #     else:
    #         raise ValueError("Expected action as dict or int, got: {}".format(action))

    #     last_score = self.state.score()
    #     # Apply the action to the state.
    #     self.state.apply_move(action)

    #     while self.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
    #         self.state.deal_random_card()

    #     observation = self._make_observation_all_players()
    #     current_player = self.state.cur_player()
    #     player_observations = observation['player_observations']

    #     available_actions = np.zeros(self.num_moves())
    #     available_actions[player_observations[current_player]['legal_moves_as_int']] = 1.0

    #     agent_turn = np.zeros(self.players, dtype=np.int).tolist()
    #     agent_turn[current_player] = 1

    #     obs = player_observations[current_player]['vectorized'] + agent_turn
    #     if self.obs_instead_of_state:
    #         share_obs = [player_observations[i]['vectorized'] for i in range(self.players)]
    #         concat_obs = np.concatenate(share_obs, axis=0)
    #         share_obs = np.concatenate((concat_obs, agent_turn), axis=0)
    #     else:
    #         share_obs = player_observations[current_player]['vectorized_ownhand'] + player_observations[current_player]['vectorized'] + agent_turn

    #     done = self.state.is_terminal()
    #     # Reward is score differential. May be large and negative at game end.
    #     reward = self.state.score() - last_score
    #     rewards = [[reward]] * self.players
    #     infos = {'score': self.state.score()}

    #     return obs, share_obs, rewards, done, infos, available_actions

    def step(self, actions):
        #     """Take one step in the game.

        #     Args:
        #       action: dict, mapping to a legal action taken by an agent. The following
        #         actions are supported:
        #           - { 'action_type': 'PLAY', 'card_index': int }
        #           - { 'action_type': 'DISCARD', 'card_index': int }
        #           - {
        #               'action_type': 'REVEAL_COLOR',
        #               'color': str,
        #               'target_offset': int >=0
        #             }
        #           - {
        #               'action_type': 'REVEAL_RANK',
        #               'rank': str,
        #               'target_offset': int >=0
        #             }
        #         Alternatively, action may be an int in range [0, num_moves()).
        status = 1
        reward = 2
        done = False
        info = "Donno what"

        return status, reward, done, info

    def reset(self, num_gnb, num_uav, num_ue):
        return status
