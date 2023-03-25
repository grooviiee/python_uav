# Multi-agent DQN (reference: https://github.com/ChristoffelDoorman/Multi-Agent-Reinforcement-Learning/tree/main/DQN)


import gym
from DQN.MADQN import MADQN

MODEL = "DQN"
ENV_NAME = "ma_gym:Switch4-v0"
N_EPISODES = 3_000
LOG_PERIOD = 200
RENDER = True
TEST_EPISODES = 10

if __name__ == "__main__":
    env = gym.make(ENV_NAME)

    if MODEL == "DQN":
        MaModel = MADQN(env)
    else:
        MaModel = "NONE"  # MAPPO(env)

    successful_run = False

    while not successful_run:
        # train agents
        train_rewards, successful_train_agents, successful_run = MaModel.train_agents(
            n_episodes=N_EPISODES, log_period=LOG_PERIOD, render=RENDER
        )

