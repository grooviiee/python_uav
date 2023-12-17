# Multiple BS environment will be used here.
# Choose training algorithms and setup RIC existing env.
# To control multiple BS and UAVs, central controller is needed.
from envs.uavenv import UAV_ENV
from envs.util import CovertToStateList
from utils.shared_buffer import SharedReplayBuffer
from utils.separated_buffer import SeparatedReplayBuffer
from runner.base_runner import Runner
from gym.spaces.utils import flatdim, flatten


class MultipleBS_runner(object):
    def __init__(self, config):
        NotImplemented

    def run(self):
        def __init__(self, arlist):
            print("run in MultipleBS environments.")
            print("Sadly, it is not implemented yet.")

            self.done = False
            self.total_reward = 0

        while not done:
            self.warmup()

            start = time.time()
            episodes = (
                int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
            )
            action = Policy(obs)

            obs, reward, done, info = env.step(action)
            total_reward += reward
            env.render()
