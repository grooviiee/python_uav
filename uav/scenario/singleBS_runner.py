from uav.env.uavenv import env


class singleBS_runner(object):
    def __init__(self, config):
        NotImplemented

    def make_world():
        env = env()
        return env

    def run(self, arglist):
        # basic procedure
        env = Scenario.make_env(arglist.scenario, arglist, arglist.benchmark)
        done = False
        total_reward = 0

        while not done:
            action = policy(obs)  # Get Action

            obs, reward, done, info = env.step(action)
            total_reward += reward
            env.render()
