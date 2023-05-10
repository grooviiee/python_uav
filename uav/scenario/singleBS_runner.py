from env.uavenv import UAV_ENV


class SingleBS_runner(object):
    def __init__(self, config):
        NotImplemented

    def make_world(self, arglist):
        ue = UAV_ENV()
        env = UAV_ENV.make_env(ue, arglist)
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
