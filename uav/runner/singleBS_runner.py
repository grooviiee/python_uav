


class singleBS_runner(object):
    def __init__(self, config):
        NotImplemented
        
    def run(self):
        # basic procedure
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        done = False
        total_reward = 0

        while not done:
            action = policy(obs)

            obs, reward, done, info = env.step(action)
            total_reward += reward
            env.render()