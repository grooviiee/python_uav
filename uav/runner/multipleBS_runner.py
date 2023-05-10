


class MultipleBS_runner(object):
    def __init__(self, config):
        NotImplemented
        
    def run(self):
        def __init__(self, arlist):
            print("run MultipleBS")
            self.done = False
            self.total_reward = 0

        while not done:
            self.warmup()   

            start = time.time()
            episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
            action = Policy(obs)

            obs, reward, done, info = env.step(action)
            total_reward += reward
            env.render()