from envs.uavenv import UAV_ENV
from utils.shared_buffer import SharedReplayBuffer

import time
# config {
#   "args": arglist,
#   "envs": envs,
#   "device": device,
# }
class SingleBS_runner(object):
    def __init__(self, config):
        print("Choose SingleBS_runner")
        self.done = False
        self.total_reward = 0
        self.envs = config['envs']
        
        # buffer we will implement this further
        # self.buffer = SharedReplayBuffer(self.all_args,
        #                                 self.num_agents,
        #                                 self.envs.observation_space[0],
        #                                 share_observation_space,
        #                                 self.envs.action_space[0])
    def run(self):
        # basic procedure
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                    
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)


        # insert obs to buffer
        self.buffer.share_obs[0] = obs.copy()
        self.buffer.obs[0] = obs.copy()

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    def reset(self):
        """Reset sth here"""