from envs.uavenv import UAV_ENV
from utils.shared_buffer import SharedReplayBuffer
from utils.separated_buffer import SeparatedReplayBuffer
from runner.base_runner import Runner
    
import time
# import wandb
import os
import numpy as np
from itertools import chain
import torch
# config {
#   "args": arglist,
#   "envs": envs,
#   "device": device,
# }
class SingleBS_runner(Runner):
    def __init__(self, config):
        super(SingleBS_runner, self).__init__(config)

        print("Choose SingleBS_runner")
        self.done = False
        self.total_reward = 0
        self.all_args = config['args']
        self.envs = config['envs']
        self.device = config['device']
        self.num_uavs = config['num_uavs']
        self.num_mbs = config['num_mbs']
        self.num_agents = self.num_uavs + self.num_mbs
        self.trainer = []
        self.buffer = []
        
        # parameters
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length  # step얼마 뒤에 train을 할지
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        
        print(f'[INIT_RUNNER] Insert Agent settings into Trainer')
        from algorithms.mappo import MAPPOAgentTrainer as TrainAlgo
        from algorithms.algorithm.mappoPolicy import MAPPOAgentPolicy as Policy
        
        print(f'[INIT_RUNNER] Make Actor Critic Policy for Agents')
        self.policy = []
        for agent_id in range(self.num_agents):
            # share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            share_observation_space = self.envs.observation_space[agent_id]

            # policy network
            policy = Policy(self.all_args,
                        self.envs.observation_space[agent_id],
                        share_observation_space,
                        self.envs.action_space[agent_id],
                        device = self.device)
            self.policy.append(policy)
        
        print(f'[INIT_RUNNER] Set Policy into Replay buffer and Trainer')
        # algorithm
        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device = self.device)
            # buffer
            # share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            share_observation_space = self.envs.observation_space[agent_id]
            bu = SeparatedReplayBuffer(self.all_args,
                                       self.envs.observation_space[agent_id],
                                       share_observation_space,
                                       self.envs.action_space[agent_id])
            self.buffer.append(bu)
            self.trainer.append(tr)
        
        print(f'[INIT_RUNNER] Insert Agent settings into Trainer Finished')

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
                
                # insert data into replay buffer
                self.insert(data)
            
            # compute return and update network
            self.compute()
            train_info = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log render information 

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        # insert obs to buffer
        for agent_id in range(self.num_uavs + self.num_mbs):
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    def reset(self):
        """Reset sth here"""
        
        
    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_cirtic = data
        
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        
        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)
        
        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(share_obs,
                                        np.array(list(obs[:, agent_id])),
                                        rnn_states[:, agent_id],
                                        rnn_states_critic[:, agent_id],
                                        actions[:, agent_id],
                                        action_log_probs[:, agent_id],
                                        values[:, agent_id],
                                        rewards[:, agent_id],
                                        masks[:, agent_id])
            
    def compute():
        NotImplemented