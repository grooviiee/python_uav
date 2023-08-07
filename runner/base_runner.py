# common functions will be moved here
import torch
import wandb
import os
import logging

from tensorboardX import SummaryWriter
from itertools import chain

class Runner(object):
    """ Base class for training recurrent policies. """
    
    def __init__(self, config):
        self.all_args = config['args']
        self.use_centralized_V = self.all_args.use_centralized_V
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.eval_interval = self.all_args.eval_interval

        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        
        if self.use_render:
            import imageio
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / 'gifs')
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
    
    def train(self):
        NotImplemented
    
    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            next_value = self.trainer[agent_id].policy.get_value()
            
    # this code follows runner/seperated/base_runner.py
    def train(self):
        train_info_list = []
        for agent_id in range(self.num_agents):
            if agent_id < self.num_mbs:
                is_uav = False
            else:
                is_uav = True
                
            self.trainer[agent_id].prep_training()
            train_info = self.trainer[agent_id].train(is_uav, self.buffer[agent_id])
            train_info_list.append(train_info)
            self.buffer[agent_id].after_update()
            
        return train_info_list
    
    
    def save(self):
        for agent_id in range(self.num_agents):
            policy_actor = self.trainer[agent_id].policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
            policy_critic = self.trainer[agent_id].policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")
            if self.trainer[agent_id]._use_valuenorm:
                policy_vnrom = self.trainer[agent_id].value_normalizer
                torch.save(policy_vnrom.state_dict(), str(self.save_dir) + "/vnrom_agent" + str(agent_id) + ".pt")