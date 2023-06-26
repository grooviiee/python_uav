# common functions will be moved here
import torch

class Runner(object):
    """ Base class for training recurrent policies. """
    
    def __init__(self, config):
        self.all_args = config['args']
        self.use_centralized_V = self.all_args.use_centralized_V
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.recurrent_N = self.all_args.recurrent_N

    
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