# common functions will be moved here
import torch

class Runner(object):
    """ Base class for training recurrent policies. """
    
    def __init__(self, config):
        self.all_args = config['all_args']
    
    def train(self):
        NotImplemented
    
    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            next_value = self.trainer[agent_id].policy.get_value()
            
    def train(self):
        train_info_list = []
        for agent_id in range(self.num_uavs + self.num_mbs):
            self.trainer[agent_id].prep_training()
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])
            train_info_list.append(train_info)
            self.buffer[agent_id].after_update()
            
        return train_info_list