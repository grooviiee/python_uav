# common functions will be moved here
import torch

class Runner(object):
    """ Base class for training recurrent policies. """
    
    def __init__(self, config):
        self.all_args = config['args']
        self.use_centralized_V = self.all_args.use_centralized_V
    
    def train(self):
        NotImplemented
    
    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            next_value = self.trainer[agent_id].policy.get_value()