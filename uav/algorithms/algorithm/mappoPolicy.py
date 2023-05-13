


class MAPPOAgentPolicy:
    def __init__(self, args, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        
        self.actor = R_Actor()
        self.cirtic = R_Critic()
        
        self.actor_optimizer = torch.optim.Adam()
        self.critic_optimizer = torch.optim.Adam()
        
    def lr_decay(self, episode, num_episodes):
        update_linear_schedule(self.actor_optimizer, episode, num_episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, num_episodes, self.critic_lr)
        
    def get_actions:
        NotImplemented
        
    def get_values:
        NotImplemented
        
    def evaluate_actions:
        NotImplemented
    
    def act:
        NotImplemented