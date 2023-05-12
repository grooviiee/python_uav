


class MAPPOAgentPolicy:
    def __init__(self, args, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr