import torch


class MAPPOAgentTrainer:
    def __init__(
        self, args, obs_space, cent_obs_space, act_space, device  # =torch.device("gpu")
    ):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

    def get_action(self, cent_bos, obs, rnn_state_actor, rnn_state_critic):
        """
        Compute actions and value function predictions for the given inputs.
        """
        actions, action_log_probs, rnn_state_actor = self.actor()

        value, rnn_state_critic = self.critic()

        return value, action, action_log_prob, rnn_state_actor, rnn_state_critic

    def act(self):
        return NotImplemented
