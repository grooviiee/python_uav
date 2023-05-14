import torch
from r_actor import R_Actor
from r_critic import R_Critic

class MAPPOAgentPolicy:
    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        
        self.obs_space = obs_space      # Individual obs space
        self.share_obs_space = cent_obs_space   # Merged obs space
        self.act_space = act_space
        
        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.cirtic = R_Critic(args, self.share_obs_space, self.device)
        
        self.actor_optimizer = torch.optim.Adam()
        self.critic_optimizer = torch.optim.Adam()
        
    def lr_decay(self, episode, num_episodes):
        update_linear_schedule(self.actor_optimizer, episode, num_episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, num_episodes, self.critic_lr)
        
    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)
  
        NotImplemented
        
    def get_values:
        NotImplemented
        
    def evaluate_actions:
        NotImplemented
    
    def act:
        NotImplemented