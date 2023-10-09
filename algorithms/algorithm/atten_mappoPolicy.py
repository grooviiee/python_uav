import torch
import numpy as np
from algorithms.algorithm.r_actor import R_Actor
from algorithms.algorithm.r_attention_critic import R_Critic
# from algorithms.algorithm.r_critic import R_Critic

class AttentionMappoAgent_Policy:
    def __init__(self, args, obs_space, cent_obs_space, act_space, agent_id, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr        
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space      # Individual Obs space
        self.share_obs_space = cent_obs_space   # Shared Obs space
        self.act_space = act_space      # Action space
        
        if args.num_mbs > agent_id:
            self.is_uav = False
        else:
            self.is_uav = True

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.is_uav, self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)        
        self.critic = R_Critic(args, self.share_obs_space, self.is_uav, self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        
    def get_actions(self, is_uav, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None, deterministic=False):
        
        if is_uav == False:
            reshape_obs = np.reshape(obs, (2,5,-1)) # (2, 5, 5)
        else:
            reshape_obs = np.reshape(obs, (1,2,-1)) # (2, 2, 17)
        
        
        cent_obs = reshape_obs

        # actions: distribution set, action_log_probs: sample probabilities
        actions, action_log_probs, rnn_states_actor = self.actor(reshape_obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)
  
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
        
    def get_values(self, is_uav, cent_obs, rnn_states_critic, masks):
        if is_uav == False:
            reshaped_cent_obs = np.reshape(cent_obs, (2,5,-1)) # (2, 5, 5)
        else:
            reshaped_cent_obs = np.reshape(cent_obs, (1,2,-1)) # (2, 2, 17)
        
        values, _ = self.critic(reshaped_cent_obs, rnn_states_critic, masks)
        return values