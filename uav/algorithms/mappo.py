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

	def train(self, buffer, update_actor=True):
        
        advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advangates = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copu)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        
        
        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        
        for _ in range(self.ppo_epoch):
            for sample in data_generator:
                
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                = self.ppo_update(sample, update_actor)
        	        
        		train_info['value_loss'] += value_loss.item()
        		train_info['policy_loss'] += policy_loss.item()
        		train_info['dist_entropy'] += dist_entropy.item()
        		train_info['actor_grad_norm'] += actor_grad_norm
        		train_info['critic_grad_norm'] += critic_grad_norm
        		train_info['ratio'] += imp_weights.mean()
        	
            num_updates = self.ppo_epoch * self.num_mini_batch
            
            for key in train_info.keys():
                train_info[key] /= num_updates
                
            return train_info
        
        
        
        
        