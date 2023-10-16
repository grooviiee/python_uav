import torch
import numpy as np
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from algorithms.utils.util import check

class MAPPOAgentTrainer:
    def __init__(self,
                 args,
                 policy,
                 is_uav,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param    
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.huber_delta = args.huber_delta

        self.data_chunk_length = args.data_chunk_length
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks  #True
        self._use_policy_active_masks = args.use_policy_active_masks    #False

        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1).to(self.device)
        else:
            self.value_normalizer = None
            

    # Train is acheived per Agent
    def train(self, is_uav, buffer, update_actor=True):
        advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        
        # Create train_info
        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        
        # Start Train (episode: 51 -> batch_size: 51)
        print(f"[TRAIN] ppo_epoch: {self.ppo_epoch}")
        for _ in range(self.ppo_epoch): # epoch만큼 random sample을 뽑는다
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:      # <-- Now we use this generator
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)    
            
            print(f"[TRAIN] data_generator ({data_generator}) num_mini_batch ({self.num_mini_batch})")
            idx = 0
            for sample in data_generator: 
                
                print(f"[TRAIN] ppo_update. Sample index ({idx}) is_uav ({is_uav})")
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(is_uav, idx, sample, update_actor)
        	        
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()
                idx = idx + 1            
            
            num_updates = self.ppo_epoch * self.num_mini_batch
            
            for key in train_info.keys():
                train_info[key] /= num_updates
                
            return train_info
        
    def ppo_update(self, is_uav, sample_index, sample, update_actor=True):
        # Update Actor and Critic Network
        # input: random generated sampled data

        # Step 1. Parse input data
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch, batch_size = sample
        
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        print(f"[PPO_UPDATE] (before_reshape) share_obs_batch ({share_obs_batch.shape}), obs_batch ({obs_batch.shape})")
        if is_uav == False:
            #share_obs_batch = np.reshape(share_obs_batch, (batch_size,8,2,-1)) # (2, 5, 5)
            #obs_batch = np.reshape(obs_batch, (batch_size,8,2,-1)) # (2, 5, 5)
            share_obs_batch = np.reshape(share_obs_batch, (batch_size, 2, 5, -1)) # (2, 5, 5)
            obs_batch = np.reshape(obs_batch, (batch_size, 2, 5, -1)) # (2, 5, 5)
        else:
            share_obs_batch = np.reshape(share_obs_batch, (batch_size,1,2,-1)) # (2, 2, 17)
            obs_batch= np.reshape(obs_batch, (batch_size,1,2,-1)) # (2, 2, 17)

        print(f"[PPO_UPDATE] (after_reshape) share_obs_batch ({share_obs_batch.shape}), obs_batch ({obs_batch.shape}), actions_batch ({actions_batch.shape})")

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(is_uav, share_obs_batch[sample_index],
                                                                              obs_batch[sample_index], 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch[sample_index], 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # Step 2. Actor update
        print(f"[PPO_UPDATE] action_log_probs ({action_log_probs.shape}), old_action_log_probs_batch ({old_action_log_probs_batch.shape})")
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch[sample_index])

        surr1 = imp_weights * adv_targ[sample_index]
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ[sample_index]

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) *
                                    active_masks_batch[sample_index]).sum() / active_masks_batch[sample_index].sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        # Step 2-1. Backpropagation
        if update_actor == False:
            print(f"[PPO_UPDATE] (update_actor) policy_loss: {policy_loss}, dist_entropy: {dist_entropy}, self.entropy_coef: {self.entropy_coef}")
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # Step 3. Critic update
        value_loss = self.cal_value_loss(values, value_preds_batch[sample_index], return_batch[sample_index], active_masks_batch[sample_index])

        self.policy.critic_optimizer.zero_grad()

        # Step 3-1. Backpropagation
        print(f"[PPO_UPDATE] (update_critic) value_loss: {value_loss}, self.value_loss_coef: {self.value_loss_coef}")
        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights     
    
    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss  
        
    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
