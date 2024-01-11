import torch
import numpy as np
from algorithms.algorithm.r_actor import R_Actor
from algorithms.algorithm.r_critic import R_Critic
from envs.rl_params.rl_params import CNN_Conv, Get_obs_shape, Adjust_list_size


class MAPPOAgentPolicy:
    def __init__(
        self,
        args,
        obs_space,
        cent_obs_space,
        act_space,
        agent_id,
        device=torch.device("cpu"),
    ):
        self.args = args
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space  # Individual Obs space
        self.share_obs_space = cent_obs_space  # Shared Obs space
        self.act_space = act_space  # Action space
        if args.num_mbs > agent_id:
            self.is_uav = False
        else:
            self.is_uav = True

        self.actor = R_Actor(
            args, self.obs_space, self.act_space, self.is_uav, self.device
        )

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

        self.critic = R_Critic(args, self.share_obs_space, self.is_uav, self.device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(self, episode, num_episodes):
        update_linear_schedule(self.actor_optimizer, episode, num_episodes, self.lr)
        update_linear_schedule(
            self.critic_optimizer, episode, num_episodes, self.critic_lr
        )

    def get_actions(
        self,
        is_uav,
        cent_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        masks,
        available_actions=None,
        deterministic=False,
    ):
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

        # CNN_Conv(is_uav, )
        print(f"obs_org_size: {len(obs[0])}")
        obs = Adjust_list_size(obs)
        channel, width, height = CNN_Conv(
            is_uav, self.args.num_uavs, self.args.num_users, self.args.num_contents
        )
        print(f"{is_uav}, {len(obs)}, {channel}, {width}, {height}")
        reshape_obs = np.reshape(obs, (channel, width, height))

        cent_obs = reshape_obs

        # actions: distribution set, action_log_probs: sample probabilities
        # self.actor = R_Actor(args, self.obs_space, self.act_space, self.is_uav, self.device)
        #     def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False)
        actions, action_log_probs, rnn_states_actor = self.actor(
            reshape_obs, rnn_states_actor, masks, available_actions, deterministic
        )

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, is_uav, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        # if is_uav == False:  # MBS
        #     reshaped_cent_obs = np.reshape(cent_obs, (2, 5, -1))  # (2, 5, 5)
        # else:  # UAV
        #     reshaped_cent_obs = np.reshape(cent_obs, (1, 2, -1))  # (2, 2, 17)
        cent_obs = Adjust_list_size(cent_obs)
        channel, width, height = CNN_Conv(
            is_uav, self.args.num_uavs, self.args.num_users, self.args.num_contents
        )
        print(f"{is_uav}, {len(cent_obs)}, {channel}, {width}, {height}")
        reshape_obs = np.reshape(cent_obs, (channel, width, height))

        values, _ = self.critic(reshape_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(
        self,
        is_uav,
        cent_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ):
        print(
            f"[POLICY] (evaluate_actions) is_uav({is_uav}), cent_obs.shape: {cent_obs.shape}, obs: {obs.shape}"
        )
        # if is_uav == False:
        #     cent_obs[1] = np.reshape(cent_obs, (2,5,-1)) # (2, 5, 5)
        #     obs[1] = np.reshape(obs, (2,5,-1)) # (2, 5, 5)
        # else:
        #     cent_obs[1] = np.reshape(cent_obs, (1,2,-1)) # (2, 2, 17)
        #     obs[1] = np.reshape(obs, (1,2,-1)) # (2, 2, 17)

        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs, rnn_states_actor, action, masks, available_actions, active_masks
        )
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(
        self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False
    ):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        return actions, rnn_states_actor
