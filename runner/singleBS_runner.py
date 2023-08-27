from envs.uavenv import UAV_ENV
from envs.util import CovertToStateList
from utils.shared_buffer import SharedReplayBuffer
from utils.separated_buffer import SeparatedReplayBuffer
from runner.base_runner import Runner
from gym.spaces.utils import flatdim, flatten


import time
# import wandb
import os
import numpy as np
from itertools import chain
import torch
# config {
#   "args": arglist,
#   "envs": envs,
#   "device": device,
# }
def _t2n(x):
    return x.detach().cpu().numpy()

class SingleBS_runner(Runner):
    def __init__(self, config):
        super(SingleBS_runner, self).__init__(config)

        print("Choose SingleBS_runner")
        self.done = False
        self.total_reward = 0
        self.logger = config['logger']
        self.all_args = config['args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.algorithm = config['algorithm']
        self.num_uavs = config['num_uavs']
        self.num_mbs = config['num_mbs']
        self.num_agents = self.num_uavs + self.num_mbs
        self.num_users = config['num_users']
        self.trainer = []
        self.buffer = []
        
        # parameters
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length  # step얼마 뒤에 train을 할지
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.logger.info("[INIT_RUNNER] Insert Agent settings into Trainer")
        
        print(f'[INIT_RUNNER] Insert Agent settings into Trainer')
        if self.algorithm == "mappo":
            from algorithms.mappo import MAPPOAgentTrainer as TrainAlgo
            from algorithms.algorithm.mappoPolicy import MAPPOAgentPolicy as Policy
        elif self.algorithm == "attention_mappo":
            raise NotImplementedError
        elif self.algorithm == "random":
            raise NotImplementedError
        else:
            raise NotImplemented        
        
        print(f'[INIT_RUNNER] Make Actor Critic Policy for Agents')
        self.policy = []
        for agent_id in range(self.num_agents):
            # share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            share_observation_space = self.envs.observation_space[agent_id]
            print(f'[INIT_RUNNER] agent_id: {agent_id}, action_space: {self.envs.action_space[agent_id]}')
            # policy network
            policy = Policy(self.all_args,
                        self.envs.observation_space[agent_id],
                        share_observation_space,
                        self.envs.action_space[agent_id],
                        agent_id,
                        device = self.device)
            self.policy.append(policy)
        
        print(f'[INIT_RUNNER] Set Policy into Replay buffer and Trainer')
        # algorithm
        self.trainer = []
        self.buffer = []

        for agent_id in range(self.num_agents):
            if agent_id < self.num_mbs:
                is_uav = False
            else:
                is_uav = True
            
            tr = TrainAlgo(self.all_args, self.policy[agent_id], is_uav, device = self.device)
            # buffer
            # share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            share_observation_space = self.envs.observation_space[agent_id]
            bu = SeparatedReplayBuffer(self.all_args,
                                       self.envs.observation_space[agent_id],
                                       share_observation_space,
                                       self.envs.action_space[agent_id],
                                       is_uav)
            self.buffer.append(bu)
            self.trainer.append(tr)
        
        # For Debugging
        for agent_id in range(self.num_agents):
            print(f'agend_id {agent_id} | {self.envs.observation_space[agent_id]} | self.buffer[{agent_id}].obs.shape {self.buffer[agent_id].obs.shape}')

        NotImplementedError
        print(f'[RUNNER] Insert Agent settings into Trainer Finished')

    def run(self):
        print(f'[RUNNER] Warm up')
        # basic procedure
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        episodes = 2
        
        print(f'[RUNNER] Run Episode')
        for episode in range(episodes):
            for step in range(self.episode_length):
                self.logger.info("[RUNNER] Step: %d", step)
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.runner_collect(step)
                
                # raise NotImplementedError("Breakpoint")
                
                # Obser reward and next obs
                obs, rewards, origin_rewards, dones, infos = self.envs.step(actions_env)
                self.logger.info("[RUNNER] Get rewards: %f", rewards)
                
                # insert data into replay buffer
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic 
                self.runner_insert(data)
                self.sum_rewards(origin_rewards)
                #raise NotImplementedError("Breakpoint")
            
            # compute GAE and update network
            print(f'[RUNNER] Compute GAE')
            self.compute_gae() 

            print(f'[RUNNER] TRAIN')
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            print(f'[RUNNER] total_num_steps: {total_num_steps}')
                        
            # save trained model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information 
            if episode % self.eval_interval == 0 and self.use_eval:
                for agent_id in range(self.num_agents):
                    individual_rewards = []
                    for into in infos:
                        for count, info in enumerate(infos):
                            if 'individual_reward' in infos[count][agent_id].keys():
                                individual_rewards.append(infos[count][agent_id].get('individual_reward', 0))

                    train_infos[agent_id].update({'individual_rewards': np.mean(individual_rewards)})
                    train_infos[agent_id].update({"average_episode_rewards": np.mean(self.buffer[agent_id].rewards) * self.episode_length})
                self.log_train(train_infos, total_num_steps)

            # print(f'[RUNNER] EVAL')
            # self.eval(total_num_steps)

                
            # eval
            # if episode % self.eval_interval == 0 and self.use_eval:
            #     print(f'[RUNNER] EVAL')
            #     self.eval(total_num_steps)

    def warmup(self):
        NotImplemented
        #TODO 
        # reset env
        # obs = self.envs.reset()
        # share_obs = []
        # for o in obs:
        #     share_obs.append(list(chain(*o)))
        # share_obs = np.array(share_obs)

        # insert obs to buffer
        # for agent_id in range(self.num_agents):
        #     self.buffer[agent_id].share_obs[0] = share_obs.copy()
        #     self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def runner_collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        # For Debugging
        if self.all_args.log_level >= 3:
            for agent_id in range(self.num_agents):
                print(f'[RUNNER_DEBUG] agent_id : {agent_id}, share_obs.shape: {self.buffer[agent_id].share_obs[step].shape}, obs.shape: {self.buffer[agent_id].obs[step].shape}')
                NotImplementedError

        for agent_id in range(self.num_agents):
            if agent_id < self.num_mbs:
                is_uav = False
            else:
                is_uav = True
                
            self.trainer[agent_id].prep_rollout()
            print(f'[RUNNER] agent_id : {agent_id}, share_obs.shape: {self.buffer[agent_id].share_obs[step].shape}, obs.shape: {self.buffer[agent_id].obs[step].shape}')
            value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer[agent_id].policy.get_actions(is_uav,
                                                                                    self.buffer[agent_id].share_obs[step],
                                                                                    self.buffer[agent_id].obs[step],
                                                                                    self.buffer[agent_id].rnn_states[step],
                                                                                    self.buffer[agent_id].rnn_states_critic[step],
                                                                                    self.buffer[agent_id].masks[step])
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)
            
            # re-arrange action
            print(f'[RUNNER] agent_id : {agent_id}, action space: {self.envs.action_space[agent_id]}')
            if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
            elif self.envs.action_space[agent_id].__class__.__name__ == 'Box':
                # MBS Case -> [RUNNER] agent_id : 0, action space: Box(False, True, (100,), bool)
                print(f'[RUNNER] BOX dType action.shape:  {action.shape}')
                action_env = action
                
            elif self.envs.action_space[agent_id].__class__.__name__ == 'Tuple':
                # [RUNNER] agent_id : 4, action space dType: Tuple value: Tuple(Box(False, True, (2, 10), bool), Box(0.0, 23.0, (2,), float32), Box(0.0, 5.0, (2,), float32), Box(0.0, 3.0, (2,), float32))
                print(f'[RUNNER] Tuple dType action.shape:  {action.shape}')
                action_env = self.envs.action_space[agent_id]
                action_env = action
                # for i in range 
            else:
                raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append( _t2n(rnn_state_critic))
            # MBS: action_env [ True]
            # UAV: action_env Tuple(Box(False, True, (2, 30), bool), Box(0.0, 23.0, (2,), float32), Box(0.0, 5.0, (2,), float32), Box(0.0, 3.0, (2,), float32))
            print(f'[RUNNER] agent_id: {agent_id} Done.. action_env.shape: {action_env.shape} / len: {len(temp_actions_env)}, n_rollout_threads: {self.n_rollout_threads}')

        # [envs, agents, dim] -> action dimension depends on num threads
        action_env_results = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env)
            action_env_results.append(one_hot_action_env)

        # values = np.array(values).transpose()
        # actions = np.transpose(actions, (1, 0, 2))
        # action_log_probs = np.array(action_log_probs).transpose()
        # rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        # rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)
        for return_action_info in action_env_results:
            NotImplemented
            # print(f'[RUNNER_COLLECT] Spit actionInfo As {return_action_info} /len: {len(action_env_results)}')


        # action_env_results will be insert into "Env".
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, action_env_results

    def reset(self):
        """Reset sth here"""
        
    def sum_rewards(self, rewards):
        total_reward = np.sum(rewards)
        print(f'[RUNNER_REWARD] individual_reward: {rewards}, tatal_reward: {total_reward}')
        
    """To get type of sturct: type(variable) or struct.__class__"""    
    def runner_insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        if self.all_args.log_level >= 3:
            print(f'[RUNNER_INSERT] (TYPE) obs.type: {obs}, reward: {type(rewards)}, dones: {type(dones)}, infos: {type(infos)}, values: {type(values)}')
            print(f'[RUNNER_INSERT] (TYPE) actions: {actions}, action_log_probs: {type(action_log_probs)}, rnn_states: {rnn_states}, rnn_states_critic: {type(rnn_states_critic)}')
        # Dones가 True인 index에 대해서는 모두 0으로 설정하나 보다. -> 이건 나중에 고려하기로.
    
        npDones = np.array(dones)
        # rnn_states[npDones == True] = np.zeros(((npDones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        # rnn_states_critic[npDones == True] = np.zeros(((npDones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones(self.num_agents, dtype=np.float32)
        # masks[npDones == True] = np.zeros(((npDones == True).sum(), 1), dtype=np.float32)
        
        share_obs = []
        for idx, o in enumerate(obs):
            # share_obs.append(list(chain(*o))) 
            #TODO: Need to Have deep copy using "func CovertToStateList"
            #state_list = CovertToStateList(obs[idx])
            share_obs.append(obs[idx])
            print(f'[RUNNER_INSERT] MAKE_SHARE_OBS: idx: {idx}, len(obs[idx]): {len(obs[idx])}, len(share_obs): {len(share_obs)}')

        # Convert array type share_obs into np.array
        share_obs = np.array(share_obs, dtype=object)
        print(f'[RUNNER_INSERT] SHARE_OBS len(share_obs): {len(share_obs)}')

        for agent_id in range(self.num_agents):
            if agent_id < self.num_mbs:
                is_uav = "MBS"
            else:
                is_uav = "UAV"

            # We use centralized V as a default
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            if self.all_args.log_level >= 3:
                print(f'[RUNNER_BUFFER_INSERT] agent_id: {agent_id} which is {is_uav}, Refined_SHARE_OBS.shape: {len(share_obs)}')
                print(f'[RUNNER_BUFFER_INSERT] {len(share_obs)} {obs[agent_id]} {len(rnn_states)} {len(rnn_states_critic)} {len(actions)} {len(action_log_probs)} {len(values)} {len(rewards)} {len(masks)}')

            # Save share_obs and other agent resource into replay buffer
            self.buffer[agent_id].buffer_insert(share_obs,
                                        list(chain(*obs[agent_id])),
                                        rnn_states[agent_id],
                                        rnn_states_critic[agent_id],
                                        actions[agent_id],
                                        action_log_probs[agent_id],
                                        values[agent_id],
                                        rewards[agent_id],
                                        masks[agent_id])
            
    @torch.no_grad()            
    def compute_gae(self):
        for agent_id in range(self.num_agents):
            if agent_id < self.num_mbs:
                is_uav = False
            else:
                is_uav = True
            
            print(f"[RUNNER_BUFFER_INSERT] agent_id:{agent_id}\nshare_obs:{self.buffer[agent_id].share_obs[-1]}\nrnn_states_critic:{self.buffer[agent_id].rnn_states_critic[-1]}, masks: {self.buffer[agent_id].masks[-1]}")

            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(is_uav, self.buffer[agent_id].share_obs[-1], 
                                                                self.buffer[agent_id].rnn_states_critic[-1],
                                                                self.buffer[agent_id].masks[-1])
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)
            
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                                eval_rnn_states[:, agent_id],
                                                                                eval_masks[:, agent_id],
                                                                                deterministic=True)

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i]+1)[eval_action[:, i]]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                    eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
                
            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        
        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards})
            print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)  