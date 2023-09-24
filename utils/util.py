import numpy as np
import math
import torch

def get_shape_from_obs_space(obs_space):
    #print(f'obs_space dType: {obs_space.__class__.__name__}')
    obs_shape = []
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    elif obs_space.__class__.__name__ == 'Discrete':
        obs_shape = 1
    elif obs_space.__class__.__name__ == 'Tuple':
        for obs_info in obs_space:
            obs_shape.append(obs_info.shape)
    else:
        raise NotImplementedError

    return obs_shape

def get_shape_from_act_space(act_space, args, is_uav):
    #print(f'act_space dType: {act_space.__class__.__name__}')
    act_shape = 0
    # if act_space.__class__.__name__ == 'Discrete':
    #     act_shape = 1
    # elif act_space.__class__.__name__ == "MultiDiscrete":
    #     act_shape = act_space.shape
    # elif act_space.__class__.__name__ == "Box":
    #     act_shape = act_space.shape[0]
    # elif act_space.__class__.__name__ == "MultiBinary":
    #     act_shape = act_space.shape[0]
    # elif act_space.__class__.__name__ == 'Tuple':
    #     for act_info in act_space:
    #         act_shape = act_shape + act_info.shape[0]
    # else:  # agar
    #     act_shape = act_space[0].shape[0] + 1  
    if is_uav == False:
        act_shape = (args.num_uavs + args.num_mbs) * args.num_users
    else:
        act_shape = 4   # tuple: {cache, power, velocity, angle}
        
    print(f"[REPLAYBUFFER_INIT] (get_shape_from_act_space) is_uav: {is_uav}, act_shape: {act_shape}")
    return act_shape

def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)