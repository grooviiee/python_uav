import numpy as np
import math
import torch


# TODO. Study gym.Space... "https://www.gymlibrary.dev/api/spaces/#general-functions"
def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape