import numpy as np

"""
params: is_uav, input_channel, input_width, input_height, num_files
"""


def Adjust_list_size(list):
    list = list[0]
    current_size = len(list)
    remainder = current_size % 10
    list_from_array = list.tolist()
    if remainder != 0:
        # 부족한 부분을 0으로 채우기
        padding_size = 10 - remainder
        list_from_array.extend([0] * padding_size)

    return np.array(list_from_array)


def Get_obs_shape(is_uav, num_uavs, num_users, num_files):
    result = []
    if is_uav is True:
        # channel, width, height
        input_size = 2 + 2 * num_users + num_users
        height = input_size // 10 + 1
        result.append(2)
        result.append(5)
        result.append(height)
        return result

    else:
        # channel, width, height
        input_size = 2 + 2 * num_uavs + 2 * num_users
        height = input_size // 10 + 1
        result.append(2)
        result.append(5)
        result.append(height)
        return result


def CNN_Conv(is_uav, num_uavs, num_users, num_files):
    # num_uav can be 2,4,6,8, 10
    # num_user can be 10 20 30 40 50 100
    # num_files can be 10 20 30 40 50

    if is_uav is True:
        input_size = 2 + 2 * num_users + num_users

        input_channel = 2
        input_height = 5
        input_width = input_size // 10 + 1

        return input_channel, input_width, input_height

    else:
        input_size = 2 + 2 * num_uavs + 2 * num_users
        input_channel = 2  # my location
        input_height = 5
        input_width = input_size // 10 + 1

        return input_channel, input_width, input_height
