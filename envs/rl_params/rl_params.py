"""
params: is_uav, input_channel, input_width, input_height, num_files
"""


def CNN_Conv(is_uav, num_uavs, num_users, num_files):
    input_channel = 0
    input_width = 0
    input_height = 0

    if is_uav == True:
        input_channel = 2  # my location
        input_height = 2 * num_users
        input_width = num_users * num_files

        return input_channel, input_width, input_height

    else:
        # obs: 2 + 2*num_uavs + 2*num_users
        input_channel = 2  # my location
        input_height = 2 * num_uavs
        input_width = 2 * num_users

        return input_channel, input_width, input_height
