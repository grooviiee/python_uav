import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from envs.rl_params.rl_params import CNN_Conv, Get_obs_shape
from algorithms.utils.util import init


class Flatten(nn.Module):
    def forward(self, x):
        # mod_x = x.view(x.size(0), -1)
        mod_x = x.view(-1)
        print(f"size x: {len(x)}, x.size(0): {x.size(0)}, mod_x: {len(mod_x)}")
        return mod_x


class CNNLayer(nn.Module):
    def __init__(
        self,
        obs_shape,
        input_size,
        hidden_size,
        use_orthogonal,
        use_ReLU,
        is_uav,
        kernel_size=3,
        stride=1,
    ):
        super(CNNLayer, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(["tanh", "relu"][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        if len(obs_shape) != 3:
            raise NotImplementedError

        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]
        conv2d_out_size = 4

        # Print cnn configurations
        print(
            f"[CNN_LAYER_INIT] is_uav: {is_uav}, input_channel {input_channel}, input_width {input_width}, input_height {input_height} hidden_size {hidden_size} kernel_size: {kernel_size}, stride: {stride}"
        )

        self.cnn = nn.Sequential(
            init_(
                nn.Conv2d(
                    in_channels=input_channel,
                    # out_channels=hidden_size // 2,
                    out_channels=hidden_size // 2,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            ),
            active_func,
            Flatten(),
            init_(
                nn.Linear(
                    in_features=hidden_size
                    // 2
                    * (input_width - kernel_size + stride)
                    * (input_height - kernel_size + stride),
                    out_features=hidden_size,
                )
            ),
            active_func,
            init_(nn.Linear(hidden_size, hidden_size)),
            active_func,
        )

        print(
            f"[INIT_CNN_LAYER] Init CNNLayer: [{input_channel},{input_width},{input_height}],{self.cnn}"
        )

    def forward(self, prev_x):
        x_norm = prev_x / 255.0 + 1e-6
        x = self.cnn(x_norm)
        print(
            f"[CNN_FORWARD]: (before) input x({prev_x.shape}): {prev_x}, (after.cnn(x)) returned x({x.shape}): {x}"
        )
        return x


class CNNBase(nn.Module):
    def __init__(self, args, obs_shape, is_uav, attention_mode):
        print(f"..Init CNNBase")
        super(CNNBase, self).__init__()

        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self.hidden_size = args.hidden_size
        self.is_uav = is_uav

        cnn_input_size = CNN_Conv(
            is_uav, args.num_uavs, args.num_users, args.num_contents
        )
        obs_shape = Get_obs_shape(
            is_uav, args.num_uavs, args.num_users, args.num_contents
        )
        self.cnn = CNNLayer(
            obs_shape,
            cnn_input_size,
            self.hidden_size,
            self._use_orthogonal,
            self._use_ReLU,
            self.is_uav,
        )

    def forward(self, x):
        x = self.cnn(x)
        return x


""" 
Attention based CNN Architecture
"""


class Attention_CNNLayer(nn.Module):
    def __init__(
        self,
        obs_shape,
        hidden_size,
        use_orthogonal,
        use_ReLU,
        is_uav,
        attention_size=32,
        kernel_size=2,
        stride=2,
    ):
        super(Attention_CNNLayer, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(["tanh", "relu"][use_ReLU])

        self.attention_layer = MultiHeadAttention(attention_size)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        input_channel = obs_shape[0][0]
        input_width = obs_shape[1][0]
        input_height = obs_shape[2][0]

        # MBS: input_channel 2, input_width 8, input_height 40, UAV: input_channel 8, input_width 40, input_height 600
        # inputs: [N(Bacth), C(Channel), W(Width), H(Height)]
        if is_uav is True:
            input_channel = 1
            input_width = 2
            input_height = 31
            conv2d_out_size = 15
        else:
            input_channel = 2
            input_width = 5
            input_height = 5
            conv2d_out_size = 4

        print(
            f"[CNN_LAYER_INIT] is_uav: {is_uav}, input_channel {input_channel}, input_width {input_width}, input_height {input_height} hidden_size {hidden_size}"
        )

        # self.attention_cnn = nn.Sequential(
        #     init_(
        #         nn.Conv2d(  # 2D Convolution function
        #             in_channels=input_channel,
        #             out_channels=hidden_size // 2,
        #             # out_channels=3,
        #             kernel_size=kernel_size,  # it was set to 2
        #             stride=stride,
        #         )
        #     ),
        #     active_func,  # call nn.ReLU()
        #     Flatten(),
        #     init_(nn.Linear(conv2d_out_size, 64)),  # nn.Linear : Fully connected layer
        #     active_func,
        #     #self.attention_layer(h, h, h),
        #     self.attention_layer,
        #     active_func,
        #     init_(nn.Linear(hidden_size, 64)),
        #     active_func,
        # )
        self.attention_cnn = nn.Sequential(
            init_(
                nn.Conv2d(
                    in_channels=input_channel,
                    # out_channels=hidden_size // 2,
                    out_channels=hidden_size // 2,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            ),
            active_func,
            Flatten(),
            init_(
                nn.Linear(
                    in_features=hidden_size
                    // 2
                    * (input_width - kernel_size + stride)
                    * (input_height - kernel_size + stride),
                    out_features=hidden_size,
                )
            ),
            active_func,
            self.attention_layer,
            active_func,
            init_(nn.Linear(hidden_size, hidden_size)),
            active_func,
        )

        print(
            f"[INIT_CNN_LAYER] Init CNNLayer: [{input_channel},{input_width},{input_height}],{self.attention_cnn}"
        )

    def forward(self, x):
        print(f"[ATTEN_CNN_FORWARD]: (forward) input x: {x.shape}")
        x = x / 255.0
        x = F.relu(self.c1(x))
        x = self.attention_layer(x, x, x)
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        print(f"[ATTEN_CNN_FORWARD]: (forward_after_self.cnn(x)) returned x: {x.shape}")
        return x


class Attention_CNNBase(nn.Module):
    def __init__(self, args, obs_shape, is_uav):
        print(f"..Init CNNBase")
        super(Attention_CNNBase, self).__init__()

        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self.hidden_size = args.hidden_size
        self.is_uav = is_uav
        self.attention_size = 32
        self.attention_layer = MultiHeadAttention(self.attention_size)
        self.cnn = Attention_CNNLayer(
            obs_shape,
            self.hidden_size,
            self._use_orthogonal,
            self._use_ReLU,
            self.is_uav,
            attention_size=self.attention_size,
        )

    def forward(self, x):
        x = self.cnn(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.w_qs = nn.Conv2d(size, size, 1)
        self.w_ks = nn.Conv2d(size, size, 1)
        self.w_vs = nn.Conv2d(size, size, 1)

        self.attention = ScaledDotProductAttention()

    def forward(self, q, k, v):
        residual = q
        q = self.w_qs(q).permute(0, 2, 3, 1)
        k = self.w_ks(k).permute(0, 2, 3, 1)
        v = self.w_vs(v).permute(0, 2, 3, 1)

        attention = self.attention(q, k, v).permute(0, 3, 1, 2)

        out = attention + residual
        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        attn = torch.matmul(q, k.transpose(2, 3))
        output = torch.matmul(attn, v)

        return output
