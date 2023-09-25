import torch.nn as nn
from .util import init

"""CNN Modules and utils."""


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNNLayer(nn.Module):
    def __init__(
        self,
        obs_shape,
        hidden_size,
        use_orthogonal,
        use_ReLU,
        is_uav,
        kernel_size=2,
        stride=2,
    ):
        super(CNNLayer, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(["tanh", "relu"][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        input_channel = obs_shape[0][0]
        input_width = obs_shape[1][0]
        input_height = obs_shape[2][0]

        # MBS: input_channel 2, input_width 8, input_height 40, UAV: input_channel 8, input_width 40, input_height 600
        # inputs: [N(Bacth), C(Channel), W(Width), H(Height)]
        if is_uav == True:
            input_channel = 1
            input_width = 2
            input_height = 31
            conv2d_out_size = 15
        else:
            input_channel = 2
            input_width = 5
            input_height = 5
            conv2d_out_size = 4


        print(f'[CNN_LAYER_INIT] is_uav: {is_uav}, input_channel {input_channel}, input_width {input_width}, input_height {input_height} hidden_size {hidden_size}')

        self.cnn = nn.Sequential(
                init_(
                    nn.Conv2d(      # 2D Convolution function
                        in_channels=input_channel,
                        out_channels=hidden_size // 2,
                        #out_channels=3,
                        kernel_size=kernel_size,    # it was set to 2
                        stride=stride,
                    )
                ),
                active_func,    # call nn.ReLU()
                Flatten(),
                init_(nn.Linear(conv2d_out_size, 64)),     # nn.Linear : Fully connected layer
                active_func,
                init_(nn.Linear(hidden_size, 64)),
                active_func,
        )

        # self.cnn = nn.Sequential(
        #     init_(
        #         nn.Conv2d(      # 2D Convolution function
        #             in_channels=input_channel,
        #             #out_channels=hidden_size // 2,
        #             out_channels=3,
        #             kernel_size=kernel_size,    # it was set to 2
        #             stride=stride,
        #         )
        #     ),
        #     active_func,    # call nn.ReLU()
        #     Flatten(),
        #     init_(nn.Linear(num_hidden_layer, 64)),     # nn.Linear : Fully connected layer
        #     active_func,
        #     init_(nn.Linear(hidden_size, 64)),
        #     active_func,
        # )

        print(
            f"[INIT_CNN_LAYER] Init CNNLayer: [{input_channel},{input_width},{input_height}],{self.cnn}"
        )

    def forward(self, x):
        print(f"[CNN FORWARD]: (forward) input x: {x.shape}")
        x = x / 255.0
        x = self.cnn(x)
        print(f"[CNN FORWARD]: (forward_after_self.cnn(x)) returned x: {x.shape}")
        return x


class CNNBase(nn.Module):
    def __init__(self, args, obs_shape, is_uav):
        print(f"..Init CNNBase")
        super(CNNBase, self).__init__()

        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self.hidden_size = args.hidden_size
        self.is_uav = is_uav
        self.cnn = CNNLayer(
            obs_shape,
            self.hidden_size,
            self._use_orthogonal,
            self._use_ReLU,
            self.is_uav,
        )

    def forward(self, x):
        x = self.cnn(x)
        return x
