import torch
import torch.nn as nn
from algorithms.utils.util import init, check
from algorithms.utils.cnn import CNNBase
from algorithms.utils.cnn import Attention_CNNBase
from algorithms.utils.mlp import MLPBase
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act import ACTLayer
from algorithms.algorithm.r_attention import MultiHeadAttention
from utils.util import get_shape_from_obs_space


class R_Attention_Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, is_uav, device=torch.device("cpu")):
        super(R_Attention_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)  # device type
        self.is_uav = is_uav
        self.cache_capa = args.cache_capa
        obs_shape = get_shape_from_obs_space(obs_space)

        # Choose base network
        if len(obs_shape) == 3:
            print(
                f"[ACTOR] returned obs_shape: {obs_shape}. CNN Base because length is 3."
            )

            # flatten(obs_shape)
            temp_list = list(chain(*obs_shape))
            print(
                f"[ACTOR] reshaped obs_shape: {temp_list} which length is {len(temp_list)}."
            )

            cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
            if len(cent_obs_shape) == 3:
                self.base = Attention_CNNBase(args, cent_obs_shape, self.is_uav)
            else:
                print(f"(We do not use this currently) [ACTOR] returned obs_shape: {obs_shape}. MLP Base because length is not 3")
                self.base = MLPBase(args, obs_shape, is_uav, False)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            print(f"self.rnn = RNNLayer")
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )

        self.act = ACTLayer(
            action_space, self.hidden_size, self._use_orthogonal, self._gain
        )

        self.to(device)

class R_Attention_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).

    :variable base: determine NN layer according to cent_obs_shape
    """

    def __init__(self, args, cent_obs_space, is_uav, device=torch.device("cpu")):
        super(R_Attention_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.is_uav = is_uav
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][
            self._use_orthogonal
        ]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        if len(cent_obs_shape) == 3:
            self.base = Attention_CNNBase(args, cent_obs_shape, self.is_uav)
        else:
            raise NotImplementedError
            self.base = MLPBase(args, cent_obs_shape, self.is_uav)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        print(
            f"[CRITIC_FORWARD] cent_obs.shape: {cent_obs.shape}, _use_naive_recurrent_policy:{self._use_naive_recurrent_policy}, _use_recurrent_policy:{self._use_recurrent_policy}"
        )

        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        print(f"[critic_features]")

        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        
        # if type(critic_features) == tuple:
        #     print(f"[critic_features] critic_features({type(critic_features[0])}), {len(critic_features[0])}")

        #     critic_features = critic_features[0].view(-1)
        values = self.v_out(critic_features[0][0])

        return values, rnn_states
