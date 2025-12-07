import torch.nn as nn
import torch.nn.functional as F
from tensordict.nn import TensorDictModule as Mod
from tensordict.nn import TensorDictSequential as Seq
from torchrl.modules import LSTMModule, ValueOperator

from jaxtrl.model.th import LambdaModule, create_mlp_layers


def mlp_critic_factory(
    obs_space, action_space, net_arch=[64, 64, 64], activation_func="tanh", device="cpu"
):
    layers = create_mlp_layers(obs_space.shape[0], net_arch, activation_func, True)
    layers.append(nn.Linear(net_arch[-1], 1))

    critic_net = nn.Sequential(*layers).to(device)

    return ValueOperator(module=critic_net, in_keys=["observation"])


def lstm_critic_factory(
    obs_space,
    action_space,
    lstm_hidden_size=64,
    head_net_arch=[64, 64, 64],
    head_activation_func="tanh",
    input_key="observation",
    input_size_override=None,
    lstm_key_id="critic",
    out_key="state_value",
    bounded=False,
    bounded_range=(0, 1),
    device="cpu",
):

    create_is_init = Mod(
        lambda step_count: step_count == 0,
        in_keys=["step_count"],
        out_keys=["is_init"],
    )

    lstm = LSTMModule(
        input_size=(
            obs_space.shape[0] if input_size_override is None else input_size_override
        ),
        hidden_size=lstm_hidden_size,
        in_keys=[input_key, f"{lstm_key_id}_rnn_h", f"{lstm_key_id}_rnn_c", "is_init"],
        out_keys=[
            f"{lstm_key_id}_lstm_embed",
            ("next", f"{lstm_key_id}_rnn_h"),
            ("next", f"{lstm_key_id}_rnn_c"),
        ],
        default_recurrent_mode=True,
    )

    layers = create_mlp_layers(
        lstm_hidden_size,
        head_net_arch + [1],
        head_activation_func,
    )

    value_net = nn.Sequential(*layers)

    if bounded:
        value_net = nn.Sequential(
            value_net,
            LambdaModule(
                lambda x: x
                + F.relu(bounded_range[0] - x)
                - F.relu(x - bounded_range[1])
            ),
        )

    value_net_module = Mod(
        value_net, in_keys=[f"{lstm_key_id}_lstm_embed"], out_keys=[out_key]
    )

    critic_module = Seq(create_is_init, lstm, value_net_module)

    critic_module.to(device)

    return critic_module
