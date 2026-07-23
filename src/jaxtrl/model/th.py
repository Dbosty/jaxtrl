from typing import Any, Callable

import numpy as np
import torch as th
from torch import nn
from torch.distributions import Categorical, Distribution, Normal
from torchrl.modules import TanhNormal


def create_mlp_layers(
    input_size,
    net_arch=[64, 64, 64],
    activation_func="tanh",
    last_layer_activation_func=False,
):
    if activation_func == "tanh":
        activation_func = nn.Tanh
    elif activation_func == "relu":
        activation_func = nn.ReLU
    else:
        raise ValueError(f"Activation function {activation_func} not supported")

    layers = [
        nn.Linear(input_size, net_arch[0]),
        activation_func(),
    ]
    for i in range(1, len(net_arch)):
        layers.append(th.nn.Linear(net_arch[i - 1], net_arch[i]))
        layers.append(activation_func())

    if not last_layer_activation_func:
        layers.pop()

    return layers


def deep_copy_state_dict(state_dict):
    return {k: v.clone().detach() for k, v in state_dict.items()}


def polyak_update(net, target_net, tau=0.005):
    with th.no_grad():
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.mul_(1 - tau).add_(param.data, alpha=tau)


# Type definition using a custom class (the recommended way)
class LambdaModule(nn.Module):
    """
    A PyTorch Module that wraps a simple Python callable (like a lambda).
    """

    # Type hinting for the constructor and attributes
    def __init__(self, lambd: Callable[[Any], Any]):
        super().__init__()
        # Assert to ensure a callable is passed, though Callable type hint is preferred
        # You can optionally store the function object
        self.lambd = lambd

    # Type hinting for the forward method
    def forward(self, x: th.Tensor) -> th.Tensor:
        # The core of the module: passing the input tensor to the wrapped callable
        return self.lambd(x)


# distributions ================================


class DiagonalNormal(Normal):
    def __init__(self, loc, scale, validate_args=None):
        super().__init__(loc, scale, validate_args=validate_args)

    def log_prob(self, value):
        return super().log_prob(value).sum(dim=-1)

    def sample(self, sample_shape=th.Size()):
        return super().sample(sample_shape)

    def rsample(self, sample_shape=th.Size()):
        return super().rsample(sample_shape)

    def cdf(self, value):
        raise NotImplementedError("cdf is not implemented for DiagonalNormal")

    def entropy(self):
        return super().entropy().sum(dim=-1)

    @property
    def deterministic_sample(self):
        return self.mean


class ClippedTanhNormal(TanhNormal):
    def __init__(self, loc, clip_low=-5, clip_high=5, *args, **kwargs):
        loc = th.clamp(loc, clip_low, clip_high)
        super().__init__(loc, *args, **kwargs)


class SingleCategorical(Distribution):
    """Wrapper around torch Categorical for use with ProbabilisticActor.

    Handles single discrete action spaces properly, where actions are
    scalar integers rather than multi-dimensional.
    """

    def __init__(self, logits):
        self.categorical = Categorical(logits=logits)

    def log_prob(self, value):
        return self.categorical.log_prob(value)

    def sample(self, sample_shape=th.Size()):
        return self.categorical.sample(sample_shape)

    def rsample(self, sample_shape=th.Size()):
        raise NotImplementedError

    def cdf(self, value):
        raise NotImplementedError

    def entropy(self):
        return self.categorical.entropy()

    @property
    def deterministic_sample(self):
        return self.categorical.mode


class MultiCategorical(Distribution):
    def __init__(self, logits, head_dim):
        cs_head_dim = np.cumsum(head_dim)
        split_logits = [logits[..., 0 : head_dim[0]]]
        for i in range(1, len(head_dim)):
            split_logits.append(logits[..., cs_head_dim[i - 1] : cs_head_dim[i]])

        self.categorical = [Categorical(logits=split_logits[i]) for i in range(len(split_logits))]

    def log_prob(self, value):
        if value.ndim == 1:
            return th.sum(
                th.hstack(
                    [
                        categorical.log_prob(value[i])
                        for i, categorical in enumerate(self.categorical)
                    ]
                ),
                axis=-1,
            )
        elif value.ndim == 2:
            return th.sum(
                th.stack(
                    [
                        categorical.log_prob(value[:, i])
                        for i, categorical in enumerate(self.categorical)
                    ],
                    axis=-1,
                ),
                axis=-1,
            )
        else:
            raise NotImplementedError

    def sample(self, sample_shape=th.Size()):
        return th.stack(
            [categorical.rsample(sample_shape) for categorical in self.categorical]
        )

    def rsample(self, sample_shape=th.Size()):
        raise NotImplementedError

    def cdf(self, value):
        raise NotImplementedError

    def entropy(self):
        return th.sum(
            th.stack(
                [categorical.entropy() for categorical in self.categorical], axis=-1
            ),
            axis=-1,
        )

    @property
    def deterministic_sample(self):
        return th.stack([categorical.mode for categorical in self.categorical])
