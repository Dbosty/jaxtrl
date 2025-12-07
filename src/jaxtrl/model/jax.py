import jax
from flax import nnx
from jax import nn as jnn
from jax import numpy as jnp


def create_mlp_layers(
    input_size,
    rngs,
    net_arch=[64, 64, 64],
    activation_func="tanh",
    last_layer_activation_func=False,
):
    if activation_func == "tanh":
        activation_func = jnp.tanh
    elif activation_func == "relu":
        activation_func = jnn.relu
    else:
        raise ValueError(f"Activation function {activation_func} not supported")

    layers = [
        nnx.Linear(input_size, net_arch[0], rngs=rngs),
        activation_func,
    ]

    for i in range(1, len(net_arch)):
        layers.append(nnx.Linear(net_arch[i - 1], net_arch[i], rngs=rngs))
        layers.append(activation_func)

    if not last_layer_activation_func:
        layers.pop()

    return layers


# distributions ================================


class NormalSampler(nnx.Module):
    def __init__(self, output_shape):
        self.output_shape = output_shape

        self.log_std = nnx.Param(jnp.zeros(output_shape))

    def __call__(self, loc, key):
        return loc + jnp.exp(self.log_std) * jax.random.normal(key, self.output_shape)

    def deterministic_sample(self, loc):
        return loc


class TanhGaussianSampler(NormalSampler):
    def __init__(self, output_shape, tanh_loc=False, upscale=5.0):
        super().__init__(output_shape)

        self.tanh_loc = tanh_loc
        self.upscale = upscale

    def __call__(self, loc, key):
        if self.tanh_loc:
            loc = jnp.tanh(loc / self.upscale) * self.upscale
        return jnp.tanh(super().__call__(loc, key))

    def deterministic_sample(self, loc):
        if self.tanh_loc:
            loc = jnp.tanh(loc / self.upscale) * self.upscale
        return jnp.tanh(loc)


class ClippedTanhGaussianSampler(TanhGaussianSampler):
    def __init__(self, output_shape, clip_low=-5, clip_high=5):
        super().__init__(output_shape)
        self.clip_low = clip_low
        self.clip_high = clip_high

    def __call__(self, loc, key):
        return super().__call__(jnp.clip(loc, self.clip_low, self.clip_high), key)


class CategoricalSampler(nnx.Module):
    def __call__(self, logits, key):
        return jax.random.categorical(key, logits)

    def deterministic_sample(self, logits):
        return jnp.argmax(logits, -1)


class MultiCategoricalSampler(nnx.Module):
    def __init__(self, rngs):
        self.rngs = rngs

    def __call__(self, logits, key):
        return jnp.hstack(
            [jax.random.categorical(key, logits[i]) for i in range(len(logits))]
        )

    def deterministic_sample(self, logits):
        raise NotImplementedError
