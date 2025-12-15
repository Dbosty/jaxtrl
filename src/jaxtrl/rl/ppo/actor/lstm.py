import copy
from typing import Literal

import jax
import torch as th
from flax import nnx
from flax.nnx.nn import recurrent
from flax.struct import dataclass
from jax import numpy as jnp
from jaxtyping import Array, Key, PyTree
from tensordict.nn import TensorDictModule as Mod
from tensordict.nn import TensorDictSequential as Seq
from torch import nn
from torchrl.modules import LSTMModule, ProbabilisticActor

from jaxtrl.model.jax import (
    ClippedTanhGaussianSampler,
    NormalSampler,
    TanhGaussianSampler,
)
from jaxtrl.model.jax import create_mlp_layers as fl_create_mlp_layers
from jaxtrl.model.th import ClippedTanhNormal, DiagonalNormal, TanhNormal
from jaxtrl.model.th import create_mlp_layers as th_create_mlp_layers
from jaxtrl.sim.policy import Policy
from jaxtrl.utils.gym import gym_to_torchrl_spec


@dataclass
class LSTMContext:
    state: nnx.State
    c: Array
    h: Array


class LSTMActor(Policy):
    def __init__(
        self,
        obs_space,
        action_space,
        jax_key,
        lstm_hidden_size=64,
        head_net_arch=[64, 64, 64],
        head_activation_func: Literal["tanh", "relu"] = "tanh",
        distribution: Literal["normal", "tanh", "clipped_tanh"] = "normal",
        distribution_kwargs: dict = {},
        device="cpu",
    ):

        self.lstm_hidden_size = lstm_hidden_size

        self.trl_actor = self._get_trl_actor(
            obs_space,
            action_space,
            lstm_hidden_size,
            head_net_arch,
            head_activation_func,
            distribution,
            distribution_kwargs,
            device,
        )

        self.fl_actor_graphdef, self.fl_actor_base_state = self._get_fl_actor_graphdef(
            obs_space,
            action_space,
            lstm_hidden_size,
            head_net_arch,
            head_activation_func,
            distribution,
            distribution_kwargs,
            jax_key,
        )

    # ================================

    def __call__(self, obs_for_pi: Array, context: PyTree, key: Key):
        state, c, h = context.state, context.c, context.h

        policy = nnx.merge(self.fl_actor_graphdef, state)
        (c, h), out = policy((c, h), obs_for_pi, key)

        context = LSTMContext(state=state, c=c, h=h)

        return out, context

    def init_context(self, _):
        return LSTMContext(
            state=self.params_trl2fl(),
            c=jnp.zeros((self.lstm_hidden_size,)),
            h=jnp.zeros((self.lstm_hidden_size,)),
        )

    def reset_context(self, policy_context: PyTree, key: Key):
        return LSTMContext(
            state=policy_context.state,
            c=jnp.zeros((self.lstm_hidden_size,)),
            h=jnp.zeros((self.lstm_hidden_size,)),
        )

    def deterministic_sample(self, obs_for_pi: Array, context: PyTree):
        state, c, h = context.state, context.c, context.h

        policy = nnx.merge(self.fl_actor_graphdef, state)

        (c, h), out = policy.deterministic_sample((c, h), obs_for_pi)

        return out, LSTMContext(state=state, c=c, h=h)

    # ================================

    def params_trl2fl(self):
        """Transfer parameters from PyTorch to Flax actor."""
        state = copy.deepcopy(
            self.fl_actor_base_state
        )  # TODO: replace this with flatten -> .replace() -> unflatten

        # Get all PyTorch parameters from the actor
        for name, param in self.trl_actor.named_parameters():
            param_jax = jax.dlpack.from_dlpack(param.clone().detach())

            if name == "module.0.module.3.module.log_std":
                # Transfer log_std parameter
                state["sampler"]["log_std"] = nnx.Param(param_jax)

            elif name == "module.0.module.1.lstm.weight_ih_l0":
                # LSTM input-to-hidden weights
                state["lstm_cell"]["dense_i"]["kernel"] = nnx.Param(param_jax.T)

            elif name == "module.0.module.1.lstm.weight_hh_l0":
                # LSTM hidden-to-hidden weights
                state["lstm_cell"]["dense_h"]["kernel"] = nnx.Param(param_jax.T)

            elif name == "module.0.module.1.lstm.bias_ih_l0":
                # Store input-to-hidden bias temporarily for combination
                self._temp_bias_ih = param_jax

            elif name == "module.0.module.1.lstm.bias_hh_l0":
                # Combine input-to-hidden and hidden-to-hidden biases
                bias_combined = self._temp_bias_ih + param_jax
                state["lstm_cell"]["dense_h"]["bias"] = nnx.Param(bias_combined)
                # Clean up temporary variable
                delattr(self, "_temp_bias_ih")

            elif name.startswith("module.0.module.2.module.") and name.endswith(
                ".weight"
            ):
                # Head MLP layer weights - extract layer index dynamically
                parts = name.split(".")
                layer_idx = int(parts[-2])  # Get layer index from second-to-last part
                state["mlp"]["layers"][layer_idx]["kernel"] = nnx.Param(param_jax.T)

            elif name.startswith("module.0.module.2.module.") and name.endswith(
                ".bias"
            ):
                # Head MLP layer bias - extract layer index dynamically
                parts = name.split(".")
                layer_idx = int(parts[-2])  # Get layer index from second-to-last part
                state["mlp"]["layers"][layer_idx]["bias"] = nnx.Param(param_jax)

            else:
                raise ValueError(f"Unsupported parameter: {name}")

        return state

    def _get_trl_actor(
        self,
        obs_space,
        action_space,
        lstm_hidden_size,
        head_net_arch,
        head_activation_func,
        distribution,
        distribution_kwargs,
        device,
    ):

        # lstm

        create_is_init = Mod(
            lambda step_count: step_count == 0,
            in_keys=["step_count"],
            out_keys=["is_init"],
        )

        lstm_mod = LSTMModule(
            input_size=obs_space.shape[0],
            hidden_size=lstm_hidden_size,
            in_keys=["observation", "actor_rnn_h", "actor_rnn_c", "is_init"],
            out_keys=[
                "actor_lstm_embed",
                ("next", "actor_rnn_h"),
                ("next", "actor_rnn_c"),
            ],
            default_recurrent_mode=True,
        )

        # head

        layers = th_create_mlp_layers(
            lstm_hidden_size,
            head_net_arch,
            head_activation_func,
            last_layer_activation_func=True,
        )
        layers.append(nn.Linear(head_net_arch[-1], action_space.shape[0]))

        head = nn.Sequential(*layers)

        head_mod = Mod(
            head,
            in_keys=["actor_lstm_embed"],
            out_keys=["loc"],
        )

        # scale

        class ScaleMod(nn.Module):
            def __init__(self):
                super().__init__()
                self.log_std = nn.Parameter(th.zeros(action_space.shape[0]))

            def forward(self, loc):
                return (
                    self.log_std.exp().repeat(loc.shape[0], 1)
                    if loc.ndim > 1
                    else self.log_std.exp()
                )

        scale_mod = Mod(
            ScaleMod(),
            in_keys=["loc"],
            out_keys=["scale"],
        )

        action_spec = gym_to_torchrl_spec(action_space, device)

        if distribution == "normal":
            dist_class = DiagonalNormal
            dist_kwargs = {}
        elif distribution == "tanh":
            dist_class = TanhNormal
            dist_kwargs = {
                "low": action_space.low,
                "high": action_space.high,
                "tanh_loc": distribution_kwargs["tanh_loc"],
            }
        elif distribution == "clipped_tanh":
            dist_class = ClippedTanhNormal
            dist_kwargs = {
                "low": action_space.low,
                "high": action_space.high,
                "tanh_loc": distribution_kwargs["tanh_loc"],
                "clip_low": distribution_kwargs["clip_low"],
                "clip_high": distribution_kwargs["clip_high"],
            }
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

        # actor

        policy = Seq(create_is_init, lstm_mod, head_mod, scale_mod)

        actor = ProbabilisticActor(
            module=policy,
            spec=action_spec,
            in_keys=["loc", "scale"],
            distribution_class=dist_class,
            distribution_kwargs=dist_kwargs,
            return_log_prob=True,
        )

        actor.to(device)

        return actor

    def _get_fl_actor_graphdef(
        self,
        obs_space,
        action_space,
        lstm_hidden_size,
        head_net_arch,
        head_activation_func,
        distribution,
        distribution_kwargs,
        jax_key,
    ):
        rngs = nnx.Rngs(params=jax_key)

        # lstm_cell = recurrent.LSTMCell(
        lstm_cell = recurrent.OptimizedLSTMCell(  # optimized better for hidden size < 2048 according to flax docs
            obs_space.shape[0], lstm_hidden_size, rngs=rngs
        )  # returns (c, h), h [vs th -> h, (h, c)]

        layers = fl_create_mlp_layers(
            lstm_hidden_size,
            rngs,
            net_arch=head_net_arch,
            activation_func=head_activation_func,
            last_layer_activation_func=True,
        )
        layers.append(nnx.Linear(head_net_arch[-1], action_space.shape[0], rngs=rngs))

        if distribution == "normal":
            sampler = NormalSampler(action_space.shape[0])
        elif distribution == "tanh":
            sampler = TanhGaussianSampler(
                action_space.shape[0],
                tanh_loc=distribution_kwargs["tanh_loc"],
            )
        elif distribution == "clipped_tanh":
            sampler = ClippedTanhGaussianSampler(
                action_space.shape[0],
                clip_low=distribution_kwargs["clip_low"],
                clip_high=distribution_kwargs["clip_high"],
            )
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

        class FLLSTMActor(nnx.Module):
            def __init__(self, lstm_cell, mlp, sampler):
                self.lstm_cell = lstm_cell
                self.mlp = mlp
                self.sampler = sampler

            def __call__(self, carry, obs, key):  # api in flax
                carry, out = self.lstm_cell(carry, obs)
                out = self.mlp(out)
                out = self.sampler(out, key)

                return carry, out

            def deterministic_sample(self, carry, obs):
                carry, out = self.lstm_cell(carry, obs)
                out = self.mlp(out)
                out = self.sampler.deterministic_sample(out)

                return carry, out

        actor = FLLSTMActor(lstm_cell, nnx.Sequential(*layers), sampler)

        fl_actor_graphdef, fl_actor_base_state = nnx.split(actor)

        return fl_actor_graphdef, fl_actor_base_state
