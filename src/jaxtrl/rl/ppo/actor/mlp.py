import copy
from typing import Literal

import jax
import torch as th
from flax import nnx
from flax.struct import dataclass
from jaxtrl.model.jax import NormalSampler, TanhGaussianSampler
from jaxtrl.model.jax import create_mlp_layers as fl_create_mlp_layers
from jaxtrl.model.th import DiagonalNormal
from jaxtrl.model.th import create_mlp_layers as th_create_mlp_layers
from jaxtrl.sim.policy import Policy
from jaxtrl.utils.gym import gym_to_torchrl_spec
from jaxtyping import Array, Key, PyTree
from tensordict.nn import TensorDictModule as Mod
from tensordict.nn import TensorDictSequential as Seq
from torch import nn
from torchrl.modules import ProbabilisticActor, TanhNormal


@dataclass
class MLPContext:
    state: nnx.State


class MLPActor(Policy):
    def __init__(
        self,
        obs_space,
        action_space,
        jax_key,
        net_arch=[64, 64, 64],
        activation_func: Literal["tanh", "relu"] = "tanh",
        distribution: Literal["normal", "tanh"] = "normal",
        distribution_kwargs: dict = {},
        device="cpu",
    ):
        self.trl_actor = self._get_trl_actor(
            obs_space,
            action_space,
            net_arch,
            activation_func,
            distribution,
            distribution_kwargs,
            device,
        )

        self.fl_actor_graphdef, self.fl_actor_base_state = self._get_fl_actor(
            obs_space,
            action_space,
            net_arch,
            activation_func,
            distribution,
            distribution_kwargs,
            jax_key,
        )

    # ================================

    def __call__(self, obs_for_pi: Array, context: PyTree, key: Key):
        policy = nnx.merge(self.fl_actor_graphdef, context.state)

        return policy(obs_for_pi, key), context

    def deterministic_sample(self, obs_for_pi: Array, context: PyTree):
        policy = nnx.merge(self.fl_actor_graphdef, context.state)
        return policy.deterministic_sample(obs_for_pi), context

    def init_context(self, _):
        return MLPContext(state=self.params_trl2fl())

    def reset_context(self, context: PyTree, key: Key):
        return context

    # ================================

    def params_trl2fl(self):
        state = copy.deepcopy(
            self.fl_actor_base_state
        )  # TODO: replace this with flatten -> .replace() -> unflatten

        for name, param in self.trl_actor_net.named_parameters():
            key = name.split(".")
            param = jax.dlpack.from_dlpack(param.clone().detach())
            if key[0] == "log_std":
                state["sampler"]["log_std"] = nnx.Param(param)
            elif key[1] == "weight":
                state["mlp"]["layers"][int(key[0])]["kernel"] = nnx.Param(param.T)
            elif key[1] == "bias":
                state["mlp"]["layers"][int(key[0])]["bias"] = nnx.Param(param)
            else:
                raise ValueError(f"Unsupported parameter: {name}")

        return state

    def _get_trl_actor(
        self,
        obs_space,
        action_space,
        net_arch,
        activation_func,
        distribution,
        distribution_kwargs,
        device,
    ):
        layers = th_create_mlp_layers(
            obs_space.shape[0],
            net_arch,
            activation_func,
            True,
        )
        layers.append(nn.Linear(net_arch[-1], action_space.shape[0]))

        self.trl_actor_net = nn.Sequential(*layers)

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

        self.trl_scale_mod = ScaleMod()
        self.trl_log_std = self.trl_scale_mod.log_std

        self.trl_log_std.to(device)

        policy_module = Mod(
            self.trl_actor_net,
            in_keys=["observation"],
            out_keys=["loc"],
        )

        scale_module = Mod(
            self.trl_scale_mod,
            in_keys=["loc"],
            out_keys=["scale"],
        )
        self.trl_actor_net.register_parameter(
            "log_std", self.trl_log_std
        )  # manual register with trl_actor_net

        self.trl_actor_net.to(device)

        action_spec = gym_to_torchrl_spec(action_space, device)

        if distribution == "normal":
            dist_class = DiagonalNormal
            dist_kwargs = {}
        elif distribution == "tanh":
            dist_class = TanhNormal
            dist_kwargs = {
                "low": action_spec.space.low,
                "high": action_spec.space.high,
                "tanh_loc": distribution_kwargs["tanh_loc"],
            }

        return ProbabilisticActor(
            module=Seq(policy_module, scale_module),
            spec=action_spec,
            in_keys=["loc", "scale"],
            distribution_class=dist_class,
            distribution_kwargs=dist_kwargs,
            return_log_prob=True,
            # safe=True,
        )

    def _get_fl_actor(
        self,
        obs_space,
        action_space,
        net_arch,
        activation_func,
        distribution,
        distribution_kwargs,
        jax_key,
    ):

        rngs = nnx.Rngs(params=jax_key)

        layers = fl_create_mlp_layers(
            obs_space.shape[0],
            rngs,
            net_arch=net_arch,
            activation_func=activation_func,
            last_layer_activation_func=True,
        )
        layers.append(nnx.Linear(net_arch[-1], action_space.shape[0], rngs=rngs))

        if distribution == "normal":
            sampler = NormalSampler(action_space.shape[0])
        elif distribution == "tanh":
            sampler = TanhGaussianSampler(
                action_space.shape[0], tanh_loc=distribution_kwargs["tanh_loc"]
            )
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

        class FLLSTMActor(nnx.Module):
            def __init__(self, mlp, sampler):
                self.mlp = mlp
                self.sampler = sampler

            def __call__(self, obs, key):  # api in flax
                out = self.mlp(obs)
                out = self.sampler(out, key)

                return out

            def deterministic_sample(self, obs):
                out = self.mlp(obs)
                out = self.sampler.deterministic_sample(out)

                return out

        actor = FLLSTMActor(nnx.Sequential(*layers), sampler)

        fl_actor_graphdef, fl_actor_base_state = nnx.split(actor)

        return fl_actor_graphdef, fl_actor_base_state
