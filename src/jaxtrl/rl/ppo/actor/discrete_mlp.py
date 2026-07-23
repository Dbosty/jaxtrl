import copy
from typing import Literal

import gymnasium as gym
import jax
import torch as th
from flax import nnx
from flax.struct import dataclass
from jaxtyping import Array, Key, PyTree
from tensordict.nn import TensorDictModule as Mod
from torch import nn
from torchrl.modules import ProbabilisticActor

from jaxtrl.model.jax import CategoricalSampler
from jaxtrl.model.jax import create_mlp_layers as fl_create_mlp_layers
from jaxtrl.model.th import MultiCategorical, SingleCategorical
from jaxtrl.model.th import create_mlp_layers as th_create_mlp_layers
from jaxtrl.sim.policy import Policy
from jaxtrl.utils.gym import gym_to_torchrl_spec


@dataclass
class DiscreteMLP_Context:
    state: nnx.State


class DiscreteMLP(Policy):
    """Discrete MLP actor for environments with discrete action spaces."""

    def __init__(
        self,
        obs_space,
        action_space,
        jax_key,
        net_arch=[64, 64],
        activation_func: Literal["tanh", "relu"] = "tanh",
        device="cpu",
    ):
        self.trl_actor = self._get_trl_actor(
            obs_space,
            action_space,
            net_arch,
            activation_func,
            device,
        )

        self.fl_actor_graphdef, self.fl_actor_base_state = self._get_fl_actor(
            obs_space,
            action_space,
            net_arch,
            activation_func,
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
        return DiscreteMLP_Context(state=self.params_trl2fl())

    def reset_context(self, context: PyTree, key: Key):
        return context

    # ================================

    def params_trl2fl(self):
        state = copy.deepcopy(self.fl_actor_base_state)

        for name, param in self.trl_actor_net.named_parameters():
            key = name.split(".")
            param = jax.dlpack.from_dlpack(param.clone().detach())
            if key[1] == "weight":
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
        device,
    ):
        # Compute output size based on action space type
        is_multi_discrete = isinstance(action_space, gym.spaces.MultiDiscrete)
        if is_multi_discrete:
            head_dim = list(action_space.nvec)
            output_size = sum(head_dim)
        else:  # Discrete
            output_size = action_space.n

        layers = th_create_mlp_layers(
            obs_space.shape[0],
            net_arch,
            activation_func,
            last_layer_activation_func=True,
        )
        layers.append(nn.Linear(net_arch[-1], output_size))

        self.trl_actor_net = nn.Sequential(*layers)

        policy_module = Mod(
            self.trl_actor_net,
            in_keys=["observation"],
            out_keys=["logits"],
        )

        self.trl_actor_net.to(device)

        action_spec = gym_to_torchrl_spec(action_space, device)

        if is_multi_discrete:
            return ProbabilisticActor(
                module=policy_module,
                spec=action_spec,
                in_keys=["logits"],
                distribution_class=MultiCategorical,
                distribution_kwargs={"head_dim": head_dim},
                return_log_prob=True,
            )
        else:
            return ProbabilisticActor(
                module=policy_module,
                spec=action_spec,
                in_keys=["logits"],
                distribution_class=SingleCategorical,
                return_log_prob=True,
            )

    def _get_fl_actor(
        self,
        obs_space,
        action_space,
        net_arch,
        activation_func,
        jax_key,
    ):
        rngs = nnx.Rngs(params=jax_key)

        # Compute output size based on action space type
        if isinstance(action_space, gym.spaces.MultiDiscrete):
            output_size = sum(action_space.nvec)
        else:  # Discrete
            output_size = action_space.n

        layers = fl_create_mlp_layers(
            obs_space.shape[0],
            rngs,
            net_arch=net_arch,
            activation_func=activation_func,
            last_layer_activation_func=True,
        )
        layers.append(nnx.Linear(net_arch[-1], output_size, rngs=rngs))

        sampler = CategoricalSampler()

        class FLDiscreteMLPActor(nnx.Module):
            def __init__(self, mlp, sampler):
                self.mlp = mlp
                self.sampler = sampler

            def __call__(self, obs, key):
                out = self.mlp(obs)
                out = self.sampler(out, key)
                return out

            def deterministic_sample(self, obs):
                out = self.mlp(obs)
                out = self.sampler.deterministic_sample(out)
                return out

        actor = FLDiscreteMLPActor(nnx.Sequential(*layers), sampler)

        fl_actor_graphdef, fl_actor_base_state = nnx.split(actor)

        return fl_actor_graphdef, fl_actor_base_state
