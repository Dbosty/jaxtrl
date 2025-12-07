from dataclasses import replace

import jax
from flax import struct
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, Key, PyTree

from jaxtrl.sim.env import Observation
from jaxtrl.sim.policy import Policy


@struct.dataclass
class Transition:
    obs: PyTree[Array]
    action: PyTree[Array]
    next_obs: PyTree[Array]
    reward: Array
    terminated: bool
    truncated: bool
    step_count: int

    def __getitem__(self, idx):
        return Transition(
            obs=self.obs[idx],
            action=self.action[idx],
            next_obs=self.next_obs[idx],
            reward=self.reward[..., idx],
            terminated=self.terminated,
            truncated=self.truncated,
            step_count=self.step_count,
        )


@struct.dataclass
class Carry:
    obs: Observation
    obs_for_pi: PyTree[Array]
    episode_reward: PyTree[float]
    policy_context: PyTree
    agent_context: PyTree
    sim_context: PyTree
    done: bool
    key: Key


def _collect_factory(sim_reset, sim_step, policy, timesteps):
    pass


def collect_factory(
    sim_reset,
    sim_step,
    policy,
    timesteps_per_env=1000,
    num_envs=1,
    flatten=True,
):
    assert isinstance(
        policy, Policy
    ), "policy must be a Policy type (ensure not JIT-ed)"

    def step_branch(carry):
        action_from_pi, policy_context = policy(...)
