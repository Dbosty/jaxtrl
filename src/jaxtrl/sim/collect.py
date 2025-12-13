from dataclasses import asdict

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
    episode_reward: Array
    terminated: bool
    truncated: bool
    step_count: int

    def __getitem__(self, idx):
        return Transition(
            obs=self.obs[idx],
            action=self.action[idx],
            next_obs=self.next_obs[idx],
            reward=self.reward[..., idx],
            episode_reward=self.episode_reward[..., idx],
            terminated=self.terminated,
            truncated=self.truncated,
            step_count=self.step_count,
        )


@struct.dataclass
class ExtendedTransition(Transition):
    raw_obs: Observation
    raw_next_obs: Observation


@struct.dataclass
class Carry:
    obs: PyTree[Array]
    obs_for_pi: PyTree[Array]
    policy_context: PyTree
    agent_context: PyTree
    sim_context: PyTree
    done: bool
    step_count: int
    episode_reward: Array
    key: Key


def _collect_factory(sim_reset, sim_step, policy, timesteps, extended_transition):
    def step_branch(carry):
        key, policy_key = jax.random.split(carry.key, 2)
        action_from_pi, policy_context = policy(
            carry.obs_for_pi, carry.policy_context, policy_key
        )

        next_obs, reward, term, trun, next_obs_for_pi, next_agent_con, next_sim_con = (
            sim_step(
                carry.obs,
                action_from_pi,
                carry.agent_context,
                carry.sim_context,
            )
        )

        transition = Transition(
            obs=carry.obs_for_pi,
            action=action_from_pi,
            next_obs=next_obs_for_pi,
            reward=reward,
            episode_reward=carry.episode_reward + reward,
            terminated=term,
            truncated=trun,
            step_count=carry.step_count + 1,
        )

        if extended_transition:
            transition = ExtendedTransition(
                raw_obs=carry.obs,
                raw_next_obs=next_obs,
                **asdict(transition),
            )

        carry = Carry(
            obs=next_obs,
            obs_for_pi=next_obs_for_pi,
            episode_reward=carry.episode_reward + reward,
            policy_context=policy_context,
            agent_context=next_agent_con,
            sim_context=next_sim_con,
            done=term | trun,
            step_count=carry.step_count + 1,
            key=key,
        )

        return carry, transition

    def reset_branch(carry):
        key, sim_key, policy_key = jax.random.split(carry.key, 3)
        obs, obs_for_pi, agent_context, sim_context = sim_reset(sim_key)

        policy_context = policy.reset_context(carry.policy_context, policy_key)

        carry = Carry(
            obs=obs,
            obs_for_pi=obs_for_pi,
            episode_reward=jnp.zeros(len(obs_for_pi)),
            policy_context=policy_context,
            agent_context=agent_context,
            sim_context=sim_context,
            done=False,
            step_count=-1,
            key=key,
        )

        return step_branch(carry)

    def _step(carry, _):
        return lax.cond(carry.done, reset_branch, step_branch, carry)

    dummy_obs, dummy_obs_for_pi, dummy_agent_context, dummy_sim_context = sim_reset(
        dummy_key := jax.random.key(0)
    )
    dummy_policy_context = policy.init_context(dummy_key)

    def _collect(
        key: Key,
        policy_context: PyTree = None,
        agent_context: PyTree = None,
        sim_context: PyTree = None,
    ):
        if policy_context is None:
            policy_context = dummy_policy_context
        if agent_context is None:
            agent_context = dummy_agent_context
        if sim_context is None:
            sim_context = dummy_sim_context

        carry = Carry(
            obs=dummy_obs,
            obs_for_pi=dummy_obs_for_pi,
            episode_reward=jnp.zeros(len(dummy_obs_for_pi)),
            policy_context=policy_context,
            agent_context=agent_context,
            sim_context=sim_context,
            done=True,
            step_count=-1,
            key=key,
        )

        _, transitions = lax.scan(_step, carry, None, timesteps)

        transitions = transitions.replace(
            truncated=transitions.truncated.at[-1].set(True)
        )

        return transitions

    return _collect


def collect_factory(
    sim_reset,
    sim_step,
    policy,
    timesteps_per_env=1000,
    num_envs=1,
    flatten=True,
    extended_transition=False,
):
    assert isinstance(
        policy, Policy
    ), "policy must be a Policy type (ensure not JIT-ed)"

    _collect = _collect_factory(
        sim_reset, sim_step, policy, timesteps_per_env, extended_transition
    )

    _collect = jax.vmap(_collect, in_axes=(0, None, None, None))

    remove_first_dim = lambda node: node.reshape(-1, *node.shape[2:])

    def collect(
        key: Key,
        policy_context: PyTree = None,
        agent_context: PyTree = None,
        sim_context: PyTree = None,
    ):
        keys = jax.random.split(key, num_envs)
        transitions = _collect(keys, policy_context, agent_context, sim_context)

        if flatten:
            transitions = jax.tree.map(remove_first_dim, transitions)

        return transitions

    return collect


def dummy_collect_factory(sim_reset, sim_step, policy, *args, **kwargs):
    dummy_obs, dummy_obs_for_pi, dummy_agent_context, dummy_sim_context = sim_reset(
        dummy_key := jax.random.key(0)
    )
    dummy_policy_context = policy.init_context(dummy_key)
    dummy_action_from_pi, dummy_policy_context = policy(
        dummy_obs_for_pi, dummy_policy_context, dummy_key
    )

    def _collect(*args, **kwargs):
        carry = Carry(
            obs=dummy_obs,
            obs_for_pi=dummy_obs_for_pi,
            episode_reward=jnp.zeros(len(dummy_obs_for_pi)),
            policy_context=dummy_policy_context,
            agent_context=dummy_agent_context,
            sim_context=dummy_sim_context,
            done=True,
            step_count=-1,
            key=dummy_key,
        )

        transition = Transition(
            obs=carry.obs_for_pi,
            action=dummy_action_from_pi,
            next_obs=dummy_obs_for_pi,
            reward=jnp.zeros(len(dummy_obs_for_pi)),
            episode_reward=jnp.zeros(len(dummy_obs_for_pi)),
            terminated=False,
            truncated=False,
            step_count=0,
        )

        transitions = jax.tree.map(lambda x: jnp.stack([x, x]), transition)

        return transitions

    return _collect
