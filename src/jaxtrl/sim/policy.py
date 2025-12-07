from abc import ABC, abstractmethod

import gymnasium as gym
import jax
from jaxtyping import Array, Key, PyTree


class Policy(ABC):
    @abstractmethod
    def __call__(
        self, obs_for_pi: Array, context: PyTree, key: Key
    ) -> tuple[Array, PyTree]:
        pass

    def init_context(self, key: Key) -> PyTree:
        return {}

    def reset_context(self, context: PyTree, key: Key) -> PyTree:
        return context

    def deterministic_sample(
        self, obs_for_pi: Array, context: PyTree
    ) -> tuple[Array, PyTree]:
        raise NotImplementedError("deterministic_sample is not implemented")


class PolicyAggregator(Policy):
    def __init__(self, policies: list[Policy]):
        self.policies = policies

    def __call__(self, obs_for_pi: PyTree, context: PyTree, key: Key):
        keys = jax.random.split(key, len(self.policies))
        actions = []
        contexts = []
        for i in range(len(self.policies)):
            action, policy_context = self.policies[i](
                obs_for_pi[i], context[i], keys[i]
            )

            actions.append(action)
            contexts.append(policy_context)

        return actions, contexts

    def init_context(self, key: Key) -> PyTree:
        keys = jax.random.split(key, len(self.policies))
        return [
            self.policies[i].init_context(keys[i]) for i in range(len(self.policies))
        ]

    def reset_context(self, context: PyTree, key: Key) -> PyTree:
        keys = jax.random.split(key, len(self.policies))
        return [
            self.policies[i].reset_context(context[i], keys[i])
            for i in range(len(self.policies))
        ]

    def deterministic_sample(self, obs_for_pi: PyTree, context: PyTree):
        actions = []
        contexts = []
        for i in range(len(self.policies)):
            action, policy_context = self.policies[i].deterministic_sample(
                obs_for_pi[i], context[i]
            )
            actions.append(action)
            contexts.append(policy_context)
        return actions, contexts


class DeterministicWrapper(Policy):
    def __init__(self, policy: Policy):
        self.policy = policy

    def __call__(
        self, obs_for_pi: Array, context: PyTree, key: Key
    ) -> tuple[Array, PyTree]:
        return self.policy.deterministic_sample(obs_for_pi, context)

    def init_context(self, key: Key) -> PyTree:
        return self.policy.init_context(key)

    def reset_context(self, context: PyTree, key: Key) -> PyTree:
        return self.policy.reset_context(context, key)

    def deterministic_sample(
        self, obs_for_pi: Array, context: PyTree
    ) -> tuple[Array, PyTree]:
        return self.policy.deterministic_sample(obs_for_pi, context)


#


class RandomPolicy(Policy):
    def __init__(self, action_space: gym.Space):
        if isinstance(action_space, gym.spaces.Box):
            self.random_fn = lambda key: jax.random.uniform(
                key,
                shape=action_space.shape,
                minval=action_space.low,
                maxval=action_space.high,
            )
        elif isinstance(action_space, gym.spaces.Discrete):
            self.random_fn = lambda key: jax.random.randint(
                key,
                shape=(),
                minval=0,
                maxval=action_space.n,
            )
        else:
            raise NotImplementedError(f"Unsupported action space: {action_space}")

    def __call__(self, obs_for_pi: Array, context: PyTree, key: Key) -> Array:
        out = self.random_fn(key)
        return out, context


class ChoicePerRolloutPolicy(Policy):  # per traj
    def __init__(self, probs: Array):
        self.probs = probs

    def __call__(self, obs_for_pi: Array, context: PyTree, key: Key):
        out = context["current_choice"]
        return out, context

    def reset_context(self, context: PyTree, key: Key):
        key, subkey = jax.random.split(key)
        current_choice = jax.random.choice(subkey, a=len(self.probs), p=self.probs)
        return {"key": key, "current_choice": current_choice}

    def init_context(self, key: Key):
        key, subkey = jax.random.split(key)
        current_choice = jax.random.choice(subkey, a=len(self.probs), p=self.probs)
        return {"current_choice": current_choice}
