import jax
from jax import numpy as jnp
from jaxtyping import Key, PyTree

from jaxtrl.sim.env import Agent, Observation, Reward


class Sim:
    def __init__(
        self, env, agents: list[Agent], reward: list[Reward], term, max_steps: int = 100
    ):
        assert len(agents) == len(
            reward
        ), "number of agents and reward must be the same"

        self.env = env
        self.agents = agents
        self.reward = reward
        self.term = term
        self.max_steps = max_steps

    def reset(self, key: Key):
        sk1, sk2, sk3 = jax.random.split(key, 3)
        obs = self.env.reset(sk1)

        keys = jax.random.split(sk2, len(self.agents))
        agent_context = [
            self.agents[i].init_context(keys[i]) for i in range(len(self.agents))
        ]

        obs_for_pi, agent_context = self._obs_env2pi(obs, agent_context)

        keys = jax.random.split(sk3, len(self.reward))
        sim_context = [
            self.reward[i].init_context(keys[i]) for i in range(len(self.reward))
        ]

        return obs, obs_for_pi, agent_context, sim_context

    def step(
        self,
        obs: Observation,
        action_from_pi: PyTree,
        agent_context: PyTree,
        sim_context: PyTree,
    ):
        action, agent_context = self._action_pi2env(action_from_pi, agent_context)

        action = jnp.vstack(jax.tree.leaves(action))

        next_obs = self.env.step(obs, action)

        reward = []
        next_sim_context = []
        for i in range(len(self.agents)):
            _reward, _sim_context = self.reward[i](
                obs=obs, action=action, next_obs=next_obs, context=sim_context[i]
            )
            reward.append(_reward)
            next_sim_context.append(_sim_context)

        terminated = self.term(next_obs)
        truncated = next_obs.timestep >= self.max_steps

        next_obs_for_pi, next_agent_context = self._obs_env2pi(next_obs, agent_context)

        return (
            next_obs,
            jnp.hstack(reward),
            terminated,
            truncated,
            next_obs_for_pi,
            next_agent_context,
            next_sim_context,
        )

    def _obs_env2pi(self, obs: Observation, obs_context: PyTree):
        val = jax.tree.map(
            lambda agent, context: agent.obs_env2pi(obs, context),
            self.agents,
            obs_context,
        )

        obs_for_pi = [xs[0] for xs in val]
        obs_context = [xs[1] for xs in val]

        return obs_for_pi, obs_context

    def _action_pi2env(self, action_from_pi: PyTree, agent_context: PyTree):
        action = []
        context = []
        for i in range(len(self.agents)):
            _action, _context = self.agents[i].action_pi2env(
                action_from_pi[i], agent_context[i]
            )
            action.append(_action)
            context.append(_context)

        return action, context
