"""
MPE Simple Spread environment components.

A cooperative multi-agent environment where N agents must spread out
to cover N landmarks while avoiding collisions.

Usage:
    from envs.mpe.simple_spread_env import (
        SimpleSpreadEnv,
        SimpleSpreadAgent,
        SimpleSpreadReward,
        SimpleSpreadTerm,
        make_simple_spread,
    )

    # Quick setup with defaults (3 agents, 3 landmarks)
    env, agents, rewards, term_fn = make_simple_spread()

    # Or customize
    env, agents, rewards, term_fn = make_simple_spread(
        num_agents=5,
        num_landmarks=5,
        world_size=1.5,
        collision_penalty=2.0,
    )
"""

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jaxtyping import Array, Key

from jaxtrl.sim.env import Agent, Env, Reward


# ================================
# Simple Spread Observation
# ================================


@struct.dataclass
class SimpleSpreadObs:
    """
    Observation state for Simple Spread environment.

    Attributes:
        agent_pos: (num_agents, 2) positions of all agents
        agent_vel: (num_agents, 2) velocities of all agents
        landmark_pos: (num_landmarks, 2) positions of all landmarks
        timestep: int - REQUIRED by Sim for truncation check
    """

    agent_pos: Array  # (num_agents, 2)
    agent_vel: Array  # (num_agents, 2)
    landmark_pos: Array  # (num_landmarks, 2)
    timestep: int


# ================================
# Simple Spread Environment
# ================================


class SimpleSpreadEnv(Env):
    """
    Pure JAX implementation of MPE Simple Spread.

    Physics:
    - World bounds: [-world_size, world_size] x [-world_size, world_size]
    - Agent accel: action * accel_scale
    - Velocity update: vel = vel * damping + accel * dt
    - Position update: pos = pos + vel * dt
    - Clamp position to world bounds

    Args:
        num_agents: Number of agents (default: 3)
        num_landmarks: Number of landmarks (default: 3)
        world_size: Half-width of the world (default: 1.0, world is [-1, 1])
        dt: Simulation timestep (default: 0.1)
        damping: Velocity damping factor (default: 0.25)
        max_speed: Maximum agent speed (default: 1.3)
        accel: Acceleration scaling (default: 5.0)
        agent_size: Agent collision radius (default: 0.15)
    """

    def __init__(
        self,
        num_agents: int = 3,
        num_landmarks: int = 3,
        world_size: float = 1.0,
        dt: float = 0.1,
        damping: float = 0.25,
        max_speed: float = 1.3,
        accel: float = 5.0,
        agent_size: float = 0.15,
    ):
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.world_size = world_size
        self.dt = dt
        self.damping = damping
        self.max_speed = max_speed
        self.accel = accel
        self.agent_size = agent_size

    def reset(self, key: Key) -> SimpleSpreadObs:
        """Initialize agents and landmarks at random positions."""
        k1, k2 = jax.random.split(key)

        agent_pos = jax.random.uniform(
            k1, (self.num_agents, 2), minval=-self.world_size, maxval=self.world_size
        )
        agent_vel = jnp.zeros((self.num_agents, 2))
        landmark_pos = jax.random.uniform(
            k2, (self.num_landmarks, 2), minval=-self.world_size, maxval=self.world_size
        )

        return SimpleSpreadObs(
            agent_pos=agent_pos,
            agent_vel=agent_vel,
            landmark_pos=landmark_pos,
            timestep=0,
        )

    def step(self, obs: SimpleSpreadObs, action: Array) -> SimpleSpreadObs:
        """
        Apply physics step.

        Args:
            obs: Current observation
            action: (num_agents, 2) array from jnp.vstack() in Sim
        """
        # action is (num_agents, 2) - forces in x/y for each agent
        # Clip action to [-1, 1] (policy outputs may exceed)
        action = jnp.clip(action, -1.0, 1.0)

        # Apply acceleration
        accel = action * self.accel

        # Update velocity with damping
        new_vel = obs.agent_vel * self.damping + accel * self.dt

        # Clamp velocity to max speed
        speed = jnp.linalg.norm(new_vel, axis=-1, keepdims=True)
        new_vel = jnp.where(
            speed > self.max_speed, new_vel * self.max_speed / (speed + 1e-8), new_vel
        )

        # Update position
        new_pos = obs.agent_pos + new_vel * self.dt

        # Clamp to world bounds
        new_pos = jnp.clip(new_pos, -self.world_size, self.world_size)

        return SimpleSpreadObs(
            agent_pos=new_pos,
            agent_vel=new_vel,
            landmark_pos=obs.landmark_pos,  # landmarks are static
            timestep=obs.timestep + 1,
        )


# ================================
# Simple Spread Agent
# ================================


class SimpleSpreadAgent(Agent):
    """
    Agent wrapper for Simple Spread.

    Observation per agent:
    - Own velocity (2D)
    - Relative positions to all landmarks (num_landmarks * 2D)
    - Relative positions to all other agents ((num_agents - 1) * 2D)

    Action (2D):
    - Continuous force in x/y, range [-1, 1]

    Args:
        agent_idx: Index of this agent (0 to num_agents-1)
        num_agents: Total number of agents (default: 3)
        num_landmarks: Number of landmarks (default: 3)
    """

    def __init__(self, agent_idx: int, num_agents: int = 3, num_landmarks: int = 3):
        super().__init__(sub_idx=agent_idx)
        self.agent_idx = agent_idx
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks

        # obs: own_vel(2) + rel_landmark(num_landmarks*2) + rel_agents((num_agents-1)*2)
        obs_dim = 2 + num_landmarks * 2 + (num_agents - 1) * 2
        self._obs_spec = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Continuous 2D action
        self._action_spec = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Pre-compute indices for other agents (static, for JAX tracing)
        self._other_indices = jnp.array(
            [j for j in range(num_agents) if j != agent_idx]
        )

    @property
    def pi_obs_spec(self):
        return self._obs_spec

    @property
    def pi_action_spec(self):
        return self._action_spec

    def obs_env2pi(self, obs: SimpleSpreadObs, context=None):
        """Convert environment observation to policy observation."""
        i = self.agent_idx

        # Own velocity
        own_vel = obs.agent_vel[i]  # (2,)

        # Relative positions to all landmarks
        rel_landmarks = obs.landmark_pos - obs.agent_pos[i]  # (num_landmarks, 2)
        rel_landmarks = rel_landmarks.flatten()

        # Relative positions to other agents using pre-computed indices
        other_pos = obs.agent_pos[self._other_indices]  # (num_agents - 1, 2)
        rel_agents = other_pos - obs.agent_pos[i]
        rel_agents = rel_agents.flatten()

        # Concatenate all observations
        pi_obs = jnp.concatenate([own_vel, rel_landmarks, rel_agents])

        return pi_obs, context

    def action_pi2env(self, action: Array, context=None):
        """Pass through action (already in correct format)."""
        return action, context

    def init_context(self, key: Key):
        return None


# ================================
# Simple Spread Reward
# ================================


class SimpleSpreadReward(Reward):
    """
    Simple Spread reward function.

    Reward = -sum(min_distance_to_each_landmark) - collision_penalty

    All agents share the same coverage reward (cooperative task),
    but each agent gets individual collision penalties.

    Args:
        agent_idx: Index of this agent
        num_agents: Total number of agents (default: 3)
        collision_penalty: Penalty per collision (default: 1.0)
        agent_size: Agent collision radius (default: 0.15)
    """

    def __init__(
        self,
        agent_idx: int,
        num_agents: int = 3,
        collision_penalty: float = 1.0,
        agent_size: float = 0.15,
    ):
        super().__init__()
        self.agent_idx = agent_idx
        self.num_agents = num_agents
        self.collision_penalty = collision_penalty
        self.agent_size = agent_size

        # Pre-compute mask for other agents (static, for JAX tracing)
        self._other_mask = jnp.arange(num_agents) != agent_idx

    def __call__(
        self,
        obs: SimpleSpreadObs,
        action: Array,
        next_obs: SimpleSpreadObs,
        context=None,
    ):
        """
        Compute reward for this agent.

        For Simple Spread, all agents get the same team reward:
        - Negative of sum of min distances from each landmark to any agent
        - Collision penalty when agents overlap
        """
        agent_pos = next_obs.agent_pos
        landmark_pos = next_obs.landmark_pos

        # Coverage reward: for each landmark, find min distance to any agent
        # Broadcast: (num_landmarks, 1, 2) - (1, num_agents, 2) -> (num_landmarks, num_agents, 2)
        diffs = landmark_pos[:, None, :] - agent_pos[None, :, :]
        distances = jnp.linalg.norm(diffs, axis=-1)  # (num_landmarks, num_agents)

        # Min distance from each landmark to nearest agent
        min_dists = jnp.min(distances, axis=1)  # (num_landmarks,)
        coverage_reward = -jnp.sum(min_dists)

        # Collision penalty: check pairwise distances between this agent and others
        i = self.agent_idx
        agent_diffs = agent_pos - agent_pos[i]  # (num_agents, 2)
        agent_dists = jnp.linalg.norm(agent_diffs, axis=-1)  # (num_agents,)

        # Count collisions with other agents
        collisions = jnp.sum(
            jnp.where(
                self._other_mask & (agent_dists < 2 * self.agent_size),
                self.collision_penalty,
                0.0,
            )
        )

        reward = coverage_reward - collisions

        return reward, context

    def init_context(self, key: Key):
        return None


# ================================
# Simple Spread Termination
# ================================


class SimpleSpreadTerm:
    """
    Termination function for Simple Spread.

    Simple Spread does not have early termination - episodes only end
    via truncation (max_steps in Sim).
    """

    def __call__(self, next_obs: SimpleSpreadObs) -> bool:
        return False  # Never terminates early


# ================================
# Factory Function
# ================================


def make_simple_spread(
    num_agents: int = 3,
    num_landmarks: int = 3,
    world_size: float = 1.0,
    dt: float = 0.1,
    damping: float = 0.25,
    max_speed: float = 1.3,
    accel: float = 5.0,
    agent_size: float = 0.15,
    collision_penalty: float = 1.0,
):
    """
    Factory function to create all Simple Spread components.

    Args:
        num_agents: Number of agents (default: 3)
        num_landmarks: Number of landmarks (default: 3)
        world_size: Half-width of the world (default: 1.0)
        dt: Simulation timestep (default: 0.1)
        damping: Velocity damping factor (default: 0.25)
        max_speed: Maximum agent speed (default: 1.3)
        accel: Acceleration scaling (default: 5.0)
        agent_size: Agent collision radius (default: 0.15)
        collision_penalty: Penalty per collision (default: 1.0)

    Returns:
        tuple: (env, agents, rewards, term_fn)
            - env: SimpleSpreadEnv instance
            - agents: List of SimpleSpreadAgent instances
            - rewards: List of SimpleSpreadReward instances
            - term_fn: SimpleSpreadTerm instance
    """
    env = SimpleSpreadEnv(
        num_agents=num_agents,
        num_landmarks=num_landmarks,
        world_size=world_size,
        dt=dt,
        damping=damping,
        max_speed=max_speed,
        accel=accel,
        agent_size=agent_size,
    )

    agents = [
        SimpleSpreadAgent(
            agent_idx=i,
            num_agents=num_agents,
            num_landmarks=num_landmarks,
        )
        for i in range(num_agents)
    ]

    rewards = [
        SimpleSpreadReward(
            agent_idx=i,
            num_agents=num_agents,
            collision_penalty=collision_penalty,
            agent_size=agent_size,
        )
        for i in range(num_agents)
    ]

    term_fn = SimpleSpreadTerm()

    return env, agents, rewards, term_fn
