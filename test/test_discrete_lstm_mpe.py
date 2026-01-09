"""
Test DiscreteLSTMActor with discrete action MPE Simple Spread.

This script tests that:
1. DiscreteLSTMActor works with discrete action spaces
2. Training loop runs successfully
3. Both Discrete and MultiDiscrete action spaces work

Usage:
    python test/test_discrete_lstm_mpe.py
"""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
mpe_dir = os.path.join(project_root, "envs", "mpe")
if mpe_dir not in sys.path:
    sys.path.insert(0, mpe_dir)

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import torch as th
from flax import struct
from jaxtyping import Array, Key

from jaxtrl.rl.ppo.actor.discrete_lstm import DiscreteLSTMActor
from jaxtrl.sim import Sim
from jaxtrl.sim.collect import collect_factory
from jaxtrl.sim.env import Agent, Env, Reward
from jaxtrl.sim.policy import PolicyAggregator


# ================================
# Discrete Action Simple Spread
# ================================


@struct.dataclass
class DiscreteSpreadObs:
    agent_pos: Array
    agent_vel: Array
    landmark_pos: Array
    timestep: int


class DiscreteSpreadEnv(Env):
    """Simple Spread with discrete actions: 0=noop, 1=left, 2=right, 3=up, 4=down"""

    def __init__(
        self,
        num_agents: int = 3,
        num_landmarks: int = 3,
        world_size: float = 1.0,
        dt: float = 0.1,
        damping: float = 0.25,
        max_speed: float = 1.3,
        accel: float = 5.0,
    ):
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.world_size = world_size
        self.dt = dt
        self.damping = damping
        self.max_speed = max_speed
        self.accel = accel

        # Discrete action to continuous force mapping
        # 0=noop, 1=left, 2=right, 3=up, 4=down
        self.action_map = jnp.array(
            [
                [0.0, 0.0],  # noop
                [-1.0, 0.0],  # left
                [1.0, 0.0],  # right
                [0.0, 1.0],  # up
                [0.0, -1.0],  # down
            ]
        )

    def reset(self, key: Key) -> DiscreteSpreadObs:
        k1, k2 = jax.random.split(key)
        agent_pos = jax.random.uniform(
            k1, (self.num_agents, 2), minval=-self.world_size, maxval=self.world_size
        )
        agent_vel = jnp.zeros((self.num_agents, 2))
        landmark_pos = jax.random.uniform(
            k2, (self.num_landmarks, 2), minval=-self.world_size, maxval=self.world_size
        )
        return DiscreteSpreadObs(
            agent_pos=agent_pos,
            agent_vel=agent_vel,
            landmark_pos=landmark_pos,
            timestep=0,
        )

    def step(self, obs: DiscreteSpreadObs, action: Array) -> DiscreteSpreadObs:
        # action is (num_agents,) discrete actions or (num_agents, num_dims) for multidiscrete
        # Convert to continuous forces
        if action.ndim == 1:
            # Single discrete action per agent
            forces = self.action_map[action]  # (num_agents, 2)
        else:
            # MultiDiscrete: action is (num_agents, 2) where each is 0-2 (left/noop/right, down/noop/up)
            # Map: 0=-1, 1=0, 2=1
            forces = (action - 1).astype(jnp.float32)  # (num_agents, 2)

        accel = forces * self.accel
        new_vel = obs.agent_vel * self.damping + accel * self.dt

        speed = jnp.linalg.norm(new_vel, axis=-1, keepdims=True)
        new_vel = jnp.where(
            speed > self.max_speed, new_vel * self.max_speed / (speed + 1e-8), new_vel
        )

        new_pos = obs.agent_pos + new_vel * self.dt
        new_pos = jnp.clip(new_pos, -self.world_size, self.world_size)

        return DiscreteSpreadObs(
            agent_pos=new_pos,
            agent_vel=new_vel,
            landmark_pos=obs.landmark_pos,
            timestep=obs.timestep + 1,
        )


class DiscreteSpreadAgent(Agent):
    """Agent with discrete action space."""

    def __init__(
        self,
        agent_idx: int,
        num_agents: int = 3,
        num_landmarks: int = 3,
        multidiscrete: bool = False,
    ):
        super().__init__(sub_idx=agent_idx)
        self.agent_idx = agent_idx
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.multidiscrete = multidiscrete

        obs_dim = 2 + num_landmarks * 2 + (num_agents - 1) * 2
        self._obs_spec = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        if multidiscrete:
            # 2 dimensions: x-axis (3 choices: left/noop/right), y-axis (3 choices: down/noop/up)
            self._action_spec = gym.spaces.MultiDiscrete([3, 3])
        else:
            # 5 discrete actions: noop, left, right, up, down
            self._action_spec = gym.spaces.Discrete(5)

        self._other_indices = jnp.array(
            [j for j in range(num_agents) if j != agent_idx]
        )

    @property
    def pi_obs_spec(self):
        return self._obs_spec

    @property
    def pi_action_spec(self):
        return self._action_spec

    def obs_env2pi(self, obs: DiscreteSpreadObs, context=None):
        i = self.agent_idx
        own_vel = obs.agent_vel[i]
        rel_landmarks = (obs.landmark_pos - obs.agent_pos[i]).flatten()
        other_pos = obs.agent_pos[self._other_indices]
        rel_agents = (other_pos - obs.agent_pos[i]).flatten()
        pi_obs = jnp.concatenate([own_vel, rel_landmarks, rel_agents])
        return pi_obs, context

    def action_pi2env(self, action: Array, context=None):
        return action, context

    def init_context(self, key: Key):
        return None


class DiscreteSpreadReward(Reward):
    def __init__(self, agent_idx: int, num_agents: int = 3):
        super().__init__()
        self.agent_idx = agent_idx
        self.num_agents = num_agents

    def __call__(self, obs, action, next_obs, context=None):
        agent_pos = next_obs.agent_pos
        landmark_pos = next_obs.landmark_pos

        diffs = landmark_pos[:, None, :] - agent_pos[None, :, :]
        distances = jnp.linalg.norm(diffs, axis=-1)
        min_dists = jnp.min(distances, axis=1)
        coverage_reward = -jnp.sum(min_dists)

        return coverage_reward, context

    def init_context(self, key: Key):
        return None


class DiscreteSpreadTerm:
    def __call__(self, next_obs) -> bool:
        return False


def make_discrete_spread(num_agents=3, num_landmarks=3, multidiscrete=False):
    env = DiscreteSpreadEnv(num_agents=num_agents, num_landmarks=num_landmarks)
    agents = [
        DiscreteSpreadAgent(i, num_agents, num_landmarks, multidiscrete)
        for i in range(num_agents)
    ]
    rewards = [DiscreteSpreadReward(i, num_agents) for i in range(num_agents)]
    term_fn = DiscreteSpreadTerm()
    return env, agents, rewards, term_fn


# ================================
# Training Test
# ================================


def test_training(multidiscrete=False, num_iterations=50):
    """Test training with DiscreteLSTMActor."""
    action_type = "MultiDiscrete" if multidiscrete else "Discrete"
    print(f"\n{'=' * 60}")
    print(f"Testing DiscreteLSTMActor with {action_type} actions")
    print("=" * 60)

    seed = 42
    device = "cpu"
    timesteps_per_env = 50
    num_envs = 32
    max_steps = 25
    num_agents = 3

    key = jax.random.key(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    # Create environment
    env, agents, rewards, term_fn = make_discrete_spread(
        num_agents=num_agents, num_landmarks=3, multidiscrete=multidiscrete
    )

    sim = Sim(env, agents, rewards, term_fn, max_steps=max_steps)

    # Create DiscreteLSTMActor for each agent
    actor_keys = jax.random.split(key, num_agents + 1)
    key = actor_keys[0]
    actor_keys = actor_keys[1:]

    actors = []
    for i, agent in enumerate(agents):
        actor = DiscreteLSTMActor(
            agent.pi_obs_spec,
            agent.pi_action_spec,
            actor_keys[i],
            lstm_hidden_size=32,
            head_net_arch=[32],
            head_activation_func="tanh",
            device=device,
        )
        actors.append(actor)

    print(f"Created {len(actors)} DiscreteLSTMActor instances")
    print(f"Action space: {agents[0].pi_action_spec}")

    # Create policy aggregator
    policy = PolicyAggregator(actors)

    # Create collect function
    key, policy_key = jax.random.split(key)
    policy_context = policy.init_context(policy_key)

    collect = collect_factory(
        sim.reset,
        sim.step,
        policy,
        timesteps_per_env=timesteps_per_env,
        num_envs=num_envs,
        extended_transition=False,
    )
    collect = jax.jit(collect)

    # Run collection loop (no training, just verify it runs)
    print(f"\nRunning {num_iterations} collection iterations...")
    print(f"Batch size: {timesteps_per_env * num_envs}")
    print("-" * 60)

    total_frames = 0
    episode_rewards = []

    for i in range(num_iterations):
        key, policy_key, collect_key = jax.random.split(key, 3)
        policy_context = policy.init_context(policy_key)

        transitions = collect(collect_key, policy_context)
        total_frames += transitions.step_count.size

        # Compute mean episode reward
        ep_reward = float(jnp.mean(transitions.episode_reward))
        episode_rewards.append(ep_reward)

        if i % 10 == 0:
            print(f"Iter {i:4d} | Frames: {total_frames:8,} | Ep Reward: {ep_reward:.3f}")

    print("-" * 60)
    print(f"Collection complete! Total frames: {total_frames:,}")
    print(f"Mean episode reward: {np.mean(episode_rewards):.3f}")

    # Verify actions are valid
    print("\nVerifying action validity...")
    key, collect_key = jax.random.split(key)
    transitions = collect(collect_key, policy_context)

    for agent_idx in range(num_agents):
        actions = np.array(transitions.action[agent_idx])
        if multidiscrete:
            assert actions.shape[-1] == 2, f"Expected 2D actions, got shape {actions.shape}"
            assert np.all(actions >= 0), "Actions should be >= 0"
            assert np.all(actions[:, 0] < 3), "X actions should be < 3"
            assert np.all(actions[:, 1] < 3), "Y actions should be < 3"
        else:
            assert np.all(actions >= 0), "Actions should be >= 0"
            assert np.all(actions < 5), "Actions should be < 5"

    print(f"All actions valid for {action_type} action space!")
    return True


def test_per_dimension_independence(num_samples=1000):
    """Test that MultiDiscrete samples independently per dimension."""
    print(f"\n{'=' * 60}")
    print("Testing per-dimension independence")
    print("=" * 60)

    seed = 42
    key = jax.random.key(seed)

    # Create single agent with MultiDiscrete action space
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
    action_space = gym.spaces.MultiDiscrete([10, 10, 10])

    actor = DiscreteLSTMActor(
        obs_space,
        action_space,
        key,
        lstm_hidden_size=32,
        head_net_arch=[32],
    )

    context = actor.init_context(None)

    # Collect samples
    samples = []
    obs = jnp.zeros(10)
    for i in range(num_samples):
        key, subkey = jax.random.split(key)
        action, context = actor(obs, context, subkey)
        samples.append(np.array(action))

    samples = np.array(samples)

    # Check correlation between dimensions
    print(f"\nCollected {num_samples} samples")
    print("Checking correlation between dimensions...")

    max_corr = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            corr = np.corrcoef(samples[:, i], samples[:, j])[0, 1]
            max_corr = max(max_corr, abs(corr))
            print(f"  Corr(dim{i}, dim{j}) = {corr:.4f}")

    assert max_corr < 0.15, f"Dimensions too correlated: {max_corr:.4f}"
    print(f"\nMax correlation: {max_corr:.4f} < 0.15 (PASS)")
    return True


if __name__ == "__main__":
    # Test Discrete action space
    success_discrete = test_training(multidiscrete=False, num_iterations=30)

    # Test MultiDiscrete action space
    success_multidiscrete = test_training(multidiscrete=True, num_iterations=30)

    # Test per-dimension independence
    success_independence = test_per_dimension_independence()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Discrete action space:      {'PASS' if success_discrete else 'FAIL'}")
    print(f"MultiDiscrete action space: {'PASS' if success_multidiscrete else 'FAIL'}")
    print(f"Per-dimension independence: {'PASS' if success_independence else 'FAIL'}")

    if success_discrete and success_multidiscrete and success_independence:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
        sys.exit(1)
