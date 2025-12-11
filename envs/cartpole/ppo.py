"""
CartPole PPO test using gymnax environment.

Uses the existing jaxtrl Sim, collect_factory, and transitions_to_tensordicts.
"""

import gymnasium as gym
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import torch as th
from flax import struct
from jaxtyping import Array, Key

from jaxtrl.rl.ppo.trainer import PPOTrainer
from jaxtrl.sim import Sim
from jaxtrl.sim.collect import collect_factory
from jaxtrl.sim.env import Agent, Env, Reward
from jaxtrl.sim.policy import PolicyAggregator
from jaxtrl.utils.tensordict import transitions_to_tensordicts


# ================================
# CartPole Observation (simple structure)
# ================================


@struct.dataclass
class CartPoleObs:
    """Observation wrapper for CartPole."""
    obs: Array  # The actual observation (4,) - [x, x_dot, theta, theta_dot]
    state: gymnax.environments.environment.EnvState  # Gymnax env state directly
    timestep: int
    done: bool = False  # Terminal flag from gymnax (needed because gymnax auto-resets)


# ================================
# CartPole Environment (matches Env interface)
# ================================


class CartPoleEnv(Env):
    """
    Gymnax CartPole wrapped to match jaxtrl Env interface.
    """

    def __init__(self):
        self.env, self.env_params = gymnax.make("CartPole-v1")

    def reset(self, key: Key) -> CartPoleObs:
        obs, state = self.env.reset(key, self.env_params)
        return CartPoleObs(obs=obs, state=state, timestep=0)

    def step(self, obs: CartPoleObs, action: Array) -> CartPoleObs:
        # Action comes as (1, 1) from vstack, extract scalar
        action_scalar = action[0, 0].astype(jnp.int32)

        next_obs_array, next_state, reward, done, info = self.env.step(
            key=jax.random.key(0),  # CartPole step is deterministic
            state=obs.state,
            action=action_scalar,
            params=self.env_params,
        )

        return CartPoleObs(
            obs=next_obs_array,
            state=next_state,
            timestep=obs.timestep + 1,
            done=done,  # Store gymnax done flag (gymnax auto-resets so we can't detect from obs)
        )


# ================================
# CartPole Agent (matches Agent interface)
# ================================


class CartPoleAgent(Agent):
    """
    Agent wrapper for CartPole.

    Continuous action space [-1, 1] discretized to {0, 1}.
    """

    def __init__(self):
        super().__init__(sub_idx=0)
        self._obs_spec = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        # Continuous action that gets discretized
        self._action_spec = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

    @property
    def pi_obs_spec(self):
        return self._obs_spec

    @property
    def pi_action_spec(self):
        return self._action_spec

    def obs_env2pi(self, obs: CartPoleObs, context=None):
        """Extract observation array for policy."""
        return obs.obs, context

    def action_pi2env(self, action: Array, context=None):
        """Convert continuous [-1, 1] action to discrete {0, 1}."""
        discrete_action = (action > 0).astype(jnp.int32)
        return discrete_action, context

    def init_context(self, key: Key):
        return None


# ================================
# CartPole Reward (matches Reward interface)
# ================================


class CartPoleReward(Reward):
    """
    Reward wrapper for CartPole.

    CartPole gives +1 reward per step survived, 0 on termination.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, obs: CartPoleObs, action, next_obs: CartPoleObs, context=None):
        # Use done flag stored in next_obs (gymnax auto-resets so can't detect from values)
        reward = jnp.where(next_obs.done, 0.0, 1.0)
        return reward, context

    def init_context(self, key: Key):
        return None


# ================================
# CartPole Termination (for Sim interface)
# ================================


class CartPoleTerm:
    """
    Termination condition for CartPole.

    Returns True only for actual terminations (pole fell), not time-limit truncations.
    Gymnax's done flag is True for both, so we distinguish by checking timestep.
    """

    def __init__(self, max_episode_steps: int = 500):
        self.max_episode_steps = max_episode_steps

    def __call__(self, next_obs: CartPoleObs) -> bool:
        # Gymnax done=True means either pole fell OR hit time limit
        # Only count as terminated if done AND not at time limit
        is_time_limit = next_obs.timestep >= self.max_episode_steps
        return jnp.logical_and(next_obs.done, jnp.logical_not(is_time_limit))


# ================================
# Main Training Loop
# ================================


def main():
    # Config
    seed = 42
    device = "cuda" if th.cuda.is_available() else "cpu"
    total_timesteps = 500_000
    timesteps_per_env = 500
    num_envs = 10
    max_steps = 500

    print(f"Device: {device}")
    print(f"Total timesteps: {total_timesteps}")

    # Set seeds
    key = jax.random.key(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    # Create environment, agent, reward, and termination condition
    env = CartPoleEnv()
    agent = CartPoleAgent()
    reward_fn = CartPoleReward()
    term_fn = CartPoleTerm()

    # Create Sim with single agent
    sim = Sim(env, agents=[agent], reward=[reward_fn], term=term_fn, max_steps=max_steps)

    # Create trainer config
    train_config = {
        "actor": "mlp",
        "critic": "mlp",
        "actor_kwargs": {
            "net_arch": [64, 64],
            "activation_func": "tanh",
            "distribution": "normal",
            "distribution_kwargs": {},
        },
        "critic_kwargs": {
            "net_arch": [64, 64],
            "activation_func": "tanh",
        },
        "rl_kwargs": {
            "entropy_eps": 0.01,
            "num_epochs": 10,
            "sub_batch_size": 256,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "max_grad_norm": 0.5,
            "critic_coef": 0.5,
            "lr": 3e-4,
            "loss_critic_type": "l2",
        },
    }

    key, trainer_key = jax.random.split(key)

    # Create PPO trainer
    trainer = PPOTrainer(agent, train_config, trainer_key, device=device)

    # Create policy aggregator (even for single agent, to match collect_factory interface)
    policy = PolicyAggregator([trainer.policy])

    # Create collect function using jaxtrl's collect_factory
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

    # Training loop
    total_frames = 0
    batch_size = timesteps_per_env * num_envs
    num_iterations = total_timesteps // batch_size + 1

    # Track statistics for plotting
    history = {
        "frames": [],
        "ep_rew_mean": [],
        "ep_len_mean": [],
        "ep_len_max": [],
        "loss": [],
        "loss_objective": [],
        "loss_critic": [],
        "loss_entropy": [],
        "entropy": [],
        "kl_approx": [],
        "clip_fraction": [],
        "explained_variance": [],
        "lr": [],
    }

    print(f"\nStarting training for {num_iterations} iterations...")
    print(f"Batch size: {batch_size}")
    print("-" * 50)

    for i in range(num_iterations):
        key, subkey = jax.random.split(key)

        # Update policy context with latest trained weights (PyTorch -> Flax)
        policy_context = policy.init_context(subkey)

        # Collect transitions using jaxtrl's collect_factory
        transitions = collect(subkey, policy_context)

        # Convert to tensordicts using jaxtrl's utility
        tds = transitions_to_tensordicts(transitions)

        total_frames += batch_size

        # Train on first (and only) agent's tensordict
        log_dict = trainer.step(tds[0])

        # Calculate max episode length (excluding partial episodes at collection boundary)
        done = tds[0]["next", "done"]
        step_counts = tds[0]["next", "step_count"][done]
        full_eps = step_counts[step_counts > 10]  # Exclude partial episodes
        max_ep_len = full_eps.max().item() if len(full_eps) > 0 else 0

        # Store statistics
        history["frames"].append(total_frames)
        history["ep_rew_mean"].append(log_dict.get("rollout/ep_rew_mean", 0))
        history["ep_len_mean"].append(log_dict.get("rollout/ep_len_mean", 0))
        history["ep_len_max"].append(max_ep_len)
        history["loss"].append(log_dict.get("train/loss", 0))
        history["loss_objective"].append(log_dict.get("train/loss_objective", 0))
        history["loss_critic"].append(log_dict.get("train/loss_critic", 0))
        history["loss_entropy"].append(log_dict.get("train/loss_entropy", 0))
        history["entropy"].append(log_dict.get("train/entropy", 0))
        history["kl_approx"].append(log_dict.get("train/kl_approx", 0))
        history["clip_fraction"].append(log_dict.get("train/clip_fraction", 0))
        history["explained_variance"].append(log_dict.get("train/explained_variance", 0))
        history["lr"].append(log_dict.get("train/lr", 0))

        # Log progress
        if i % 10 == 0:
            ep_rew = log_dict.get("rollout/ep_rew_mean", 0)
            ep_len = log_dict.get("rollout/ep_len_mean", 0)
            loss = log_dict.get("train/loss", 0)

            print(
                f"Iter {i:4d} | Frames: {total_frames:8d} | "
                f"Ep Rew: {ep_rew:7.2f} | Ep Len: {ep_len:6.1f} | "
                f"Max Ep: {max_ep_len:5.0f} | Loss: {loss:.4f}"
            )

    print("-" * 50)
    print("Training complete!")

    # Plot statistics
    plot_training_stats(history)


def plot_training_stats(history):
    """Plot training statistics from log_dict history."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle("CartPole PPO Training Statistics", fontsize=14, fontweight="bold")

    frames = np.array(history["frames"]) / 1000  # Convert to thousands

    # Row 1: Rewards and Episode Length
    ax = axes[0, 0]
    ax.plot(frames, history["ep_len_max"], color="tab:blue", linewidth=1.5, label="Max Ep Len")
    ax.set_xlabel("Frames (k)")
    ax.set_ylabel("Episode Length")
    ax.set_title("Max Episode Length (Solved)")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=500, color="tab:green", linestyle="--", alpha=0.7, label="Max (500)")
    ax.set_ylim(0, 550)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(frames, history["ep_rew_mean"], color="tab:orange", linewidth=1.5)
    ax.set_xlabel("Frames (k)")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Mean Episode Reward")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(frames, history["loss"], color="tab:red", linewidth=1.5)
    ax.set_xlabel("Frames (k)")
    ax.set_ylabel("Total Loss")
    ax.set_title("Total Loss")
    ax.grid(True, alpha=0.3)

    # Row 2: Loss Components
    ax = axes[1, 0]
    ax.plot(frames, history["loss_objective"], color="tab:purple", linewidth=1.5)
    ax.set_xlabel("Frames (k)")
    ax.set_ylabel("Policy Loss")
    ax.set_title("Policy (Objective) Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(frames, history["loss_critic"], color="tab:brown", linewidth=1.5)
    ax.set_xlabel("Frames (k)")
    ax.set_ylabel("Critic Loss")
    ax.set_title("Critic (Value) Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(frames, history["entropy"], color="tab:cyan", linewidth=1.5)
    ax.set_xlabel("Frames (k)")
    ax.set_ylabel("Entropy")
    ax.set_title("Policy Entropy")
    ax.grid(True, alpha=0.3)

    # Row 3: PPO Diagnostics
    ax = axes[2, 0]
    ax.plot(frames, history["kl_approx"], color="tab:olive", linewidth=1.5)
    ax.set_xlabel("Frames (k)")
    ax.set_ylabel("KL Divergence")
    ax.set_title("Approx KL Divergence")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(frames, history["clip_fraction"], color="tab:pink", linewidth=1.5)
    ax.set_xlabel("Frames (k)")
    ax.set_ylabel("Clip Fraction")
    ax.set_title("Clip Fraction")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    ax.plot(frames, history["explained_variance"], color="tab:gray", linewidth=1.5)
    ax.set_xlabel("Frames (k)")
    ax.set_ylabel("Explained Variance")
    ax.set_title("Explained Variance")
    ax.axhline(y=1.0, color="tab:green", linestyle="--", alpha=0.7, label="Optimal (1.0)")
    ax.axhline(y=0.0, color="tab:red", linestyle="--", alpha=0.7, label="Random (0.0)")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("envs/cartpole/ppo_training_stats.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: envs/cartpole/ppo_training_stats.png")
    plt.show()


if __name__ == "__main__":
    main()
