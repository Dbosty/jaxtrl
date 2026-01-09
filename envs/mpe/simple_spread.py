"""
MPE Simple Spread HAPPO training script.

Usage:
    python envs/mpe/simple_spread.py
"""

import os
import sys

# Add envs/mpe to path for imports when running from project root
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import jax
import jax.numpy as jnp
import numpy as np
import torch as th

from simple_spread_env import make_simple_spread
from jaxtrl.marl.happo import HAPPOTrainer
from jaxtrl.sim import Sim
from jaxtrl.sim.collect import collect_factory
from jaxtrl.sim.policy import PolicyAggregator
from jaxtrl.utils.tensordict import transitions_to_ma_tensordict


def main():
    # Configuration
    seed = 42
    device = "cuda" if th.cuda.is_available() else "cpu"
    total_timesteps = 1_000_000
    timesteps_per_env = 100
    num_envs = 100
    max_steps = 25  # Standard for Simple Spread

    print("=" * 60)
    print("MPE Simple Spread with HAPPO")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Timesteps per env: {timesteps_per_env}")
    print(f"Num envs: {num_envs}")
    print(f"Max steps per episode: {max_steps}")

    # Set seeds
    key = jax.random.key(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    # Create environment components using factory
    env, agents, rewards, term_fn = make_simple_spread(
        num_agents=3,
        num_landmarks=3,
    )

    # Create Sim
    sim = Sim(env, agents, rewards, term_fn, max_steps=max_steps)

    # Training config for HAPPO
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
            "sub_batch_size": 2048,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "max_grad_norm": 0.5,
            "critic_coef": 0.5,
            "lr": 3e-4,
            "loss_critic_type": "l2",
        },
    }

    # Create HAPPO trainer
    key, trainer_key = jax.random.split(key)
    trainer = HAPPOTrainer(agents, train_config, trainer_key, device=device)

    # Create policy aggregator
    policy = PolicyAggregator(trainer.policies)

    # Create collect function with extended_transition=True (required for HAPPO)
    key, policy_key = jax.random.split(key)
    policy_context = policy.init_context(policy_key)

    collect = collect_factory(
        sim.reset,
        sim.step,
        policy,
        timesteps_per_env=timesteps_per_env,
        num_envs=num_envs,
        extended_transition=True,
    )
    collect = jax.jit(collect)

    # Training statistics
    history = {
        "frames": [],
        "ep_rew_mean": [],
        "ep_len_mean": [],
        "loss_objective": [],
        "loss_critic": [],
        "loss_entropy": [],
        "avg_kappa_0": [],
        "avg_kappa_1": [],
        "avg_kappa_2": [],
    }

    # Training loop
    total_frames = 0
    batch_size = timesteps_per_env * num_envs
    num_iterations = total_timesteps // batch_size + 1

    print(f"\nStarting training for {num_iterations} iterations...")
    print(f"Batch size: {batch_size:,}")
    print("-" * 60)

    for i in range(num_iterations):
        key, policy_key, collect_key = jax.random.split(key, 3)

        # Update policy context with latest weights
        policy_context = policy.init_context(policy_key)

        # Collect transitions
        transitions = collect(collect_key, policy_context)

        # Convert to multi-agent tensordict
        ma_td = transitions_to_ma_tensordict(transitions)

        total_frames += transitions.step_count.size

        # Train
        log_dict = trainer.step(ma_td)

        # Log statistics
        history["frames"].append(total_frames)
        history["ep_rew_mean"].append(log_dict.get("rollout/ep_rew_mean", 0))
        history["ep_len_mean"].append(log_dict.get("rollout/ep_len_mean", 0))
        history["loss_objective"].append(
            log_dict.get("train/agent_0/loss_objective", 0)
        )
        history["loss_critic"].append(log_dict.get("train/critic/loss_critic", 0))
        history["loss_entropy"].append(log_dict.get("train/agent_0/loss_entropy", 0))
        history["avg_kappa_0"].append(log_dict.get("train/agent_0/avg_kappa", 1.0))
        history["avg_kappa_1"].append(log_dict.get("train/agent_1/avg_kappa", 1.0))
        history["avg_kappa_2"].append(log_dict.get("train/agent_2/avg_kappa", 1.0))

        # Console logging
        if i % 10 == 0:
            ep_rew = log_dict.get("rollout/ep_rew_mean", 0)
            ep_len = log_dict.get("rollout/ep_len_mean", 0)
            loss_obj = log_dict.get("train/agent_0/loss_objective", 0)
            loss_crit = log_dict.get("train/critic/loss_critic", 0)
            kappa_0 = log_dict.get("train/agent_0/avg_kappa", 1.0)
            kappa_1 = log_dict.get("train/agent_1/avg_kappa", 1.0)
            kappa_2 = log_dict.get("train/agent_2/avg_kappa", 1.0)

            print(
                f"Iter {i:4d} | Frames: {total_frames:8,} | "
                f"Ep Rew: {ep_rew:8.3f} | Ep Len: {ep_len:5.1f} | "
                f"Loss: {loss_obj:.4f} | Crit: {loss_crit:.4f} | "
                f"Kappa: [{kappa_0:.3f}, {kappa_1:.3f}, {kappa_2:.3f}]"
            )

    print("-" * 60)
    print("Training complete!")

    # Plot and save statistics
    plot_training_stats(history)


def plot_training_stats(history):
    """Plot training statistics and save to PNG."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        "MPE Simple Spread HAPPO Training Statistics", fontsize=14, fontweight="bold"
    )

    frames = np.array(history["frames"]) / 1000  # Convert to thousands

    # Plot 1: Episode Reward
    ax = axes[0, 0]
    ax.plot(frames, history["ep_rew_mean"], color="tab:blue", linewidth=1.5)
    ax.set_xlabel("Frames (k)")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Mean Episode Reward (Coverage)")
    ax.grid(True, alpha=0.3)

    # Plot 2: Episode Length
    ax = axes[0, 1]
    ax.plot(frames, history["ep_len_mean"], color="tab:green", linewidth=1.5)
    ax.set_xlabel("Frames (k)")
    ax.set_ylabel("Episode Length")
    ax.set_title("Mean Episode Length")
    ax.axhline(y=25, color="tab:red", linestyle="--", alpha=0.7, label="Max (25)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Policy Loss
    ax = axes[0, 2]
    ax.plot(frames, history["loss_objective"], color="tab:purple", linewidth=1.5)
    ax.set_xlabel("Frames (k)")
    ax.set_ylabel("Policy Loss")
    ax.set_title("Policy (Objective) Loss")
    ax.grid(True, alpha=0.3)

    # Plot 4: Critic Loss
    ax = axes[1, 0]
    ax.plot(frames, history["loss_critic"], color="tab:red", linewidth=1.5)
    ax.set_xlabel("Frames (k)")
    ax.set_ylabel("Critic Loss")
    ax.set_title("Shared Critic Loss")
    ax.grid(True, alpha=0.3)

    # Plot 5: Entropy Loss
    ax = axes[1, 1]
    ax.plot(frames, history["loss_entropy"], color="tab:cyan", linewidth=1.5)
    ax.set_xlabel("Frames (k)")
    ax.set_ylabel("Entropy Loss")
    ax.set_title("Entropy Loss")
    ax.grid(True, alpha=0.3)

    # Plot 6: Kappa values (HAPPO-specific)
    ax = axes[1, 2]
    ax.plot(frames, history["avg_kappa_0"], label="Agent 0", linewidth=1.5)
    ax.plot(frames, history["avg_kappa_1"], label="Agent 1", linewidth=1.5)
    ax.plot(frames, history["avg_kappa_2"], label="Agent 2", linewidth=1.5)
    ax.set_xlabel("Frames (k)")
    ax.set_ylabel("Avg Kappa")
    ax.set_title("HAPPO Advantage Scaling (Kappa)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "envs/mpe/simple_spread_training_stats.png", dpi=150, bbox_inches="tight"
    )
    print(f"\nPlot saved to: envs/mpe/simple_spread_training_stats.png")
    plt.show()


if __name__ == "__main__":
    main()
