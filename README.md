# jaxtrl

Reinforcement learning library for JAX-based environments with PyTorch / TorchRL training. Supports single-agent and multi-agent algorithms.

Personal research library built around one idea: collect rollouts in JAX (jit-compiled, vectorized env stepping) while keeping the training loop in PyTorch / TorchRL. Policies exist as paired Flax/Torch modules with weights synced between them, so you get JAX's simulation throughput without giving up TorchRL's loss modules, tensordict ergonomics, or the PyTorch debugging experience. Built for my own experiments (MPE, custom grid envs) rather than as a general framework — shared in case the JAX-collection / Torch-training bridge is useful to others.

## Algorithms

- **PPO** - Proximal Policy Optimization
- **IPPO** - Independent PPO (multi-agent)
- **MAPPO** - Multi-Agent PPO (shared critic)
- **HAPPO** - Heterogeneous-Agent PPO (importance-sampling weighted MAPPO)

Actors: MLP and LSTM variants for continuous (`TanhNormal`) and discrete (`Categorical` / `MultiCategorical`) action spaces.

## Installation

```bash
pip install -e .
```

`torch` and `jax` are not pinned as dependencies — install the builds (CUDA / CPU / Metal) that match your platform first.

## Examples

Runnable end-to-end scripts live in `envs/`:

- `envs/cartpole/ppo.py` - single-agent PPO on CartPole
- `envs/mpe/simple_spread.py` - HAPPO on MPE simple spread
- `envs/pgj/` - PPO / IPPO / MAPPO / HAPPO on a custom grid environment

Configuration is handled with Hydra; see `conf/`.

## Usage

### 1. Define your simulation

```python
from jaxtrl.sim import Sim
from jaxtrl.sim.collect import collect_factory
from jaxtrl.sim.policy import PolicyAggregator

# Create simulation with env, agents, rewards, and termination function
sim = Sim(env, agents, reward_fns, is_terminated)

# Create collector
collect = collect_factory(
    sim.reset,
    sim.step,
    policy,
    timesteps_per_env=128,
    num_envs=64,
)
collect = jax.jit(collect)
```

### 2. Single-agent (PPO)

```python
from jaxtrl.rl.ppo.trainer import PPOTrainer
from jaxtrl.utils.tensordict import transitions_to_tensordicts

trainer = PPOTrainer(agent, cfg.train, key, device="cuda")

# Training loop
transitions = collect(key, policy_context)
tds = transitions_to_tensordicts(transitions)
log_dict = trainer.step(tds[0])
```

### 3. Multi-agent (IPPO/MAPPO/HAPPO)

```python
from jaxtrl.marl.ippo import IPPOTrainer  # or MAPPOTrainer, HAPPOTrainer
from jaxtrl.utils.tensordict import transitions_to_ma_tensordict

trainer = IPPOTrainer(agents, cfg.train, key, device="cuda")

# Aggregate policies
policy = PolicyAggregator([*trainer.policies, other_policy])

# Training loop
transitions = collect(key, policy_context)
ma_td = transitions_to_ma_tensordict(transitions)
log_dict = trainer.step(ma_td)
```

## Logging

```python
from jaxtrl.utils.logger import ExtWandbLogger

logger = ExtWandbLogger(exp_name="experiment", **wandb_cfg)
logger.log_scalars(log_dict, timestep=total_frames)
```
