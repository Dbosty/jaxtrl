# jaxtrl

Reinforcement learning library for JAX-based environments with PyTorch training. Supports single-agent and multi-agent algorithms.

## Algorithms

- **PPO** - Proximal Policy Optimization
- **IPPO** - Independent PPO (multi-agent)
- **MAPPO** - Multi-Agent PPO (shared critic)
- **HAPPO** - Heterogeneous-Agent PPO (importance-sampling weighted MAPPO)

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
