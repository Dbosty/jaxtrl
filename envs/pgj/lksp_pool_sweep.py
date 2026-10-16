"""New experiment: does RLBO's achievable performance against a FIXED
evader rationality range increase as we grow the PURSUER expert pool it's
allowed to orchestrate over -- even when that pool stays smaller than the
evader's own max rationality level?

Normal lksp.py training couples pursuer pool size and evader sampling range
to the same cfg.level_k.K (meta-controller choice space = K+1 pursuer
levels, opponent = evader levels 1..K). This script decouples them: the
evader ALWAYS samples uniformly from levels 1..--evader-k (fixed across
every run in the sweep), while the pursuer's available expert pool is
truncated to the first --pool-size levels (pi_1..pi_{pool_size} in the
paper's Section III.B notation -- our 0-indexed pursuer_levels[:pool_size]).
Everything else (architecture, PPO hyperparameters, training budget) is
identical to the real RLBO training run, for a fair comparison across pool
sizes.

If RLBO's final/training performance improves as pool_size grows from 2 up
toward evader_k, even while pool_size < evader_k (pursuer pool strictly
smaller than the evader's rationality range), that supports the claim that
dynamic orchestration synthesizes effective rationality beyond what any
single pretrained pursuer policy in the pool provides on its own.

Run with, e.g. (one call per pool size, 2..6):
    python envs/pgj/lksp_pool_sweep.py \\
        level_k.K=8 level_k.checkpoint_dir=checkpoints/level_k_40M_8k \\
        collect.total_timesteps=40_000_000 \\
        --pool-size 2 --evader-k 6
"""

import argparse
import sys

import gymnasium as gym
import _compat  # noqa: F401
import hydra
import jax
import numpy as np
import torch as th
from tensordict import TensorDict

from pgj.env import ODEnv
from pgj.jaxtrl.agent import IndEvader
from pgj.jaxtrl.reward import SparseEvader, SparsePursuer
from pgj.jaxtrl.term import is_terminated

from jaxtrl.rl.ppo.trainer import PPOTrainer
from jaxtrl.sim import Sim
from jaxtrl.utils.tensordict import tdl
from jaxtrl.sim.policy import PolicyAggregator

from level_k import EDR, HLP, LLP, SIM_MAX_STEPS, run_stage, save_actor
from lksp import LKSPPolicy, MetaActorSpec, TeamPursuerAgent, load_pursuer_levels, meta_tensordict
from pap import UniformLevelKEvaderPolicy, load_evader_levels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-size", type=int, required=True, help="number of pursuer experts RLBO may choose from (pi_1..pi_{pool_size})")
    parser.add_argument("--evader-k", type=int, default=6, help="fixed evader sampling range, levels 1..evader_k, held constant across the sweep")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    pool_size = args.pool_size
    evader_k = args.evader_k

    @hydra.main(version_base=None, config_path="../../conf", config_name="lksp")
    def _main(cfg):
        if cfg.seed is None:
            cfg.seed = np.random.randint(0, 2**32 - 1)

        key = jax.random.key(cfg.seed)
        key, seed_key = jax.random.split(key)
        np_seed, th_seed = jax.random.randint(seed_key, (2,), 0, 2**16 - 1)
        np.random.seed(int(np_seed))
        th.manual_seed(int(th_seed))

        env = ODEnv()

        key, pursuer_key, evader_key = jax.random.split(key, 3)
        pursuer_levels_full = load_pursuer_levels(cfg, env, pursuer_key)  # FULL K+1, trained checkpoint size
        evader_levels_full = load_evader_levels(cfg, env, evader_key)  # FULL K

        pursuer_levels = pursuer_levels_full[:pool_size]  # pi_1 .. pi_{pool_size}
        evader_levels = evader_levels_full[:evader_k]  # fixed evader range, same across every pool_size in the sweep
        evader_policy = UniformLevelKEvaderPolicy(evader_levels)

        team_agent = TeamPursuerAgent(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape)
        edr_agent = IndEvader(EDR, LLP, HLP, SIM_MAX_STEPS, env.grid.shape)

        meta_spec = MetaActorSpec(team_agent.pi_obs_spec, pool_size)  # action space = pool_size, NOT K+1

        key, trainer_key, run_key = jax.random.split(key, 3)
        trainer = PPOTrainer(meta_spec, cfg.train_meta, trainer_key, device=cfg.device)

        lksp_policy = LKSPPolicy(trainer.policy, pursuer_levels)
        policy = PolicyAggregator([lksp_policy, evader_policy])

        sim = Sim(
            env,
            [team_agent, edr_agent],
            [SparsePursuer(LLP, HLP, EDR, 1, 0), SparseEvader(EDR, LLP, HLP, 1, 0)],
            is_terminated,
            max_steps=SIM_MAX_STEPS,
        )

        def train_step(transitions):
            return trainer.step(meta_tensordict(transitions, 0))

        exp_name = f"pgj_lksp_poolsweep_n{pool_size}_evk{evader_k}"
        checkpoint_dir = f"checkpoints/lksp_poolsweep_n{pool_size}_evk{evader_k}"
        meta_path = f"{checkpoint_dir}/lksp_meta.pt"
        run_stage(
            cfg, sim, policy, trainer, train_step, exp_name, run_key,
            save_targets={meta_path: trainer.policy}, best_metric="rollout/pursuer_winrate", best_mode="max",
        )

        save_actor(trainer.policy, meta_path)

    _main()


if __name__ == "__main__":
    main()
