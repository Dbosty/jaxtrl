"""PAP trained against the evader restricted to levels 1..--evader-k (fixed,
uniform), matching the pool-sweep RLBO runs' training distribution exactly
-- unlike the original PAP checkpoint (pap_40M_8k), which trained against
the FULL evader range (levels 1-8). Same architecture/PPO hyperparameters
and training budget as the original PAP run; the only difference is which
evader levels load_evader_levels() is truncated to.

Run with, e.g.:
    python envs/pgj/pap_evader_fixed.py \\
        level_k.K=8 level_k.checkpoint_dir=checkpoints/level_k_40M_8k \\
        collect.total_timesteps=160_000 seed=0 --evader-k 6
"""
import argparse
import sys

import _compat  # noqa: F401
import hydra
import jax
import numpy as np
import torch as th

from pgj.env import ODEnv
from pgj.jaxtrl.agent import IndEvader

from level_k import EDR, HLP, LLP, SIM_MAX_STEPS, save_actor, train_pursuer_stage
from pap import UniformLevelKEvaderPolicy, load_evader_levels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evader-k", type=int, default=6, help="fixed evader sampling range, levels 1..evader_k")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    evader_k = args.evader_k

    @hydra.main(version_base=None, config_path="../../conf", config_name="pap")
    def _main(cfg):
        if cfg.seed is None:
            cfg.seed = np.random.randint(0, 2**32 - 1)

        key = jax.random.key(cfg.seed)
        key, seed_key = jax.random.split(key)
        np_seed, th_seed = jax.random.randint(seed_key, (2,), 0, 2**16 - 1)
        np.random.seed(int(np_seed))
        th.manual_seed(int(th_seed))

        env = ODEnv()

        key, load_key = jax.random.split(key)
        evader_levels_full = load_evader_levels(cfg, env, load_key)  # FULL K, trained checkpoint size
        evader_levels = evader_levels_full[:evader_k]  # fixed range, matches pool-sweep RLBO runs
        evader_policy = UniformLevelKEvaderPolicy(evader_levels)

        checkpoint_dir = f"checkpoints/pap_evk{evader_k}"
        llp_path = f"{checkpoint_dir}/pap_llp.pt"
        hlp_path = f"{checkpoint_dir}/pap_hlp.pt"

        key, stage_key = jax.random.split(key)
        llp, hlp = train_pursuer_stage(
            cfg, env, IndEvader(EDR, LLP, HLP, SIM_MAX_STEPS, env.grid.shape), evader_policy, stage_key,
            f"pgj_pap_evk{evader_k}", llp_path, hlp_path,
        )

        save_actor(llp, llp_path)
        save_actor(hlp, hlp_path)

    _main()


if __name__ == "__main__":
    main()
