"""Adds a "Random" bar to an already-computed fig6_K{L}.png -- the paper's
TDO-random "classifier" (U([n])), used standalone as a policy-type baseline:
at every timestep, picks a pursuer level uniformly at random from the first
`--num-levels` frozen levels, evaluated against the same mixed evader-level
distribution PAP/MPO/CBO-DAgger/CBO-Offline already were.

Reuses PAP/MPO/CBO-DAgger/CBO-Offline's numbers from the existing
fig6_K{L}.txt instead of re-simulating them -- only Random's rollout is new.

Run with, e.g.:
    python envs/pgj/add_random_to_fig6.py \\
        level_k.K=8 level_k.checkpoint_dir=checkpoints/level_k_40M_8k \\
        --existing figures_acc/fig6_K8.txt --num-levels 9 \\
        --out figures_acc/fig6_K8.png
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
from pgj.jaxtrl.reward import SparseEvader, SparsePursuer

from lksp import TeamPursuerAgent, load_pursuer_levels
from pap import UniformLevelKEvaderPolicy, load_evader_levels

from level_k import EDR, HLP, LLP, SIM_MAX_STEPS
from evaluate import rollout
from evaluate_acc import RandomDispatchPolicy, discounted_reward, plot_figure_6
from plot_utils import save_values_txt
from jaxtrl.sim.policy import PolicyAggregator


def read_existing(path):
    results = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, value = line.split("\t")
            results[name] = float(value)
    return results


def main():
    # Hydra's @hydra.main consumes all argv as its own overrides, so parse
    # our own flags out first and leave the rest (the level_k.K=... style
    # overrides) for hydra.
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing", required=True, help="path to the existing fig6_K{L}.txt to reuse")
    parser.add_argument("--num-levels", type=int, required=True, help="L+1 -- how many of the first pursuer levels Random draws from")
    parser.add_argument("--out", required=True, help="output .png path (paired .txt written alongside)")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    @hydra.main(version_base=None, config_path="../../conf", config_name="evaluate")
    def _main(cfg):
        np.random.seed(cfg.seed)
        th.manual_seed(cfg.seed)
        key = jax.random.key(cfg.seed)
        env = ODEnv()

        key, k1, k2 = jax.random.split(key, 3)
        pursuer_levels = load_pursuer_levels(cfg, env, k1)
        evader_levels = load_evader_levels(cfg, env, k2)[: args.num_levels - 1]

        evader_policy = UniformLevelKEvaderPolicy(evader_levels)
        evader_agent = IndEvader(EDR, LLP, HLP, SIM_MAX_STEPS, env.grid.shape)
        team_agent = TeamPursuerAgent(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape)
        agents2 = [team_agent, evader_agent]
        reward2 = [SparsePursuer(LLP, HLP, EDR, 1, 0), SparseEvader(EDR, LLP, HLP, 1, 0)]

        random_policy = RandomDispatchPolicy(pursuer_levels, num_levels=args.num_levels)
        policy2 = PolicyAggregator([random_policy, evader_policy])
        key, rk = jax.random.split(key)
        transitions = rollout(env, agents2, reward2, policy2, cfg.eval.num_envs, rk)
        random_reward = discounted_reward(transitions, agent_idx=0)
        print(f"Random -> expected_discounted_reward={random_reward:.4f}")

        results = read_existing(args.existing)
        results["Random"] = random_reward

        plot_figure_6(results, args.out)
        txt_path = args.out.rsplit(".", 1)[0] + ".txt"
        save_values_txt(txt_path, expected_discounted_reward=results)
        print(f"wrote {args.out}")
        print(f"wrote {txt_path}")

    _main()


if __name__ == "__main__":
    main()
