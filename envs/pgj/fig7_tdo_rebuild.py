"""Rebuilds fig7 (TDO: Expected Discounted Reward vs TDO Delay) per today's
revision list, item 5:
  - legend relabeled TDO-DAgger / TDO-Offline (reusing the CBO-DAgger/
    CBO-Offline sweeps already computed in figures_acc/fig10_K5.txt --
    identical underlying TDOPolicy computation, TDO-* is just the correct
    name for it: it's a TDO instance parameterized by that classifier, not
    the classifier deployed standalone).
  - adds TDO-Random (the paper's U([n]) pre-oracle "classifier"), computed
    fresh here.
  - X axis capped at delay=40 (still computes/saves the full delay grid
    including 60/100 to the .txt, just doesn't plot past 40).
  - adds DFO / CBO-DAgger / CBO-Offline's own (non-TDO, always-deployed)
    performance as static dotted reference lines, reusing figures_acc/
    fig6_K5.txt's already-computed numbers.

Run with, e.g.:
    python envs/pgj/fig7_tdo_rebuild.py \\
        level_k.K=8 level_k.checkpoint_dir=checkpoints/level_k_40M_8k \\
        pap.checkpoint_dir=checkpoints/pap_40M_8k \\
        lksp.checkpoint_dir=checkpoints/lksp_40M_8k \\
        dagger.checkpoint_dir=checkpoints/dagger_40M_8k \\
        offline.checkpoint_dir=checkpoints/offline_40M_8k \\
        eval.num_envs=500 --view-k 5 --out figures_acc/fig7.png
"""

import argparse
import copy
import logging
import sys

import _compat  # noqa: F401
import hydra
import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as th

from pgj.env import ODEnv
from pgj.jaxtrl.agent import IndEvader
from pgj.jaxtrl.reward import SparseEvader, SparsePursuer

from lksp import TeamPursuerAgent, load_pursuer_levels
from pap import load_evader_levels

from level_k import EDR, HLP, LLP, SIM_MAX_STEPS
from evaluate import rollout
from evaluate_acc import RandomMetaActor, TDOPolicy, discounted_reward
from plot_utils import ALGO_COLORS, save_values_txt
from jaxtrl.sim.policy import PolicyAggregator


def read_series(path, name):
    with open(path) as f:
        lines = f.read().split("\n\n")
    for block in lines:
        block_lines = block.strip().split("\n")
        if not block_lines or not block_lines[0].startswith(f"# {name}"):
            continue
        xs, ys = [], []
        for line in block_lines[1:]:
            x, y = line.split("\t")
            xs.append(float(x))
            ys.append(float(y))
        return xs, ys
    raise ValueError(f"series {name!r} not found in {path}")


def read_scalar(path, name):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(name + "\t"):
                return float(line.split("\t")[1])
    raise ValueError(f"{name!r} not found in {path}")


def view_cfg(cfg, L):
    vc = copy.deepcopy(cfg)
    vc.level_k.K = L
    return vc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view-k", type=int, required=True, help="K view to evaluate at (reporting base, e.g. 5)")
    parser.add_argument("--fig10-txt", default=None, help="existing fig10_K{view_k}.txt to reuse CBO-DAgger/CBO-Offline TDO sweeps from")
    parser.add_argument("--fig6-txt", default=None, help="existing fig6_K{view_k}.txt to reuse PAP/MPO/CBO-DAgger/CBO-Offline static values from")
    parser.add_argument("--out", required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    L = args.view_k
    fig10_txt = args.fig10_txt or f"figures_acc/fig10_K{L}.txt"
    fig6_txt = args.fig6_txt or f"figures_acc/fig6_K{L}.txt"

    @hydra.main(version_base=None, config_path="../../conf", config_name="evaluate")
    def _main(cfg):
        np.random.seed(cfg.seed)
        th.manual_seed(cfg.seed)
        key = jax.random.key(cfg.seed)
        env = ODEnv()
        vc = view_cfg(cfg, L)

        key, k1, k2 = jax.random.split(key, 3)
        pursuer_levels = load_pursuer_levels(cfg, env, k1)
        evader_levels = load_evader_levels(cfg, env, k2)[:L]
        sub_levels = pursuer_levels[1:]

        delays = list(cfg.eval.tdo_delays)
        random_meta = RandomMetaActor(num_levels=L)

        random_by_delay = []
        for delay in delays:
            level_returns = []
            for k in range(1, L + 1):
                agents = [TeamPursuerAgent(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape), IndEvader(EDR, LLP, HLP, SIM_MAX_STEPS, env.grid.shape)]
                reward = [SparsePursuer(LLP, HLP, EDR, 1, 0), SparseEvader(EDR, LLP, HLP, 1, 0)]
                tdo_policy = TDOPolicy(random_meta, sub_levels, delay, k - 1)
                policy = PolicyAggregator([tdo_policy, evader_levels[k - 1]])
                key, rk = jax.random.split(key)
                transitions = rollout(env, agents, reward, policy, cfg.eval.num_envs, rk)
                level_returns.append(discounted_reward(transitions, agent_idx=0))
            mean_dr = float(np.mean(level_returns))
            random_by_delay.append(mean_dr)
            print(f"TDO-Random delay={delay} -> expected_discounted_reward={mean_dr:.4f}")

        dagger_delays, dagger_vals = read_series(fig10_txt, "CBO-DAgger (expected_discounted_reward by delay)")
        offline_delays, offline_vals = read_series(fig10_txt, "CBO-Offline (expected_discounted_reward by delay)")
        assert dagger_delays == delays == offline_delays, "delay grids don't match between existing fig10 data and this run's cfg.eval.tdo_delays"

        dfo_static = read_scalar(fig6_txt, "MPO")
        dagger_static = read_scalar(fig6_txt, "CBO-DAgger")
        offline_static = read_scalar(fig6_txt, "CBO-Offline")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(delays, dagger_vals, marker="o", color=ALGO_COLORS["CBO-DAgger"], label="TDO-DAgger")
        ax.plot(delays, offline_vals, marker="o", color=ALGO_COLORS["CBO-Offline"], label="TDO-Offline")
        ax.plot(delays, random_by_delay, marker="o", color=ALGO_COLORS["Random"], label="TDO-Random")
        ax.axhline(dfo_static, color=ALGO_COLORS["MPO"], linestyle=":", label="DFO")
        ax.axhline(dagger_static, color=ALGO_COLORS["CBO-DAgger"], linestyle=":", label="CBO-DAgger")
        ax.axhline(offline_static, color=ALGO_COLORS["CBO-Offline"], linestyle=":", label="CBO-Offline")
        ax.set_xlim(0, 40)
        ax.set_xlabel("TDO Delay (trajectory steps before switching to oracle)")
        ax.set_ylabel("Expected Discounted Reward")
        ax.set_title("Expected Discounted Reward vs TDO Delay")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out, dpi=150)
        plt.close(fig)
        print(f"wrote {args.out}")

        txt_path = args.out.rsplit(".", 1)[0] + ".txt"
        save_values_txt(
            txt_path,
            delays=delays,
            **{
                "TDO-DAgger (expected_discounted_reward by delay)": (delays, dagger_vals),
                "TDO-Offline (expected_discounted_reward by delay)": (delays, offline_vals),
                "TDO-Random (expected_discounted_reward by delay)": (delays, random_by_delay),
                "DFO (static)": dfo_static,
                "CBO-DAgger (static)": dagger_static,
                "CBO-Offline (static)": offline_static,
            },
        )
        print(f"wrote {txt_path}")

    _main()


if __name__ == "__main__":
    main()
