"""Adds a denser delay grid to fig7 (Expected Discounted Reward vs TDO
Delay): every integer 0..20, plus 30 -- reusing the already-computed
delays (0, 5, 10, 20) from figures_acc_new/fig7.txt and only computing the
new ones fresh, for all three TDO variants (DAgger, Offline, Random).

The X-axis stays capped at 40 (unchanged); this just adds resolution in the
0-20 region plus one point at 30, and keeps the existing 40/60/100 points
too so the full shape (including the tail, even past the visual cutoff) is
still in the underlying .txt data.

Run with, e.g.:
    python envs/pgj/fig7_denser_delays.py \\
        level_k.K=8 level_k.checkpoint_dir=checkpoints/level_k_40M_8k \\
        pap.checkpoint_dir=checkpoints/pap_40M_8k \\
        lksp.checkpoint_dir=checkpoints/lksp_40M_8k \\
        dagger.checkpoint_dir=checkpoints/dagger_40M_8k \\
        offline.checkpoint_dir=checkpoints/offline_40M_8k \\
        eval.num_envs=500 --view-k 5 --out figures_acc_new/fig7.png
"""

import argparse
import copy
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
from evaluate import rollout, load_dagger, load_offline
from evaluate_acc import RandomMetaActor, TDOPolicy, discounted_reward
from plot_utils import ALGO_COLORS, save_values_txt
from jaxtrl.sim.policy import PolicyAggregator

NEW_DELAYS = list(range(1, 20)) + [30]  # 1..19, plus 30 (0, 5, 10, 20 already exist)


def read_series(path, name):
    with open(path) as f:
        blocks = f.read().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if lines and lines[0] == f"# {name}":
            xs, ys = [], []
            for line in lines[1:]:
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


def compute_delay_points(cfg, env, pursuer_levels, evader_levels, classifier_or_meta, delays, L, is_random):
    sub_levels = pursuer_levels[1:]
    results = []
    for delay in delays:
        level_returns = []
        for k in range(1, L + 1):
            agents = [TeamPursuerAgent(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape), IndEvader(EDR, LLP, HLP, SIM_MAX_STEPS, env.grid.shape)]
            reward = [SparsePursuer(LLP, HLP, EDR, 1, 0), SparseEvader(EDR, LLP, HLP, 1, 0)]
            meta = RandomMetaActor(num_levels=L) if is_random else classifier_or_meta
            tdo_policy = TDOPolicy(meta, sub_levels, delay, k - 1)
            policy = PolicyAggregator([tdo_policy, evader_levels[k - 1]])
            key = jax.random.key(cfg.seed + delay * 1000 + k)
            transitions = rollout(env, agents, reward, policy, cfg.eval.num_envs, key)
            level_returns.append(discounted_reward(transitions, agent_idx=0))
        mean_dr = float(np.mean(level_returns))
        results.append(mean_dr)
        print(f"delay={delay} -> expected_discounted_reward={mean_dr:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view-k", type=int, required=True)
    parser.add_argument("--existing", default=None, help="existing fig7.txt to merge new delay points into")
    parser.add_argument("--out", required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    L = args.view_k
    existing = args.existing or "figures_acc_new/fig7.txt"

    @hydra.main(version_base=None, config_path="../../conf", config_name="evaluate")
    def _main(cfg):
        np.random.seed(cfg.seed)
        th.manual_seed(cfg.seed)
        env = ODEnv()
        key = jax.random.key(cfg.seed)

        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        pursuer_levels = load_pursuer_levels(cfg, env, k1)
        evader_levels = load_evader_levels(cfg, env, k2)[:L]
        dagger_classifier = load_dagger(cfg, env, k3)
        offline_classifier = load_offline(cfg, env, k4)

        print("=== TDO-DAgger new delays ===")
        dagger_new = compute_delay_points(cfg, env, pursuer_levels, evader_levels, dagger_classifier, NEW_DELAYS, L, is_random=False)
        print("=== TDO-Offline new delays ===")
        offline_new = compute_delay_points(cfg, env, pursuer_levels, evader_levels, offline_classifier, NEW_DELAYS, L, is_random=False)
        print("=== TDO-Random new delays ===")
        random_new = compute_delay_points(cfg, env, pursuer_levels, evader_levels, None, NEW_DELAYS, L, is_random=True)

        old_delays, dagger_old = read_series(existing, "TDO-DAgger (expected_discounted_reward by delay)")
        _, offline_old = read_series(existing, "TDO-Offline (expected_discounted_reward by delay)")
        _, random_old = read_series(existing, "TDO-Random (expected_discounted_reward by delay)")
        dfo_static = read_scalar(existing, "DFO (static)")
        dagger_static = read_scalar(existing, "CBO-DAgger (static)")
        offline_static = read_scalar(existing, "CBO-Offline (static)")

        all_delays = old_delays + NEW_DELAYS
        all_dagger = dagger_old + dagger_new
        all_offline = offline_old + offline_new
        all_random = random_old + random_new

        order = np.argsort(all_delays)
        delays_sorted = [all_delays[i] for i in order]
        dagger_sorted = [all_dagger[i] for i in order]
        offline_sorted = [all_offline[i] for i in order]
        random_sorted = [all_random[i] for i in order]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(delays_sorted, dagger_sorted, marker="o", markersize=3, color=ALGO_COLORS["CBO-DAgger"], label="TDO-DAgger")
        ax.plot(delays_sorted, offline_sorted, marker="o", markersize=3, color=ALGO_COLORS["CBO-Offline"], label="TDO-Offline")
        ax.plot(delays_sorted, random_sorted, marker="o", markersize=3, color=ALGO_COLORS["Random"], label="TDO-Random")
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
            delays=delays_sorted,
            **{
                "TDO-DAgger (expected_discounted_reward by delay)": (delays_sorted, dagger_sorted),
                "TDO-Offline (expected_discounted_reward by delay)": (delays_sorted, offline_sorted),
                "TDO-Random (expected_discounted_reward by delay)": (delays_sorted, random_sorted),
                "DFO (static)": dfo_static,
                "CBO-DAgger (static)": dagger_static,
                "CBO-Offline (static)": offline_static,
            },
        )
        print(f"wrote {txt_path}")

    _main()


if __name__ == "__main__":
    main()
