"""Rebuilds fig5 (classifier accuracy vs trajectory step) per today's
revision list, item 3: X axis capped at 40, CDF of trajectory (episode)
lengths added on a right-hand axis, static dotted line at the classifier's
TRUE random-guess baseline.

NOTE on the "1/6" baseline: per clarification, that number is exactly right
for fig6 (DFO's selector, which picks among K+1=6 pursuer levels at K=5),
but NOT for this figure -- CBO-DAgger/CBO-Offline's classifiers each predict
among K=5 evader levels (1..K), so their random baseline is 1/5=0.20, not
1/6. Drawing the line at the classifier's true baseline (1/5) here rather
than 1/6, since 1/6 would be a wrong number for what this figure measures.

Trajectory lengths are sampled from TWO separate fresh rollouts -- one under
CBO-DAgger, one under CBO-Offline, each against the same mixed evader-level
distribution (the opponent PAP/DFO were trained against) -- so each
accuracy curve is paired with the length distribution it was actually
measured under, rather than both sharing a single (DAgger-only) CDF.

Run with, e.g.:
    python envs/pgj/fig5_cdf_add.py \\
        level_k.K=8 level_k.checkpoint_dir=checkpoints/level_k_40M_8k \\
        pap.checkpoint_dir=checkpoints/pap_40M_8k \\
        lksp.checkpoint_dir=checkpoints/lksp_40M_8k \\
        dagger.checkpoint_dir=checkpoints/dagger_40M_8k \\
        offline.checkpoint_dir=checkpoints/offline_40M_8k \\
        eval.num_envs=500 --view-k 5 --out figures_acc/fig5.png
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

from lksp import LKSPPolicy, TeamPursuerAgent, load_pursuer_levels
from pap import UniformLevelKEvaderPolicy, load_evader_levels

from level_k import EDR, HLP, LLP, SIM_MAX_STEPS
from evaluate import rollout, episode_lengths, load_dagger, load_offline
from plot_utils import ALGO_COLORS, save_values_txt
from jaxtrl.sim.policy import PolicyAggregator


def read_series_block(path, block_name):
    with open(path) as f:
        blocks = f.read().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if lines and lines[0] == f"# {block_name}":
            xs, ys, sems = [], [], []
            for line in lines[1:]:
                x, y, s = line.split("\t")
                xs.append(float(x))
                ys.append(float(y))
                sems.append(float(s))
            return np.array(xs), np.array(ys), np.array(sems)
    raise ValueError(f"{block_name!r} not found in {path}")


def view_cfg(cfg, L):
    vc = copy.deepcopy(cfg)
    vc.level_k.K = L
    return vc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view-k", type=int, required=True)
    parser.add_argument("--fig8-txt", default=None, help="existing fig8_K{view_k}.txt with classifier accuracy curves")
    parser.add_argument("--out", default="figures_acc/fig5.png")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    L = args.view_k
    fig8_txt = args.fig8_txt or f"figures_acc/fig8_K{L}.txt"

    @hydra.main(version_base=None, config_path="../../conf", config_name="evaluate")
    def _main(cfg):
        np.random.seed(cfg.seed)
        th.manual_seed(cfg.seed)
        key = jax.random.key(cfg.seed)
        env = ODEnv()
        vc = view_cfg(cfg, L)

        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        pursuer_levels = load_pursuer_levels(cfg, env, k1)
        evader_levels = load_evader_levels(cfg, env, k2)[:L]
        dagger_classifier = load_dagger(cfg, env, k3)
        offline_classifier = load_offline(cfg, env, k4)
        dagger_policy = LKSPPolicy(dagger_classifier, pursuer_levels[1:])
        offline_policy = LKSPPolicy(offline_classifier, pursuer_levels[1:])

        evader_policy = UniformLevelKEvaderPolicy(evader_levels)
        evader_agent = IndEvader(EDR, LLP, HLP, SIM_MAX_STEPS, env.grid.shape)
        team_agent = TeamPursuerAgent(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape)
        agents = [team_agent, evader_agent]
        reward = [SparsePursuer(LLP, HLP, EDR, 1, 0), SparseEvader(EDR, LLP, HLP, 1, 0)]

        episode_cdfs = {}
        for name, clf_policy in [("CBO-DAgger", dagger_policy), ("CBO-Offline", offline_policy)]:
            policy = PolicyAggregator([clf_policy, evader_policy])
            key, rk = jax.random.split(key)
            transitions = rollout(env, agents, reward, policy, cfg.eval.num_envs, rk)
            lengths, _won = episode_lengths(transitions, EDR)
            lengths = np.sort(np.array(lengths))
            cdf = np.arange(1, len(lengths) + 1) / len(lengths)
            episode_cdfs[name] = (lengths, cdf)
            print(f"{name}: collected {len(lengths)} episode lengths, mean={lengths.mean():.1f}, median={np.median(lengths):.1f}")

        classifier_accuracy = {}
        for name in ["CBO-DAgger", "CBO-Offline"]:
            block = f"{name} (step, accuracy, sem)"
            try:
                steps, means, sems = read_series_block(fig8_txt, block)
                classifier_accuracy[name] = (steps, means, sems)
            except ValueError:
                pass

        fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
        for name, (steps, means, sems) in classifier_accuracy.items():
            mask = steps <= 40
            color = ALGO_COLORS[name]
            ax1.plot(steps[mask], means[mask], color=color, label=name)
            ax1.fill_between(steps[mask], (means - sems)[mask], (means + sems)[mask], alpha=0.3, color=color)
        # Per group feedback: use 1/6 here too, matching fig6's DFO-selector
        # random baseline (1/(K+1) at K=5), for visual/conceptual consistency
        # across the two figures -- even though this classifier's own output
        # space is technically K-way (1/L = 1/5), not (K+1)-way.
        random_baseline = 1.0 / (L + 1)
        ax1.axhline(random_baseline, color="black", linestyle=":", label=f"Random (1/{L + 1})")
        ax1.set_xlim(0, 40)
        ax1.set_xlabel("Trajectory Step")
        ax1.set_ylabel("Classification Accuracy")
        ax1.set_title("Classifier Accuracy vs Trajectory Step")

        ax2 = ax1.twinx()
        for name, (lengths, cdf) in episode_cdfs.items():
            mask2 = lengths <= 40
            ax2.plot(lengths[mask2], cdf[mask2], color=ALGO_COLORS[name], linestyle="-.", label=f"{name} Episode Length CDF")
        ax2.set_ylabel("CDF of Trajectory Length")
        ax2.set_ylim(0, 1)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")

        fig.tight_layout()
        fig.savefig(args.out, dpi=150)
        plt.close(fig)
        print(f"wrote {args.out}")

        txt_path = args.out.rsplit(".", 1)[0] + ".txt"
        save_values_txt(
            txt_path,
            random_baseline=random_baseline,
            **{
                f"{name} (episode_lengths_sorted, cdf)": (lengths.tolist(), cdf.tolist())
                for name, (lengths, cdf) in episode_cdfs.items()
            },
        )
        print(f"wrote {txt_path}")

    _main()


if __name__ == "__main__":
    main()
