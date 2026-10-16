"""Jax-free re-render of fig5 from already-collected data -- renames
CBO-DAgger/CBO-Offline to CBO-On/CBO-Off (per the group's latest naming
pass) and shortens the right-axis CDF legend text, without re-running the
GPU rollouts that produced figures_acc_new/fig5.txt (CBO-On/CBO-Off episode
length + CDF, from today's dual-rollout fix) and figures_acc/fig8_K5.txt
(classifier accuracy curves, unchanged source).

Run with: python envs/pgj/replot_fig5_from_txt.py
"""
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_utils import ALGO_COLORS

FIG5_TXT = "figures_acc_new/fig5.txt"
FIG8_TXT = "figures_acc/fig8_K5.txt"
OUT = "figures_acc_new/fig5.png"

RENAME = {"CBO-DAgger": "CBO-On", "CBO-Offline": "CBO-Off"}


def read_series_block(path, block_name, ncols):
    with open(path) as f:
        blocks = f.read().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if lines and lines[0] == f"# {block_name}":
            cols = [[] for _ in range(ncols)]
            for line in lines[1:]:
                parts = line.split("\t")
                for c, p in zip(cols, parts):
                    c.append(float(p))
            return [np.array(c) for c in cols]
    raise ValueError(f"{block_name!r} not found in {path}")


def read_scalar(path, name):
    with open(path) as f:
        blocks = f.read().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if lines and lines[0] == f"# {name}":
            return float(lines[1])
    raise ValueError(f"{name!r} not found in {path}")


def main():
    random_baseline = read_scalar(FIG5_TXT, "random_baseline")

    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    for name in ["CBO-DAgger", "CBO-Offline"]:
        color = ALGO_COLORS[name]
        steps, means, sems = read_series_block(FIG8_TXT, f"{name} (step, accuracy, sem)", 3)
        mask = steps <= 40
        ax1.plot(steps[mask], means[mask], color=color, label=RENAME[name])
        ax1.fill_between(steps[mask], (means - sems)[mask], (means + sems)[mask], alpha=0.3, color=color)

    ax1.axhline(random_baseline, color="black", linestyle=":", label="Random (1/6)")
    ax1.set_xlim(0, 40)
    ax1.set_xlabel("Trajectory Step")
    ax1.set_ylabel("Classification Accuracy")
    ax1.set_title("Classifier Accuracy vs Trajectory Step")

    ax2 = ax1.twinx()
    for name in ["CBO-DAgger", "CBO-Offline"]:
        color = ALGO_COLORS[name]
        lengths, cdf = read_series_block(FIG5_TXT, f"{name} (episode_lengths_sorted, cdf)", 2)
        mask2 = lengths <= 40
        ax2.plot(lengths[mask2], cdf[mask2], color=color, linestyle="-.", label=f"{RENAME[name]} CDF")
    ax2.set_ylabel("CDF of Trajectory Length")
    ax2.set_ylim(0, 1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
