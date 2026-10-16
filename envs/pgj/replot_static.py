"""Jax-free replots for today's revision list -- items 1, 4, 6 (partial) --
reading straight from already-computed .txt data, no rollouts, no jax
import (importing jax/evaluate_acc.py greedily grabs GPU memory even for a
plain replot, which would contend with any GPU job currently running).

Run with: python envs/pgj/replot_static.py
"""
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_utils import ALGO_COLORS, save_values_txt

SRCDIR = "figures_acc"  # where the per-view raw data (fig*_K5.txt) still lives
FIGDIR = "figures_acc_new"  # where today's revised figures get written


def read_dict_block(path, block_name):
    with open(path) as f:
        blocks = f.read().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if lines and lines[0] == f"# {block_name}":
            out = {}
            for line in lines[1:]:
                k, v = line.split("\t")
                out[k] = float(v)
            return out
    raise ValueError(f"{block_name!r} not found in {path}")


def read_series_block(path, block_name):
    with open(path) as f:
        blocks = f.read().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if lines and lines[0] == f"# {block_name}":
            xs, ys = [], []
            for line in lines[1:]:
                parts = line.split("\t")
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
            return xs, ys
    raise ValueError(f"{block_name!r} not found in {path}")


def read_matrix(path):
    with open(path) as f:
        blocks = f.read().split("\n\n")
    rows = {}
    for block in blocks:
        lines = block.strip().split("\n")
        if not lines or not lines[0].startswith("# pursuer L"):
            continue
        i = int(re.search(r"pursuer L(\d+)", lines[0]).group(1))
        rows[i] = [float(v) for v in lines[1:]]
    n = len(rows)
    return np.array([rows[i] for i in range(n)])


# ---------------------------------------------------------------------------
# Item 1 (fig3 = our fig6): rename MPO->DFO, tight y-scale, title->Orchestrator
# ---------------------------------------------------------------------------

def replot_fig3(K=5):
    path = f"{SRCDIR}/fig6_K{K}.txt"
    results = read_dict_block(path, "expected_discounted_reward")

    rename = {"MPO": "RLBO", "CBO-DAgger": "CBO-On", "CBO-Offline": "CBO-Off"}
    fig, ax = plt.subplots(figsize=(6, 4))
    orig_names = list(results.keys())
    display_names = [rename.get(n, n) for n in orig_names]
    values = list(results.values())
    colors = [ALGO_COLORS[n] for n in orig_names]
    ax.bar(display_names, values, color=colors)
    ax.set_ylabel("Expected Discounted Reward")
    ax.set_xlabel("Orchestrator")
    ax.set_title("Expected Discounted Reward vs Orchestrator")
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.15 or 0.05
    ax.set_ylim(max(0, lo - pad), min(1, hi + pad))
    fig.tight_layout()
    out = f"{FIGDIR}/fig3.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    txt_out = f"{FIGDIR}/fig3.txt"
    save_values_txt(txt_out, expected_discounted_reward=dict(zip(display_names, values)))
    print(f"wrote {txt_out}")


# ---------------------------------------------------------------------------
# Item 4 (fig6 = our fig9): X cut to 40, title change, random line at 1/6
# ---------------------------------------------------------------------------

def replot_fig6_selection_rate(K=5):
    path = f"{SRCDIR}/fig9_K{K}.txt"
    steps, means = read_series_block(path, "step, mpo_selection_rate, sem")
    # sems live in the 3rd column -- re-parse directly since read_series_block only keeps 2
    sems = []
    with open(path) as f:
        blocks = f.read().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if lines and lines[0] == "# step, mpo_selection_rate, sem":
            for line in lines[1:]:
                sems.append(float(line.split("\t")[2]))
            break
    steps, means, sems = np.array(steps), np.array(means), np.array(sems)
    mask = steps <= 40

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps[mask], means[mask], color=ALGO_COLORS["MPO"])
    ax.fill_between(steps[mask], (means - sems)[mask], (means + sems)[mask], alpha=0.3, color=ALGO_COLORS["MPO"], label="+/- 1 SE")
    ax.axhline(1 / 6, color="black", linestyle=":", label="Random (1/6)")
    ax.set_xlim(0, 40)
    ax.set_xlabel("Trajectory Step")
    ax.set_ylabel("Selection Rate")
    ax.set_title("Best-Response Selection Rate of RLBO vs Trajectory Timestep")
    ax.legend()
    fig.tight_layout()
    out = f"{FIGDIR}/fig6.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    txt_out = f"{FIGDIR}/fig6.txt"
    save_values_txt(txt_out, **{"step, selection_rate, sem": (steps[mask].tolist(), means[mask].tolist(), sems[mask].tolist())})
    print(f"wrote {txt_out}")


# ---------------------------------------------------------------------------
# Item 6 (fig8 = our fig11): mark BOTH the pursuer's and evader's true
# Algorithm-1 "designed" partner, not a naive i==j diagonal, AND relabel
# levels to match the new paper's alternating hero/adversary schema.
#
# Ground truth from level_k.py's actual generation loop (verified by
# reading it directly): evader_0 (baseline) -> pursuer_0 = BR(evader_0) ->
# evader_1 = BR(pursuer_0) -> pursuer_1 = BR(evader_1) -> evader_2 =
# BR(pursuer_1) -> ... i.e. pursuer level i is a best response to evader
# level (i-1) [designed partner = column (i-1)], and evader level j is a
# best response to pursuer level j [designed partner = row j].
#
# The new paper's schema numbers this same alternating sequence as
# adversary_1 (base) -> hero_2 = BR(adversary_1) -> adversary_3 = BR(hero_2)
# -> hero_4 = BR(adversary_3) -> ..., i.e. hero/pursuer gets only EVEN
# levels, adversary/evader gets only ODD levels, starting from 1. Matching
# term-by-term against our own sequence gives:
#   our pursuer level i (0-indexed)      -> paper's hero level      (2i + 2)
#   our evader level j (0=baseline..K)   -> paper's adversary level (2j + 1)
# (Sanity check: our pursuer_i = BR(evader_i) under this map becomes
# hero_(2i+2) = BR(adversary_(2i+1)), which is exactly the paper's
# pi_{k+1} = BR(pi_bar_k) rule with k = 2i+1. Our evader_i = BR(pursuer_{i-1})
# becomes adversary_(2i+1) = BR(hero_(2i)), matching pi_bar_{k+1} = BR(pi_k)
# with k = 2i.)
# ---------------------------------------------------------------------------

def replot_fig8_matrix(K=5):
    path = f"{SRCDIR}/fig11_K{K}.txt"
    matrix = read_matrix(path)
    n = matrix.shape[0]

    # LaTeX/mathtext symbols matching the CURRENT paper draft's notation
    # (APEGS_ACC_27.pdf, Section III.B): T := {pi_1, ..., pi_{K+1}} (hero,
    # all K+1 TRAINED via best-response -- our pursuer level i, 0-indexed
    # 0..K, is simply shifted by 1) and T-bar := {pi_bar_0, ..., pi_bar_K}
    # (adversary, where pi_bar_0 is the PRESCRIBED baseline -- our evader
    # level 0, the hand-designed A*+random mixture, not RL-trained -- and
    # pi_bar_1..pi_bar_K are the K trained ones). No shift needed for the
    # evader axis since our internal indexing already starts at 0 = baseline,
    # matching pi_bar_0 exactly. (Supersedes an earlier even/odd-alternating
    # labeling scheme that matched a *previous* draft's now-superseded
    # problem formulation, not this one.)
    pursuer_labels = [rf"$\pi_{{{i + 1}}}$" for i in range(n)]  # pi_1, pi_2, ..., pi_{K+1}
    evader_labels = [rf"$\bar\pi_{{{j}}}$" for j in range(n)]  # pi_bar_0 (prescribed), pi_bar_1, ...

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n), evader_labels)
    ax.set_yticks(range(n), pursuer_labels)
    ax.set_xlabel("Evader (adversary)")
    ax.set_ylabel("Pursuer (hero)")
    ax.set_title("Pursuer Winrate by Policy Matchup")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="win rate")
    fig.tight_layout()
    out = f"{FIGDIR}/fig8.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    txt_out = f"{FIGDIR}/fig8.txt"
    save_values_txt(txt_out, **{f"pursuer {pursuer_labels[i]} winrate by evader level ({', '.join(evader_labels)})": row for i, row in enumerate(matrix.tolist())})
    print(f"wrote {txt_out}")


if __name__ == "__main__":
    replot_fig3()
    replot_fig6_selection_rate()
    replot_fig8_matrix(K=5)
