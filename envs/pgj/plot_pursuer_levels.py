"""Discounted reward training curves for each of the 9 individually-trained
pursuer levels (L0-L8) from the K=8 level_k library build, overlaid in one
plot.

These CSVs predate this session's level_k.py fix that logs
rollout/pursuer_discounted_reward directly, so -- same as the original PAP
curve in fig4 -- we approximate via winrate * gamma**ep_len_mean (a lower
bound on the true E[gamma^L] by Jensen's inequality, not exact).

Each level was trained for ~20.025M steps (see conversation: level_k used
collect.total_timesteps=80_000 per-env, half of PAP/RLBO/DAgger's 160_000),
against its OWN fixed/perturbed opponent (Algorithm 1: pursuer L_i vs
evader L_i) -- so unlike the pool-sweep plot, these 9 curves are NOT
trained against a shared opponent distribution; each is its own matchup.

Run with: python envs/pgj/plot_pursuer_levels.py
"""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LEVELS = list(range(9))  # L0..L8
GAMMA = 0.99
CSV_FMT = "metrics/level_k_40M_8k/pgj_level_k_pursuer_L{i}_40M_8k.csv"
CMAP = plt.get_cmap("viridis")


def read_curve(level):
    path = CSV_FMT.format(i=level)
    timesteps, values = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            wr = row.get("rollout/pursuer_winrate")
            ep_len = row.get("rollout/ep_len_mean")
            if not wr or ep_len is None:
                continue
            v = float(wr) * GAMMA ** float(ep_len)
            timesteps.append(float(row["timestep"]))
            values.append(v)
    return np.array(timesteps), np.array(values)


def main():
    curves = {lvl: read_curve(lvl) for lvl in LEVELS}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, lvl in enumerate(LEVELS):
        t, v = curves[lvl]
        color = CMAP(i / (len(LEVELS) - 1))
        ax.plot(t, v, color=color, linewidth=1.0, label=rf"$\pi_{{{lvl + 1}}}$ (L{lvl})")
    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Expected Discounted Reward (approx.)")
    ax.set_title("Discounted Reward by Pursuer Level (K=8 library build)\neach vs its own fixed/perturbed opponent")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out = "figures_acc_new/fig_pursuer_levels_discounted_reward.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    for lvl in LEVELS:
        t, v = curves[lvl]
        tail = v[t >= t.max() - t.max() * 0.05]
        print(f"L{lvl} (pi_{lvl + 1}): final_discounted_reward_approx={tail.mean():.4f} +/- {tail.std():.4f}")


if __name__ == "__main__":
    main()
