"""Plots for the pursuer-pool-size sweep experiment: does RLBO's achievable
performance against a FIXED evader rationality range (levels 1-6, uniform)
increase as its pursuer expert pool grows from 2 up through 6 -- even while
staying at or below the evader's own max rationality level?

Produces, for each metric in METRICS (both logged directly in
metrics/pgj_lksp_poolsweep_n{2..6}_evk6.csv -- no approximation needed):
  1. fig_poolsweep_curves_{metric}.png -- training curves overlaid, one per
     pool size
  2. fig_poolsweep_final_{metric}.png -- final (last-5%-mean) value vs pool
     size, with error bars (std over the final window)

Run with: python envs/pgj/plot_pool_sweep.py
"""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

POOL_SIZES = [2, 3, 4, 5, 6]
EVADER_K = 6
CSV_FMT = "metrics/pgj_lksp_poolsweep_n{n}_evk{evk}.csv"
PAP_CSV = f"metrics/pgj_pap_evk{EVADER_K}.csv"  # PAP retrained with evader fixed to 1-6, matching the pool sweep exactly (same budget, same seed=0)
CMAP = plt.get_cmap("viridis")

METRICS = {
    "discounted_reward": ("rollout/pursuer_discounted_reward", "Expected Discounted Reward"),
    "winrate": ("rollout/pursuer_winrate", "Pursuer Win Rate"),
}


def read_curve(pool_size, column):
    return read_curve_from(CSV_FMT.format(n=pool_size, evk=EVADER_K), column)


def read_pap_curve(metric_key):
    """PAP retrained from scratch with the evader fixed to levels 1-6
    (pap_evader_fixed.py), same training budget and seed=0 as the pool
    sweep -- a genuine matched baseline, not the old full-K=8 reference.
    Logs rollout/pursuer_discounted_reward directly, same as the pool-sweep
    CSVs, so no approximation needed for either metric."""
    return read_curve_from(PAP_CSV, METRICS[metric_key][0])


def read_curve_from(path, column):
    timesteps, values = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            v = row.get(column)
            if not v:
                continue
            timesteps.append(float(row["timestep"]))
            values.append(float(v))
    return np.array(timesteps), np.array(values)


def plot_metric(metric_key, column, ylabel):
    curves = {n: read_curve(n, column) for n in POOL_SIZES}

    # --- training curves overlay ---
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = {n: CMAP(i / (len(POOL_SIZES) - 1)) for i, n in enumerate(POOL_SIZES)}
    for n in POOL_SIZES:
        t, v = curves[n]
        ax.plot(t, v, color=colors[n], label=rf"pool size {n} ($\pi_1..\pi_{n}$)", linewidth=1.2)
    pap_t, pap_v = read_pap_curve(metric_key)
    ax.plot(pap_t, pap_v, color="black", linestyle="--", linewidth=1.2, label="PAP (matched: evader 1-6, same budget/seed)")
    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel(ylabel)
    ax.set_title(f"RLBO Training Performance vs Pursuer Pool Size\n(evader fixed to levels 1-{EVADER_K}, uniform)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = f"figures_acc_new/fig_poolsweep_curves_{metric_key}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    # --- final value vs pool size, with error bars ---
    means, stds = [], []
    for n in POOL_SIZES:
        t, v = curves[n]
        tail = v[t >= t.max() - t.max() * 0.05]  # last ~5% of training
        means.append(tail.mean())
        stds.append(tail.std())
    means, stds = np.array(means), np.array(stds)

    pap_t, pap_v = read_pap_curve(metric_key)
    pap_tail = pap_v[pap_t >= pap_t.max() - pap_t.max() * 0.05]
    pap_mean, pap_std = pap_tail.mean(), pap_tail.std()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(POOL_SIZES, means, yerr=stds, marker="o", color="tab:blue", capsize=4, label="RLBO")
    ax.axhline(pap_mean, color="black", linestyle="--", label="PAP (matched: evader 1-6)")
    ax.fill_between([min(POOL_SIZES), max(POOL_SIZES)], pap_mean - pap_std, pap_mean + pap_std, color="black", alpha=0.1)
    ax.axvline(EVADER_K, color="gray", linestyle=":", label=f"evader max level ({EVADER_K})")
    ax.set_xlabel("Pursuer Pool Size (number of experts RLBO may choose from)")
    ax.set_ylabel(f"Final {ylabel}")
    ax.set_title(f"RLBO Final Performance vs Pursuer Pool Size\n(evader fixed to levels 1-{EVADER_K}, uniform)")
    ax.set_xticks(POOL_SIZES)
    ax.legend()
    fig.tight_layout()
    out = f"figures_acc_new/fig_poolsweep_final_{metric_key}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    for n, m, s in zip(POOL_SIZES, means, stds):
        print(f"pool_size={n}: final_{metric_key}={m:.4f} +/- {s:.4f}")


def main():
    for metric_key, (column, ylabel) in METRICS.items():
        plot_metric(metric_key, column, ylabel)


if __name__ == "__main__":
    main()
