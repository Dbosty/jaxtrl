"""Figure 13 from the APEGS_ACC draft: winrate gap (PAP - MPO) and expert
behavioral diversity ("action variance") vs number of experts K.

MPO in the new draft is our LKSP (envs/pgj/lksp.py) -- same meta-controller,
renamed. This script does not re-run any rollouts; it aggregates numbers
already produced by evaluate.py for two separate level_k.K runs:

  - PAP/LKSP per-evader-level win rates: evaluate.py's Figure 5 step logs
    these (`Figure 5: evader Lj -> PAP=... LKSP=...`) but never saves them to
    disk, so this script parses them back out of evaluate.py's saved run log.
  - action diversity: evaluate.py's Figure 7 step saves
    figures/figure_7_matrix{suffix}.npy, a K x K matrix of pairwise L2
    action-norm differences between evader levels, *normalized by its own
    max* (see evaluate.py's figure_7()). Because normalization is per-run,
    the absolute scale of "action variance" below is not directly
    comparable across K -- only the relative trend is. Flagged in the plot.

Run with, e.g.:
    python envs/pgj/figure_13.py \\
        --run 4 /tmp/evaluate_final.log figures/figure_7_matrix.npy \\
        --run 8 /tmp/evaluate_40M_8k.log figures/figure_7_matrix_40M_8K.npy \\
        --out figures/figure_13.png
"""

import argparse
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_utils import save_values_txt

FIGURE_5_LINE = re.compile(r"Figure 5: evader L(\d+) -> PAP=([\d.]+) LKSP=([\d.]+)")


def read_pap_lksp_rates(log_path):
    """Returns (pap_rates, lksp_rates) arrays ordered by evader level, parsed
    from evaluate.py's own INFO-level rollout log."""
    pap_rates, lksp_rates = {}, {}
    with open(log_path) as f:
        for line in f:
            m = FIGURE_5_LINE.search(line)
            if m:
                level, pap, lksp = int(m.group(1)), float(m.group(2)), float(m.group(3))
                pap_rates[level], lksp_rates[level] = pap, lksp
    if not pap_rates:
        raise ValueError(f"no 'Figure 5: evader L...' lines found in {log_path}")
    levels = sorted(pap_rates)
    return np.array([pap_rates[k] for k in levels]), np.array([lksp_rates[k] for k in levels])


def action_variance(matrix_path):
    """Mean of the off-diagonal entries of the (already max-normalized) L2
    action-difference matrix -- a single scalar summary of how behaviorally
    distinct the K expert (evader) policies are from one another."""
    diff_matrix = np.load(matrix_path)
    k = diff_matrix.shape[0]
    off_diag = diff_matrix[~np.eye(k, dtype=bool)]
    return float(off_diag.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", nargs=3, action="append", metavar=("K", "EVAL_LOG", "FIGURE_7_MATRIX"), required=True,
        help="one per K value, e.g. --run 4 /tmp/evaluate_final.log figures/figure_7_matrix.npy",
    )
    parser.add_argument("--out", default="figures/figure_13.png")
    args = parser.parse_args()

    ks, gaps, variances = [], [], []
    for k_str, log_path, matrix_path in args.run:
        k = int(k_str)
        pap_rates, lksp_rates = read_pap_lksp_rates(log_path)
        gap = float((pap_rates - lksp_rates).mean())
        var = action_variance(matrix_path)
        ks.append(k)
        gaps.append(gap)
        variances.append(var)
        print(
            f"K={k}: mean PAP={pap_rates.mean():.3f} mean MPO={lksp_rates.mean():.3f} "
            f"gap(PAP-MPO)={gap:+.3f} action_variance={var:.3f}"
        )

    order = np.argsort(ks)
    ks = np.array(ks)[order]
    gaps = np.array(gaps)[order]
    variances = np.array(variances)[order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax1.plot(ks, gaps, marker="o", color="tab:purple")
    ax1.set_xlabel("Number of Experts (K)")
    ax1.set_ylabel("Winrate Gap (PAP - MPO)")
    ax1.set_title("MPO vs PAP Performance Gap")
    ax1.set_xticks(ks)

    ax2.plot(ks, variances, marker="o", color="teal")
    ax2.set_xlabel("Number of Experts (K)")
    ax2.set_ylabel("Mean Pairwise Action Difference\n(normalized per-run, not cross-run scale)")
    ax2.set_title("Expert Behavioral Diversity")
    ax2.set_xticks(ks)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"wrote {args.out}")

    txt_path = args.out.rsplit(".", 1)[0] + ".txt"
    save_values_txt(
        txt_path,
        **{
            "K, winrate_gap(PAP-MPO), action_variance": (ks, gaps, variances),
        },
    )
    print(f"wrote {txt_path}")


if __name__ == "__main__":
    main()
