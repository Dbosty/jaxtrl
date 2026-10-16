"""Figure 4 from the APEGS paper: PAP/LKSP pursuer win-rate over training,
plus (optionally) the DAgger and offline classifiers for comparison.

Reads the CSV files run_stage (in level_k.py) writes during pap.py's,
lksp.py's, and dagger.py's training runs -- pure plotting over
already-logged metrics, no checkpoint loading or rollouts needed.

Note on the x-axis: unlike pap/lksp/dagger, where "timestep" is fresh
environment frames collected, offline_classifier.py's "timestep" is
cumulative *training samples seen*, repeatedly drawn from one fixed dataset
collected once upfront (see offline_classifier.py's docstring) -- not new
environment interaction, and its dataset/epoch schedule makes its raw
timestep count run ~4.5x higher than pap/lksp/dagger's (180M vs 40M for the
K=8 run). Plotting raw timesteps on a shared axis would make the offline
curve stretch ~4.5x wider than the others for a reason that has nothing to
do with how "far along" it is. Instead every curve's x-axis is normalized
to its own [0, 1] training-progress fraction, so all curves span the same
width and "further right" consistently means "further into its own
training" regardless of what a timestep means for that method. The offline
curve stays dashed as a reminder it's not the same underlying unit.

Run with, e.g.:
    python envs/pgj/plot_training_curves.py \\
        --pap metrics/pap/pgj_pap.csv \\
        --lksp metrics/lksp/pgj_lksp.csv \\
        --dagger metrics/dagger/pgj_dagger.csv \\
        --offline metrics/offline/pgj_offline_classifier.csv \\
        --out figures/figure_4.png
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_utils import ALGO_COLORS, save_values_txt


def read_curve(path, column="rollout/pursuer_winrate"):
    timesteps, values = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get(column):
                timesteps.append(float(row["timestep"]))
                values.append(float(row[column]))
    return timesteps, values


def normalize_progress(timesteps):
    """Rescales to [0, 1] by this curve's own max -- see module docstring on
    why a shared raw-timestep axis isn't a fair comparison here."""
    max_t = max(timesteps)
    return [t / max_t for t in timesteps]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pap", required=True, help="path to pap.py's run_stage CSV")
    parser.add_argument("--lksp", required=True, help="path to lksp.py's run_stage CSV")
    parser.add_argument("--dagger", default=None, help="path to dagger.py's run_stage CSV (optional)")
    parser.add_argument("--offline", default=None, help="path to offline_classifier.py's CSV (optional)")
    parser.add_argument("--out", default="figures/figure_4.png")
    args = parser.parse_args()

    pap_t, pap_v = read_curve(args.pap)
    lksp_t, lksp_v = read_curve(args.lksp)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(normalize_progress(lksp_t), lksp_v, label="MPO", color=ALGO_COLORS["MPO"])
    ax.plot(normalize_progress(pap_t), pap_v, label="PAP", color=ALGO_COLORS["PAP"])
    if args.dagger:
        dagger_t, dagger_v = read_curve(args.dagger)
        ax.plot(normalize_progress(dagger_t), dagger_v, label="CBO-DAgger", color=ALGO_COLORS["CBO-DAgger"])
    if args.offline:
        offline_t, offline_v = read_curve(args.offline)
        ax.plot(
            normalize_progress(offline_t), offline_v, label="CBO-Offline (samples seen)",
            color=ALGO_COLORS["CBO-Offline"], linestyle="--",
        )
    ax.set_xlabel("Fraction of Training Complete")
    ax.set_ylabel("Winrate")
    ax.set_title("Pursuer Winrate vs Training Progress")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")

    series = {"MPO": (lksp_t, lksp_v), "PAP": (pap_t, pap_v)}
    if args.dagger:
        series["CBO-DAgger"] = (dagger_t, dagger_v)
    if args.offline:
        series["CBO-Offline (samples seen)"] = (offline_t, offline_v)
    txt_path = args.out.rsplit(".", 1)[0] + ".txt"
    save_values_txt(txt_path, **series)
    print(f"saved {txt_path}")


if __name__ == "__main__":
    main()
