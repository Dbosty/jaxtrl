"""Jax-free re-render of fig10 (Expected Discounted Reward vs Max Level K)
from the already-collected figures_acc_new/fig10.txt sweep data -- renames
DFO/CBO-DAgger/CBO-Offline to RLBO/CBO-On/CBO-Off without re-running the
expensive 7-K-view x 5-method GPU sweep (fig10_kscan.py).

Run with: python envs/pgj/replot_fig10_from_txt.py
"""
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_utils import ALGO_COLORS

TXT = "figures_acc_new/fig10.txt"
OUT = "figures_acc_new/fig10.png"

RENAME = {"DFO": "RLBO", "CBO-DAgger": "CBO-On", "CBO-Offline": "CBO-Off"}
COLOR_KEY = {"DFO": "MPO", "CBO-DAgger": "CBO-DAgger", "CBO-Offline": "CBO-Offline", "PAP": "PAP", "Random": "Random"}


def read_plain_list(path, name):
    with open(path) as f:
        blocks = f.read().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if lines and lines[0] == f"# {name}":
            return np.array([float(v) for v in lines[1:]])
    raise ValueError(f"{name!r} not found in {path}")


def main():
    ks = read_plain_list(TXT, "K")

    fig, ax = plt.subplots(figsize=(6, 4))
    for name in ["PAP", "DFO", "CBO-DAgger", "CBO-Offline", "Random"]:
        values = read_plain_list(TXT, name)
        ax.plot(ks, values, marker="o", color=ALGO_COLORS[COLOR_KEY[name]], label=RENAME.get(name, name))
    ax.set_xlabel("Max Level K")
    ax.set_ylabel("Expected Discounted Reward")
    ax.set_title("Expected Discounted Reward vs Max Level K")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
