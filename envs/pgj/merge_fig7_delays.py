"""Merges the new TDO delay 1-5 points into fig7, replacing the old
delay=5 point (computed with the same methodology, just re-derived fresh
here) rather than duplicating it. Jax-free -- pure text/plot merge.

Run with: python envs/pgj/merge_fig7_delays.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_utils import ALGO_COLORS, save_values_txt

EXISTING = "figures_acc_new/fig7.txt"
NEW = "figures_acc_new/fig7_new_delays.txt"
OUT = "figures_acc_new/fig7.png"


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
    raise ValueError(f"{name!r} not found in {path}")


def read_scalar(path, name):
    """Reads a plain-scalar block (header line '# name', then a single bare
    value line) -- the format save_values_txt uses for a non-dict, non-tuple
    scalar, as opposed to a dict block's 'key\\tvalue' lines."""
    with open(path) as f:
        blocks = f.read().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if lines and lines[0] == f"# {name}":
            return float(lines[1])
    raise ValueError(f"{name!r} not found in {path}")


def main():
    dfo_static = read_scalar(EXISTING, "DFO (static)")
    dagger_static = read_scalar(EXISTING, "CBO-DAgger (static)")
    offline_static = read_scalar(EXISTING, "CBO-Offline (static)")

    merged = {}
    for name in ["TDO-DAgger", "TDO-Offline", "TDO-Random"]:
        old_delays, old_vals = read_series(EXISTING, f"{name} (expected_discounted_reward by delay)")
        new_delays, new_vals = read_series(NEW, f"{name} (expected_discounted_reward by delay)")

        combined = dict(zip(old_delays, old_vals))
        combined.update(dict(zip(new_delays, new_vals)))  # new values win on overlap (delay=5)
        sorted_delays = sorted(combined.keys())
        merged[name] = (sorted_delays, [combined[d] for d in sorted_delays])
        print(f"{name}: {len(sorted_delays)} points, delay=5 -> {combined[5.0]:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(*merged["TDO-DAgger"], marker="o", markersize=3, color=ALGO_COLORS["CBO-DAgger"], label="TDO-On")
    ax.plot(*merged["TDO-Offline"], marker="o", markersize=3, color=ALGO_COLORS["CBO-Offline"], label="TDO-Off")
    ax.plot(*merged["TDO-Random"], marker="o", markersize=3, color=ALGO_COLORS["Random"], label="TDO-Rand")
    ax.axhline(dfo_static, color=ALGO_COLORS["MPO"], linestyle=":", label="RLBO")
    ax.axhline(dagger_static, color=ALGO_COLORS["CBO-DAgger"], linestyle=":", label="CBO-On")
    ax.axhline(offline_static, color=ALGO_COLORS["CBO-Offline"], linestyle=":", label="CBO-Off")
    ax.set_xlim(0, 40)
    ax.set_xlabel("TDO Delay (trajectory steps before switching to oracle)")
    ax.set_ylabel("Expected Discounted Reward")
    ax.set_title("Expected Discounted Reward vs TDO Delay")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}")

    txt_path = OUT.rsplit(".", 1)[0] + ".txt"
    save_values_txt(
        txt_path,
        delays=merged["TDO-DAgger"][0],
        **{
            "TDO-DAgger (expected_discounted_reward by delay)": merged["TDO-DAgger"],
            "TDO-Offline (expected_discounted_reward by delay)": merged["TDO-Offline"],
            "TDO-Random (expected_discounted_reward by delay)": merged["TDO-Random"],
            "DFO (static)": dfo_static,
            "CBO-DAgger (static)": dagger_static,
            "CBO-Offline (static)": offline_static,
        },
    )
    print(f"wrote {txt_path}")


if __name__ == "__main__":
    main()
