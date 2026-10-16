"""Rebuilds fig4 (training curves), now attempting real Expected Discounted
Reward per the group's repeated request, rather than plain win rate.

METHODOLOGY (read before trusting this for the paper): run_stage never
logged true discounted reward during training, and only PAP/DFO's CSVs log
rollout/ep_len_mean at all (DAgger's never does). True per-checkpoint
discounted reward would require re-rolling-out the policy AS IT EXISTED at
each logged interval, which isn't possible after the fact (only final/best
weights were saved, not one snapshot per interval).

Given that, this script uses the best available approximation, applied as
consistently as possible across all three curves:
  - PAP, DFO: winrate(t) * gamma**ep_len_mean(t), using each interval's own
    logged mean episode length (time-varying, real data, just not the full
    length distribution -- see the Jensen's-inequality caveat discussed
    with the user: gamma**ep_len_mean(t) is a lower bound on the true
    E[gamma^L], not the exact value).
  - CBO-DAgger: since its CSV has no per-interval episode length at all, we
    use a CONSTANT gamma**ep_len_mean estimate instead -- specifically the
    REAL, MEASURED mean episode length of DAgger's actual final checkpoint
    (figures_acc_new/dagger_mean_ep_len.txt, from a fresh eval rollout, not
    a guess), applied uniformly across DAgger's whole curve. This is worse
    than PAP/DFO's time-varying version (episode length does change over
    training) but strictly better than either raw winrate (no discounting
    at all) or a fabricated/arbitrary constant.
  - CBO-Offline: unlike the other three, this one doesn't need approximating
    at all -- it's a single point, not a curve, and we already have its
    REAL, exact discounted reward from a genuine post-training rollout
    (figures_acc_new/fig6.txt's "CBO-Offline" entry, via evaluate_acc.py's
    discounted_reward()).

Run with: python envs/pgj/fig4_reward_rebuild.py --view-k 5
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_utils import ALGO_COLORS, save_values_txt


def approx_discounted_curve(csv_path, gamma, constant_ep_len=None):
    """If constant_ep_len is given, uses that fixed value for every point
    (DAgger's case); otherwise reads rollout/ep_len_mean per row (PAP/DFO)."""
    timesteps, values = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            wr = row.get("rollout/pursuer_winrate")
            if not wr:
                continue
            ep_len = constant_ep_len if constant_ep_len is not None else row.get("rollout/ep_len_mean")
            if ep_len is None:
                continue
            timesteps.append(float(row["timestep"]))
            values.append(float(wr) * gamma ** float(ep_len))
    return timesteps, values


def read_dict_scalar(path, block_name, key):
    with open(path) as f:
        blocks = f.read().split("\n\n")
    for b in blocks:
        lines = b.strip().split("\n")
        if lines and lines[0] == f"# {block_name}":
            for line in lines[1:]:
                k, v = line.split("\t")
                if k == key:
                    return float(v)
    raise ValueError(f"{key!r} not found in block {block_name!r} of {path}")


def read_plain_scalar(path, name):
    with open(path) as f:
        blocks = f.read().split("\n\n")
    for b in blocks:
        lines = b.strip().split("\n")
        if lines and lines[0] == f"# {name}":
            return float(lines[1])
    raise ValueError(f"{name!r} not found in {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view-k", type=int, required=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--pap", default="metrics/pap_40M_8k/pgj_pap_40M_8k.csv")
    parser.add_argument("--dfo", default="metrics/lksp_40M_8k/pgj_lksp_40M_8k.csv")
    parser.add_argument("--dagger", default="metrics/dagger_40M_8k/pgj_dagger_40M_8k.csv")
    parser.add_argument("--dagger-ep-len-txt", default="figures_acc_new/dagger_mean_ep_len.txt")
    parser.add_argument("--fig3-txt", default=None, help="fig3's real discounted-reward-vs-policy-type data (figures_acc_new/fig3.txt), for CBO-Offline's static line")
    parser.add_argument("--out", default="figures_acc_new/fig4.png")
    args = parser.parse_args()

    fig3_txt = args.fig3_txt or "figures_acc_new/fig3.txt"
    offline_final = read_dict_scalar(fig3_txt, "expected_discounted_reward", "CBO-Off")
    dagger_ep_len = read_plain_scalar(args.dagger_ep_len_txt, "dagger_mean_ep_len")

    pap_t, pap_v = approx_discounted_curve(args.pap, args.gamma)
    dfo_t, dfo_v = approx_discounted_curve(args.dfo, args.gamma)
    dagger_t, dagger_v = approx_discounted_curve(args.dagger, args.gamma, constant_ep_len=dagger_ep_len)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pap_t, pap_v, label="PAP", color=ALGO_COLORS["PAP"])
    ax.plot(dfo_t, dfo_v, label="RLBO", color=ALGO_COLORS["MPO"])
    ax.plot(dagger_t, dagger_v, label="CBO-On", color=ALGO_COLORS["CBO-DAgger"])
    ax.axhline(offline_final, color=ALGO_COLORS["CBO-Offline"], linestyle="--", label="CBO-Off (final)")
    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Expected Discounted Reward")
    ax.set_title("Expected Discounted Reward vs Training Timesteps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")

    txt_path = args.out.rsplit(".", 1)[0] + ".txt"
    save_values_txt(
        txt_path,
        PAP=(pap_t, pap_v), DFO=(dfo_t, dfo_v), CBO_DAgger=(dagger_t, dagger_v),
        CBO_Offline_final=offline_final,
        DAgger_constant_ep_len_used=dagger_ep_len,
    )
    print(f"wrote {txt_path}")


if __name__ == "__main__":
    main()
