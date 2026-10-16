"""New fig10 per today's revision list, item 10: replaces the old winrate-
gap/action-variance-vs-K plot with a direct Expected Discounted Reward vs
Max Level K sweep across all 5 policy types (PAP, DFO, CBO-DAgger,
CBO-Offline, Random), for K=2..8. Reuses evaluate_acc.py's figure_6 (single
mixed-opponent rollout per policy type) at each K "view" of the same
checkpoints trained once at K=8 -- see evaluate_multi_view.py's docstring
for why this is valid (no retraining needed).

Run with, e.g.:
    python envs/pgj/fig10_kscan.py \\
        level_k.K=8 level_k.checkpoint_dir=checkpoints/level_k_40M_8k \\
        pap.checkpoint_dir=checkpoints/pap_40M_8k \\
        lksp.checkpoint_dir=checkpoints/lksp_40M_8k \\
        dagger.checkpoint_dir=checkpoints/dagger_40M_8k \\
        offline.checkpoint_dir=checkpoints/offline_40M_8k \\
        eval.num_envs=500 --out figures_acc/fig10.png
"""

import argparse
import copy
import logging
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

from lksp import LKSPPolicy, load_pursuer_levels
from pap import load_evader_levels

from evaluate import load_pap, load_lksp_meta, load_dagger, load_offline
from evaluate_acc import RandomDispatchPolicy, figure_6 as figure_6acc
from plot_utils import ALGO_COLORS, save_values_txt

log = logging.getLogger(__name__)

KS = [2, 3, 4, 5, 6, 7, 8]


def view_cfg(cfg, L):
    vc = copy.deepcopy(cfg)
    vc.level_k.K = L
    return vc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    @hydra.main(version_base=None, config_path="../../conf", config_name="evaluate")
    def _main(cfg):
        np.random.seed(cfg.seed)
        th.manual_seed(cfg.seed)
        key = jax.random.key(cfg.seed)
        env = ODEnv()

        key, k1, k2, k3, k4, k5, k6 = jax.random.split(key, 7)
        pursuer_levels = load_pursuer_levels(cfg, env, k1)
        evader_levels_full = load_evader_levels(cfg, env, k2)
        pap_llp, pap_hlp = load_pap(cfg, env, k3)
        lksp_meta = load_lksp_meta(cfg, env, cfg.level_k.K + 1, k4)
        lksp_policy = LKSPPolicy(lksp_meta, pursuer_levels)
        dagger_classifier = load_dagger(cfg, env, k5)
        dagger_policy = LKSPPolicy(dagger_classifier, pursuer_levels[1:])
        offline_classifier = load_offline(cfg, env, k6)
        offline_policy = LKSPPolicy(offline_classifier, pursuer_levels[1:])

        results_by_method = {"PAP": [], "DFO": [], "CBO-DAgger": [], "CBO-Offline": [], "Random": []}

        for L in KS:
            vc = view_cfg(cfg, L)
            evader_levels = evader_levels_full[:L]
            random_policy = RandomDispatchPolicy(pursuer_levels, num_levels=L + 1)

            key, fk = jax.random.split(key)
            raw_results, key = figure_6acc(
                vc, env, pap_llp, pap_hlp, lksp_policy, dagger_policy, offline_policy, evader_levels, fk,
                random_policy=random_policy,
            )
            results_by_method["PAP"].append(raw_results["PAP"])
            results_by_method["DFO"].append(raw_results["MPO"])  # MPO renamed to DFO throughout
            results_by_method["CBO-DAgger"].append(raw_results["CBO-DAgger"])
            results_by_method["CBO-Offline"].append(raw_results["CBO-Offline"])
            results_by_method["Random"].append(raw_results["Random"])
            log.info(f"K={L}: " + ", ".join(f"{k}={v[-1]:.4f}" for k, v in results_by_method.items()))

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = {**ALGO_COLORS, "DFO": ALGO_COLORS["MPO"]}
        for name, values in results_by_method.items():
            ax.plot(KS, values, marker="o", color=colors.get(name, "tab:gray"), label=name)
        ax.set_xlabel("Max Level K")
        ax.set_ylabel("Expected Discounted Reward")
        ax.set_title("Expected Discounted Reward vs Max Level K")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out, dpi=150)
        plt.close(fig)
        log.info(f"wrote {args.out}")

        txt_path = args.out.rsplit(".", 1)[0] + ".txt"
        save_values_txt(txt_path, K=KS, **results_by_method)
        log.info(f"wrote {txt_path}")

    _main()


if __name__ == "__main__":
    main()
