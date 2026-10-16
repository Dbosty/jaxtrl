"""Computes the data fig13 needs (evaluate.py's Figure 5 win rates and
Figure 7 action-diff matrix) at each of several "K views" of the SAME
checkpoints trained once at K=8 -- see evaluate_multi_view.py's module
docstring for why this is valid (Level-K generation is sequential, so the
first L pursuer/evader levels of a K=8 run are already valid, freestanding
"K=L" experts; no retraining needed).

Only computes Figure 5 + Figure 7 (fig13's two inputs), skipping every other
figure evaluate_multi_view.py would otherwise also compute -- much cheaper
per view when all that's wanted is fig13.

Writes figures_acc/figure_5_K{L}.log and figures_acc/figure_7_matrix_K{L}.npy
for each requested L, skipping any that already exist (so re-running after a
partial failure only fills in what's missing).

Run with, e.g.:
    python envs/pgj/compute_fig13_views.py \\
        level_k.K=8 level_k.checkpoint_dir=checkpoints/level_k_40M_8k \\
        pap.checkpoint_dir=checkpoints/pap_40M_8k \\
        lksp.checkpoint_dir=checkpoints/lksp_40M_8k \\
        dagger.checkpoint_dir=checkpoints/dagger_40M_8k \\
        offline.checkpoint_dir=checkpoints/offline_40M_8k \\
        eval.num_envs=500 eval.views=[1,2,3,4,6,7]
"""

import copy
import logging
import os

import _compat  # noqa: F401
import hydra
import jax
import numpy as np
import torch as th

from pgj.env import ODEnv

from lksp import LKSPPolicy, load_pursuer_levels
from pap import load_evader_levels

from evaluate import figure_5, figure_7, load_pap, load_lksp_meta, load_dagger, load_offline

log = logging.getLogger(__name__)


def view_cfg(cfg, L):
    vc = copy.deepcopy(cfg)
    vc.level_k.K = L
    return vc


@hydra.main(version_base=None, config_path="../../conf", config_name="evaluate")
def main(cfg):
    np.random.seed(cfg.seed)
    th.manual_seed(cfg.seed)
    key = jax.random.key(cfg.seed)

    env = ODEnv()
    figures_dir = cfg.eval.figures_dir
    os.makedirs(figures_dir, exist_ok=True)
    views = list(cfg.eval.views)

    key, k1, k2, k3, k4, k5, k6 = jax.random.split(key, 7)
    pursuer_levels = load_pursuer_levels(cfg, env, k1)  # FULL K+1 entries, never sliced
    evader_levels_full = load_evader_levels(cfg, env, k2)  # FULL K entries
    pap_llp, pap_hlp = load_pap(cfg, env, k3)
    lksp_meta = load_lksp_meta(cfg, env, cfg.level_k.K + 1, k4)
    lksp_policy = LKSPPolicy(lksp_meta, pursuer_levels)
    dagger_classifier = load_dagger(cfg, env, k5)
    dagger_policy = LKSPPolicy(dagger_classifier, pursuer_levels[1:])
    try:
        offline_classifier = load_offline(cfg, env, k6)
        offline_policy = LKSPPolicy(offline_classifier, pursuer_levels[1:])
    except (RuntimeError, FileNotFoundError) as e:
        log.warning(f"no offline classifier: {e}")
        offline_policy = None

    for L in views:
        log_path = f"{figures_dir}/figure_5_K{L}.log"
        matrix_path = f"{figures_dir}/figure_7_matrix_K{L}.npy"
        if os.path.exists(log_path) and os.path.exists(matrix_path):
            log.info(f"view K={L}: both outputs already exist, skipping")
            continue

        vc = view_cfg(cfg, L)
        evader_levels = evader_levels_full[:L]
        log.info(f"=== view K={L} ===")

        if not os.path.exists(log_path):
            key, fk = jax.random.split(key)
            pap_rates, lksp_rates, dagger_rates, offline_rates, key = figure_5(
                vc, env, pap_llp, pap_hlp, lksp_policy, dagger_policy, offline_policy, evader_levels, fk
            )
            with open(log_path, "w") as f5log:
                for j in range(L + 1):
                    off_str = f"{offline_rates[j]:.3f}" if offline_rates is not None else "n/a"
                    line = (
                        f"Figure 5: evader L{j} -> PAP={pap_rates[j]:.3f} LKSP={lksp_rates[j]:.3f} "
                        f"DAgger={dagger_rates[j]:.3f} Offline={off_str}"
                    )
                    log.info(line)
                    f5log.write(line + "\n")
        else:
            log.info(f"view K={L}: {log_path} already exists, skipping Figure 5")

        if not os.path.exists(matrix_path):
            key, fk = jax.random.split(key)
            diff_matrix = figure_7(vc, env, pursuer_levels, evader_levels, fk)
            np.save(matrix_path, diff_matrix)
            log.info(f"view K={L}: wrote {matrix_path}")
        else:
            log.info(f"view K={L}: {matrix_path} already exists, skipping Figure 7")

        log.info(f"=== view K={L} done ===")

    log.info(f"done: fig13 data for views {views} written to {figures_dir}/")


if __name__ == "__main__":
    main()
