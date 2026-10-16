"""Minimal driver for figures_acc's fig9 (= evaluate.py's Figure 6, MPO
selection rate by trajectory step) and fig11 (= evaluate.py's Figure 2,
pursuer x evader winrate matrix) only -- skips evaluate.py's Figure 3/5/7,
which aren't part of figures_acc and (Figure 7 especially) would redundantly
recompute nearly the same rollouts evaluate_acc.py's Figure 12 already
covers.

Run with, e.g.:
    python envs/pgj/fig9_fig11.py level_k.K=5 \\
        level_k.checkpoint_dir=checkpoints/level_k_40M_5k \\
        pap.checkpoint_dir=checkpoints/pap_40M_5k \\
        lksp.checkpoint_dir=checkpoints/lksp_40M_5k \\
        eval.figures_dir=figures_acc eval.suffix=_K5
"""

import logging

import _compat  # noqa: F401
import hydra
import jax
import numpy as np
import torch as th

from pgj.env import ODEnv

from lksp import LKSPPolicy, load_pursuer_levels
from pap import load_evader_levels

from evaluate import figure_2, plot_figure_2, figure_6, plot_figure_6, load_lksp_meta
from plot_utils import save_values_txt

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="evaluate")
def main(cfg):
    np.random.seed(cfg.seed)
    th.manual_seed(cfg.seed)
    key = jax.random.key(cfg.seed)

    env = ODEnv()
    suffix = cfg.eval.get("suffix", "") or ""

    key, k1, k2, k3 = jax.random.split(key, 4)
    pursuer_levels = load_pursuer_levels(cfg, env, k1)
    evader_levels = load_evader_levels(cfg, env, k2)
    lksp_meta = load_lksp_meta(cfg, env, cfg.level_k.K + 1, k3)
    lksp_policy = LKSPPolicy(lksp_meta, pursuer_levels)

    log.info("=== fig11 (Figure 2): pursuer x evader winrate matrix ===")
    key, fk = jax.random.split(key)
    matrix, key = figure_2(cfg, env, pursuer_levels, evader_levels, fk)
    plot_figure_2(matrix, cfg.level_k.K, f"{cfg.eval.figures_dir}/fig11{suffix}.png")
    save_values_txt(
        f"{cfg.eval.figures_dir}/fig11{suffix}.txt",
        **{f"pursuer L{i} winrate by evader level": row for i, row in enumerate(matrix.tolist())},
    )

    log.info("=== fig9 (Figure 6): MPO selection rate by trajectory step ===")
    key, fk = jax.random.split(key)
    steps, means, sems = figure_6(cfg, env, lksp_policy, evader_levels, fk)
    plot_figure_6(steps, means, sems, f"{cfg.eval.figures_dir}/fig9{suffix}.png")
    save_values_txt(
        f"{cfg.eval.figures_dir}/fig9{suffix}.txt",
        **{"step, mpo_selection_rate, sem": (steps, means, sems)},
    )

    log.info(f"done: fig9/fig11 written to {cfg.eval.figures_dir}/ (suffix={suffix!r})")


if __name__ == "__main__":
    main()
