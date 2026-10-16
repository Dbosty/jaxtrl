"""Produces figures_acc's full figure set at multiple "K views" from a
SINGLE trained checkpoint set (checkpoints/*_40M_8k, trained once at K=8),
instead of training a separate pipeline per K.

Why this works: level_k.py's Algorithm-1 generation is sequential -- level
k's pursuer/evader was trained as a best response to level k-1, independent
of whatever higher levels came later. So pursuer_llp/hlp_L0..L5 and
evader_L1..L5 from the K=8 run are already valid, freestanding "K=5"
experts; no retraining needed to look at just the first 5 of them.

For figures that don't go through a trained discrete selector (Figure 2's
matchup matrix), restricting the view is just a loop-bound change. For ones
that DO (MPO's meta-controller, CBO-DAgger/CBO-Offline's classifiers), the
policy objects always keep the FULL pursuer_levels list (they were trained
with a 9-way / 8-way action space over the real K=8 system, so truncating
that list would crash on any index >= L) -- only the *opponent* set
(evader_levels) gets sliced to the first L entries, i.e. "how does the
already-trained K=8 system behave when we only ever pit it against evader
levels 1..L". That's a real, meaningful restriction (and the one place it
matters structurally -- Figure 7/12's cross-level diff matrix and Figure
6acc's opponent-sampling distribution -- slicing evader_levels is required,
not just cosmetic; the other figures would behave identically whether or
not evader_levels is sliced, since they only ever index its first L-1
entries anyway, so slicing everywhere uniformly is simplest and always
correct).

Run with, e.g.:
    python envs/pgj/evaluate_multi_view.py \\
        level_k.K=8 level_k.checkpoint_dir=checkpoints/level_k_40M_8k \\
        pap.checkpoint_dir=checkpoints/pap_40M_8k \\
        lksp.checkpoint_dir=checkpoints/lksp_40M_8k \\
        dagger.checkpoint_dir=checkpoints/dagger_40M_8k \\
        offline.checkpoint_dir=checkpoints/offline_40M_8k \\
        eval.num_envs=500 eval.figures_dir=figures_acc eval.views=[5,8]
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

from lksp import LKSPPolicy, TeamPursuerAgent, load_pursuer_levels
from pap import UniformLevelKEvaderPolicy, load_evader_levels

from evaluate import (
    figure_2, plot_figure_2,
    figure_5, plot_figure_5,
    figure_6, plot_figure_6,
    figure_7, plot_figure_7,
    load_pap, load_lksp_meta, load_dagger, load_offline,
)
from evaluate_acc import (
    RandomDispatchPolicy,
    figure_6 as figure_6acc, plot_figure_6 as plot_figure_6acc,
    figure_8 as figure_8acc, plot_figure_8 as plot_figure_8acc,
    figure_10 as figure_10acc, plot_figure_10 as plot_figure_10acc,
    figure_12 as figure_12acc, plot_figure_12 as plot_figure_12acc,
)
from plot_utils import save_values_txt

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
    os.makedirs(cfg.eval.figures_dir, exist_ok=True)
    views = list(cfg.eval.get("views") or [cfg.level_k.K])

    key, k1, k2, k3, k4, k5, k6 = jax.random.split(key, 7)
    pursuer_levels = load_pursuer_levels(cfg, env, k1)  # FULL K+1 entries -- never sliced, see module docstring
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
        log.warning(f"no offline classifier at {cfg.offline.checkpoint_dir}: {e}")
        offline_classifier, offline_policy = None, None

    for L in views:
        suffix = f"_K{L}"
        vc = view_cfg(cfg, L)
        evader_levels = evader_levels_full[:L]
        log.info(f"=== view K={L} (from checkpoints trained at K={cfg.level_k.K}) ===")

        log.info(f"[{suffix}] fig11 (Figure 2): pursuer x evader winrate matrix")
        key, fk = jax.random.split(key)
        matrix, key = figure_2(vc, env, pursuer_levels, evader_levels, fk)
        plot_figure_2(matrix, L, f"{cfg.eval.figures_dir}/fig11{suffix}.png")
        save_values_txt(
            f"{cfg.eval.figures_dir}/fig11{suffix}.txt",
            **{f"pursuer L{i} winrate by evader level": row for i, row in enumerate(matrix.tolist())},
        )

        log.info(f"[{suffix}] fig9 (Figure 6): MPO selection rate by trajectory step")
        key, fk = jax.random.split(key)
        steps, means, sems = figure_6(vc, env, lksp_policy, evader_levels, fk)
        plot_figure_6(steps, means, sems, f"{cfg.eval.figures_dir}/fig9{suffix}.png")
        save_values_txt(f"{cfg.eval.figures_dir}/fig9{suffix}.txt", **{"step, mpo_selection_rate, sem": (steps, means, sems)})

        log.info(f"[{suffix}] Figure 5 (feeds fig13): PAP/MPO winrate by evader level")
        key, fk = jax.random.split(key)
        pap_rates, lksp_rates, dagger_rates, offline_rates, key = figure_5(
            vc, env, pap_llp, pap_hlp, lksp_policy, dagger_policy, offline_policy, evader_levels, fk
        )
        plot_figure_5(pap_rates, lksp_rates, dagger_rates, offline_rates, L, f"{cfg.eval.figures_dir}/figure_5{suffix}.png")
        fig5_series = {"PAP": pap_rates, "MPO": lksp_rates, "CBO-DAgger": dagger_rates}
        if offline_rates is not None:
            fig5_series["CBO-Offline"] = offline_rates
        save_values_txt(f"{cfg.eval.figures_dir}/figure_5{suffix}.txt", evader_levels=list(range(L + 1)), **fig5_series)
        # Written to a dedicated per-view file, not just log.info -- figure_13.py
        # greps "Figure 5: evader L..." lines back out by level number, and
        # levels 0..5 appear in BOTH the K=5 and K=8 views, so a single shared
        # log would let the later view's lines silently overwrite the earlier
        # view's values for those overlapping levels.
        with open(f"{cfg.eval.figures_dir}/figure_5{suffix}.log", "w") as f5log:
            for j in range(L + 1):
                off_str = f"{offline_rates[j]:.3f}" if offline_rates is not None else "n/a"
                line = f"Figure 5: evader L{j} -> PAP={pap_rates[j]:.3f} LKSP={lksp_rates[j]:.3f} DAgger={dagger_rates[j]:.3f} Offline={off_str}"
                log.info(line)
                f5log.write(line + "\n")

        log.info(f"[{suffix}] Figure 7 (feeds fig13): action-diff matrix among evader levels 1..{L}")
        key, fk = jax.random.split(key)
        diff_matrix = figure_7(vc, env, pursuer_levels, evader_levels, fk)
        np.save(f"{cfg.eval.figures_dir}/figure_7_matrix{suffix}.npy", diff_matrix)

        log.info(f"[{suffix}] fig6: expected discounted reward vs policy type")
        key, fk = jax.random.split(key)
        random_policy = RandomDispatchPolicy(pursuer_levels, num_levels=L + 1)
        fig6_results, key = figure_6acc(
            vc, env, pap_llp, pap_hlp, lksp_policy, dagger_policy, offline_policy, evader_levels, fk,
            random_policy=random_policy,
        )
        plot_figure_6acc(fig6_results, f"{cfg.eval.figures_dir}/fig6{suffix}.png")
        save_values_txt(f"{cfg.eval.figures_dir}/fig6{suffix}.txt", expected_discounted_reward=fig6_results)

        log.info(f"[{suffix}] fig8: classifier accuracy by trajectory step")
        classifier_policies = {"CBO-DAgger": dagger_policy}
        if offline_policy is not None:
            classifier_policies["CBO-Offline"] = offline_policy
        key, fk = jax.random.split(key)
        fig8_results, key = figure_8acc(vc, env, classifier_policies, evader_levels, fk)
        plot_figure_8acc(fig8_results, f"{cfg.eval.figures_dir}/fig8{suffix}.png")
        save_values_txt(
            f"{cfg.eval.figures_dir}/fig8{suffix}.txt",
            **{f"{name} (step, accuracy, sem)": v for name, v in fig8_results.items()},
        )

        log.info(f"[{suffix}] fig10: expected discounted reward vs TDO delay")
        classifiers = {"CBO-DAgger": dagger_classifier}
        if offline_classifier is not None:
            classifiers["CBO-Offline"] = offline_classifier
        delays = list(cfg.eval.tdo_delays)
        key, fk = jax.random.split(key)
        fig10_results, key = figure_10acc(vc, env, pursuer_levels, evader_levels, classifiers, fk, delays)
        plot_figure_10acc(fig10_results, delays, f"{cfg.eval.figures_dir}/fig10{suffix}.png")
        save_values_txt(
            f"{cfg.eval.figures_dir}/fig10{suffix}.txt",
            delays=delays,
            **{f"{name} (expected_discounted_reward by delay)": (delays, v) for name, v in fig10_results.items()},
        )

        log.info(f"[{suffix}] fig12: expert action divergence by trajectory step")
        key, fk = jax.random.split(key)
        steps12, means12, sems12 = figure_12acc(vc, env, pursuer_levels, evader_levels, fk)
        plot_figure_12acc(steps12, means12, sems12, f"{cfg.eval.figures_dir}/fig12{suffix}.png")
        save_values_txt(f"{cfg.eval.figures_dir}/fig12{suffix}.txt", **{"step, mean_sq_action_diff, sem": (steps12, means12, sems12)})

        log.info(f"=== view K={L} done ===")

    log.info(f"done: all views {views} written to {cfg.eval.figures_dir}/")


if __name__ == "__main__":
    main()
