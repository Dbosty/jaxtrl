"""Lean K=8 figures_acc pass that reuses everything already computed today
instead of re-simulating it:

  - fig11 (Figure 2 matrix): untouched -- doesn't reference MPO/classifiers
    at all, the existing file from this morning's evaluate.py K=8 run is
    already correct.
  - fig13's action-diff-matrix input: reuses
    figures/figure_7_matrix_40M_8K.npy directly (also has zero classifier
    dependency, so nothing about today's changes affects it).
  - fig10's CBO-DAgger sweep (7 delays): reused verbatim from the run that
    crashed at Figure 12 -- only CBO-DAgger had finished sweeping before it
    crashed, and that data doesn't change just because CBO-Offline now
    exists. Only CBO-Offline's sweep is computed fresh here.
  - fig6's PAP/MPO/CBO-DAgger bars: also reused verbatim from that same
    crashed run's log (it got through Figure 6 successfully before dying at
    Figure 12); only CBO-Offline's bar is computed fresh.

Genuinely new work (no prior K=8 run ever computed these, or the run
crashed before writing them): Figure 5 (now with CBO-Offline available,
feeds fig13), fig9 (needs the MPO title/color fix, independent of
classifiers), fig8 (never had per-classifier raw data saved), fig12 (the
run that would have produced it crashed at 71/72 combos with zero output).

Run with, e.g.:
    python envs/pgj/evaluate_K8_lean.py level_k.K=8 \\
        level_k.checkpoint_dir=checkpoints/level_k_40M_8k \\
        pap.checkpoint_dir=checkpoints/pap_40M_8k \\
        lksp.checkpoint_dir=checkpoints/lksp_40M_8k \\
        dagger.checkpoint_dir=checkpoints/dagger_40M_8k \\
        offline.checkpoint_dir=checkpoints/offline_40M_8k
"""

import logging

import _compat  # noqa: F401
import hydra
import jax
import numpy as np
import torch as th

from pgj.env import ODEnv

from lksp import LKSPPolicy, TeamPursuerAgent, load_pursuer_levels
from pap import UniformLevelKEvaderPolicy, load_evader_levels

from evaluate import (
    figure_5, plot_figure_5,
    figure_6, plot_figure_6,
    load_pap, load_lksp_meta, load_dagger, load_offline,
)
from evaluate_acc import (
    figure_6 as figure_6acc, plot_figure_6 as plot_figure_6acc,
    figure_8 as figure_8acc, plot_figure_8 as plot_figure_8acc,
    plot_figure_10 as plot_figure_10acc,
    figure_12 as figure_12acc, plot_figure_12 as plot_figure_12acc,
)
from plot_utils import save_values_txt

log = logging.getLogger(__name__)

# Reused verbatim from /tmp/evaluate_acc_K8.log (the run that crashed at
# Figure 12) -- CBO-DAgger had already finished Figure 10's full 7-delay
# sweep before the crash, and adding CBO-Offline doesn't change these.
DAGGER_TDO_BY_DELAY = {
    0: 0.8995, 5: 0.8071, 10: 0.7686, 20: 0.7607, 40: 0.7585, 60: 0.7595, 100: 0.7589,
}
# Reused verbatim from the same log -- Figure 6's PAP/MPO/CBO-DAgger bars.
REUSED_FIG6 = {"PAP": 0.8536, "MPO": 0.8478, "CBO-DAgger": 0.7540}


@hydra.main(version_base=None, config_path="../../conf", config_name="evaluate")
def main(cfg):
    np.random.seed(cfg.seed)
    th.manual_seed(cfg.seed)
    key = jax.random.key(cfg.seed)

    env = ODEnv()
    K = cfg.level_k.K
    figures_dir = cfg.eval.figures_dir
    delays = list(cfg.eval.tdo_delays)
    assert delays == list(DAGGER_TDO_BY_DELAY.keys()), (
        f"cfg.eval.tdo_delays {delays} doesn't match the reused DAgger sweep's delays "
        f"{list(DAGGER_TDO_BY_DELAY.keys())} -- can't merge mismatched delay grids."
    )

    key, k1, k2, k3, k4, k5, k6 = jax.random.split(key, 7)
    pursuer_levels = load_pursuer_levels(cfg, env, k1)
    evader_levels = load_evader_levels(cfg, env, k2)
    pap_llp, pap_hlp = load_pap(cfg, env, k3)
    lksp_meta = load_lksp_meta(cfg, env, K + 1, k4)
    lksp_policy = LKSPPolicy(lksp_meta, pursuer_levels)
    dagger_classifier = load_dagger(cfg, env, k5)
    dagger_policy = LKSPPolicy(dagger_classifier, pursuer_levels[1:])
    offline_classifier = load_offline(cfg, env, k6)
    offline_policy = LKSPPolicy(offline_classifier, pursuer_levels[1:])

    log.info("=== Figure 5 (feeds fig13): PAP/MPO winrate by evader level, now with CBO-Offline ===")
    key, fk = jax.random.split(key)
    pap_rates, lksp_rates, dagger_rates, offline_rates, key = figure_5(
        cfg, env, pap_llp, pap_hlp, lksp_policy, dagger_policy, offline_policy, evader_levels, fk
    )
    plot_figure_5(pap_rates, lksp_rates, dagger_rates, offline_rates, K, f"{figures_dir}/figure_5_K8.png")
    with open(f"{figures_dir}/figure_5_K8.log", "w") as f5log:
        for j in range(K + 1):
            line = (
                f"Figure 5: evader L{j} -> PAP={pap_rates[j]:.3f} LKSP={lksp_rates[j]:.3f} "
                f"DAgger={dagger_rates[j]:.3f} Offline={offline_rates[j]:.3f}"
            )
            log.info(line)
            f5log.write(line + "\n")

    log.info("=== fig9 (Figure 6): MPO selection rate by trajectory step ===")
    key, fk = jax.random.split(key)
    steps, means, sems = figure_6(cfg, env, lksp_policy, evader_levels, fk)
    plot_figure_6(steps, means, sems, f"{figures_dir}/fig9_K8.png")
    save_values_txt(f"{figures_dir}/fig9_K8.txt", **{"step, mpo_selection_rate, sem": (steps, means, sems)})

    log.info("=== fig6: expected discounted reward vs policy type (CBO-Offline computed fresh, rest reused) ===")
    evader_policy = UniformLevelKEvaderPolicy(evader_levels)
    from pgj.jaxtrl.agent import IndEvader
    from pgj.jaxtrl.reward import SparseEvader, SparsePursuer
    from level_k import EDR, HLP, LLP, SIM_MAX_STEPS
    from evaluate_acc import discounted_reward
    from jaxtrl.sim.policy import PolicyAggregator

    evader_agent = IndEvader(EDR, LLP, HLP, SIM_MAX_STEPS, env.grid.shape)
    team_agent = TeamPursuerAgent(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape)
    agents2 = [team_agent, evader_agent]
    reward2 = [SparsePursuer(LLP, HLP, EDR, 1, 0), SparseEvader(EDR, LLP, HLP, 1, 0)]
    from evaluate import rollout

    policy_offline = PolicyAggregator([offline_policy, evader_policy])
    key, fk = jax.random.split(key)
    transitions_offline = rollout(env, agents2, reward2, policy_offline, cfg.eval.num_envs, fk)
    fig6_results = dict(REUSED_FIG6)
    fig6_results["CBO-Offline"] = discounted_reward(transitions_offline, agent_idx=0)
    log.info(f"Figure 6: CBO-Offline -> expected_discounted_reward={fig6_results['CBO-Offline']:.4f}")
    plot_figure_6acc(fig6_results, f"{figures_dir}/fig6_K8.png")
    save_values_txt(f"{figures_dir}/fig6_K8.txt", expected_discounted_reward=fig6_results)

    log.info("=== fig8: classifier accuracy by trajectory step (both classifiers, never had raw data saved before) ===")
    classifier_policies = {"CBO-DAgger": dagger_policy, "CBO-Offline": offline_policy}
    key, fk = jax.random.split(key)
    fig8_results, key = figure_8acc(cfg, env, classifier_policies, evader_levels, fk)
    plot_figure_8acc(fig8_results, f"{figures_dir}/fig8_K8.png")
    save_values_txt(
        f"{figures_dir}/fig8_K8.txt",
        **{f"{name} (step, accuracy, sem)": v for name, v in fig8_results.items()},
    )

    log.info("=== fig10: expected discounted reward vs TDO delay (CBO-DAgger reused, CBO-Offline computed fresh) ===")
    from evaluate_acc import TDOPolicy

    sub_levels = pursuer_levels[1:]
    offline_by_delay = []
    for delay in delays:
        level_returns = []
        for k in range(1, K + 1):
            agents = [TeamPursuerAgent(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape), IndEvader(EDR, LLP, HLP, SIM_MAX_STEPS, env.grid.shape)]
            reward = [SparsePursuer(LLP, HLP, EDR, 1, 0), SparseEvader(EDR, LLP, HLP, 1, 0)]
            tdo_policy = TDOPolicy(offline_classifier, sub_levels, delay, k - 1)
            policy = PolicyAggregator([tdo_policy, evader_levels[k - 1]])
            key, rk = jax.random.split(key)
            transitions = rollout(env, agents, reward, policy, cfg.eval.num_envs, rk)
            level_returns.append(discounted_reward(transitions, agent_idx=0))
        mean_dr = float(np.mean(level_returns))
        offline_by_delay.append(mean_dr)
        log.info(f"Figure 10: CBO-Offline delay={delay} -> expected_discounted_reward={mean_dr:.4f}")

    fig10_results = {"CBO-DAgger": [DAGGER_TDO_BY_DELAY[d] for d in delays], "CBO-Offline": offline_by_delay}
    plot_figure_10acc(fig10_results, delays, f"{figures_dir}/fig10_K8.png")
    save_values_txt(
        f"{figures_dir}/fig10_K8.txt",
        delays=delays,
        **{f"{name} (expected_discounted_reward by delay)": (delays, v) for name, v in fig10_results.items()},
    )

    log.info("=== fig12: expert action divergence by trajectory step (crashed with zero output before, redone) ===")
    key, fk = jax.random.split(key)
    steps12, means12, sems12 = figure_12acc(cfg, env, pursuer_levels, evader_levels, fk)
    plot_figure_12acc(steps12, means12, sems12, f"{figures_dir}/fig12_K8.png")
    save_values_txt(f"{figures_dir}/fig12_K8.txt", **{"step, mean_sq_action_diff, sem": (steps12, means12, sems12)})

    log.info("done: lean K=8 pass complete")


if __name__ == "__main__":
    main()
