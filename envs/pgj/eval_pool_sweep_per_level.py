"""Per-evader-level diagnostic for the pursuer-pool-size sweep: does a
RESTRICTED pursuer pool (e.g. pool_size=2, only pi_1/pi_2 available) hold up
against evader levels FAR beyond its own pool, or does its aggregate
training-time win rate just average over a few good levels and several bad
ones?

Training sampled the evader level uniformly at random each episode, so the
training CSVs only give a pooled metric. This script instead evaluates each
ALREADY-TRAINED checkpoint (checkpoints/lksp_poolsweep_n{2..6}_evk6) against
each evader level 1..6 held FIXED for the whole rollout -- same technique as
fig8's pursuer-vs-evader matrix, just reusing the pool-sweep checkpoints
instead of retraining anything. 5 pool sizes x 6 evader levels = 30 fresh
evaluation rollouts.

Run with, e.g.:
    python envs/pgj/eval_pool_sweep_per_level.py \\
        level_k.K=8 level_k.checkpoint_dir=checkpoints/level_k_40M_8k \\
        eval.num_envs=500
"""
import copy

import _compat  # noqa: F401
import hydra
import jax
import matplotlib
import numpy as np
import torch as th

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pgj.env import ODEnv
from pgj.jaxtrl.agent import IndEvader
from pgj.jaxtrl.reward import SparseEvader, SparsePursuer

from lksp import LKSPPolicy, TeamPursuerAgent, load_pursuer_levels
from pap import load_evader_levels

from level_k import EDR, HLP, LLP, SIM_MAX_STEPS
from evaluate import rollout, load_lksp_meta, episode_lengths
from jaxtrl.sim.policy import PolicyAggregator
from plot_utils import save_values_txt

POOL_SIZES = [2, 3, 4, 5, 6]
EVADER_LEVELS = [1, 2, 3, 4, 5, 6]
GAMMA = 0.99
CMAP = plt.get_cmap("viridis")

# Resume data: the first run (background task b68ac0mvs) completed these 20
# of 30 combos cleanly before dying to an unrelated segfault (exit 139, no
# python traceback -- transient XLA/CUDA crash, not a logic bug: every value
# up to this point was sane and monotonic-ish). Hardcoded from that run's
# own printed output to avoid re-paying ~20 GPU-minutes recomputing them.
RESUME = {
    (2, 1): (0.7995, 0.0048, 0.7530, 0.0045), (2, 2): (0.7855, 0.0044, 0.7551, 0.0042),
    (2, 3): (0.8064, 0.0048, 0.7587, 0.0045), (2, 4): (0.8767, 0.0037, 0.8307, 0.0036),
    (2, 5): (0.8458, 0.0042, 0.8001, 0.0040), (2, 6): (0.8579, 0.0040, 0.8140, 0.0038),
    (3, 1): (0.8845, 0.0036, 0.8387, 0.0034), (3, 2): (0.8563, 0.0036, 0.8231, 0.0035),
    (3, 3): (0.8113, 0.0048, 0.7626, 0.0045), (3, 4): (0.9118, 0.0031, 0.8669, 0.0030),
    (3, 5): (0.8414, 0.0043, 0.7951, 0.0041), (3, 6): (0.9069, 0.0031, 0.8656, 0.0030),
    (4, 1): (0.8435, 0.0043, 0.7952, 0.0041), (4, 2): (0.8660, 0.0034, 0.8337, 0.0033),
    (4, 3): (0.8561, 0.0042, 0.8055, 0.0040), (4, 4): (0.9089, 0.0031, 0.8660, 0.0030),
    (4, 5): (0.8376, 0.0044, 0.7892, 0.0042), (4, 6): (0.9051, 0.0032, 0.8635, 0.0030),
    (5, 1): (0.8329, 0.0045, 0.7836, 0.0042), (5, 2): (0.8692, 0.0034, 0.8374, 0.0033),
}


def per_episode_discounted(transitions, agent_idx=0, gamma=GAMMA):
    """Per-completed-episode sum_t gamma^t * r_t (not just the pooled mean
    discounted_reward() in evaluate_acc.py gives -- keeping every episode's
    value lets us report a real SEM instead of just a point estimate)."""
    reward = np.array(transitions.reward[..., agent_idx])
    step_count = np.array(transitions.step_count)
    done = np.array(transitions.terminated | transitions.truncated)
    discounted = reward * (gamma**step_count)
    return discounted[done]


@hydra.main(version_base=None, config_path="../../conf", config_name="evaluate")
def main(cfg):
    np.random.seed(cfg.seed)
    th.manual_seed(cfg.seed)
    key = jax.random.key(cfg.seed)
    env = ODEnv()

    key, pk, ek = jax.random.split(key, 3)
    pursuer_levels_full = load_pursuer_levels(cfg, env, pk)  # FULL K+1=9, trained checkpoint size
    evader_levels_full = load_evader_levels(cfg, env, ek)  # FULL K=8

    n_p, n_e = len(POOL_SIZES), len(EVADER_LEVELS)
    winrate = np.zeros((n_p, n_e))
    winrate_sem = np.zeros((n_p, n_e))
    disc_reward = np.zeros((n_p, n_e))
    disc_reward_sem = np.zeros((n_p, n_e))

    for i, pool_size in enumerate(POOL_SIZES):
        if all((pool_size, lvl) in RESUME for lvl in EVADER_LEVELS):
            for j, lvl in enumerate(EVADER_LEVELS):
                winrate[i, j], winrate_sem[i, j], disc_reward[i, j], disc_reward_sem[i, j] = RESUME[(pool_size, lvl)]
                print(f"pool={pool_size} evader_L{lvl}: [resumed] winrate={winrate[i, j]:.4f}+-{winrate_sem[i, j]:.4f}  discounted_reward={disc_reward[i, j]:.4f}+-{disc_reward_sem[i, j]:.4f}")
            continue

        cfg_i = copy.deepcopy(cfg)
        cfg_i.lksp.checkpoint_dir = f"checkpoints/lksp_poolsweep_n{pool_size}_evk6"
        key, mk = jax.random.split(key)
        lksp_meta = load_lksp_meta(cfg_i, env, pool_size, mk)
        lksp_policy = LKSPPolicy(lksp_meta, pursuer_levels_full[:pool_size])

        for j, lvl in enumerate(EVADER_LEVELS):
            if (pool_size, lvl) in RESUME:
                winrate[i, j], winrate_sem[i, j], disc_reward[i, j], disc_reward_sem[i, j] = RESUME[(pool_size, lvl)]
                print(f"pool={pool_size} evader_L{lvl}: [resumed] winrate={winrate[i, j]:.4f}+-{winrate_sem[i, j]:.4f}  discounted_reward={disc_reward[i, j]:.4f}+-{disc_reward_sem[i, j]:.4f}")
                continue

            team_agent = TeamPursuerAgent(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape)
            edr_agent = IndEvader(EDR, LLP, HLP, SIM_MAX_STEPS, env.grid.shape)
            agents = [team_agent, edr_agent]
            reward = [SparsePursuer(LLP, HLP, EDR, 1, 0), SparseEvader(EDR, LLP, HLP, 1, 0)]
            policy = PolicyAggregator([lksp_policy, evader_levels_full[lvl - 1]])

            key, rk = jax.random.split(key)
            transitions = rollout(env, agents, reward, policy, cfg.eval.num_envs, rk)

            _, win = episode_lengths(transitions, LLP)
            winrate[i, j] = win.mean()
            winrate_sem[i, j] = win.std() / np.sqrt(len(win))

            dvals = per_episode_discounted(transitions)
            disc_reward[i, j] = dvals.mean()
            disc_reward_sem[i, j] = dvals.std() / np.sqrt(len(dvals))

            print(
                f"pool={pool_size} evader_L{lvl}: "
                f"winrate={winrate[i, j]:.4f}+-{winrate_sem[i, j]:.4f}  "
                f"discounted_reward={disc_reward[i, j]:.4f}+-{disc_reward_sem[i, j]:.4f}  "
                f"(n={len(win)})"
            )

    def plot(matrix, sem, ylabel, out_path):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for i, pool_size in enumerate(POOL_SIZES):
            color = CMAP(i / (n_p - 1))
            ax.errorbar(
                EVADER_LEVELS, matrix[i], yerr=sem[i], marker="o", capsize=3,
                color=color, label=rf"pool size {pool_size} ($\pi_1..\pi_{pool_size}$)",
            )
        ax.set_xlabel("Evader Level (fixed opponent)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by Evader Level, per Pursuer Pool Size")
        ax.set_xticks(EVADER_LEVELS)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {out_path}")

    plot(winrate, winrate_sem, "Pursuer Win Rate", "figures_acc_new/fig_poolsweep_per_level_winrate.png")
    plot(disc_reward, disc_reward_sem, "Expected Discounted Reward", "figures_acc_new/fig_poolsweep_per_level_discounted_reward.png")

    save_values_txt(
        "figures_acc_new/fig_poolsweep_per_level.txt",
        evader_levels=EVADER_LEVELS,
        **{f"pool_{n} winrate by evader level": row.tolist() for n, row in zip(POOL_SIZES, winrate)},
        **{f"pool_{n} winrate_sem by evader level": row.tolist() for n, row in zip(POOL_SIZES, winrate_sem)},
        **{f"pool_{n} discounted_reward by evader level": row.tolist() for n, row in zip(POOL_SIZES, disc_reward)},
        **{f"pool_{n} discounted_reward_sem by evader level": row.tolist() for n, row in zip(POOL_SIZES, disc_reward_sem)},
    )
    print("wrote figures_acc_new/fig_poolsweep_per_level.txt")


if __name__ == "__main__":
    main()
