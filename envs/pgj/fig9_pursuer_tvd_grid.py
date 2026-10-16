"""Rebuilds fig9 (was: Mean Total Variational Distance vs Trajectory Step)
per the latest meeting feedback: turn it into a GRID over PURSUER policies
(not evader policies, and not a per-step line), where cell (i, j) is the
average pairwise TVD(pi_{i+1}, pi_{j+1}) between pursuer level i's and
level j's action distributions, pooled over the real interaction histories
encountered when DFO, CBO-DAgger, CBO-Offline, and Random are actually
deployed against the mixed evader distribution.

Methodology: for each of the four orchestrators, roll out eval.num_envs
episodes against the mixed evader-level distribution (same opponent PAP/DFO
were trained against). At every environment step along every one of those
real trajectories, evaluate ALL L+1 pursuer levels' actual (llp, hlp)
policies on that SAME shared team observation (only the orchestrator's own
dispatched choice actually drives the environment -- exactly LKSPPolicy's
own "evaluate every candidate, index into the chosen one" pattern, just
keeping every candidate's stochastic output instead of discarding it), with
n_samples real stochastic action draws per level per step (not the
deterministic mean action -- see fig9_tvd.py's TVD rationale, reused
unchanged here). Samples from all four orchestrators' rollouts are then
pooled per level (this is what "given histories seen at deployment of DFO,
CBO-DAgger, CBO-Offline, and Random" means -- the sampling distribution over
histories h is the union of what these four methods actually encounter, not
an artificial fixed-opponent sweep), subsampled down to --max-pool for KDE
tractability, and used to estimate every pairwise TVD(level_i, level_j) via
the same KDE + Monte Carlo estimator as before.

Axes use the same pi_1..pi_{K+1} notation as fig8 (Section III.B of the
current draft), since this grid lives entirely on the pursuer/hero side --
both axes share the identical label set (TVD is symmetric; the diagonal is
exactly 0 by definition and is not estimated).

Run with, e.g.:
    python envs/pgj/fig9_pursuer_tvd_grid.py \\
        level_k.K=8 level_k.checkpoint_dir=checkpoints/level_k_40M_8k \\
        pap.checkpoint_dir=checkpoints/pap_40M_8k \\
        lksp.checkpoint_dir=checkpoints/lksp_40M_8k \\
        dagger.checkpoint_dir=checkpoints/dagger_40M_8k \\
        offline.checkpoint_dir=checkpoints/offline_40M_8k \\
        eval.num_envs=200 --view-k 5 --n-samples 16 --out figures_acc_new/fig9.png
"""

import argparse
import copy
import logging
import sys

import _compat  # noqa: F401
import hydra
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as th

from pgj.env import ODEnv
from pgj.jaxtrl.agent import IndEvader
from pgj.jaxtrl.reward import SparseEvader, SparsePursuer

from lksp import LKSPPolicy, TeamPursuerAgent, load_pursuer_levels
from pap import UniformLevelKEvaderPolicy, load_evader_levels

from level_k import EDR, HLP, LLP, SIM_MAX_STEPS
from evaluate import rollout, load_pap, load_lksp_meta, load_dagger, load_offline  # noqa: F401 (load_pap unused but kept for symmetry with sibling scripts)
from evaluate_acc import RandomDispatchPolicy
from fig9_tvd import tvd_via_kde
from jaxtrl.sim.policy import Policy, PolicyAggregator
from plot_utils import save_values_txt

log = logging.getLogger(__name__)


class MultiLevelPursuerSamplingObserverPolicy(Policy):
    """Wraps a pursuer-team orchestrator (anything returning a {"env_action":
    ...} dict -- LKSPPolicy or RandomDispatchPolicy) plus a fixed list of
    (llp, hlp) pursuer-level pairs. Every step, draws n_samples real
    stochastic action pairs from EVERY level on the same shared team
    observation the orchestrator itself saw, for post-hoc pairwise TVD
    estimation; only the orchestrator's own dispatched env_action actually
    drives the environment."""

    def __init__(self, driving_policy, level_pairs, n_samples):
        self.driving_policy = driving_policy
        self.level_pairs = level_pairs
        self.n_samples = n_samples

    def init_context(self, key):
        key, dkey = jax.random.split(key)
        subs = [(llp.init_context(key), hlp.init_context(key)) for llp, hlp in self.level_pairs]
        return {"driving": self.driving_policy.init_context(dkey), "levels": subs}

    def reset_context(self, context, key):
        key, dkey = jax.random.split(key)
        subs = [
            (llp.reset_context(lctx, key), hlp.reset_context(hctx, key))
            for (llp, hlp), (lctx, hctx) in zip(self.level_pairs, context["levels"])
        ]
        return {"driving": self.driving_policy.reset_context(context["driving"], dkey), "levels": subs}

    def __call__(self, obs_for_pi, context, key):
        key, driving_key = jax.random.split(key)
        driving_action, driving_ctx = self.driving_policy(obs_for_pi, context["driving"], driving_key)

        level_samples, new_subs = [], []
        for (llp, hlp), (lctx, hctx) in zip(self.level_pairs, context["levels"]):
            key, *sample_keys = jax.random.split(key, self.n_samples + 1)
            samples = []
            new_lctx, new_hctx = lctx, hctx
            for sk in sample_keys:
                lk, hk = jax.random.split(sk)
                # Same (lctx, hctx) reused for every draw -- context updates
                # are action-independent (see fig9_tvd.py), so only the last
                # draw's updated context is kept to advance recurrent state.
                llp_action, new_lctx = llp(obs_for_pi, lctx, lk)
                hlp_action, new_hctx = hlp(obs_for_pi, hctx, hk)
                samples.append(jnp.concatenate([llp_action, hlp_action]))
            level_samples.append(jnp.stack(samples))  # (n_samples, action_dim*2)
            new_subs.append((new_lctx, new_hctx))

        action = {"env_action": driving_action["env_action"], "level_samples": jnp.stack(level_samples)}
        return action, {"driving": driving_ctx, "levels": new_subs}


def view_cfg(cfg, L):
    vc = copy.deepcopy(cfg)
    vc.level_k.K = L
    return vc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view-k", type=int, required=True)
    parser.add_argument("--n-samples", type=int, default=16, help="stochastic action draws per level per step")
    parser.add_argument("--n-mc", type=int, default=2000, help="Monte Carlo integration points per TVD estimate")
    parser.add_argument("--max-step", type=int, default=45, help="drop transitions past this step (matches fig9_tvd's padding-safety default)")
    parser.add_argument("--max-pool", type=int, default=4000, help="cap pooled samples per level before KDE fit, for tractability")
    parser.add_argument("--out", default="figures_acc_new/fig9.png")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    L = args.view_k

    @hydra.main(version_base=None, config_path="../../conf", config_name="evaluate")
    def _main(cfg):
        np.random.seed(cfg.seed)
        th.manual_seed(cfg.seed)
        key = jax.random.key(cfg.seed)
        env = ODEnv()

        key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
        pursuer_levels = load_pursuer_levels(cfg, env, k1)  # FULL K+1 entries, trained checkpoint size
        evader_levels_full = load_evader_levels(cfg, env, k2)
        evader_levels = evader_levels_full[:L]
        level_pairs = pursuer_levels[: L + 1]  # the L+1 levels this grid actually measures (pi_1..pi_{L+1})

        lksp_meta = load_lksp_meta(cfg, env, cfg.level_k.K + 1, k3)
        dfo_policy = LKSPPolicy(lksp_meta, pursuer_levels)
        dagger_classifier = load_dagger(cfg, env, k4)
        dagger_policy = LKSPPolicy(dagger_classifier, pursuer_levels[1:])
        offline_classifier = load_offline(cfg, env, k5)
        offline_policy = LKSPPolicy(offline_classifier, pursuer_levels[1:])
        random_policy = RandomDispatchPolicy(pursuer_levels, num_levels=L + 1)

        evader_policy = UniformLevelKEvaderPolicy(evader_levels)
        evader_agent = IndEvader(EDR, LLP, HLP, SIM_MAX_STEPS, env.grid.shape)
        team_agent = TeamPursuerAgent(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape)
        agents = [team_agent, evader_agent]
        reward = [SparsePursuer(LLP, HLP, EDR, 1, 0), SparseEvader(EDR, LLP, HLP, 1, 0)]

        # level_idx -> list of (n_t, n_samples, action_dim*2) arrays, pooled
        # across ALL four orchestrators' real deployment histories.
        pooled_level_samples = [[] for _ in range(L + 1)]

        for name, driving in [
            ("DFO", dfo_policy), ("CBO-DAgger", dagger_policy),
            ("CBO-Offline", offline_policy), ("Random", random_policy),
        ]:
            observer = MultiLevelPursuerSamplingObserverPolicy(driving, level_pairs, args.n_samples)
            policy = PolicyAggregator([observer, evader_policy])

            key, rk = jax.random.split(key)
            transitions = rollout(env, agents, reward, policy, cfg.eval.num_envs, rk)

            level_samples = np.array(transitions.action[0]["level_samples"])  # (N, L+1, n_samples, d)
            step_count = np.array(transitions.step_count)
            mask = step_count <= args.max_step
            level_samples = level_samples[mask]

            for lvl in range(L + 1):
                pooled_level_samples[lvl].append(level_samples[:, lvl])  # (n_t, n_samples, d)
            log.info(f"TVD grid: {name} deployment histories collected ({mask.sum()} transitions)")

        rng = np.random.default_rng(0)
        pooled = []
        for lvl in range(L + 1):
            full = np.concatenate(pooled_level_samples[lvl], axis=0).reshape(-1, pooled_level_samples[lvl][0].shape[-1])
            if len(full) > args.max_pool:
                idx = rng.choice(len(full), args.max_pool, replace=False)
                full = full[idx]
            pooled.append(full)
            log.info(f"pi_{lvl + 1}: pooled {len(full)} action samples")

        n = L + 1
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                tvd = tvd_via_kde(pooled[i], pooled[j], rng, n_mc=args.n_mc)
                matrix[i, j] = matrix[j, i] = tvd
                log.info(f"TVD(pi_{i + 1}, pi_{j + 1}) = {tvd:.4f}")

        labels = [rf"$\pi_{{{i + 1}}}$" for i in range(n)]
        fig, ax = plt.subplots(figsize=(6, 5.2))
        im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(n), labels)
        ax.set_yticks(range(n), labels)
        ax.set_xlabel("Pursuer Policy")
        ax.set_ylabel("Pursuer Policy")
        ax.set_title("Mean Pairwise Action TVD Between Pursuer Policies")
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if matrix[i, j] < 0.6 else "black")
        fig.colorbar(im, ax=ax, label="TVD")
        fig.tight_layout()
        fig.savefig(args.out, dpi=150)
        plt.close(fig)
        log.info(f"wrote {args.out}")

        txt_path = args.out.rsplit(".", 1)[0] + ".txt"
        save_values_txt(txt_path, **{f"{labels[i]} TVD by pursuer policy ({', '.join(labels)})": row for i, row in enumerate(matrix.tolist())})
        log.info(f"wrote {txt_path}")

    _main()


if __name__ == "__main__":
    main()
