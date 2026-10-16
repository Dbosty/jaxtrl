"""Rebuilds fig9 (was: Squared Difference in Action vs Trajectory Timesteps)
per today's revision list, item 7: Y axis -> Mean Total Variational
Distance, X axis capped at 40, title -> "Mean Total Variational Distance vs
Trajectory Step", where mean TVD at a given step = mean over evader levels
k of TVD(pi_{k+1}, pi_{!=k+1}) (that level's action distribution vs. the
pooled mixture of every OTHER level's action distribution), at that step.

Why sampling + KDE instead of a closed-form Gaussian TVD: our evader
policies use tanh-squashed Gaussians (see lstm.py's distribution="tanh"
option), so an untransformed-Gaussian TVD formula would be measuring the
wrong distribution. Instead: at each shared history h (the same history all
K level-policies see, exactly as evaluate.py's MultiLevelObserverPolicy
already does for the old squared-action-difference metric), draw N_SAMPLES
real stochastic action draws from EACH level's actual policy (not the
deterministic mean action), fit a KDE to each level's pooled samples, and
estimate TVD via Monte Carlo integration against a mixture proposal:

    TVD(P, Q) = 0.5 * E_{x ~ 0.5(P+Q)}[ |p(x) - q(x)| / (0.5*(p(x)+q(x))) ]

which is an unbiased importance-sampling estimator of 0.5*integral|p-q|dx,
using the actual empirical (KDE-fitted) densities -- correct regardless of
the true distribution's shape.

Run with, e.g.:
    python envs/pgj/fig9_tvd.py \\
        level_k.K=8 level_k.checkpoint_dir=checkpoints/level_k_40M_8k \\
        pap.checkpoint_dir=checkpoints/pap_40M_8k \\
        lksp.checkpoint_dir=checkpoints/lksp_40M_8k \\
        dagger.checkpoint_dir=checkpoints/dagger_40M_8k \\
        offline.checkpoint_dir=checkpoints/offline_40M_8k \\
        eval.num_envs=200 --view-k 5 --n-samples 16 --out figures_acc/fig9.png
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
from scipy.stats import gaussian_kde

from pgj.env import ODEnv

from lksp import load_pursuer_levels
from pap import load_evader_levels

from level_k import EDR
from evaluate import rollout, pursuer_vs_evader_agents_reward, DiffTrackingEvaderAgent
from jaxtrl.sim.policy import Policy, PolicyAggregator
from plot_utils import save_values_txt

log = logging.getLogger(__name__)


class MultiLevelSamplingObserverPolicy(Policy):
    """Like evaluate.py's MultiLevelObserverPolicy, but each level policy is
    called n_samples times (different keys, same obs/context -- state update
    is action-independent, see module docstring) to get real stochastic
    action draws instead of one deterministic mean action per step."""

    def __init__(self, driving_policy: Policy, level_policies: list[Policy], n_samples: int):
        self.driving_policy = driving_policy
        self.level_policies = level_policies
        self.n_samples = n_samples

    def init_context(self, key):
        keys = jax.random.split(key, len(self.level_policies) + 1)
        return {
            "driving": self.driving_policy.init_context(keys[0]),
            "levels": [lvl.init_context(k) for lvl, k in zip(self.level_policies, keys[1:])],
        }

    def reset_context(self, context, key):
        keys = jax.random.split(key, len(self.level_policies) + 1)
        return {
            "driving": self.driving_policy.reset_context(context["driving"], keys[0]),
            "levels": [
                lvl.reset_context(c, k)
                for lvl, c, k in zip(self.level_policies, context["levels"], keys[1:])
            ],
        }

    def __call__(self, obs_for_pi, context, key):
        key, driving_key = jax.random.split(key)
        driving_action, driving_ctx = self.driving_policy(obs_for_pi, context["driving"], driving_key)

        level_samples, new_level_ctxs = [], []
        for lvl, ctx in zip(self.level_policies, context["levels"]):
            key, *sample_keys = jax.random.split(key, self.n_samples + 1)
            samples, new_ctx = [], None
            for sk in sample_keys:
                action, new_ctx = lvl(obs_for_pi, ctx, sk)
                samples.append(action)
            level_samples.append(jnp.stack(samples))  # (n_samples, action_dim)
            new_level_ctxs.append(new_ctx)

        action = {"env_action": driving_action, "level_samples": jnp.stack(level_samples)}  # (K, n_samples, action_dim)
        return action, {"driving": driving_ctx, "levels": new_level_ctxs}


def tvd_via_kde(samples_p, samples_q, rng, n_mc=2000):
    """samples_p, samples_q: (n, d) arrays of real action draws. Returns a
    Monte Carlo estimate of TVD(P, Q) using KDE fits to each sample set."""
    d = samples_p.shape[1]
    try:
        kde_p = gaussian_kde(samples_p.T)
        kde_q = gaussian_kde(samples_q.T)
    except np.linalg.LinAlgError:
        return np.nan  # degenerate (near-zero variance) sample set

    half = n_mc // 2
    idx_p = rng.integers(0, len(samples_p), half)
    idx_q = rng.integers(0, len(samples_q), n_mc - half)
    x = np.concatenate([samples_p[idx_p], samples_q[idx_q]], axis=0)  # (n_mc, d)

    p = kde_p(x.T)
    q = kde_q(x.T)
    r = 0.5 * (p + q)
    r = np.where(r <= 0, 1e-12, r)
    integrand = np.abs(p - q) / r
    return float(0.5 * integrand.mean())


def view_cfg(cfg, L):
    vc = copy.deepcopy(cfg)
    vc.level_k.K = L
    return vc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view-k", type=int, required=True)
    parser.add_argument("--n-samples", type=int, default=16, help="stochastic action draws per level per step")
    parser.add_argument("--n-mc", type=int, default=2000, help="Monte Carlo integration points per TVD estimate")
    parser.add_argument("--max-step", type=int, default=45, help="only bin/estimate TVD for steps <= this (plot itself is further capped at 40)")
    parser.add_argument("--out", default="figures_acc/fig9.png")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    L = args.view_k

    @hydra.main(version_base=None, config_path="../../conf", config_name="evaluate")
    def _main(cfg):
        np.random.seed(cfg.seed)
        key = jax.random.key(cfg.seed)
        env = ODEnv()
        vc = view_cfg(cfg, L)

        key, k1, k2 = jax.random.split(key, 3)
        pursuer_levels = load_pursuer_levels(cfg, env, k1)
        evader_levels = load_evader_levels(cfg, env, k2)[:L]

        # step_bin -> level_idx -> list of (n_samples, 4) arrays collected across combos/envs
        step_level_samples = {}

        for ev_idx in range(L):
            for p_idx in range(L + 1):
                llp_p, hlp_p = pursuer_levels[p_idx]
                edr_agent = DiffTrackingEvaderAgent(EDR, "LLP", "HLP", 100, env.grid.shape) if False else None
                from level_k import LLP, HLP, SIM_MAX_STEPS
                edr_agent = DiffTrackingEvaderAgent(EDR, LLP, HLP, SIM_MAX_STEPS, env.grid.shape)
                agents, reward = pursuer_vs_evader_agents_reward(env, edr_agent)
                policy = PolicyAggregator([
                    llp_p, hlp_p,
                    MultiLevelSamplingObserverPolicy(evader_levels[ev_idx], evader_levels, args.n_samples),
                ])

                key, rk = jax.random.split(key)
                transitions = rollout(env, agents, reward, policy, cfg.eval.num_envs, rk)

                level_samples = np.array(transitions.action[EDR]["level_samples"])  # (N, K, n_samples, 4)
                step_count = np.array(transitions.step_count)

                mask = step_count <= args.max_step
                level_samples, step_count = level_samples[mask], step_count[mask]
                for t in np.unique(step_count):
                    tmask = step_count == t
                    bucket = step_level_samples.setdefault(int(t), [[] for _ in range(L)])
                    for lvl in range(L):
                        bucket[lvl].append(level_samples[tmask, lvl])  # (n_t, n_samples, 4)

                log.info(f"TVD data: driving=L{ev_idx + 1} vs pursuer L{p_idx} collected")

        rng = np.random.default_rng(0)
        steps_sorted = sorted(step_level_samples.keys())
        mean_tvd_by_step, sem_tvd_by_step = [], []

        for t in steps_sorted:
            bucket = step_level_samples[t]
            pooled = [np.concatenate(bucket[lvl], axis=0).reshape(-1, 4) for lvl in range(L)]
            tvds = []
            for k in range(L):
                p_samples = pooled[k]
                q_samples = np.concatenate([pooled[j] for j in range(L) if j != k], axis=0)
                if len(p_samples) < 10 or len(q_samples) < 10:
                    continue
                tvds.append(tvd_via_kde(p_samples, q_samples, rng, n_mc=args.n_mc))
            tvds = np.array([v for v in tvds if not np.isnan(v)])
            if len(tvds) == 0:
                continue
            mean_tvd_by_step.append(tvds.mean())
            sem_tvd_by_step.append(tvds.std() / np.sqrt(len(tvds)))
            log.info(f"step={t}: mean_tvd={tvds.mean():.4f} (n_levels={len(tvds)})")

        steps_arr = np.array(steps_sorted[: len(mean_tvd_by_step)])
        means_arr = np.array(mean_tvd_by_step)
        sems_arr = np.array(sem_tvd_by_step)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(steps_arr, means_arr, color="tab:purple")
        ax.fill_between(steps_arr, means_arr - sems_arr, means_arr + sems_arr, alpha=0.3, color="tab:purple", label="+/- 1 SE")
        ax.set_xlim(0, 40)
        ax.set_xlabel("Trajectory Step")
        ax.set_ylabel("Mean Total Variational Distance")
        ax.set_title("Mean Total Variational Distance vs Trajectory Step")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out, dpi=150)
        plt.close(fig)
        log.info(f"wrote {args.out}")

        txt_path = args.out.rsplit(".", 1)[0] + ".txt"
        save_values_txt(txt_path, **{"step, mean_tvd, sem": (steps_arr.tolist(), means_arr.tolist(), sems_arr.tolist())})
        log.info(f"wrote {txt_path}")

    _main()


if __name__ == "__main__":
    main()
