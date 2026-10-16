"""Figures 6, 8, 10, and 12 from the APEGS_ACC draft ("Dynamic Orchestration
of Expert Strategies Against Unknown Opponents").

That draft renames evaluate.py's LKSP -> MPO, DAgger classifier -> CBO-DAgger,
offline classifier -> CBO-Offline, and introduces one new baseline not built
anywhere else in this repo: TDO (Time-Delayed Oracle) -- a classifier
(CBO-DAgger or CBO-Offline) drives pursuer-level selection for the first
`delay` steps of an episode, then switching to the ground-truth active
evader level for the rest of the episode.

Figures 4, 7, 9, 11, 13 of that draft are NOT produced here:
  - Fig 4 is a static environment diagram (not derived from data).
  - Figs 1/2/3/5 are algorithm-pseudocode boxes, not plots (see APPENDIX.md
    and paper/dagger_algorithm.tex).
  - Fig 7 (reward vs training timesteps) = plot_training_curves.py's output.
  - Fig 9 (selection rate vs trajectory step) = evaluate.py's Figure 6.
  - Fig 11 (pursuer vs evader winrate matrix) = evaluate.py's Figure 2.
  - Fig 13 (winrate gap / action variance vs number of experts) =
    envs/pgj/figure_13.py's output.
These are just copied into figures_acc/ under this draft's numbering.

Requires the same checkpoints as evaluate.py (level_k.py, pap.py, lksp.py,
dagger.py, and optionally offline_classifier.py, all at matching level_k.K).

Run with, e.g.:
    python envs/pgj/evaluate_acc.py level_k.K=4 \\
        level_k.checkpoint_dir=checkpoints/level_k \\
        pap.checkpoint_dir=checkpoints/pap \\
        lksp.checkpoint_dir=checkpoints/lksp \\
        dagger.checkpoint_dir=checkpoints/dagger \\
        offline.checkpoint_dir=checkpoints/offline \\
        eval.figures_dir=figures_acc eval.suffix=_K4
"""

import logging
import os

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
from pgj.jaxtrl.agent import IndEvader, IndTeamHlp, IndTeamLlp
from pgj.jaxtrl.reward import SparseEvader, SparsePursuer
from pgj.jaxtrl.term import is_terminated  # noqa: F401 (kept for parity with evaluate.py's imports)

from jaxtrl.sim.policy import Policy, PolicyAggregator

from level_k import EDR, HLP, LLP, SIM_MAX_STEPS  # noqa: F401 (HLP kept for parity)
from pap import UniformLevelKEvaderPolicy, load_evader_levels
from lksp import LKSPPolicy, TeamPursuerAgent, load_pursuer_levels

from evaluate import (
    load_pap, load_lksp_meta, load_dagger, load_offline,
    rollout, pursuer_vs_evader_agents_reward,
)
from plot_utils import ALGO_COLORS, save_values_txt

log = logging.getLogger(__name__)

GAMMA = 0.99  # matches conf/train/ppo.yaml's train.gamma


# ---------------------------------------------------------------------------
# shared helper: expected discounted reward
# ---------------------------------------------------------------------------


def discounted_reward(transitions, agent_idx=0, gamma=GAMMA):
    """Mean, over every completed episode in the rollout, of
    sum_t gamma^t * r_t. Reward here (SparsePursuer with l0=1, l1=0) is
    nonzero only at the single terminal step where the pursuer wins, so this
    reduces to mean(gamma^(win_step) over won episodes, 0 over lost/timed-out
    episodes) -- rewarding both winning AND winning fast."""
    reward = np.array(transitions.reward[..., agent_idx])
    step_count = np.array(transitions.step_count)
    terminated = np.array(transitions.terminated)
    truncated = np.array(transitions.truncated)
    done = terminated | truncated

    discounted = reward * (gamma**step_count)
    n_episodes = int(done.sum())
    return float(discounted.sum() / n_episodes)


# ---------------------------------------------------------------------------
# Figure 6: expected discounted reward vs policy type
# ---------------------------------------------------------------------------


def figure_6(cfg, env, pap_llp, pap_hlp, lksp_policy, dagger_policy, offline_policy, evader_levels, key, random_policy=None):
    """Single rollout per policy type against the mixed evader-level
    opponent (levels 1..K, resampled every episode) -- the same opponent
    distribution PAP/LKSP were trained against (pap.py/lksp.py), so this is
    "policy type" performance against its natural target distribution,
    rather than the per-evader-level breakdown evaluate.py's Figure 5 gives."""
    evader_policy = UniformLevelKEvaderPolicy(evader_levels)
    evader_agent = IndEvader(EDR, LLP, HLP, SIM_MAX_STEPS, env.grid.shape)

    results = {}

    agents3, reward3 = pursuer_vs_evader_agents_reward(env, evader_agent)
    policy3 = PolicyAggregator([pap_llp, pap_hlp, evader_policy])
    key, rk = jax.random.split(key)
    transitions = rollout(env, agents3, reward3, policy3, cfg.eval.num_envs, rk)
    results["PAP"] = discounted_reward(transitions, agent_idx=0)

    team_agent = TeamPursuerAgent(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape)
    agents2 = [team_agent, evader_agent]
    reward2 = [SparsePursuer(LLP, HLP, EDR, 1, 0), SparseEvader(EDR, LLP, HLP, 1, 0)]

    named_policies = [("MPO", lksp_policy), ("CBO-DAgger", dagger_policy)]
    if offline_policy is not None:
        named_policies.append(("CBO-Offline", offline_policy))
    if random_policy is not None:
        named_policies.append(("Random", random_policy))

    for name, pol in named_policies:
        policy2 = PolicyAggregator([pol, evader_policy])
        key, rk = jax.random.split(key)
        transitions2 = rollout(env, agents2, reward2, policy2, cfg.eval.num_envs, rk)
        results[name] = discounted_reward(transitions2, agent_idx=0)

    for name, val in results.items():
        log.info(f"Figure 6: {name} -> expected_discounted_reward={val:.4f}")

    return results, key


def plot_figure_6(results, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    display_names = ["DFO" if n == "MPO" else n for n in results.keys()]
    values = list(results.values())
    colors = ALGO_COLORS
    ax.bar(
        display_names, values,
        color=[colors.get(n, colors.get("MPO") if n == "DFO" else "gray") for n in display_names],
    )
    ax.set_ylabel("Expected Discounted Reward")
    ax.set_xlabel("Orchestrator")
    ax.set_title("Expected Discounted Reward vs Orchestrator")
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.15 or 0.05
    ax.set_ylim(max(0, lo - pad), min(1, hi + pad))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8: classifier accuracy vs trajectory timestep
# ---------------------------------------------------------------------------


def figure_8(cfg, env, classifier_policies, evader_levels, key):
    """Same measurement as evaluate.py's Figure 6 (selection rate by
    trajectory step), but for CBO-DAgger/CBO-Offline instead of MPO.
    classifier_policies values are LKSPPolicy(classifier, pursuer_levels[1:])
    instances -- a K-way (not K+1-way) action space, 0-indexed over evader
    levels 1..K, so the "correct" label for evader level k is (k - 1), not k
    as in evaluate.py's Figure 6."""
    K = cfg.level_k.K
    results = {}

    for name, policy in classifier_policies.items():
        all_steps, all_hits = [], []
        for k in range(1, K + 1):
            team_agent = TeamPursuerAgent(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape)
            edr_agent = IndEvader(EDR, LLP, HLP, SIM_MAX_STEPS, env.grid.shape)
            agents = [team_agent, edr_agent]
            reward = [SparsePursuer(LLP, HLP, EDR, 1, 0), SparseEvader(EDR, LLP, HLP, 1, 0)]
            pa = PolicyAggregator([policy, evader_levels[k - 1]])

            key, rk = jax.random.split(key)
            transitions = rollout(env, agents, reward, pa, cfg.eval.num_envs, rk)

            step_count = np.array(transitions.step_count)
            selected_k = np.array(transitions.action[0]["k"])
            all_steps.append(step_count)
            all_hits.append((selected_k == (k - 1)).astype(np.float32))
            log.info(f"Figure 8: {name} evader L{k} rollout collected")

        all_steps = np.concatenate(all_steps)
        all_hits = np.concatenate(all_hits)

        steps, means, sems = [], [], []
        for t in range(int(all_steps.max()) + 1):
            mask = all_steps == t
            n = int(mask.sum())
            if n == 0:
                continue
            steps.append(t)
            means.append(float(all_hits[mask].mean()))
            sems.append(float(all_hits[mask].std() / np.sqrt(n)))
        results[name] = (np.array(steps), np.array(means), np.array(sems))

    return results, key


def plot_figure_8(results, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ALGO_COLORS
    for name, (steps, means, sems) in results.items():
        color = colors.get(name, "tab:blue")
        ax.plot(steps, means, color=color, label=name)
        ax.fill_between(steps, means - sems, means + sems, alpha=0.3, color=color)
    ax.set_xlabel("Trajectory Step")
    ax.set_ylabel("Classification Accuracy")
    ax.set_title("Classifier Accuracy by Trajectory Step")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Random policy: the paper's TDO-random "classifier" (U([n])), used
# standalone (no oracle switch) as a policy-type baseline in its own right --
# isolates how much of a classifier-based orchestrator's performance comes
# from the classifier vs. the selection mechanism itself.
# ---------------------------------------------------------------------------


class RandomDispatchPolicy(Policy):
    """Uniform-random pursuer-level selector -- same dispatch mechanism as
    LKSPPolicy/TDOPolicy (deterministically evaluate every frozen candidate's
    action every step, index into it by the chosen k), except k is drawn
    uniformly at random each step instead of coming from a trained network.
    pursuer_levels is always the FULL list (never truncated -- indexing must
    stay in bounds); num_levels controls how many of its FIRST entries are
    eligible to be drawn, letting this match a restricted "K view" the same
    way evaluate_multi_view.py's other figures do."""

    def __init__(self, pursuer_levels, num_levels=None):
        self.pursuer_levels = pursuer_levels
        self.num_levels = num_levels or len(pursuer_levels)

    def init_context(self, key):
        subs = [(llp.init_context(key), hlp.init_context(key)) for llp, hlp in self.pursuer_levels]
        return {"subs": subs}

    def reset_context(self, context, key):
        subs = [
            (llp.reset_context(lctx, key), hlp.reset_context(hctx, key))
            for (llp, hlp), (lctx, hctx) in zip(self.pursuer_levels, context["subs"])
        ]
        return {"subs": subs}

    def __call__(self, obs_for_pi, context, key):
        key, sel_key = jax.random.split(key)
        k = jax.random.randint(sel_key, (), 0, self.num_levels)

        llp_actions, hlp_actions, new_subs = [], [], []
        for (llp, hlp), (lctx, hctx) in zip(self.pursuer_levels, context["subs"]):
            llp_action, lctx = llp.deterministic_sample(obs_for_pi, lctx)
            hlp_action, hctx = hlp.deterministic_sample(obs_for_pi, hctx)
            llp_actions.append(llp_action)
            hlp_actions.append(hlp_action)
            new_subs.append((lctx, hctx))

        env_action = jnp.stack([jnp.stack(llp_actions)[k], jnp.stack(hlp_actions)[k]])
        return {"k": k, "env_action": env_action}, {"subs": new_subs}


class RandomMetaActor:
    """Lightweight random "classifier" -- the paper's TDO-random baseline
    (U([n]) used as TDOPolicy's pre-oracle "classifier"). Matches the same
    minimal interface a real classifier (DiscreteMLP) exposes to TDOPolicy:
    __call__(obs_for_pi, context, key) -> (k, new_context); no dispatch
    logic of its own, since TDOPolicy does the dispatch."""

    def __init__(self, num_levels):
        self.num_levels = num_levels

    def init_context(self, key):
        return None

    def reset_context(self, context, key):
        return context

    def __call__(self, obs_for_pi, context, key):
        k = jax.random.randint(key, (), 0, self.num_levels)
        return k, context


# ---------------------------------------------------------------------------
# Figure 10: expected discounted reward vs TDO delay
# ---------------------------------------------------------------------------


class TDOPolicy(Policy):
    """Time-Delayed Oracle: identical dispatch mechanism to lksp.py's
    LKSPPolicy, except for the first `delay` steps of each episode the
    classifier (meta_actor) picks which frozen pursuer level acts; after
    that it switches to the ground-truth active evader level (oracle_k).
    oracle_k is a fixed python int for the whole rollout batch, matching
    the fixed-opponent-level pattern evaluate.py's Figures 5/6/8 already use
    (we're the ones choosing which evader level is "driving", so the ground
    truth is always known here, exactly as dagger.py's training loop knows
    it -- see that module's docstring)."""

    def __init__(self, meta_actor, pursuer_levels, delay, oracle_k):
        self.meta_actor = meta_actor
        self.pursuer_levels = pursuer_levels
        self.delay = delay
        self.oracle_k = oracle_k

    def init_context(self, key):
        key, meta_key = jax.random.split(key)
        subs = [(llp.init_context(key), hlp.init_context(key)) for llp, hlp in self.pursuer_levels]
        return {"meta": self.meta_actor.init_context(meta_key), "subs": subs, "t": jnp.array(0, dtype=jnp.int32)}

    def reset_context(self, context, key):
        key, meta_key = jax.random.split(key)
        subs = [
            (llp.reset_context(lctx, key), hlp.reset_context(hctx, key))
            for (llp, hlp), (lctx, hctx) in zip(self.pursuer_levels, context["subs"])
        ]
        return {
            "meta": self.meta_actor.reset_context(context["meta"], meta_key),
            "subs": subs,
            "t": jnp.array(0, dtype=jnp.int32),
        }

    def __call__(self, obs_for_pi, context, key):
        pred_k, meta_ctx = self.meta_actor(obs_for_pi, context["meta"], key)
        t = context["t"]
        k = jnp.where(t < self.delay, pred_k, self.oracle_k)

        llp_actions, hlp_actions, new_subs = [], [], []
        for (llp, hlp), (lctx, hctx) in zip(self.pursuer_levels, context["subs"]):
            llp_action, lctx = llp.deterministic_sample(obs_for_pi, lctx)
            hlp_action, hctx = hlp.deterministic_sample(obs_for_pi, hctx)
            llp_actions.append(llp_action)
            hlp_actions.append(hlp_action)
            new_subs.append((lctx, hctx))

        env_action = jnp.stack([jnp.stack(llp_actions)[k], jnp.stack(hlp_actions)[k]])
        return {"k": k, "env_action": env_action}, {"meta": meta_ctx, "subs": new_subs, "t": t + 1}


def figure_10(cfg, env, pursuer_levels, evader_levels, classifiers, key, delays):
    """classifiers: {"CBO-DAgger": dagger_classifier, "CBO-Offline": ... (optional)}
    (the raw classifier nets, NOT wrapped in LKSPPolicy -- TDOPolicy does its
    own wrapping). Reports mean discounted reward across evader levels 1..K
    (equal-weighted, matching UniformLevelKEvaderPolicy's uniform sampling)
    for each delay. delays is cfg.eval.tdo_delays."""
    K = cfg.level_k.K
    sub_levels = pursuer_levels[1:]  # classifier's own 0-indexed levels 1..K
    results = {name: [] for name in classifiers}

    for name, classifier in classifiers.items():
        for delay in delays:
            level_returns = []
            for k in range(1, K + 1):
                team_agent = TeamPursuerAgent(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape)
                edr_agent = IndEvader(EDR, LLP, HLP, SIM_MAX_STEPS, env.grid.shape)
                agents = [team_agent, edr_agent]
                reward = [SparsePursuer(LLP, HLP, EDR, 1, 0), SparseEvader(EDR, LLP, HLP, 1, 0)]
                tdo_policy = TDOPolicy(classifier, sub_levels, delay, k - 1)
                policy = PolicyAggregator([tdo_policy, evader_levels[k - 1]])

                key, rk = jax.random.split(key)
                transitions = rollout(env, agents, reward, policy, cfg.eval.num_envs, rk)
                level_returns.append(discounted_reward(transitions, agent_idx=0))

            mean_dr = float(np.mean(level_returns))
            results[name].append(mean_dr)
            log.info(f"Figure 10: {name} delay={delay} -> expected_discounted_reward={mean_dr:.4f}")

    return results, key


def plot_figure_10(results, delays, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ALGO_COLORS
    for name, values in results.items():
        ax.plot(delays, values, marker="o", color=colors.get(name, "tab:blue"), label=name)
    ax.set_xlabel("TDO Delay (trajectory steps before switching to oracle)")
    ax.set_ylabel("Expected Discounted Reward")
    ax.set_title("Expected Discounted Reward vs TDO Delay")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 12: squared action difference vs trajectory timestep
# ---------------------------------------------------------------------------

from evaluate import MultiLevelObserverPolicy, DiffTrackingEvaderAgent  # noqa: E402


def figure_12(cfg, env, pursuer_levels, evader_levels, key):
    """Time-resolved version of evaluate.py's Figure 7: instead of summing
    pairwise squared action differences between evader levels into a single
    K x K matrix, bins the same per-sample squared differences by trajectory
    step. Same rollout structure/cost as Figure 7 (K x (K+1) combos)."""
    K = cfg.level_k.K
    all_steps, all_sqdiffs = [], []

    for ev_idx in range(K):
        for p_idx in range(K + 1):
            llp_p, hlp_p = pursuer_levels[p_idx]
            edr_agent = DiffTrackingEvaderAgent(EDR, LLP, HLP, SIM_MAX_STEPS, env.grid.shape)
            agents, reward = pursuer_vs_evader_agents_reward(env, edr_agent)
            policy = PolicyAggregator([
                llp_p, hlp_p,
                MultiLevelObserverPolicy(evader_levels[ev_idx], evader_levels),
            ])

            key, rk = jax.random.split(key)
            transitions = rollout(env, agents, reward, policy, cfg.eval.num_envs, rk)

            level_actions = np.array(transitions.action[EDR]["level_actions"])  # (N, K, 4)
            step_count = np.array(transitions.step_count)
            diff = level_actions[:, :, None, :] - level_actions[:, None, :, :]  # (N, K, K, 4)
            sq = np.sum(diff**2, axis=-1)  # (N, K, K)
            off_diag = ~np.eye(K, dtype=bool)
            per_sample_mean = sq[:, off_diag].mean(axis=-1)  # (N,)

            all_steps.append(step_count)
            all_sqdiffs.append(per_sample_mean)
            log.info(f"Figure 12: driving=L{ev_idx + 1} vs pursuer L{p_idx} collected")

    all_steps = np.concatenate(all_steps)
    all_sqdiffs = np.concatenate(all_sqdiffs)

    steps, means, sems = [], [], []
    for t in range(int(all_steps.max()) + 1):
        mask = all_steps == t
        n = int(mask.sum())
        if n == 0:
            continue
        steps.append(t)
        means.append(float(all_sqdiffs[mask].mean()))
        sems.append(float(all_sqdiffs[mask].std() / np.sqrt(n)))

    return np.array(steps), np.array(means), np.array(sems)


def plot_figure_12(steps, means, sems, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, means, color="tab:purple")
    ax.fill_between(steps, means - sems, means + sems, alpha=0.3, color="tab:purple", label="+/- 1 SE")
    ax.set_xlabel("Trajectory Step")
    ax.set_ylabel("Mean Pairwise Squared Action Difference")
    ax.set_title("Expert Action Divergence by Trajectory Step")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../../conf", config_name="evaluate")
def main(cfg):
    np.random.seed(cfg.seed)
    th.manual_seed(cfg.seed)
    key = jax.random.key(cfg.seed)

    env = ODEnv()
    os.makedirs(cfg.eval.figures_dir, exist_ok=True)
    suffix = cfg.eval.get("suffix", "") or ""

    key, k1, k2, k3, k4, k5, k6 = jax.random.split(key, 7)
    pursuer_levels = load_pursuer_levels(cfg, env, k1)
    evader_levels = load_evader_levels(cfg, env, k2)
    pap_llp, pap_hlp = load_pap(cfg, env, k3)
    lksp_meta = load_lksp_meta(cfg, env, cfg.level_k.K + 1, k4)
    lksp_policy = LKSPPolicy(lksp_meta, pursuer_levels)
    dagger_classifier = load_dagger(cfg, env, k5)
    dagger_policy = LKSPPolicy(dagger_classifier, pursuer_levels[1:])

    try:
        offline_classifier = load_offline(cfg, env, k6)
        offline_policy = LKSPPolicy(offline_classifier, pursuer_levels[1:])
    except (RuntimeError, FileNotFoundError) as e:
        log.warning(
            f"could not load offline classifier from {cfg.offline.checkpoint_dir} "
            f"(likely trained at a different K than {cfg.level_k.K}) -- "
            f"Figures 6/8/10 will show MPO/CBO-DAgger only. Error: {e}"
        )
        offline_classifier, offline_policy = None, None

    log.info("=== Figure 6: expected discounted reward vs policy type ===")
    key, fk = jax.random.split(key)
    fig6_results, key = figure_6(
        cfg, env, pap_llp, pap_hlp, lksp_policy, dagger_policy, offline_policy, evader_levels, fk
    )
    plot_figure_6(fig6_results, f"{cfg.eval.figures_dir}/fig6{suffix}.png")
    save_values_txt(f"{cfg.eval.figures_dir}/fig6{suffix}.txt", expected_discounted_reward=fig6_results)

    log.info("=== Figure 8: classifier accuracy by trajectory step ===")
    classifier_policies = {"CBO-DAgger": dagger_policy}
    if offline_policy is not None:
        classifier_policies["CBO-Offline"] = offline_policy
    key, fk = jax.random.split(key)
    fig8_results, key = figure_8(cfg, env, classifier_policies, evader_levels, fk)
    plot_figure_8(fig8_results, f"{cfg.eval.figures_dir}/fig8{suffix}.png")
    save_values_txt(
        f"{cfg.eval.figures_dir}/fig8{suffix}.txt",
        **{f"{name} (step, accuracy, sem)": (steps, means, sems) for name, (steps, means, sems) in fig8_results.items()},
    )

    log.info("=== Figure 10: expected discounted reward vs TDO delay ===")
    classifiers = {"CBO-DAgger": dagger_classifier}
    if offline_classifier is not None:
        classifiers["CBO-Offline"] = offline_classifier
    delays = list(cfg.eval.tdo_delays)
    key, fk = jax.random.split(key)
    fig10_results, key = figure_10(cfg, env, pursuer_levels, evader_levels, classifiers, fk, delays)
    plot_figure_10(fig10_results, delays, f"{cfg.eval.figures_dir}/fig10{suffix}.png")
    save_values_txt(
        f"{cfg.eval.figures_dir}/fig10{suffix}.txt",
        delays=delays,
        **{f"{name} (expected_discounted_reward by delay)": (delays, values) for name, values in fig10_results.items()},
    )

    log.info("=== Figure 12: expert action divergence by trajectory step ===")
    key, fk = jax.random.split(key)
    steps, means, sems = figure_12(cfg, env, pursuer_levels, evader_levels, fk)
    plot_figure_12(steps, means, sems, f"{cfg.eval.figures_dir}/fig12{suffix}.png")
    save_values_txt(
        f"{cfg.eval.figures_dir}/fig12{suffix}.txt",
        **{"step, mean_sq_action_diff, sem": (steps, means, sems)},
    )

    log.info(f"done: figures_acc figures written to {cfg.eval.figures_dir}/ (suffix={suffix!r})")


if __name__ == "__main__":
    main()
