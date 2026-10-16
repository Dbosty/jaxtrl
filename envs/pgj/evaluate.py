"""Reproduces Figures 2, 3, 5, 6, and 7 from the APEGS paper by running
deterministic-free (i.e. ordinary stochastic, matching how these policies
were trained and how "win rate" is naturally an expectation over the
policy's own action distribution) evaluation rollouts against the
checkpoints trained by level_k.py, pap.py, and lksp.py.

Figure 4 (PAP/LKSP training curves) is NOT produced here -- it needs metrics
captured *during* training, not from a post-hoc rollout. See
plot_training_curves.py, which reads the CSV files run_stage already writes.

Requires level_k.py (K+1 pursuer levels, K evader levels), pap.py, and
lksp.py to have already been run with matching level_k.K / checkpoint_dir /
train (network architecture) settings.

Run with, e.g.:
    python envs/pgj/evaluate.py level_k.K=4 \\
        level_k.checkpoint_dir=checkpoints/level_k \\
        pap.checkpoint_dir=checkpoints/pap \\
        lksp.checkpoint_dir=checkpoints/lksp \\
        eval.figures_dir=figures
"""

import logging
import os

import _compat  # noqa: F401
import gymnasium as gym
import hydra
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as th

from pgj.env import ODEnv
from pgj.jaxtrl.agent import GreedyEvader, IndEvader, IndTeamHlp, IndTeamLlp
from pgj.jaxtrl.policy import GreedyPolicy
from pgj.jaxtrl.reward import SparseEvader, SparsePursuer
from pgj.jaxtrl.term import is_terminated

from jaxtrl.rl.ppo.actor.discrete_mlp import DiscreteMLP
from jaxtrl.rl.ppo.trainer import ACTOR_MAP
from jaxtrl.sim import Sim
from jaxtrl.sim.collect import collect_factory
from jaxtrl.sim.policy import Policy, PolicyAggregator, RandomPolicy

from level_k import EDR, HLP, LLP, SIM_MAX_STEPS, MixtureEvaderPolicy
from plot_utils import ALGO_COLORS, save_values_txt
from pap import UniformLevelKEvaderPolicy, load_evader_levels  # noqa: F401 (UniformLevelKEvaderPolicy re-export)
from lksp import LKSPPolicy, TeamPursuerAgent, load_pursuer_levels

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# checkpoint loading
# ---------------------------------------------------------------------------


def load_pap(cfg, env, key):
    llp_spec = IndTeamLlp(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape)
    hlp_spec = IndTeamHlp(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape)
    key, llp_key, hlp_key = jax.random.split(key, 3)

    llp = ACTOR_MAP[cfg.train.actor](
        llp_spec.pi_obs_spec, llp_spec.pi_action_spec, llp_key,
        **{**cfg.train.get("actor_kwargs", {}), "device": cfg.device},
    )
    hlp = ACTOR_MAP[cfg.train.actor](
        hlp_spec.pi_obs_spec, hlp_spec.pi_action_spec, hlp_key,
        **{**cfg.train.get("actor_kwargs", {}), "device": cfg.device},
    )
    llp.trl_actor.load_state_dict(th.load(f"{cfg.pap.checkpoint_dir}/pap_llp.pt", map_location=cfg.device))
    hlp.trl_actor.load_state_dict(th.load(f"{cfg.pap.checkpoint_dir}/pap_hlp.pt", map_location=cfg.device))
    return llp, hlp


def load_lksp_meta(cfg, env, num_levels, key):
    obs_spec = IndTeamLlp(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape).pi_obs_spec
    meta = ACTOR_MAP[cfg.train_meta.actor](
        obs_spec, gym.spaces.Discrete(num_levels), key,
        **{**cfg.train_meta.get("actor_kwargs", {}), "device": cfg.device},
    )
    meta.trl_actor.load_state_dict(th.load(f"{cfg.lksp.checkpoint_dir}/lksp_meta.pt", map_location=cfg.device))
    return meta


def load_dagger(cfg, env, key):
    obs_spec = IndTeamLlp(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape).pi_obs_spec
    classifier = DiscreteMLP(
        obs_spec, gym.spaces.Discrete(cfg.level_k.K), key,
        net_arch=list(cfg.classifier.net_arch),
        activation_func=cfg.classifier.activation_func,
        device=cfg.device,
    )
    classifier.trl_actor.load_state_dict(
        th.load(f"{cfg.dagger.checkpoint_dir}/dagger_classifier.pt", map_location=cfg.device)
    )
    return classifier


def load_offline(cfg, env, key):
    obs_spec = IndTeamLlp(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape).pi_obs_spec
    classifier = DiscreteMLP(
        obs_spec, gym.spaces.Discrete(cfg.level_k.K), key,
        net_arch=list(cfg.classifier.net_arch),
        activation_func=cfg.classifier.activation_func,
        device=cfg.device,
    )
    classifier.trl_actor.load_state_dict(
        th.load(f"{cfg.offline.checkpoint_dir}/offline_classifier.pt", map_location=cfg.device)
    )
    return classifier


def evader_agent_and_policy(cfg, env, evader_levels, j):
    """j in 0..K. j=0 is the fixed level-0 mixture (Algorithm 1 line 1,
    reconstructed identically to level_k.py, since it's not a checkpoint);
    j>=1 is evader_levels[j - 1], loaded from evader_L{j}.pt."""
    if j == 0:
        evader_agent = GreedyEvader(EDR)
        astar_policy = GreedyPolicy(env, EDR)
        random_policy = RandomPolicy(evader_agent.pi_action_spec)
        return evader_agent, MixtureEvaderPolicy(astar_policy, random_policy, cfg.level_k.alpha_n)
    return IndEvader(EDR, LLP, HLP, SIM_MAX_STEPS, env.grid.shape), evader_levels[j - 1]


# ---------------------------------------------------------------------------
# rollouts
# ---------------------------------------------------------------------------


class ZeroPolicy(Policy):
    """Always outputs the zero action -- used as a non-interfering
    "stationary pursuer" baseline for Figure 3's shortest-path measurement."""

    def __init__(self, action_dim=4):
        self.action_dim = action_dim

    def __call__(self, obs_for_pi, context, key):
        return jnp.zeros(self.action_dim), context

    def init_context(self, key):
        return None

    def reset_context(self, context, key):
        return context


def rollout(env, agents, reward, policy, num_envs, key):
    sim = Sim(env, agents, reward, is_terminated, max_steps=SIM_MAX_STEPS)
    collect = jax.jit(
        collect_factory(
            sim.reset, sim.step, policy,
            timesteps_per_env=SIM_MAX_STEPS, num_envs=num_envs,
            extended_transition=True,
        )
    )
    policy_context = policy.init_context(key)
    key, collect_key = jax.random.split(key)
    transitions = collect(collect_key, policy_context)

    # every call builds a brand-new Sim/policy-specific jax.jit closure that's
    # used exactly once -- without this, jax accumulates the compiled XLA
    # program for every one of the ~40+ (level, level) combinations across
    # figures 2/3/5/6 for the rest of the process, which is what exhausted
    # host RAM and got this script OOM-killed on a first attempt.
    jax.clear_caches()

    return transitions


def pursuer_winrate(transitions):
    return float(
        jnp.sum(transitions.raw_next_obs.state.agent_win[..., LLP])
        / jnp.sum(transitions.terminated)
    )


def episode_lengths(transitions, win_idx):
    """Per-episode (length, won) for every completed episode in a flat
    rollout stream -- one num_envs x timesteps_per_env block can contain
    many episodes back to back, since envs auto-reset on termination."""
    done = np.array(transitions.terminated | transitions.truncated)
    step_count = np.array(transitions.step_count)
    win = np.array(transitions.raw_next_obs.state.agent_win[..., win_idx])
    return (step_count + 1)[done], win[done].astype(bool)


def pursuer_vs_evader_agents_reward(env, evader_agent):
    agents = [
        IndTeamLlp(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape),
        IndTeamHlp(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape),
        evader_agent,
    ]
    reward = [
        SparsePursuer(LLP, HLP, EDR, 1, 0),
        SparsePursuer(LLP, HLP, EDR, 1, 0),
        SparseEvader(EDR, LLP, HLP, 1, 0),
    ]
    return agents, reward


# ---------------------------------------------------------------------------
# Figure 2: pursuer-level x evader-level win-rate matrix
# ---------------------------------------------------------------------------


def figure_2(cfg, env, pursuer_levels, evader_levels, key):
    K = cfg.level_k.K
    matrix = np.zeros((K + 1, K + 1))

    for i in range(K + 1):
        llp_i, hlp_i = pursuer_levels[i]
        for j in range(K + 1):
            evader_agent, evader_policy = evader_agent_and_policy(cfg, env, evader_levels, j)
            agents, reward = pursuer_vs_evader_agents_reward(env, evader_agent)
            policy = PolicyAggregator([llp_i, hlp_i, evader_policy])

            key, rk = jax.random.split(key)
            transitions = rollout(env, agents, reward, policy, cfg.eval.num_envs, rk)
            matrix[i, j] = pursuer_winrate(transitions)
            log.info(f"Figure 2: pursuer L{i} vs evader L{j} -> winrate={matrix[i, j]:.3f}")

    return matrix, key


def plot_figure_2(matrix, K, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1)
    labels = [f"L{i}" for i in range(K + 1)]
    ax.set_xticks(range(K + 1), labels)
    ax.set_yticks(range(K + 1), labels)
    ax.set_xlabel("Evader")
    ax.set_ylabel("Pursuer")
    ax.set_title("Pursuer Winrate by Policy Matchup")
    for i in range(K + 1):
        for j in range(K + 1):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, label="win rate")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: relative increase in evader trajectory length
# ---------------------------------------------------------------------------


def figure_3(cfg, env, pursuer_levels, evader_levels, key):
    """Level 0 is included by pairing it with pursuer level 0 (there is no
    "level -1" pursuer to pair it with) -- the paper's own text describes
    level 0's ratio as "expected to be close to one" without specifying an
    opponent, and pursuer level 0 is the only pursuer level meaningfully
    associated with it. For k>=1 this is otherwise unchanged: evader level k
    vs its Algorithm-1 training opponent, pursuer level k-1."""
    K = cfg.level_k.K

    # shortest-path baseline: pure pi_A* evader vs stationary pursuers
    agents, reward = pursuer_vs_evader_agents_reward(env, GreedyEvader(EDR))
    policy = PolicyAggregator([ZeroPolicy(), ZeroPolicy(), GreedyPolicy(env, EDR)])
    key, rk = jax.random.split(key)
    transitions = rollout(env, agents, reward, policy, cfg.eval.num_envs, rk)
    lengths, won = episode_lengths(transitions, EDR)
    shortest_path_len = lengths[won].mean()
    log.info(f"Figure 3: shortest-path baseline = {shortest_path_len:.2f} steps")

    rel_lengths, rel_length_sems = [], []
    for k in range(0, K + 1):
        pursuer_idx = max(k - 1, 0)
        llp_p, hlp_p = pursuer_levels[pursuer_idx]
        evader_agent, evader_policy = evader_agent_and_policy(cfg, env, evader_levels, k)
        agents, reward = pursuer_vs_evader_agents_reward(env, evader_agent)
        policy = PolicyAggregator([llp_p, hlp_p, evader_policy])

        key, rk = jax.random.split(key)
        transitions = rollout(env, agents, reward, policy, cfg.eval.num_envs, rk)
        lengths, won = episode_lengths(transitions, EDR)
        won_lengths = lengths[won]

        if len(won_lengths) > 1:
            mean_len = won_lengths.mean()
            sem_len = won_lengths.std(ddof=1) / np.sqrt(len(won_lengths))
        elif len(won_lengths) == 1:
            mean_len, sem_len = won_lengths.mean(), 0.0
        else:
            mean_len, sem_len = float("nan"), 0.0

        rel_lengths.append(mean_len / shortest_path_len)
        rel_length_sems.append(sem_len / shortest_path_len)
        log.info(
            f"Figure 3: evader L{k} vs pursuer L{pursuer_idx} -> "
            f"rel_length={rel_lengths[-1]:.3f} +/- {rel_length_sems[-1]:.3f} SE"
        )

    return rel_lengths, rel_length_sems, key


def plot_figure_3(rel_lengths, rel_length_sems, K, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
        [f"L{k}" for k in range(0, K + 1)], rel_lengths,
        yerr=rel_length_sems, capsize=4, color="tab:green",
    )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Evader Level")
    ax.set_ylabel("Normalized Average Path Length")
    ax.set_title("Relative Increase in Length of Evader Trajectory\nvs Level-(k-1) Pursuer")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: PAP / LKSP win-rate vs evader level
# ---------------------------------------------------------------------------


def figure_5(cfg, env, pap_llp, pap_hlp, lksp_policy, dagger_policy, offline_policy, evader_levels, key):
    """offline_policy may be None (e.g. its checkpoint was trained at a
    different K and can't be loaded for this run) -- offline_rates comes
    back as None in that case, and plot_figure_5 drops that series."""
    K = cfg.level_k.K
    pap_rates, lksp_rates, dagger_rates = [], [], []
    offline_rates = [] if offline_policy is not None else None

    for j in range(K + 1):
        evader_agent, evader_policy = evader_agent_and_policy(cfg, env, evader_levels, j)

        agents, reward = pursuer_vs_evader_agents_reward(env, evader_agent)
        policy = PolicyAggregator([pap_llp, pap_hlp, evader_policy])
        key, rk = jax.random.split(key)
        transitions = rollout(env, agents, reward, policy, cfg.eval.num_envs, rk)
        pap_rates.append(pursuer_winrate(transitions))

        team_agent = TeamPursuerAgent(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape)
        agents2 = [team_agent, evader_agent]
        reward2 = [SparsePursuer(LLP, HLP, EDR, 1, 0), SparseEvader(EDR, LLP, HLP, 1, 0)]

        policy2 = PolicyAggregator([lksp_policy, evader_policy])
        key, rk = jax.random.split(key)
        transitions2 = rollout(env, agents2, reward2, policy2, cfg.eval.num_envs, rk)
        lksp_rates.append(pursuer_winrate(transitions2))

        policy3 = PolicyAggregator([dagger_policy, evader_policy])
        key, rk = jax.random.split(key)
        transitions3 = rollout(env, agents2, reward2, policy3, cfg.eval.num_envs, rk)
        dagger_rates.append(pursuer_winrate(transitions3))

        offline_str = "n/a"
        if offline_policy is not None:
            policy4 = PolicyAggregator([offline_policy, evader_policy])
            key, rk = jax.random.split(key)
            transitions4 = rollout(env, agents2, reward2, policy4, cfg.eval.num_envs, rk)
            offline_rates.append(pursuer_winrate(transitions4))
            offline_str = f"{offline_rates[-1]:.3f}"

        log.info(
            f"Figure 5: evader L{j} -> PAP={pap_rates[-1]:.3f} "
            f"LKSP={lksp_rates[-1]:.3f} DAgger={dagger_rates[-1]:.3f} "
            f"Offline={offline_str}"
        )

    return pap_rates, lksp_rates, dagger_rates, offline_rates, key


def plot_figure_5(pap_rates, lksp_rates, dagger_rates, offline_rates, K, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(K + 1)
    has_offline = offline_rates is not None
    width = 0.2 if has_offline else 0.25
    if has_offline:
        ax.bar(x - 1.5 * width, pap_rates, width, label="PAP", color=ALGO_COLORS["PAP"])
        ax.bar(x - 0.5 * width, lksp_rates, width, label="MPO", color=ALGO_COLORS["MPO"])
        ax.bar(x + 0.5 * width, dagger_rates, width, label="CBO-DAgger", color=ALGO_COLORS["CBO-DAgger"])
        ax.bar(x + 1.5 * width, offline_rates, width, label="CBO-Offline", color=ALGO_COLORS["CBO-Offline"])
    else:
        ax.bar(x - width, pap_rates, width, label="PAP", color=ALGO_COLORS["PAP"])
        ax.bar(x, lksp_rates, width, label="MPO", color=ALGO_COLORS["MPO"])
        ax.bar(x + width, dagger_rates, width, label="CBO-DAgger", color=ALGO_COLORS["CBO-DAgger"])
    ax.set_xticks(x, [f"L{j}" for j in range(K + 1)])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Evader Level")
    ax.set_ylabel("Win Rate")
    ax.set_title("Adaptive Pursuer vs Evader Winrate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: LKSP best-response selection rate by trajectory step
# ---------------------------------------------------------------------------


def figure_6(cfg, env, lksp_policy, evader_levels, key):
    """"Best response" to evader level k is pursuer level k (same index --
    Figure 2/Q1 in the paper confirms level-k pursuer is the best response to
    level-k evader), so this measures, at each trajectory step, how often
    LKSP's meta-controller selects k when playing evader level k. Each
    evader level is run in its own fixed (non-mixed) batch and the results
    pooled, rather than resampling the evader level mid-training the way
    pap.py/lksp.py do -- this avoids needing to smuggle per-episode evader
    identity through the collected transition stream (only the pursuer
    team's own action dict, {"k": ..., "env_action": ...}, needs that)."""
    K = cfg.level_k.K
    all_steps, all_hits = [], []

    for k in range(1, K + 1):
        team_agent = TeamPursuerAgent(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape)
        edr_agent = IndEvader(EDR, LLP, HLP, SIM_MAX_STEPS, env.grid.shape)
        agents = [team_agent, edr_agent]
        reward = [SparsePursuer(LLP, HLP, EDR, 1, 0), SparseEvader(EDR, LLP, HLP, 1, 0)]
        policy = PolicyAggregator([lksp_policy, evader_levels[k - 1]])

        key, rk = jax.random.split(key)
        transitions = rollout(env, agents, reward, policy, cfg.eval.num_envs, rk)

        step_count = np.array(transitions.step_count)
        selected_k = np.array(transitions.action[0]["k"])
        all_steps.append(step_count)
        all_hits.append((selected_k == k).astype(np.float32))
        log.info(f"Figure 6: evader L{k} rollout collected ({len(step_count)} steps)")

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

    return steps, means, sems


def plot_figure_6(steps, means, sems, path):
    steps, means, sems = np.array(steps), np.array(means), np.array(sems)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, means, color=ALGO_COLORS["MPO"])
    ax.fill_between(steps, means - sems, means + sems, alpha=0.3, color=ALGO_COLORS["MPO"], label="+/- 1 SE")
    ax.set_xlabel("Trajectory Step")
    ax.set_ylabel("Selection Rate")
    ax.set_title("Best Response Selection Rate of MPO by Trajectory Step")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7 (appendix): L2 action-norm difference matrix between evader levels
# ---------------------------------------------------------------------------


class MultiLevelObserverPolicy(Policy):
    """Wraps a "driving" evader policy (whichever level is actually
    controlling the episode) plus all K trained evader-level actors. Every
    step it advances all K candidates' own recurrent state on the true
    observation history -- exactly LKSPPolicy's pattern -- so that
    Diff_ij = ||pi_EDR^(i)(h) - pi_EDR^(j)(h)|| can be computed with every
    level evaluated on the *same* history h, per the paper's definition.
    Only the driving policy's action actually reaches the environment; the
    other K-1 candidates' (deterministic_sample) actions are stashed in the
    returned action dict for post-hoc analysis."""

    def __init__(self, driving_policy: Policy, level_policies: list[Policy]):
        self.driving_policy = driving_policy
        self.level_policies = level_policies

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
        driving_action, driving_ctx = self.driving_policy(obs_for_pi, context["driving"], key)

        level_actions, new_level_ctxs = [], []
        for lvl, ctx in zip(self.level_policies, context["levels"]):
            action, new_ctx = lvl.deterministic_sample(obs_for_pi, ctx)
            level_actions.append(action)
            new_level_ctxs.append(new_ctx)

        action = {"env_action": driving_action, "level_actions": jnp.stack(level_actions)}
        return action, {"driving": driving_ctx, "levels": new_level_ctxs}


class DiffTrackingEvaderAgent(IndEvader):
    """Sim-agent for figure_7's evader slot -- same obs as IndEvader, but the
    incoming action is MultiLevelObserverPolicy's {"env_action",
    "level_actions"} dict; only env_action is real."""

    def action_pi2env(self, action, context=None):
        return jnp.clip(action["env_action"], -1, 1), context


def figure_7(cfg, env, pursuer_levels, evader_levels, key):
    """D = histories from all (evader level, pursuer level) combinations,
    per the paper's "dataset of histories generated using all possible
    combinations of evaders pitted against all possible combinations of
    pursuers generated in Algorithm 1"."""
    K = cfg.level_k.K
    sq_diff_sum = np.zeros((K, K))
    n_samples = 0

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
            diff = level_actions[:, :, None, :] - level_actions[:, None, :, :]  # (N, K, K, 4)
            sq_diff_sum += np.sum(diff**2, axis=-1).sum(axis=0)
            n_samples += level_actions.shape[0]

            log.info(f"Figure 7: driving=L{ev_idx + 1} vs pursuer L{p_idx} collected")

    diff_matrix = np.sqrt(sq_diff_sum / n_samples)
    diff_matrix = diff_matrix / diff_matrix.max()
    return diff_matrix


def plot_figure_7(diff_matrix, K, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(diff_matrix, cmap="viridis", vmin=0, vmax=diff_matrix.max())
    labels = [f"L{i}" for i in range(1, K + 1)]
    ax.set_xticks(range(K), labels)
    ax.set_yticks(range(K), labels)
    ax.set_xlabel("Evader Level")
    ax.set_ylabel("Evader Level")
    ax.set_title("L2 Action Norm Difference Matrix")
    for i in range(K):
        for j in range(K):
            color = "white" if diff_matrix[i, j] < diff_matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{diff_matrix[i, j]:.2f}", ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax)
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

    suffix = cfg.eval.get("suffix", "") or ""  # e.g. "_40M_8K", appended to every figure filename

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
            f"Figure 5 will show PAP/LKSP/DAgger only. Error: {e}"
        )
        offline_policy = None

    log.info("=== Figure 2: pursuer x evader winrate matrix ===")
    key, fk = jax.random.split(key)
    matrix, key = figure_2(cfg, env, pursuer_levels, evader_levels, fk)
    np.save(f"{cfg.eval.figures_dir}/figure_2_matrix{suffix}.npy", matrix)
    plot_figure_2(matrix, cfg.level_k.K, f"{cfg.eval.figures_dir}/figure_2{suffix}.png")
    save_values_txt(
        f"{cfg.eval.figures_dir}/figure_2{suffix}.txt",
        **{f"pursuer L{i} winrate by evader level": row for i, row in enumerate(matrix.tolist())},
    )

    log.info("=== Figure 3: relative evader path length ===")
    key, fk = jax.random.split(key)
    rel_lengths, rel_length_sems, key = figure_3(cfg, env, pursuer_levels, evader_levels, fk)
    plot_figure_3(rel_lengths, rel_length_sems, cfg.level_k.K, f"{cfg.eval.figures_dir}/figure_3{suffix}.png")
    save_values_txt(
        f"{cfg.eval.figures_dir}/figure_3{suffix}.txt",
        **{"evader_level, rel_path_length, sem": (list(range(cfg.level_k.K + 1)), rel_lengths, rel_length_sems)},
    )

    log.info("=== Figure 5: PAP/LKSP/DAgger/Offline winrate vs evader level ===")
    key, fk = jax.random.split(key)
    pap_rates, lksp_rates, dagger_rates, offline_rates, key = figure_5(
        cfg, env, pap_llp, pap_hlp, lksp_policy, dagger_policy, offline_policy, evader_levels, fk
    )
    plot_figure_5(
        pap_rates, lksp_rates, dagger_rates, offline_rates,
        cfg.level_k.K, f"{cfg.eval.figures_dir}/figure_5{suffix}.png",
    )
    fig5_series = {"PAP": pap_rates, "MPO": lksp_rates, "CBO-DAgger": dagger_rates}
    if offline_rates is not None:
        fig5_series["CBO-Offline"] = offline_rates
    save_values_txt(f"{cfg.eval.figures_dir}/figure_5{suffix}.txt", evader_levels=list(range(cfg.level_k.K + 1)), **fig5_series)

    log.info("=== Figure 6: LKSP selection rate by trajectory step ===")
    key, fk = jax.random.split(key)
    steps, means, sems = figure_6(cfg, env, lksp_policy, evader_levels, fk)
    plot_figure_6(steps, means, sems, f"{cfg.eval.figures_dir}/figure_6{suffix}.png")
    save_values_txt(
        f"{cfg.eval.figures_dir}/figure_6{suffix}.txt",
        **{"step, mpo_selection_rate, sem": (steps, means, sems)},
    )

    log.info("=== Figure 7: L2 action-norm difference matrix ===")
    key, fk = jax.random.split(key)
    diff_matrix = figure_7(cfg, env, pursuer_levels, evader_levels, fk)
    np.save(f"{cfg.eval.figures_dir}/figure_7_matrix{suffix}.npy", diff_matrix)
    plot_figure_7(diff_matrix, cfg.level_k.K, f"{cfg.eval.figures_dir}/figure_7{suffix}.png")
    save_values_txt(
        f"{cfg.eval.figures_dir}/figure_7{suffix}.txt",
        **{f"evader L{i + 1} action diff by evader level": row for i, row in enumerate(diff_matrix.tolist())},
    )

    log.info(f"done: figures written to {cfg.eval.figures_dir}/ (suffix={suffix!r})")


if __name__ == "__main__":
    main()
