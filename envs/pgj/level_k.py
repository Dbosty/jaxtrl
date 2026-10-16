"""Algorithm 1 (APEGS paper): generation of K levels of evader policies.

    1: pi_EDR^(0) = alpha * pi_A* + (1 - alpha) * pi_random, alpha ~ Unif(A)
    2: B_eta(x) = {x' : ||x - x'|| <= eta}
    3: train (pi_HLP^(0), pi_LLP^(0)) against pi_EDR^(0)
    4: for i in 1..K:
    5:     train pi_EDR^(i) against (pi_HLP, pi_LLP) ~ Unif(B_eta(pi_HLP^(i-1), pi_LLP^(i-1)))
    6:     train (pi_HLP^(i), pi_LLP^(i)) against pi_EDR ~ Unif(B_eta(pi_EDR^(i)))

Each stage reuses the same HAPPO/PPO trainers and Sim/collect plumbing as
envs/pgj/happo.py. What's new here is (a) the level-0 evader mixture and
(b) FrozenBallPolicy, which freezes a previously trained actor and resamples
a fresh eta-ball perturbation of its weights every training iteration so the
*next* policy is trained against a neighborhood of the previous level rather
than a single point estimate.

Run with, e.g.:
    python envs/pgj/level_k.py env_var=dummy level_k.K=2 collect.total_timesteps=2_000_000
"""

import copy
import csv
import logging
import os

import _compat  # noqa: F401  (checks required env vars, e.g. COMPOSITE_LP_AGGREGATE, at import time)
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import torch as th
from omegaconf import OmegaConf

from pgj.env import ODEnv
from pgj.jaxtrl.agent import GreedyEvader, IndEvader, IndTeamHlp, IndTeamLlp
from pgj.jaxtrl.policy import GreedyPolicy
from pgj.jaxtrl.reward import SparseEvader, SparsePursuer
from pgj.jaxtrl.term import is_terminated

from jaxtrl.marl.happo import HAPPOTrainer
from jaxtrl.rl.ppo.trainer import PPOTrainer
from jaxtrl.sim import Sim
from jaxtrl.sim.collect import collect_factory
from jaxtrl.sim.policy import Policy, PolicyAggregator, RandomPolicy
from jaxtrl.utils.logger import ExtWandbLogger
from jaxtrl.utils.tensordict import (
    transitions_to_ma_tensordict,
    transitions_to_tensordicts,
)

log = logging.getLogger(__name__)

LLP, HLP, EDR = 0, 1, 2
SIM_MAX_STEPS = 100


# ---------------------------------------------------------------------------
# Algorithm 1, line 1: level-0 evader mixture
# ---------------------------------------------------------------------------


class MixtureEvaderPolicy(Policy):
    """pi_EDR^(0) = alpha * pi_A* + (1 - alpha) * pi_random, resampling alpha
    per episode from A = {l/N : l in [N]}."""

    def __init__(self, astar_policy: Policy, random_policy: Policy, n: int):
        self.astar_policy = astar_policy
        self.random_policy = random_policy
        self.n = n

    def _sample_alpha(self, key):
        ell = jax.random.randint(key, (), 1, self.n + 1)
        return ell / self.n

    def init_context(self, key):
        key, alpha_key, astar_key, random_key = jax.random.split(key, 4)
        return {
            "alpha": self._sample_alpha(alpha_key),
            "astar": self.astar_policy.init_context(astar_key),
            "random": self.random_policy.init_context(random_key),
        }

    def reset_context(self, context, key):
        key, alpha_key, astar_key, random_key = jax.random.split(key, 4)
        return {
            "alpha": self._sample_alpha(alpha_key),
            "astar": self.astar_policy.reset_context(context["astar"], astar_key),
            "random": self.random_policy.reset_context(context["random"], random_key),
        }

    def __call__(self, obs_for_pi, context, key):
        pick_key, astar_key, random_key = jax.random.split(key, 3)

        astar_action, astar_ctx = self.astar_policy(
            obs_for_pi, context["astar"], astar_key
        )
        random_action, random_ctx = self.random_policy(
            obs_for_pi, context["random"], random_key
        )

        use_astar = jax.random.uniform(pick_key) < context["alpha"]
        action = jnp.where(use_astar, astar_action, random_action)

        return action, {
            "alpha": context["alpha"],
            "astar": astar_ctx,
            "random": random_ctx,
        }


# ---------------------------------------------------------------------------
# Algorithm 1, lines 2, 5, 6: B_eta(.) opponent sampling around a frozen actor
# ---------------------------------------------------------------------------


def perturb_(module: th.nn.Module, eta: float, generator: th.Generator):
    """In-place: shift every parameter of `module` by one sample drawn
    uniformly from the L2 ball of radius `eta` (i.e. a single point in
    B_eta(0), added to the current weights)."""
    if eta <= 0:
        return

    params = list(module.parameters())
    numels = [p.numel() for p in params]
    total = sum(numels)

    direction = th.randn(total, generator=generator)
    direction = direction / direction.norm()

    # radius ~ eta * U^(1/d) samples uniformly by *volume* inside the ball,
    # not just uniformly in radius.
    radius = eta * th.rand((), generator=generator).item() ** (1.0 / total)
    noise = (direction * radius).split(numels)

    with th.no_grad():
        for p, chunk in zip(params, noise):
            p.add_(chunk.view(p.shape).to(device=p.device, dtype=p.dtype))


class FrozenBallPolicy(Policy):
    """Wraps an already-trained actor that is never updated again. Every
    time `init_context` is called (once per outer training iteration in the
    scripts below, exactly like the live/trained actors already do to remirror
    torch weights into their jax copy) the actor's torch weights are reset to
    the frozen checkpoint and re-perturbed by a fresh sample from
    Unif(B_eta(.)). This gives the "opponent sampled from the eta-ball"
    behavior in Algorithm 1 lines 5-6."""

    def __init__(self, actor: Policy, eta: float, seed: int = 0):
        self.actor = actor
        self.eta = eta
        self._frozen_state = copy.deepcopy(actor.trl_actor.state_dict())
        self._generator = th.Generator().manual_seed(seed)

    def init_context(self, key):
        self.actor.trl_actor.load_state_dict(self._frozen_state)
        perturb_(self.actor.trl_actor, self.eta, self._generator)
        return self.actor.init_context(key)

    def reset_context(self, context, key):
        return self.actor.reset_context(context, key)

    def __call__(self, obs_for_pi, context, key):
        return self.actor(obs_for_pi, context, key)

    def deterministic_sample(self, obs_for_pi, context):
        return self.actor.deterministic_sample(obs_for_pi, context)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _winrate_metrics(transitions, gamma=0.99):
    """Shared by every training script's run_stage call (level_k.py's own
    pursuer/evader stages, pap.py, lksp.py, dagger.py), so adding a real
    discounted-reward summary here -- not an approximation from winrate and
    mean episode length after the fact -- gives every training curve a
    directly-comparable, exact rollout/pursuer_discounted_reward column for
    free at every logged interval, with no extra rollouts. This is what
    evaluate_acc.py's discounted_reward() computes post-hoc from a fresh
    eval rollout; agent index 0 is always a pursuer-side agent (the LLP in
    pap.py's 3-agent setup, or the team agent in lksp.py's/dagger.py's
    2-agent setup) since SparsePursuer's reward is 1 exactly at the terminal
    step a pursuer wins, else 0, in both cases."""
    reward = jnp.asarray(transitions.reward)[..., 0]
    discounted = reward * (gamma ** jnp.asarray(transitions.step_count))
    n_episodes = jnp.sum(transitions.terminated | transitions.truncated)
    return {
        "rollout/pursuer_winrate": jnp.sum(
            transitions.raw_next_obs.state.agent_win[..., LLP]
        )
        / jnp.sum(transitions.terminated),
        "rollout/evader_winrate": jnp.sum(
            transitions.raw_next_obs.state.agent_win[..., EDR]
        )
        / jnp.sum(transitions.terminated),
        "rollout/pursuer_discounted_reward": jnp.sum(discounted) / n_episodes,
    }


def save_actor(actor: Policy, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    th.save(actor.trl_actor.state_dict(), path)
    log.info(f"saved checkpoint to {path}")


def best_checkpoint_path(path: str) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}_best{ext}"


class CsvMetricsWriter:
    """Appends one row per logged iteration to <metrics_dir>/<exp_name>.csv,
    so training curves (e.g. Figure 4's pursuer-winrate-vs-timesteps) can be
    plotted later without needing a live wandb account -- ExtWandbLogger's
    own PrintLogger only prints to stdout, it doesn't persist anything."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fieldnames = None
        self._file = open(path, "w", newline="")
        self._writer = None

    def log(self, timestep: int, log_dict: dict):
        row = {"timestep": timestep, **{k: float(v) for k, v in log_dict.items()}}
        if self._writer is None:
            self._fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()
        # tolerate a later iteration introducing a key the header didn't have
        row = {k: row.get(k, "") for k in self._fieldnames}
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()


def run_stage(
    cfg, sim, policy, trainer, train_step, exp_name, key,
    save_targets: dict[str, Policy] | None = None,
    best_metric: str = "rollout/pursuer_winrate",
    best_mode: str = "max",
):
    """Shared collect -> train -> log loop used by every stage below.

    save_targets, if given, is a {final_checkpoint_path: actor} mapping.
    Whenever best_metric improves (per best_mode, "max" or "min"), each
    actor's *current* weights are snapshotted to
    best_checkpoint_path(final_checkpoint_path) immediately -- not just
    whatever the weights happen to be at the last iteration, which the
    caller saves separately after this returns."""
    logger = ExtWandbLogger(
        exp_name=exp_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        **{**cfg.env_var.wandb, **cfg.collect.wandb},
    )
    metrics_dir = cfg.get("metrics_dir", "metrics")
    csv_writer = CsvMetricsWriter(f"{metrics_dir}/{exp_name}.csv")

    policy_context = policy.init_context(key)

    collect = jax.jit(
        collect_factory(
            sim.reset,
            sim.step,
            policy,
            timesteps_per_env=cfg.collect.batch.timesteps_per_env,
            num_envs=cfg.collect.batch.num_envs,
            extended_transition=True,
        )
    )

    total_frames = 0
    best_value = None
    n_iters = cfg.collect.total_timesteps // cfg.collect.batch.timesteps_per_env + 1
    for _ in range(n_iters):
        key, policy_key, collect_key = jax.random.split(key, 3)

        policy_context = policy.init_context(policy_key)

        transitions = collect(collect_key, policy_context)
        total_frames += transitions.step_count.size

        log_dict = train_step(transitions)
        log_dict.update(_winrate_metrics(transitions))

        if save_targets and best_metric in log_dict:
            value = log_dict[best_metric]
            is_better = best_value is None or (
                value > best_value if best_mode == "max" else value < best_value
            )
            if is_better:
                best_value = value
                for path, actor in save_targets.items():
                    save_actor(actor, best_checkpoint_path(path))

        logger.log_scalars(log_dict, timestep=total_frames)
        csv_writer.log(total_frames, _flatten_scalars(log_dict))

    csv_writer.close()
    return key


def _flatten_scalars(log_dict):
    """Drop non-scalar / non-numeric log_dict entries (e.g. HAPPO's nested
    per-agent dicts already come pre-flattened with "/" in their keys by
    PrintLogger's convention -- but log_dict itself, before that flattening,
    can hold plain python ints/None; this keeps CsvMetricsWriter simple by
    only ever seeing floatable scalars)."""
    flat = {}
    for k, v in log_dict.items():
        try:
            flat[k] = float(v)
        except (TypeError, ValueError):
            continue
    return flat


# ---------------------------------------------------------------------------
# Algorithm 1, lines 3 & 6: train (pi_HLP, pi_LLP) against a fixed evader
# ---------------------------------------------------------------------------


def train_pursuer_stage(cfg, env, evader_agent, evader_policy, key, exp_name, llp_path=None, hlp_path=None):
    llp = IndTeamLlp(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape)
    hlp = IndTeamHlp(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape)

    agents = [llp, hlp, evader_agent]
    reward = [
        SparsePursuer(LLP, HLP, EDR, 1, 0),
        SparsePursuer(LLP, HLP, EDR, 1, 0),
        SparseEvader(EDR, LLP, HLP, 1, 0),
    ]

    key, trainer_key, run_key = jax.random.split(key, 3)

    trainer = HAPPOTrainer([llp, hlp], cfg.train, trainer_key, device=cfg.device)
    policy = PolicyAggregator([*trainer.policies, evader_policy])

    sim = Sim(env, agents, reward, is_terminated, max_steps=SIM_MAX_STEPS)

    def train_step(transitions):
        ma_td = transitions_to_ma_tensordict(transitions)
        return trainer.step(ma_td)

    save_targets = None
    if llp_path and hlp_path:
        save_targets = {llp_path: trainer.policies[0], hlp_path: trainer.policies[1]}

    run_stage(
        cfg, sim, policy, trainer, train_step, exp_name, run_key,
        save_targets=save_targets, best_metric="rollout/pursuer_winrate", best_mode="max",
    )

    return trainer.policies[0], trainer.policies[1]  # llp_actor, hlp_actor


# ---------------------------------------------------------------------------
# Algorithm 1, line 5: train pi_EDR against fixed (perturbed) pursuers
# ---------------------------------------------------------------------------


def train_evader_stage(cfg, env, pursuer_llp_policy, pursuer_hlp_policy, key, exp_name, edr_path=None):
    llp = IndTeamLlp(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape)
    hlp = IndTeamHlp(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape)
    edr = IndEvader(EDR, LLP, HLP, SIM_MAX_STEPS, env.grid.shape)

    agents = [llp, hlp, edr]
    reward = [
        SparsePursuer(LLP, HLP, EDR, 1, 0),
        SparsePursuer(LLP, HLP, EDR, 1, 0),
        SparseEvader(EDR, LLP, HLP, 1, 0),
    ]

    key, trainer_key, run_key = jax.random.split(key, 3)

    trainer = PPOTrainer(edr, cfg.train, trainer_key, device=cfg.device)
    policy = PolicyAggregator([pursuer_llp_policy, pursuer_hlp_policy, trainer.policy])

    sim = Sim(env, agents, reward, is_terminated, max_steps=SIM_MAX_STEPS)

    def train_step(transitions):
        tds = transitions_to_tensordicts(transitions)
        return trainer.step(tds[EDR])

    save_targets = {edr_path: trainer.policy} if edr_path else None

    run_stage(
        cfg, sim, policy, trainer, train_step, exp_name, run_key,
        save_targets=save_targets, best_metric="rollout/evader_winrate", best_mode="max",
    )

    return trainer.policy  # evader_actor


# ---------------------------------------------------------------------------
# main: Algorithm 1
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../../conf", config_name="level_k")
def main(cfg):
    if cfg.seed is None:
        cfg.seed = np.random.randint(0, 2**32 - 1)

    key = jax.random.key(cfg.seed)
    key, seed_key = jax.random.split(key)

    np_seed, th_seed = jax.random.randint(seed_key, (2,), 0, 2**16 - 1)
    np.random.seed(int(np_seed))
    th.manual_seed(int(th_seed))

    env = ODEnv()
    ckpt_dir = cfg.level_k.checkpoint_dir
    suffix = cfg.get("exp_name_suffix") or ""  # e.g. "_40M_8k", appended to every stage's wandb/CSV name

    # --- level 0 -----------------------------------------------------------
    astar_policy = GreedyPolicy(env, EDR)
    random_policy = RandomPolicy(GreedyEvader(EDR).pi_action_spec)
    evader_policy_0 = MixtureEvaderPolicy(
        astar_policy, random_policy, cfg.level_k.alpha_n
    )

    llp_0_path, hlp_0_path = f"{ckpt_dir}/pursuer_llp_L0.pt", f"{ckpt_dir}/pursuer_hlp_L0.pt"
    key, stage_key = jax.random.split(key)
    llp_0, hlp_0 = train_pursuer_stage(
        cfg, env, GreedyEvader(EDR), evader_policy_0, stage_key,
        f"pgj_level_k_pursuer_L0{suffix}", llp_0_path, hlp_0_path,
    )
    save_actor(llp_0, llp_0_path)
    save_actor(hlp_0, hlp_0_path)

    pursuer_levels = [(llp_0, hlp_0)]
    evader_levels = [None]  # level 0 evader is the fixed mixture, not a checkpoint

    # --- levels 1..K ---------------------------------------------------------
    for i in range(1, cfg.level_k.K + 1):
        prev_llp, prev_hlp = pursuer_levels[i - 1]

        frozen_llp = FrozenBallPolicy(prev_llp, cfg.level_k.eta_pursuer, seed=cfg.seed + i)
        frozen_hlp = FrozenBallPolicy(prev_hlp, cfg.level_k.eta_pursuer, seed=cfg.seed + i)

        edr_i_path = f"{ckpt_dir}/evader_L{i}.pt"
        key, stage_key = jax.random.split(key)
        edr_i = train_evader_stage(
            cfg, env, frozen_llp, frozen_hlp, stage_key,
            f"pgj_level_k_evader_L{i}{suffix}", edr_i_path,
        )
        save_actor(edr_i, edr_i_path)
        evader_levels.append(edr_i)

        frozen_edr = FrozenBallPolicy(edr_i, cfg.level_k.eta_evader, seed=cfg.seed + 100 + i)

        llp_i_path, hlp_i_path = f"{ckpt_dir}/pursuer_llp_L{i}.pt", f"{ckpt_dir}/pursuer_hlp_L{i}.pt"
        key, stage_key = jax.random.split(key)
        llp_i, hlp_i = train_pursuer_stage(
            cfg,
            env,
            IndEvader(EDR, LLP, HLP, SIM_MAX_STEPS, env.grid.shape),
            frozen_edr,
            stage_key,
            f"pgj_level_k_pursuer_L{i}{suffix}",
            llp_i_path,
            hlp_i_path,
        )
        save_actor(llp_i, llp_i_path)
        save_actor(hlp_i, hlp_i_path)
        pursuer_levels.append((llp_i, hlp_i))

    log.info(f"done: trained {len(pursuer_levels)} pursuer levels, "
              f"{len(evader_levels) - 1} evader levels. checkpoints in {ckpt_dir}/")


if __name__ == "__main__":
    main()
