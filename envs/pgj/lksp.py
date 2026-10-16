"""Algorithm 3 (APEGS paper): Level-K Switching Policy (LKSP).

Section 4.2: "we train the Level-K Switching Policy, pi_LKSP, by solving a
POMDP. The only difference between the POMDP for pi_LKSP and pi_PAP is that
the action space in pi_LKSP is the discrete set {0,1,...,K}, denoting the
set of level-K Pursuer policies trained in Algorithm 1." Section 5.3 trains
it against the same opponent distribution as PAP: evader levels 1..K,
resampled every episode.

So LKSP training = a *new*, tiny discrete meta-controller (PPO, Categorical
over {0,...,K}) that, every timestep, picks which of the K+1 already-frozen
(pi_LLP^(k), pi_HLP^(k)) pairs from envs/pgj/level_k.py gets to act. The
frozen pairs are never trained here; only the meta-controller is.

Requires both envs/pgj/level_k.py (for pursuer_{llp,hlp}_L{0..K}.pt) and
envs/pgj/pap.py's evader-level loader (re-imported from pap.py, for
evader_L{1..K}.pt) checkpoints to already exist.

Implementation note on why this needs custom Agent/Policy classes instead of
reusing train_pursuer_stage directly: LKSP's single discrete decision must
simultaneously drive *two* environment agents (LLP and HLP), but this
framework's Sim/collect_factory expects one Policy per Sim-agent slot. The
fix used below: collapse the pursuer team into a single Sim-agent
(TeamPursuerAgent, sharing IndTeamLlp's observation -- the paper's own
o^t_HLP = o^t_LLP), whose "action" is a {"k": ..., "env_action": ...} dict.
`k` is what PPO needs to train the meta-controller (see meta_tensordict);
`env_action` is the resolved (2, 4) action -- from deterministically
executing the K+1 frozen candidate pairs and indexing by k -- that actually
gets stepped into the environment. All K+1 candidates are evaluated (via
deterministic_sample, not stochastic sampling -- these are frozen, already-
trained policies being deployed, not explored) every step regardless of
whether selected, so their recurrent state keeps advancing on the true
history; Figure 6 in the paper shows the selected level changing mid-
rollout, which requires this.

Run with, e.g.:
    python envs/pgj/lksp.py env_var=dummy level_k.K=4 \\
        level_k.checkpoint_dir=checkpoints/level_k
"""

import gymnasium as gym
import _compat  # noqa: F401
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import torch as th
from tensordict import TensorDict

from pgj.env import ODEnv
from pgj.jaxtrl.agent import IndEvader, IndTeamHlp, IndTeamLlp
from pgj.jaxtrl.reward import SparseEvader, SparsePursuer
from pgj.jaxtrl.term import is_terminated

from jaxtrl.rl.ppo.trainer import ACTOR_MAP, PPOTrainer
from jaxtrl.sim import Sim
from jaxtrl.utils.tensordict import tdl
from jaxtrl.sim.policy import Policy, PolicyAggregator

from level_k import EDR, HLP, LLP, SIM_MAX_STEPS, FrozenBallPolicy, run_stage, save_actor
from pap import UniformLevelKEvaderPolicy, load_evader_levels


class TeamPursuerAgent(IndTeamLlp):
    """Sim-agent for the pursuer team: same shared LLP/HLP observation as
    IndTeamLlp/IndTeamHlp (the paper's o^t_HLP = o^t_LLP), but the incoming
    "action" is LKSPPolicy's {"k": ..., "env_action": ...} dict -- only
    env_action is a real environment action; k is carried only so training
    can recover the meta-controller's discrete choice."""

    def action_pi2env(self, action, context=None):
        return jnp.clip(action["env_action"], -1, 1), context


class MetaActorSpec:
    """Just enough of the Agent interface for PPOTrainer.get_actor/get_critic
    to size the LKSP meta-controller's networks. Never used as a Sim-agent
    (TeamPursuerAgent is)."""

    def __init__(self, pi_obs_spec, num_levels):
        self.pi_obs_spec = pi_obs_spec
        self.pi_action_spec = gym.spaces.Discrete(num_levels)


class LKSPPolicy(Policy):
    """Algorithm 3's meta-controller: a trainable discrete actor (choice of
    pursuer level) plus the K+1 frozen (llp, hlp) pursuer pairs it switches
    between. See module docstring for why the returned action is a dict."""

    def __init__(self, meta_actor: Policy, pursuer_levels: list[tuple[Policy, Policy]]):
        self.meta_actor = meta_actor
        self.pursuer_levels = pursuer_levels

    def init_context(self, key):
        key, meta_key = jax.random.split(key)
        subs = [
            (llp.init_context(key), hlp.init_context(key))
            for llp, hlp in self.pursuer_levels
        ]
        return {"meta": self.meta_actor.init_context(meta_key), "subs": subs}

    def reset_context(self, context, key):
        key, meta_key = jax.random.split(key)
        subs = [
            (llp.reset_context(lctx, key), hlp.reset_context(hctx, key))
            for (llp, hlp), (lctx, hctx) in zip(self.pursuer_levels, context["subs"])
        ]
        return {"meta": self.meta_actor.reset_context(context["meta"], meta_key), "subs": subs}

    def __call__(self, obs_for_pi, context, key):
        k, meta_ctx = self.meta_actor(obs_for_pi, context["meta"], key)

        llp_actions, hlp_actions, new_subs = [], [], []
        for (llp, hlp), (lctx, hctx) in zip(self.pursuer_levels, context["subs"]):
            llp_action, lctx = llp.deterministic_sample(obs_for_pi, lctx)
            hlp_action, hctx = hlp.deterministic_sample(obs_for_pi, hctx)
            llp_actions.append(llp_action)
            hlp_actions.append(hlp_action)
            new_subs.append((lctx, hctx))

        env_action = jnp.stack([jnp.stack(llp_actions)[k], jnp.stack(hlp_actions)[k]])

        return {"k": k, "env_action": env_action}, {"meta": meta_ctx, "subs": new_subs}


def load_pursuer_levels(cfg, env, key):
    """Load the K+1 frozen (llp, hlp) pursuer pairs (levels 0..K) saved by
    envs/pgj/level_k.py, wrapped in FrozenBallPolicy(eta=0) purely to reuse
    its "reload the checkpoint every outer training iteration" bookkeeping.
    eta=0 means perturb_ is a no-op -- these are deployed exactly as
    trained, not perturbed."""
    llp_spec = IndTeamLlp(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape)
    hlp_spec = IndTeamHlp(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape)

    levels = []
    for i in range(cfg.level_k.K + 1):
        key, llp_key, hlp_key = jax.random.split(key, 3)

        llp_actor = ACTOR_MAP[cfg.train.actor](
            llp_spec.pi_obs_spec,
            llp_spec.pi_action_spec,
            llp_key,
            **{**cfg.train.get("actor_kwargs", {}), "device": cfg.device},
        )
        hlp_actor = ACTOR_MAP[cfg.train.actor](
            hlp_spec.pi_obs_spec,
            hlp_spec.pi_action_spec,
            hlp_key,
            **{**cfg.train.get("actor_kwargs", {}), "device": cfg.device},
        )

        llp_actor.trl_actor.load_state_dict(
            th.load(f"{cfg.level_k.checkpoint_dir}/pursuer_llp_L{i}.pt", map_location=cfg.device)
        )
        hlp_actor.trl_actor.load_state_dict(
            th.load(f"{cfg.level_k.checkpoint_dir}/pursuer_hlp_L{i}.pt", map_location=cfg.device)
        )

        levels.append((FrozenBallPolicy(llp_actor, eta=0.0), FrozenBallPolicy(hlp_actor, eta=0.0)))
    return levels


def meta_tensordict(transitions, team_idx):
    """Like jaxtrl.utils.tensordict.transitions_to_tensordicts()[team_idx],
    but pulls the discrete meta choice out of the {"k", "env_action"} dict
    LKSPPolicy stashes as this agent's action -- PPO needs to be trained
    against k (what the meta-controller actually chose), not the resolved
    box action."""
    transition = transitions[team_idx]

    obs = tdl(transition.obs)
    next_obs = tdl(transition.next_obs)
    action = tdl(transition.action["k"])
    step_count = tdl(transition.step_count)
    reward = tdl(transition.reward)
    terminated = tdl(transition.terminated)
    truncated = tdl(transition.truncated)
    episode_reward = tdl(transition.episode_reward)

    return TensorDict(
        source={
            "observation": obs,
            "action": action,
            "next": {
                "observation": next_obs,
                "reward": reward,
                "episode_reward": episode_reward,
                "terminated": terminated,
                "truncated": truncated,
                "done": terminated | truncated,
                "step_count": step_count + 1,
            },
            "step_count": step_count,
        },
        batch_size=obs.shape[0],
        device=obs.device,
    )


@hydra.main(version_base=None, config_path="../../conf", config_name="lksp")
def main(cfg):
    if cfg.seed is None:
        cfg.seed = np.random.randint(0, 2**32 - 1)

    key = jax.random.key(cfg.seed)
    key, seed_key = jax.random.split(key)

    np_seed, th_seed = jax.random.randint(seed_key, (2,), 0, 2**16 - 1)
    np.random.seed(int(np_seed))
    th.manual_seed(int(th_seed))

    env = ODEnv()

    key, pursuer_key, evader_key = jax.random.split(key, 3)
    pursuer_levels = load_pursuer_levels(cfg, env, pursuer_key)
    evader_levels = load_evader_levels(cfg, env, evader_key)
    evader_policy = UniformLevelKEvaderPolicy(evader_levels)

    team_agent = TeamPursuerAgent(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape)
    edr_agent = IndEvader(EDR, LLP, HLP, SIM_MAX_STEPS, env.grid.shape)

    meta_spec = MetaActorSpec(team_agent.pi_obs_spec, cfg.level_k.K + 1)

    key, trainer_key, run_key = jax.random.split(key, 3)
    trainer = PPOTrainer(meta_spec, cfg.train_meta, trainer_key, device=cfg.device)

    lksp_policy = LKSPPolicy(trainer.policy, pursuer_levels)
    policy = PolicyAggregator([lksp_policy, evader_policy])

    sim = Sim(
        env,
        [team_agent, edr_agent],
        [SparsePursuer(LLP, HLP, EDR, 1, 0), SparseEvader(EDR, LLP, HLP, 1, 0)],
        is_terminated,
        max_steps=SIM_MAX_STEPS,
    )

    def train_step(transitions):
        return trainer.step(meta_tensordict(transitions, 0))

    meta_path = f"{cfg.lksp.checkpoint_dir}/lksp_meta.pt"
    run_stage(
        cfg, sim, policy, trainer, train_step, cfg.get("exp_name") or "pgj_lksp", run_key,
        save_targets={meta_path: trainer.policy}, best_metric="rollout/pursuer_winrate", best_mode="max",
    )

    save_actor(trainer.policy, meta_path)


if __name__ == "__main__":
    main()
