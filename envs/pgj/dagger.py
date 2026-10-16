"""Alternative to LKSP: a supervised classifier, trained with DAgger, that
predicts which evader level (1..K) is active from the shared pursuer-team
observation, then deploys the correspondingly-indexed frozen pursuer level
(from envs/pgj/level_k.py) -- reusing LKSPPolicy/TeamPursuerAgent's exact
deployment mechanism (evaluate all K candidate pursuer pairs every step,
pick one by index) unchanged from lksp.py.

Why this is a classification problem: Figure 2 / the paper's Q1 finding is
that pursuer level k is the best response to evader level k. During
training we always know the ground truth -- we're the ones sampling which
evader level is active (see pap.py's UniformLevelKEvaderPolicy) -- so
"which pursuer level should act" reduces to "which evader level am I
facing", a supervised label we get for free instead of having to learn it
through reward (as LKSP's PPO-trained meta-controller does).

Why DAgger and not plain supervised learning: the classifier's own
predictions determine which pursuer level acts, which changes what states
get visited next. A classifier trained once on a fixed rollout's states
will see states during deployment its training data never covered, and
errors compound (classic imitation-learning distribution shift). DAgger's
loop -- roll out the CURRENT classifier, label the states it actually
visits (label = the true active evader level, which we always know),
aggregate into the training set, retrain -- closes that gap. This
implementation approximates "aggregate all data forever" with a bounded
FIFO replay buffer (see conf/dagger.yaml's dagger.buffer_capacity), since
retraining on the full ~40M-sample run every iteration is infeasible.

Requires envs/pgj/level_k.py and envs/pgj/pap.py's evader-level loader to
have already produced their checkpoints.

Run with, e.g.:
    python envs/pgj/dagger.py env_var=dummy level_k.K=4 \\
        level_k.checkpoint_dir=checkpoints/level_k
"""

import gymnasium as gym
import _compat  # noqa: F401
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import torch as th
import torch.nn.functional as F

from pgj.env import ODEnv
from pgj.jaxtrl.agent import IndEvader
from pgj.jaxtrl.reward import SparseEvader, SparsePursuer
from pgj.jaxtrl.term import is_terminated

from jaxtrl.rl.ppo.actor.discrete_mlp import DiscreteMLP
from jaxtrl.sim import Sim
from jaxtrl.sim.policy import PolicyAggregator

from level_k import EDR, HLP, LLP, SIM_MAX_STEPS, run_stage, save_actor
from pap import UniformLevelKEvaderPolicy, load_evader_levels
from lksp import LKSPPolicy, TeamPursuerAgent, load_pursuer_levels


class LabeledUniformLevelKEvaderPolicy(UniformLevelKEvaderPolicy):
    """Same opponent as PAP/LKSP training (resamples which evader level 1..K
    to play every episode), but also stashes the true label -- the active
    evader level, 0-indexed -- in the returned action so DAgger's data
    collection can read it off the transition stream, the same
    {"env_action", ...} dict trick LKSPPolicy/TeamPursuerAgent already use."""

    def __call__(self, obs_for_pi, context, key):
        action, new_context = super().__call__(obs_for_pi, context, key)
        return {"env_action": action, "level_label": new_context["idx"]}, new_context


class LabeledEvaderAgent(IndEvader):
    """Sim-agent for the evader slot -- same obs as IndEvader, but the
    incoming action is LabeledUniformLevelKEvaderPolicy's {"env_action",
    "level_label"} dict; only env_action is real."""

    def action_pi2env(self, action, context=None):
        return jnp.clip(action["env_action"], -1, 1), context


def train_classifier_step(classifier, optimizer, obs_batch, label_batch, device):
    obs_t = th.as_tensor(obs_batch, dtype=th.float32, device=device)
    label_t = th.as_tensor(label_batch, dtype=th.long, device=device)

    logits = classifier.trl_actor_net(obs_t)
    loss = F.cross_entropy(logits, label_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc = (logits.argmax(dim=-1) == label_t).float().mean().item()
    return loss.item(), acc


class ReplayBuffer:
    """Fixed-size FIFO buffer of (obs, label) pairs -- see module docstring
    for why this approximates DAgger's "aggregate all data" within a
    bounded memory budget."""

    def __init__(self, obs_dim, capacity):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.labels = np.zeros((capacity,), dtype=np.int64)
        self.size = 0
        self.ptr = 0

    def add(self, obs_batch, label_batch):
        n = obs_batch.shape[0]
        idx = (self.ptr + np.arange(n)) % self.capacity
        self.obs[idx] = obs_batch
        self.labels[idx] = label_batch
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def epoch_batches(self, batch_size, rng):
        perm = rng.permutation(self.size)
        for start in range(0, self.size, batch_size):
            batch_idx = perm[start : start + batch_size]
            yield self.obs[batch_idx], self.labels[batch_idx]


def train_step_factory(cfg, classifier, optimizer, buffer, rng):
    def train_step(transitions):
        obs_batch = np.array(transitions.obs[0])  # team agent, index 0
        label_batch = np.array(transitions.action[1]["level_label"])  # evader, index 1
        buffer.add(obs_batch, label_batch)

        losses, accs = [], []
        for _ in range(cfg.dagger.epochs_per_iter):
            for obs_mb, label_mb in buffer.epoch_batches(cfg.dagger.batch_size, rng):
                loss, acc = train_classifier_step(classifier, optimizer, obs_mb, label_mb, cfg.device)
                losses.append(loss)
                accs.append(acc)

        return {
            "train/loss": float(np.mean(losses)),
            "train/accuracy": float(np.mean(accs)),
            "dagger/buffer_size": float(buffer.size),
        }

    return train_step


@hydra.main(version_base=None, config_path="../../conf", config_name="dagger")
def main(cfg):
    if cfg.seed is None:
        cfg.seed = np.random.randint(0, 2**32 - 1)

    key = jax.random.key(cfg.seed)
    key, seed_key = jax.random.split(key)
    np_seed, th_seed = jax.random.randint(seed_key, (2,), 0, 2**16 - 1)
    np.random.seed(int(np_seed))
    th.manual_seed(int(th_seed))
    rng = np.random.default_rng(int(np_seed))

    env = ODEnv()

    key, pk, ek, ck = jax.random.split(key, 4)
    pursuer_levels = load_pursuer_levels(cfg, env, pk)[1:]  # levels 1..K only, matching training opponents
    evader_levels = load_evader_levels(cfg, env, ek)

    team_agent = TeamPursuerAgent(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape)
    edr_agent = LabeledEvaderAgent(EDR, LLP, HLP, SIM_MAX_STEPS, env.grid.shape)

    classifier = DiscreteMLP(
        team_agent.pi_obs_spec, gym.spaces.Discrete(cfg.level_k.K), ck,
        net_arch=list(cfg.classifier.net_arch),
        activation_func=cfg.classifier.activation_func,
        device=cfg.device,
    )
    optimizer = th.optim.Adam(classifier.trl_actor_net.parameters(), lr=cfg.dagger.lr)
    buffer = ReplayBuffer(obs_dim=team_agent.pi_obs_spec.shape[0], capacity=cfg.dagger.buffer_capacity)

    sim = Sim(
        env,
        [team_agent, edr_agent],
        [SparsePursuer(LLP, HLP, EDR, 1, 0), SparseEvader(EDR, LLP, HLP, 1, 0)],
        is_terminated,
        max_steps=SIM_MAX_STEPS,
    )
    policy = PolicyAggregator([
        LKSPPolicy(classifier, pursuer_levels),
        LabeledUniformLevelKEvaderPolicy(evader_levels),
    ])

    train_step = train_step_factory(cfg, classifier, optimizer, buffer, rng)

    key, run_key = jax.random.split(key)
    classifier_path = f"{cfg.dagger.checkpoint_dir}/dagger_classifier.pt"
    run_stage(
        cfg, sim, policy, None, train_step, cfg.get("exp_name") or "pgj_dagger", run_key,
        save_targets={classifier_path: classifier}, best_metric="rollout/pursuer_winrate", best_mode="max",
    )

    save_actor(classifier, classifier_path)


if __name__ == "__main__":
    main()
