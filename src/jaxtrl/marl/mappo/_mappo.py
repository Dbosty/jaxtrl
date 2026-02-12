from dataclasses import dataclass

import torch as th
from jaxtrl.marl.mappo.utils import add_critic_obs
from jaxtrl.rl.ppo import Hyperparameters as PPOHyperparameters
from tensordict import TensorDict, merge_tensordicts
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE


@dataclass
class Hyperparameters(PPOHyperparameters):
    critic_obs_key: tuple[str, ...] | str = ("agent_0", "observation")
    critic_obs_size: int | None = None
    use_huber_loss: bool = False
    huber_delta: float = 1.0


class MAPPO:
    def __init__(self, actors, critic, device=None, **hp):
        self.device = device or "cpu"

        self.actors = actors
        self.critic = critic

        self.hp = Hyperparameters(**hp)

        self.actor_parameters = [list(actor.trl_actor.parameters()) for actor in actors]
        self.critic_parameters = list(critic.parameters())

        self.actor_optims = [
            th.optim.Adam(actor_parameters, lr=self.hp.lr)
            for actor_parameters in self.actor_parameters
        ]

        self.critic_optim = th.optim.Adam(self.critic_parameters, lr=self.hp.lr)

        self.advantage_module = GAE(
            gamma=self.hp.gamma,
            lmbda=self.hp.gae_lambda,
            value_network=self.critic,
            average_gae=self.hp.average_gae,
            deactivate_vmap=True,
        )

        self.loss_modules = [
            ClipPPOLoss(
                actor_network=actor.trl_actor,
                # critic_network=self.critic,
                clip_epsilon=self.hp.clip_epsilon,
                entropy_bonus=bool(self.hp.entropy_eps),
                entropy_coeff=self.hp.entropy_eps,
                # critic_coeff=self.hp.critic_coef,
                # loss_critic_type=self.hp.loss_critic_type,
            )
            for actor in self.actors[:-1]
        ]
        self.loss_modules.append(
            ClipPPOLoss(
                actor_network=self.actors[-1].trl_actor,
                critic_network=self.critic,
                clip_epsilon=self.hp.clip_epsilon,
                entropy_bonus=bool(self.hp.entropy_eps),
                entropy_coeff=self.hp.entropy_eps,
                critic_coeff=self.hp.critic_coef,
                loss_critic_type=self.hp.loss_critic_type,
            )
        )

    def step(self, ma_td):
        base_td = TensorDict(
            source={
                "next": {
                    "terminated": ma_td["next", "terminated"],
                    "truncated": ma_td["next", "truncated"],
                    "done": ma_td["next", "done"],
                    "step_count": ma_td["next", "step_count"],
                },
                "step_count": ma_td["step_count"],
            },
            batch_size=ma_td.batch_size,
            device=ma_td.device,
        )

        base_td = add_critic_obs(
            base_td,
            self.hp.critic_obs_key,
            ma_td[self.hp.critic_obs_key],
            ma_td["next", self.hp.critic_obs_key],
        )

        tds = []
        for i in range(len(self.actors)):
            td = merge_tensordicts(
                base_td,
                ma_td[f"agent_{i}"],
            )
            td["next"].update(ma_td["next", f"agent_{i}"])
            tds.append(td)

        for i in range(len(self.actors)):
            td = tds[i]
            td["action_log_prob"] = (
                self.actors[i].trl_actor.get_dist(td).log_prob(td["action"]).detach()
            )
            td = self.advantage_module(td)
            tds[i] = td

        agents_loss_vals = [None] * len(self.actors)

        for i in range(len(self.actors)):
            for _ in range(self.hp.num_epochs):
                td = tds[i]
                loss_vals = self.loss_modules[i](td)
                loss_value = loss_vals["loss_objective"] + (
                    loss_vals["loss_entropy"] if self.hp.entropy_eps else 0
                )
                self.actor_optims[i].zero_grad()
                loss_value.backward()
                if self.hp.clip_grad_norm:
                    th.nn.utils.clip_grad_norm_(
                        self.actor_parameters[i], self.hp.max_grad_norm
                    )
                self.actor_optims[i].step()

                if i == len(self.actors) - 1:
                    self.critic_optim.zero_grad()
                    if self.hp.use_huber_loss:
                        critic_loss = (
                            th.nn.functional.huber_loss(
                                td["state_value"],
                                td["value_target"],
                                delta=self.hp.huber_delta,
                            )
                            * self.hp.critic_coef
                        )
                        loss_vals["loss_critic"] = critic_loss
                    loss_vals["loss_critic"].backward()
                    if self.hp.clip_grad_norm:
                        th.nn.utils.clip_grad_norm_(
                            self.critic_parameters, self.hp.max_grad_norm
                        )
                    self.critic_optim.step()

                agents_loss_vals[i] = loss_vals.detach()

        return self._log_dict(tds, agents_loss_vals)

    def _log_dict(self, tds, agents_loss_vals):
        # Use first agent's td for shared rollout stats (all agents share same episode info)
        td = tds[0]

        # Use terminated for stats, but fall back to done (terminated | truncated)
        # when no episodes terminate (e.g., agent is very good and all eps truncate)
        terminated = td.get(("next", "terminated"))
        done = td.get(("next", "done"))

        if terminated.any():
            step_counts = td["next", "step_count"][terminated].float()
            episode_rewards = td["next", "episode_reward"][terminated]
        elif done.any():
            # Fall back to all done episodes
            step_counts = td["next", "step_count"][done].float()
            episode_rewards = td["next", "episode_reward"][done]
        else:
            # Last resort: use last frame
            step_counts = td["next", "step_count"][-1:].float()
            episode_rewards = td["next", "episode_reward"][-1:]

        rollout_data = {
            "ep_len_mean": step_counts.mean().item(),
            "ep_rew_mean": episode_rewards.mean().item(),
            "step_rew_mean": td["next", "reward"].mean().item(),
            "ep_len_min": step_counts.min().item(),
            "ep_len_max": step_counts.max().item(),
        }

        # train data - calculate explained_variance using last agent's td (has critic values)
        last_td = tds[-1]
        explained_variance = -1
        if th.var(last_td.get("state_value")) > 0:
            explained_variance = max(
                -1.0,
                1
                - (
                    th.var(last_td.get("value_target") - last_td.get("state_value"))
                    / th.var(last_td.get("value_target"))
                ),
            )

        # Build log_dict with rollout data
        log_dict = {f"rollout/{k}": v for k, v in rollout_data.items()}

        # Add per-agent loss statistics
        if agents_loss_vals is not None:
            for i, loss_vals in enumerate(agents_loss_vals):
                if loss_vals is not None:
                    agent_loss = loss_vals["loss_objective"] + (
                        loss_vals.get("loss_entropy", 0) if self.hp.entropy_eps else 0
                    )
                    log_dict[f"train/agent_{i}/loss"] = agent_loss.item()
                    log_dict[f"train/agent_{i}/loss_objective"] = loss_vals[
                        "loss_objective"
                    ].item()
                    log_dict[f"train/agent_{i}/kl_approx"] = loss_vals[
                        "kl_approx"
                    ].item()
                    log_dict[f"train/agent_{i}/clip_fraction"] = loss_vals[
                        "clip_fraction"
                    ].item()

                    if self.hp.entropy_eps:
                        log_dict[f"train/agent_{i}/loss_entropy"] = loss_vals.get(
                            "loss_entropy", 0
                        ).item()
                        log_dict[f"train/agent_{i}/entropy"] = loss_vals.get(
                            "entropy", 0
                        ).item()

                    # Critic loss is only computed for the last agent
                    if i == len(self.actors) - 1 and "loss_critic" in loss_vals:
                        log_dict["train/critic/loss_critic"] = loss_vals[
                            "loss_critic"
                        ].item()

                    # Per-agent std and lr
                    agent_td = tds[i]
                    agent_std = (
                        agent_td["scale"].mean().item()
                        if "scale" in agent_td.keys()
                        else 0
                    )
                    log_dict[f"train/agent_{i}/std"] = agent_std

        # Add shared training statistics
        log_dict["train/critic/explained_variance"] = explained_variance

        return log_dict
