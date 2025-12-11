from dataclasses import dataclass

import torch as th
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE


@dataclass
class Hyperparameters:
    sub_batch_size: int = 64
    num_epochs: int = 10
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    max_grad_norm: float = 0.5
    entropy_eps: float = 0.005
    critic_coef: float = 0.5
    loss_critic_type: str = "l2"
    average_gae: bool = False
    clip_grad_norm: bool = False


class PPO:
    def __init__(
        self,
        actor,
        critic,
        # ,
        optim=None,
        scheduler=None,
        # ,
        device=None,
        # ,
        **hp,
    ):
        self.device = device or "cpu"

        self.actor = actor
        self.critic = critic

        self.hp = Hyperparameters(**hp)

        ###

        if optim is None:
            self.parameters = list(self.actor.trl_actor.parameters()) + list(
                self.critic.parameters()
            )
            optim = th.optim.Adam(self.parameters, lr=self.hp.lr)

        self.optim = optim

        if scheduler is None:
            self.scheduler = th.optim.lr_scheduler.LambdaLR(
                self.optim, lambda _: 1.0
            )  # Constant LR scheduler
        self.scheduler = scheduler

        ###

        self.advantage_module = GAE(
            gamma=self.hp.gamma,
            lmbda=self.hp.gae_lambda,
            value_network=self.critic,
            average_gae=self.hp.average_gae,
            deactivate_vmap=True,
            # shifted=True, #TODO: perf improvement
        )

        self.loss_module = ClipPPOLoss(
            actor_network=actor.trl_actor,
            critic_network=self.critic,
            clip_epsilon=self.hp.clip_epsilon,
            entropy_bonus=bool(self.hp.entropy_eps),
            entropy_coeff=self.hp.entropy_eps,
            critic_coeff=self.hp.critic_coef,
            loss_critic_type=self.hp.loss_critic_type,  # l2 used in sb3 over default smooth_l1 in torchrl impl
        )

    def step(self, td):
        # replacing this with deactivate_vmap = True
        # td = self.critic(td)
        # td["next"] = self.critic(td["next"])

        # td["state_value"] = td["state_value"].detach()
        # td["next", "state_value"] = td["next", "state_value"].detach()

        td["action_log_prob"] = (
            self.actor.trl_actor.get_dist(td).log_prob(td["action"]).detach()
        )
        td = self.advantage_module(td)

        # Store final loss values for logging
        final_loss_value = None
        final_loss_vals = None

        for _ in range(self.hp.num_epochs):
            for i in range(max(td.shape[0] // self.hp.sub_batch_size, 1)):
                subdata = td[
                    self.hp.sub_batch_size * i : self.hp.sub_batch_size * (i + 1)
                ]  # doing tbptt

                # with set_recurrent_mode(True):  # recurrent mode is true by default
                loss_vals = self.loss_module(subdata)

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + (loss_vals["loss_entropy"] if self.hp.entropy_eps else 0)
                )

                self.optim.zero_grad()
                loss_value.backward()
                if self.hp.clip_grad_norm:
                    th.nn.utils.clip_grad_norm_(self.parameters, self.hp.max_grad_norm)
                self.optim.step()

                # Store final loss values for logging
                final_loss_value = loss_value.detach()
                final_loss_vals = {k: v.detach() for k, v in loss_vals.items()}

        # Use original_td for logging to avoid holding references to modified tensordict
        log_dict = self._log_dict(td)

        # Update log_dict with actual loss values
        if final_loss_value is not None and final_loss_vals is not None:
            log_dict.update(
                {
                    "train/loss": final_loss_value.item(),
                    "train/loss_objective": final_loss_vals["loss_objective"].item(),
                    "train/loss_critic": final_loss_vals["loss_critic"].item(),
                    "train/kl_approx": final_loss_vals["kl_approx"].item(),
                    "train/clip_fraction": final_loss_vals["clip_fraction"].item(),
                }
            )
            if self.hp.entropy_eps:
                log_dict.update(
                    {
                        "train/loss_entropy": final_loss_vals["loss_entropy"].item(),
                        "train/entropy": final_loss_vals["entropy"].item(),
                    }
                )

        return log_dict

    def _log_dict(self, tensordict_data):
        # Use terminated for stats, but fall back to done (terminated | truncated)
        # when no episodes terminate (e.g., agent is very good and all eps truncate)
        terminated = tensordict_data.get(("next", "terminated"))
        done = tensordict_data.get(("next", "done"))

        if terminated.any():
            step_counts = tensordict_data["next", "step_count"][terminated].float()
            episode_rewards = tensordict_data["next", "episode_reward"][terminated]
        elif done.any():
            # Fall back to all done episodes
            step_counts = tensordict_data["next", "step_count"][done].float()
            episode_rewards = tensordict_data["next", "episode_reward"][done]
        else:
            # Last resort: use last frame
            step_counts = tensordict_data["next", "step_count"][-1:].float()
            episode_rewards = tensordict_data["next", "episode_reward"][-1:]

        rollout_data = {
            "ep_len_mean": step_counts.mean().item(),
            "ep_rew_mean": episode_rewards.mean().item(),
            "step_rew_mean": tensordict_data["next", "reward"].mean().item(),
            "ep_len_min": step_counts.min().item(),
            "ep_len_max": step_counts.max().item(),
        }

        # train data - calculate all metrics that need tensordict_data before deleting it
        explained_variance = -1
        if th.var(tensordict_data.get("state_value")) > 0:
            explained_variance = max(
                -1.0,
                1
                - (
                    th.var(
                        tensordict_data.get("value_target")
                        - tensordict_data.get("state_value")
                    )
                    / th.var(tensordict_data.get("value_target"))
                ),
            )

        action_std = (
            tensordict_data.get("scale").mean().item()
            if "scale" in tensordict_data
            else 0
        )  # TODO: make compatible with discrete

        # Clean up tensordict reference immediately after extracting needed values
        del tensordict_data
        del terminated

        train_data = {
            "loss": 0.0,  # Will be filled by caller
            "loss_objective": 0.0,  # Will be filled by caller
            "loss_critic": 0.0,  # Will be filled by caller
            **(
                {
                    "loss_entropy": 0.0,  # Will be filled by caller
                    "entropy": 0.0,  # Will be filled by caller
                }
                if self.hp.entropy_eps
                else {}
            ),
            "lr": self.optim.param_groups[0]["lr"],
            "kl_approx": 0.0,  # Will be filled by caller
            "clip_fraction": 0.0,  # Will be filled by caller
            "explained_variance": explained_variance,
            "std": action_std,
        }

        data = {
            "rollout": rollout_data,
            "train": train_data,
        }

        log_dict = {
            f"{key}/{k}": v for key, value in data.items() for k, v in value.items()
        }

        return log_dict
