import jax
import torch as th
from tensordict import TensorDict, merge_tensordicts

from jaxtrl.marl.mappo import MAPPO
from jaxtrl.marl.mappo import Hyperparameters as MAPPOHyperarameters
from jaxtrl.marl.mappo import MAPPOTrainer, add_critic_obs


class Hyperparameters(MAPPOHyperarameters):
    pass


class HAPPO(MAPPO):
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
            ma_td[*self.hp.critic_obs_key],
            ma_td["next", *self.hp.critic_obs_key],
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

        avg_kappas = []  # for logging

        kappa = 1
        for i in range(len(self.actors)):
            avg_kappas.append(1 if isinstance(kappa, int) else kappa.mean().item())
            for _ in range(self.hp.num_epochs):
                td = tds[i]

                td["advantage"] *= kappa

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
                    loss_vals["loss_critic"].backward()
                    if self.hp.clip_grad_norm:
                        th.nn.utils.clip_grad_norm_(
                            self.critic_parameters, self.hp.max_grad_norm
                        )
                    self.critic_optim.step()

                agents_loss_vals[i] = loss_vals.detach()

            kappa *= (
                self.actors[i].trl_actor.get_dist(td).log_prob(td["action"]).detach()
                / td["action_log_prob"]
            )[..., 0]

        log_dict = self._log_dict(tds, agents_loss_vals)

        for i in range(len(self.actors)):
            log_dict[f"train/agent_{i}/avg_kappa"] = avg_kappas[i]

        return log_dict


class HAPPOTrainer(MAPPOTrainer):
    def __init__(self, agents, config, jax_key, device=None):
        super().__init__(agents, config, jax_key, device)

        rl_kwargs = config["rl_kwargs"] if "rl_kwargs" in config else {}

        self.algo = HAPPO(
            self.actors,
            self.critic,
            device=device,
            **rl_kwargs,
        )
