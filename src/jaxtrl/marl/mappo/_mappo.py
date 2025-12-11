from dataclasses import dataclass

import torch as th
from tensordict import TensorDict, merge_tensordicts
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from jaxtrl.rl.ppo import Hyperparameters as PPOHyperparameters


@dataclass
class Hyperparameters(PPOHyperparameters):
    pass


class MAPPO:
    def __init__(self, actors, critic, device=None, **hp):
        self.device = device or "cpu"

        self.actors = actors
        self.critic = critic

        self.hp = Hyperparameters(**hp)

        self.actor_parameters = [actor.parameters() for actor in actors]
        self.critic_parameters = critic.parameters()

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

    def step(self, tds):
        base_td = TensorDict(
            source={
                "next": {
                    "terminated": tds[0]["next", "terminated"],
                    "truncated": tds[0]["next", "truncated"],
                    "done": tds[0]["next", "done"],
                    "step_count": tds[0]["next", "step_count"],
                },
                "step_count": tds[0]["step_count"],
            },
            batch_size=tds[0].batch_size,
            device=tds[0].device,
        )
        tds = [merge_tensordicts(td, base_td) for td in tds]
        log_dicts = [ppo.step(td) for ppo, td in zip(self.ppos, tds)]
        return {
            **{
                f"agent_{i}/{k}": v
                for i, log_dict in enumerate(log_dicts)
                for k, v in log_dict.items()
            },
        }
