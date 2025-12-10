from dataclasses import dataclass

import torch as th


@dataclass
class Hyperparameters:
    entropy_eps: float = 5e-3
    num_epochs: int = 10
    sub_batch_size: int = 50_000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    max_grad_norm: float = 0.5
    critic_coef: float = 0.5
    lr: float = 3e-4
    loss_critic_type: str = "l2"
    average_gae: bool = False


class IPPO:
    def __init__(self, actors, critics, device=None, **hp):
        self.device = device or "cpu"
        self.total_frames = 0

        self.actors = actors
        self.critics = critics

        self.hp = Hyperparameters(**hp)

        self.parameters = [
            list(actor.trl_actor.parameters()) + list(critic.parameters())
            for actor, critic in zip(self.actors, self.critics)
            for actor, critic in zip(self.actors, self.critics)
        ]
        self.optims = [
            th.optim.Adam(parameters, lr=self.hp.lr) for parameters in self.parameters
        ]

    def step(self, tds):
        pass
