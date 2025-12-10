from dataclasses import dataclass

import torch as th

from jaxtrl.rl.ppo import PPO
from jaxtrl.rl.ppo import Hyperparameters as PPOHyperparameters


@dataclass
class Hyperparameters(PPOHyperparameters):
    pass


class IPPO:
    def __init__(self, actors, critics, device=None, **hp):
        self.ppos = [
            PPO(actor, critic, device=device, **hp)
            for actor, critic in zip(actors, critics)
        ]

    def step(self, tds):
        pass
