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
        log_dicts = [ppo.step(td) for ppo, td in zip(self.ppos, tds)]
        return {
            **{
                f"agent_{i}/{k}": v
                for i, log_dict in enumerate(log_dicts)
                for k, v in log_dict.items()
            },
        }


class IPPOTrainer:
    pass
