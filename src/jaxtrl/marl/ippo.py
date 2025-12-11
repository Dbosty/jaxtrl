from dataclasses import dataclass

import torch as th

from jaxtrl.rl.ppo import PPO
from jaxtrl.rl.ppo import Hyperparameters as PPOHyperparameters
from jaxtrl.rl.ppo import PPOTrainer


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


class IPPOTrainer(PPOTrainer):
    def __init__(self, agents, config, jax_key, device=None):
        if device is not None:
            pass
        elif "device" in config:
            device = config["device"]
        else:
            device = "cuda" if th.cuda.is_available() else "cpu"

        self.device = device

        actor_kwargs = config["actor_kwargs"] if "actor_kwargs" in config else {}
        critic_kwargs = config["critic_kwargs"] if "critic_kwargs" in config else {}
        rl_kwargs = config["rl_kwargs"] if "rl_kwargs" in config else {}

        self.actors = [
            self.get_actor(agent, config["actor"], actor_kwargs, jax_key, device)
            for agent in agents
        ]

        self.critics = [
            self.get_critic(agent, config["critic"], critic_kwargs, jax_key, device)
            for agent in agents
        ]

        self.ippo = IPPO(
            self.actors,
            self.critics,
            **rl_kwargs,
        )

    def step(self, tds):
        return self.ippo.step(tds)

    @property
    def policies(self):
        return self.actors
