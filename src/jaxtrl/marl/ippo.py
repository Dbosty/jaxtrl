from dataclasses import dataclass

import torch as th
from tensordict import TensorDict, merge_tensordicts

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

        tds = []
        for i in range(len(self.ppos)):
            td = merge_tensordicts(
                base_td,
                ma_td[f"agent_{i}"],
            )
            td["next"].update(ma_td["next", f"agent_{i}"])
            tds.append(td)

        log_dicts = [ppo.step(td) for ppo, td in zip(self.ppos, tds)]

        # Transform keys from "category/metric" to "category/agent_i/metric"
        log_dict = {}
        for i, ld in enumerate(log_dicts):
            for k, v in ld.items():
                parts = k.split("/", 1)
                if len(parts) == 2:
                    new_key = f"{parts[0]}/agent_{i}/{parts[1]}"
                else:
                    new_key = f"agent_{i}/{k}"
                log_dict[new_key] = v

        # Use agent_0's episode length stats as shared rollout stats
        log_dict["rollout/ep_len_mean"] = log_dict.get("rollout/agent_0/ep_len_mean")
        log_dict["rollout/ep_len_min"] = log_dict.get("rollout/agent_0/ep_len_min")
        log_dict["rollout/ep_len_max"] = log_dict.get("rollout/agent_0/ep_len_max")

        for i in range(len(self.ppos)):
            del log_dict[f"rollout/agent_{i}/ep_len_mean"]
            del log_dict[f"rollout/agent_{i}/ep_len_min"]
            del log_dict[f"rollout/agent_{i}/ep_len_max"]

        return log_dict


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
            self.get_critic(agent, config["critic"], critic_kwargs, device)
            for agent in agents
        ]

        self.algo = IPPO(
            self.actors,
            self.critics,
            **rl_kwargs,
        )

    def step(self, ma_td):
        return self.algo.step(ma_td)

    @property
    def policies(self):
        return self.actors
