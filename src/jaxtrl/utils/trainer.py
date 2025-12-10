import importlib

import jax
import torch as th

RL_MAP = {"ppo": "jaxtrl.rl.ppo"}


class RLTrainer:
    def __init__(self, agent, config, jax_key, device=None):
        if device is not None:
            pass
        elif "device" in config:
            device = config["device"]
        else:
            device = "cuda" if th.cuda.is_available() else "cpu"

        self.device = device

        rl_class = importlib.import_module(RL_MAP[config["rl"]])

        actor_kwargs = config["actor_kwargs"] if "actor_kwargs" in config else {}
        critic_kwargs = config["critic_kwargs"] if "critic_kwargs" in config else {}
        rl_kwargs = config["rl_kwargs"] if "rl_kwargs" in config else {}

        self.actor = rl_class.ACTOR_MAP[config["actor"]](
            agent.pi_obs_spec,
            agent.pi_action_spec,
            jax_key,
            **{
                **actor_kwargs,
                "device": device,
            },
        )

        self.critic = rl_class.CRITIC_FACTORY_MAP[config["critic"]](
            agent.pi_obs_spec,
            agent.pi_action_spec,
            **{
                **critic_kwargs,
                "device": device,
            },
        )

        self.rl_algo = rl_class.ALGO(
            self.actor,
            self.critic,
            **rl_kwargs,
        )

    def step(self, td):
        return self.rl_algo.step(td)

    @property
    def policy(self):
        return self.actor


def get_base_actor(agent, config, device=None, jax_key=None):
    if jax_key is None:
        jax_key = jax.random.key(0)

    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"

    rl_class = importlib.import_module(RL_MAP[config["rl"]])

    actor = rl_class.ACTOR_MAP[config["actor"]](
        agent.pi_obs_spec,
        agent.pi_action_spec,
        jax_key,
        **{
            **(config["actor_kwargs"] if "actor_kwargs" in config else {}),
            "device": device,
        },
    )

    return actor
