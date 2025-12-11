import torch as th

from jaxtrl.rl.ppo._ppo import PPO
from jaxtrl.rl.ppo.actor.lstm import LSTMActor
from jaxtrl.rl.ppo.actor.mlp import MLPActor
from jaxtrl.rl.ppo.critic import lstm_critic_factory, mlp_critic_factory

ACTOR_MAP = {
    "mlp": MLPActor,
    "lstm": LSTMActor,
}


CRITIC_FACTORY_MAP = {
    "mlp": mlp_critic_factory,
    "lstm": lstm_critic_factory,
}


class PPOTrainer:
    def __init__(self, agent, config, jax_key, device=None):
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

        self.actor = self.get_actor(
            agent, config["actor"], actor_kwargs, jax_key, device
        )

        self.critic = self.get_critic(agent, config["critic"], critic_kwargs, device)

        self.algo = PPO(
            self.actor,
            self.critic,
            **rl_kwargs,
        )

    def step(self, td):
        return self.algo.step(td)

    @property
    def policy(self):
        return self.actor

    @staticmethod
    def get_actor(agent, agent_type, agent_kwargs, key, device):
        return ACTOR_MAP[agent_type](
            agent.pi_obs_spec,
            agent.pi_action_spec,
            key,
            **{
                **agent_kwargs,
                "device": device,
            },
        )

    @staticmethod
    def get_critic(agent, critic_type, critic_kwargs, device):
        return CRITIC_FACTORY_MAP[critic_type](
            agent.pi_obs_spec,
            agent.pi_action_spec,
            **{
                **critic_kwargs,
                "device": device,
            },
        )
