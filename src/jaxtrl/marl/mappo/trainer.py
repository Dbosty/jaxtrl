import jax
import torch as th

from jaxtrl.marl.mappo import MAPPO, Hyperparameters
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


class MAPPOTrainer:
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

        critic_obs_key = rl_kwargs.get("critic_obs_key", Hyperparameters.critic_obs_key)
        critic_obs_size = rl_kwargs.get(
            "critic_obs_size", Hyperparameters.critic_obs_size
        )

        actor_keys = jax.random.split(jax_key, len(agents))

        self.actors = [
            self.get_actor(agent, config["actor"], actor_kwargs, actor_key, device)
            for agent, actor_key in zip(agents, actor_keys)
        ]

        self.critic = self.get_critic(
            agents[0],
            config["critic"],
            critic_kwargs,
            critic_obs_key,
            critic_obs_size,
            device,
        )

        self.algo = MAPPO(
            self.actors,
            self.critic,
            device=device,
            **rl_kwargs,
        )

    def step(self, ma_td):
        return self.algo.step(ma_td)

    @property
    def policies(self):
        return self.actors

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
    def get_critic(
        agent, critic_type, critic_kwargs, critic_obs_key, critic_obs_size, device
    ):
        # Normalize critic_obs_key to tuple format for consistent handling
        if isinstance(critic_obs_key, str):
            critic_obs_key = (critic_obs_key,)
        elif isinstance(critic_obs_key, (list, tuple)):
            critic_obs_key = tuple(critic_obs_key)

        critic_kwargs = {
            **critic_kwargs,
            "input_key": critic_obs_key,
            "input_size_override": critic_obs_size,
            "lstm_key_id": "critic",  # Use "critic" as the LSTM key identifier
        }

        # Create the critic using the factory
        critic = CRITIC_FACTORY_MAP[critic_type](
            agent.pi_obs_spec,
            agent.pi_action_spec,
            **{
                **critic_kwargs,
                "device": device,
            },
        )

        return critic
