from ._ppo import PPO
from .actor.lstm import LSTMActor
from .actor.mlp import MLPActor
from .critic import lstm_critic_factory, mlp_critic_factory

ALGO = PPO

ACTOR_MAP = {
    "mlp": MLPActor,
    "lstm": LSTMActor,
}
CRITIC_FACTORY_MAP = {"mlp": mlp_critic_factory, "lstm": lstm_critic_factory}
