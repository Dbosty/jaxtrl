from abc import ABC, abstractmethod

from jaxtyping import Array, Key, PyTree


class Observation(PyTree):
    """observation from environment"""


class Env(ABC):
    @abstractmethod
    def reset(self, key: Key) -> Observation:
        pass

    @abstractmethod
    def step(self, obs: Observation, action: PyTree[Array]) -> Observation:
        pass


class Agent:
    def __init__(self, sub_idx):
        self.sub_idx = sub_idx

    @property
    def pi_obs_spec(self):
        raise NotImplementedError

    @property
    def pi_action_spec(self):
        raise NotImplementedError

    def obs_env2pi(self, obs: Observation, context=None):
        raise NotImplementedError

    def action_pi2env(self, action: Array, context=None):
        raise NotImplementedError

    def init_context(self, key: Key):
        return None


class Reward:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, obs, action, next_obs, context=None):
        return 0.0, None

    def init_context(self, key: Key):
        return None
