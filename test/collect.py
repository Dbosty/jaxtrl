import logging

import hydra
import jax
import numpy as np
import torch as th

#
from pgj.env import ODEnv
from pgj.jaxtrl.agent import GreedyEvader, IndTeamHlp, IndTeamLlp
from pgj.jaxtrl.policy import GreedyPolicy
from pgj.jaxtrl.reward import SparseEvader, SparsePursuer

from jaxtrl.sim import Sim
from jaxtrl.sim.collect import collect_factory
from jaxtrl.sim.policy import PolicyAggregator, RandomPolicy

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    if cfg.seed is None:
        cfg.seed = np.random.randint(0, 2**32 - 1)

    key = jax.random.key(cfg.seed)

    key, seed_key = jax.random.split(key)

    np_seed, th_seed = jax.random.randint(seed_key, (2,), 0, 2**16 - 1)

    np.random.seed(np_seed)
    th.manual_seed(th_seed)

    env = ODEnv()

    agents = [
        IndTeamLlp(0, 1, 2, 100, env.grid.shape),
        IndTeamHlp(
            0,
            1,
            2,
            100,
            env.grid.shape,
        ),
        GreedyEvader(2),
    ]

    policy = PolicyAggregator(
        [
            RandomPolicy(agents[0].pi_action_spec),
            RandomPolicy(agents[1].pi_action_spec),
            GreedyPolicy(env, 2),
        ]
    )

    reward = [
        SparsePursuer(0, 1, 2, 1, 0),
        SparsePursuer(0, 1, 2, 1, 0),
        SparseEvader(2, 0, 1, 1, 0),
    ]

    key, trainer_key, policy_key = jax.random.split(key, 3)

    sim = Sim(env, agents, reward)

    collect = collect_factory(sim.reset, sim.step, policy, timesteps_per_env=100)

    collect = jax.jit(collect)

    transitions = collect(key)

    print(transitions)


if __name__ == "__main__":
    main()
