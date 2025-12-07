import logging

import hydra
import jax
import numpy as np
import torch as th
from omegaconf import DictConfig, OmegaConf

#
from pgj.env import ODEnv
from pgj.jaxtrl.agent import GreedyEvader, IndTeamHlp, IndTeamLlp
from pgj.jaxtrl.reward import SparseEvader, SparsePursuer

from jaxtrl.utils.trainer import RLTrainer

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    if cfg.seed is None:
        cfg.seed = np.random.randint(0, 2**32 - 1)

    key = jax.random.key(cfg.seed)

    key, np_seed, th_seed = jax.random.split(key, 3)

    np.random.seed(np_seed)
    th.manual_seed(th_seed)

    env = ODEnv()

    agents = [
        IndTeamLlp(0, 1, 2, 100, env.grid.shape),
        IndTeamHlp(0, 1, 2, 100, env.grid.shape),
        GreedyEvader(2),
    ]

    reward = [
        SparsePursuer(0, 1, 2, 1, 0),
        SparsePursuer(0, 1, 2, 1, 0),
        SparseEvader(0, 1, 2, 1, 0),
    ]

    key, trainer_key, policy_key = jax.random.split(key, 3)

    trainer = RLTrainer(agents[0], cfg.train, trainer_key, device=cfg.device)

    policy = trainer.policy
    policy_context = policy.init_context(policy_key)


if __name__ == "__main__":
    main()
