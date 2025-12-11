import hydra
import jax
import jax.numpy as jnp
import numpy as np
import torch as th
from omegaconf import DictConfig, OmegaConf

#
from pgj.env import ODEnv
from pgj.jaxtrl.agent import GreedyEvader, IndTeamHlp, IndTeamLlp
from pgj.jaxtrl.policy import GreedyPolicy
from pgj.jaxtrl.reward import SparseEvader, SparsePursuer
from pgj.jaxtrl.term import is_terminated

from jaxtrl.rl.ppo.trainer import PPOTrainer
from jaxtrl.sim import Sim
from jaxtrl.sim.collect import collect_factory
from jaxtrl.sim.policy import PolicyAggregator, RandomPolicy
from jaxtrl.utils.logger import ExtWandbLogger
from jaxtrl.utils.tensordict import transitions_to_tensordicts


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
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
        IndTeamHlp(0, 1, 2, 100, env.grid.shape),
        GreedyEvader(2),
    ]

    reward = [
        SparsePursuer(0, 1, 2, 1, 0),
        SparsePursuer(0, 1, 2, 1, 0),
        SparseEvader(0, 1, 2, 1, 0),
    ]

    key, trainer_key, policy_key = jax.random.split(key, 3)

    trainer = PPOTrainer(agents[0], cfg.train, trainer_key, device=cfg.device)

    policy = PolicyAggregator(
        [
            trainer.policy,
            RandomPolicy(agents[1].pi_action_spec),
            GreedyPolicy(env, 2),
        ]
    )
    policy_context = policy.init_context(policy_key)

    sim = Sim(env, agents, reward, is_terminated)

    collect = collect_factory(
        sim.reset,
        sim.step,
        policy,
        timesteps_per_env=cfg.collect.batch.timesteps_per_env,
        num_envs=cfg.collect.batch.num_envs,
        extended_transition=True,
    )
    collect = jax.jit(collect)

    logger = ExtWandbLogger(
        exp_name="pgj_ppo_test",
        **{**cfg.env_var.wandb, **cfg.collect.wandb},
    )

    total_frames = 0
    for i in range(
        cfg.collect.total_timesteps // cfg.collect.batch.timesteps_per_env + 1
    ):
        key, policy_key, collect_key = jax.random.split(key, 3)

        policy_context = policy.init_context(policy_key)

        # collect
        transitions = collect(collect_key, policy_context)

        tds = transitions_to_tensordicts(transitions)

        total_frames += transitions.step_count.size

        # train
        log_dict = trainer.step(tds[0])

        log_dict.update(
            {
                "rollout/pursuer_winrate": jnp.sum(
                    transitions.raw_next_obs.state.agent_win[..., 0]
                )
                / jnp.sum(transitions.terminated),
                "rollout/evader_winrate": jnp.sum(
                    transitions.raw_next_obs.state.agent_win[..., 2]
                )
                / jnp.sum(transitions.terminated),
            }
        )

        logger.log_scalars(log_dict, timestep=total_frames)


if __name__ == "__main__":
    main()
