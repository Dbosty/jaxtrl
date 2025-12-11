import time

import jax
import numpy as np
from pgj.env import ODEnv
from pgj.jaxtrl.agent import GreedyEvader, IndTeamHlp, IndTeamLlp
from pgj.jaxtrl.policy import GreedyPolicy
from pgj.jaxtrl.reward import SparseEvader, SparsePursuer
from pgj.jaxtrl.term import is_terminated
from pgj.visualizer import Visualizer

from jaxtrl.sim import Sim
from jaxtrl.sim.policy import PolicyAggregator, RandomPolicy


def main(jit, sleep_time, seed):
    env = ODEnv("rust")

    key = jax.random.key(seed)

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

    policy = PolicyAggregator(
        [
            RandomPolicy(agents[0].pi_action_spec),
            RandomPolicy(agents[1].pi_action_spec),
            GreedyPolicy(env, 2),
        ]
    )

    key, subkey = jax.random.split(key)
    policy_context = policy.init_context(subkey)

    sim = Sim(env, agents, reward, is_terminated)
    reset = sim.reset
    step = sim.step
    policy_call = policy.__call__
    if jit:
        reset = jax.jit(reset)
        step = jax.jit(step)
        policy_call = jax.jit(policy_call)

    visualizer = Visualizer()
    visualizer.start()

    obs, obs_for_pi, agent_context, sim_context = reset(env.init_key)

    done = False
    while True:
        visualizer.render(obs)

        time.sleep(sleep_time)

        if done:
            obs, obs_for_pi, agent_context, sim_context = reset(obs.state.rng.next_key)
            done = False
        else:
            action_from_pi, policy_context = policy_call(
                obs_for_pi, policy_context, obs.state.rng.policy_key
            )
            obs, reward, term, trunc, obs_for_pi, agent_context, sim_context = step(
                obs, action_from_pi, agent_context, sim_context
            )
            if term or trunc:
                done = True


if __name__ == "__main__":
    import argparse

    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--jit", action="store_true", help="Enable JIT compilation")
        parser.add_argument(
            "--sleep_time", type=float, default=0.01, help="Sleep time between steps"
        )
        parser.add_argument("--seed", type=int, default=None, help="Random seed")
        args = parser.parse_args()

        if args.seed is None:
            args.seed = np.random.randint(0, 2**16 - 1)

        main(args.jit, args.sleep_time, args.seed)
