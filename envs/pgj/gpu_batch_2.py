"""Combined GPU batch for tonight's meeting-feedback items:
  1. Random's per-level win rate at K=8 (fills the missing 5th column for
     Table 1's per-level win-rate table -- PAP/DFO/DAgger/Offline already
     exist in figures_acc/figure_5_K8.log).
  2. TDO delays 1,2,3,4,5 for TDO-DAgger/TDO-Offline/TDO-Random at K=5
     (fig7's "exact points for delay 1-5" request).
  3. DAgger's actual mean episode length at K=5 (a single real measurement,
     not an approximation), for fig4's per-curve discounted-reward fix.

Run with, e.g.:
    python envs/pgj/gpu_batch_2.py \\
        level_k.K=8 level_k.checkpoint_dir=checkpoints/level_k_40M_8k \\
        pap.checkpoint_dir=checkpoints/pap_40M_8k \\
        lksp.checkpoint_dir=checkpoints/lksp_40M_8k \\
        dagger.checkpoint_dir=checkpoints/dagger_40M_8k \\
        offline.checkpoint_dir=checkpoints/offline_40M_8k \\
        eval.num_envs=500
"""

import copy
import sys

import _compat  # noqa: F401
import hydra
import jax
import numpy as np
import torch as th

from pgj.env import ODEnv
from pgj.jaxtrl.agent import IndEvader
from pgj.jaxtrl.reward import SparseEvader, SparsePursuer

from lksp import LKSPPolicy, TeamPursuerAgent, load_pursuer_levels
from pap import load_evader_levels

from level_k import EDR, HLP, LLP, SIM_MAX_STEPS
from evaluate import rollout, pursuer_winrate, episode_lengths, load_dagger, load_offline
from evaluate_acc import RandomDispatchPolicy, RandomMetaActor, TDOPolicy, discounted_reward
from plot_utils import save_values_txt
from jaxtrl.sim.policy import PolicyAggregator

sys.stdout.reconfigure(line_buffering=True)  # avoid the earlier block-buffering blind spot


def view_cfg(cfg, L):
    vc = copy.deepcopy(cfg)
    vc.level_k.K = L
    return vc


@hydra.main(version_base=None, config_path="../../conf", config_name="evaluate")
def main(cfg):
    np.random.seed(cfg.seed)
    th.manual_seed(cfg.seed)
    key = jax.random.key(cfg.seed)
    env = ODEnv()

    # === Part 1: Random's per-level win rate at K=8 ===
    print("=== [1/3] Random per-level win rate at K=8 ===")
    key, k1, k2 = jax.random.split(key, 3)
    pursuer_levels_8 = load_pursuer_levels(cfg, env, k1)  # cfg.level_k.K == 8 already
    evader_levels_8 = load_evader_levels(cfg, env, k2)

    random_rates_k8 = {}
    for j in range(cfg.level_k.K + 1):
        team_agent = TeamPursuerAgent(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape)
        if j == 0:
            from evaluate import evader_agent_and_policy
            evader_agent, evader_policy = evader_agent_and_policy(cfg, env, evader_levels_8, 0)
        else:
            evader_agent, evader_policy = IndEvader(EDR, LLP, HLP, SIM_MAX_STEPS, env.grid.shape), evader_levels_8[j - 1]
        agents = [team_agent, evader_agent]
        reward = [SparsePursuer(LLP, HLP, EDR, 1, 0), SparseEvader(EDR, LLP, HLP, 1, 0)]
        random_policy = RandomDispatchPolicy(pursuer_levels_8, num_levels=cfg.level_k.K + 1)
        policy = PolicyAggregator([random_policy, evader_policy])
        key, rk = jax.random.split(key)
        transitions = rollout(env, agents, reward, policy, cfg.eval.num_envs, rk)
        wr = pursuer_winrate(transitions)
        random_rates_k8[j] = wr
        print(f"Table1: evader L{j} -> Random={wr:.4f}")

    save_values_txt("figures_acc_new/table1_random_k8.txt", **{"evader_level, random_winrate": (list(random_rates_k8.keys()), list(random_rates_k8.values()))})

    # === Part 2: TDO delays 1-5 at K=5 ===
    print("=== [2/3] TDO delays 1-5 at K=5 ===")
    L = 5
    vc = view_cfg(cfg, L)
    key, k3, k4, k5, k6 = jax.random.split(key, 5)
    pursuer_levels_5 = load_pursuer_levels(cfg, env, k3)  # FULL K=8 checkpoint, used as a K=5 "view"
    evader_levels_5 = load_evader_levels(cfg, env, k4)[:L]
    dagger_classifier = load_dagger(cfg, env, k5)
    offline_classifier = load_offline(cfg, env, k6)
    sub_levels_5 = pursuer_levels_5[1:]

    new_delays = [1, 2, 3, 4, 5]
    tdo_results = {}
    for name, meta_factory in [
        ("TDO-DAgger", lambda: dagger_classifier),
        ("TDO-Offline", lambda: offline_classifier),
        ("TDO-Random", lambda: RandomMetaActor(num_levels=L)),
    ]:
        meta = meta_factory()
        vals = []
        for delay in new_delays:
            level_returns = []
            for k in range(1, L + 1):
                agents = [TeamPursuerAgent(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape), IndEvader(EDR, LLP, HLP, SIM_MAX_STEPS, env.grid.shape)]
                reward = [SparsePursuer(LLP, HLP, EDR, 1, 0), SparseEvader(EDR, LLP, HLP, 1, 0)]
                tdo_policy = TDOPolicy(meta, sub_levels_5, delay, k - 1)
                policy = PolicyAggregator([tdo_policy, evader_levels_5[k - 1]])
                key, rk = jax.random.split(key)
                transitions = rollout(env, agents, reward, policy, cfg.eval.num_envs, rk)
                level_returns.append(discounted_reward(transitions, agent_idx=0))
            mean_dr = float(np.mean(level_returns))
            vals.append(mean_dr)
            print(f"{name} delay={delay} -> expected_discounted_reward={mean_dr:.4f}")
        tdo_results[name] = vals

    save_values_txt(
        "figures_acc_new/fig7_new_delays.txt",
        new_delays=new_delays,
        **{f"{name} (expected_discounted_reward by delay)": (new_delays, vals) for name, vals in tdo_results.items()},
    )

    # === Part 3: DAgger's real mean episode length at K=5 ===
    print("=== [3/3] DAgger mean episode length at K=5 ===")
    from pap import UniformLevelKEvaderPolicy
    dagger_policy = LKSPPolicy(dagger_classifier, sub_levels_5)
    evader_policy_mixed = UniformLevelKEvaderPolicy(evader_levels_5)
    evader_agent = IndEvader(EDR, LLP, HLP, SIM_MAX_STEPS, env.grid.shape)
    team_agent = TeamPursuerAgent(LLP, HLP, EDR, SIM_MAX_STEPS, env.grid.shape)
    agents = [team_agent, evader_agent]
    reward = [SparsePursuer(LLP, HLP, EDR, 1, 0), SparseEvader(EDR, LLP, HLP, 1, 0)]
    policy = PolicyAggregator([dagger_policy, evader_policy_mixed])
    key, rk = jax.random.split(key)
    transitions = rollout(env, agents, reward, policy, cfg.eval.num_envs, rk)
    lengths, _ = episode_lengths(transitions, EDR)
    mean_len = float(np.mean(lengths))
    print(f"DAgger mean episode length (K=5, final checkpoint) = {mean_len:.2f}")
    save_values_txt("figures_acc_new/dagger_mean_ep_len.txt", dagger_mean_ep_len=mean_len)

    print("=== ALL DONE ===")


if __name__ == "__main__":
    main()
