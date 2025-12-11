import torch as th
from tensordict import TensorDict

from jaxtrl.sim.collect import Transition


def tdl(array):
    return th.from_dlpack(array)


def transitions_to_tensordicts(transitions: Transition):
    tds = []
    for i in range(len(transitions.obs)):
        transition = transitions[i]

        obs = tdl(transition.obs)
        action = tdl(transition.action)
        step_count = tdl(transition.step_count)

        next_obs = tdl(transition.next_obs)
        reward = tdl(transition.reward)
        terminated = tdl(transition.terminated)
        truncated = tdl(transition.truncated)
        episode_reward = tdl(transition.episode_reward)

        td = TensorDict(
            source={
                "observation": obs,
                "action": action,
                "next": {
                    "observation": next_obs,
                    "reward": reward,
                    "episode_reward": episode_reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "done": terminated | truncated,
                    "step_count": step_count + 1,
                },
                "step_count": step_count,
            },
            batch_size=obs.shape[0],
            device=obs.device,
        )

        tds.append(td)

    return tds


def transitions_to_ma_tensordict(transitions: Transition):
    tds = transitions_to_tensordicts(transitions)

    ma_source = {
        "next": {
            "terminated": tds[0]["next", "terminated"],
            "truncated": tds[0]["next", "truncated"],
            "done": tds[0]["next", "done"],
            "step_count": tds[0]["next", "step_count"],
        },
        "step_count": tds[0]["step_count"],
    }

    for i in range(len(tds)):
        ma_source[f"agent_{i}"] = {
            "observation": tds[i]["observation"],
            "action": tds[i]["action"],
        }

        ma_source["next"] |= {
            f"agent_{i}": {
                "observation": tds[i]["next", "observation"],
                "reward": tds[i]["next", "reward"],
                "episode_reward": tds[i]["next", "episode_reward"],
            },
        }

    return TensorDict(
        source=ma_source,
        batch_size=tds[0].batch_size,
        device=tds[0].device,
    )
