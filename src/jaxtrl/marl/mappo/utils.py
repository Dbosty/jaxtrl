from jaxtrl.utils.tensordict import tdl


def add_critic_obs(td, critic_obs_key, obs, next_obs):
    if isinstance(critic_obs_key, str):
        critic_obs_key = [critic_obs_key]
    td.set(critic_obs_key, tdl(obs))
    td.set(("next", *critic_obs_key), tdl(next_obs))

    return td
