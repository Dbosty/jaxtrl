import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from jaxtrl.model.jax import CategoricalSampler, MultiCategoricalSampler


def test_categorical_sampler():
    """Test CategoricalSampler with single Discrete action space."""
    sampler = CategoricalSampler()
    key = jax.random.key(42)

    # Test with uniform logits
    logits = jnp.zeros(5)
    for i in range(100):
        key, subkey = jax.random.split(key)
        action = sampler(logits, subkey)
        assert 0 <= int(action) < 5, f"Invalid action: {action}"

    # Test deterministic sampling - should return argmax
    logits = jnp.array([0.0, 0.0, 10.0, 0.0, 0.0])  # Strongly prefer action 2
    action = sampler.deterministic_sample(logits)
    assert int(action) == 2, f"Expected action 2, got {action}"

    print("test_categorical_sampler passed")


def test_multicategorical_sampler():
    """Test MultiCategoricalSampler with MultiDiscrete action space."""
    nvec = [3, 4, 5]
    sampler = MultiCategoricalSampler(nvec)
    key = jax.random.key(42)

    # Test with uniform logits
    logits = jnp.zeros(sum(nvec))  # 3 + 4 + 5 = 12
    for i in range(100):
        key, subkey = jax.random.split(key)
        action = sampler(logits, subkey)
        assert action.shape == (3,), f"Wrong shape: {action.shape}"
        assert 0 <= int(action[0]) < 3, f"Dim 0 invalid: {action[0]}"
        assert 0 <= int(action[1]) < 4, f"Dim 1 invalid: {action[1]}"
        assert 0 <= int(action[2]) < 5, f"Dim 2 invalid: {action[2]}"

    # Test deterministic sampling
    # Logits: dim0=[0,0,10], dim1=[0,10,0,0], dim2=[0,0,0,10,0]
    logits = jnp.array([0.0, 0.0, 10.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0])
    action = sampler.deterministic_sample(logits)
    assert int(action[0]) == 2, f"Dim 0 expected 2, got {action[0]}"
    assert int(action[1]) == 1, f"Dim 1 expected 1, got {action[1]}"
    assert int(action[2]) == 3, f"Dim 2 expected 3, got {action[2]}"

    print("test_multicategorical_sampler passed")


def test_per_dimension_randomness():
    """Test that each dimension samples independently (not correlated)."""
    nvec = [10, 10, 10]
    sampler = MultiCategoricalSampler(nvec)
    key = jax.random.key(42)

    # Collect samples
    samples = []
    logits = jnp.zeros(sum(nvec))
    for i in range(1000):
        key, subkey = jax.random.split(key)
        action = sampler(logits, subkey)
        samples.append(np.array(action))

    samples = np.array(samples)

    # Check correlation between dimensions - should be low if independent
    for i in range(3):
        for j in range(i + 1, 3):
            corr = np.corrcoef(samples[:, i], samples[:, j])[0, 1]
            assert (
                abs(corr) < 0.15
            ), f"Dims {i} and {j} are too correlated: {corr:.4f}"

    print(f"test_per_dimension_randomness passed (correlations < 0.15)")


def test_discrete_lstm_actor_discrete():
    """Test DiscreteLSTMActor with Discrete action space."""
    from jaxtrl.rl.ppo.actor.discrete_lstm import DiscreteLSTMActor

    obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Discrete(5)
    key = jax.random.key(42)

    actor = DiscreteLSTMActor(obs_space, action_space, key)
    context = actor.init_context(None)

    # Sample multiple times and verify actions are in valid range
    for i in range(100):
        key, subkey = jax.random.split(key)
        obs = jnp.zeros(4)
        action, context = actor(obs, context, subkey)
        assert 0 <= int(action) < 5, f"Invalid action: {action}"

    # Test deterministic sampling
    obs = jnp.zeros(4)
    action1, _ = actor.deterministic_sample(obs, context)
    action2, _ = actor.deterministic_sample(obs, context)
    assert jnp.array_equal(action1, action2), "Deterministic samples should be equal"

    print("test_discrete_lstm_actor_discrete passed")


def test_discrete_lstm_actor_multidiscrete():
    """Test DiscreteLSTMActor with MultiDiscrete action space."""
    from jaxtrl.rl.ppo.actor.discrete_lstm import DiscreteLSTMActor

    obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.MultiDiscrete([3, 4, 5])
    key = jax.random.key(42)

    actor = DiscreteLSTMActor(obs_space, action_space, key)
    context = actor.init_context(None)

    # Sample multiple times and verify each dimension is in valid range
    for i in range(100):
        key, subkey = jax.random.split(key)
        obs = jnp.zeros(4)
        action, context = actor(obs, context, subkey)
        assert action.shape == (3,), f"Wrong shape: {action.shape}"
        assert 0 <= int(action[0]) < 3, f"Dim 0 invalid: {action[0]}"
        assert 0 <= int(action[1]) < 4, f"Dim 1 invalid: {action[1]}"
        assert 0 <= int(action[2]) < 5, f"Dim 2 invalid: {action[2]}"

    # Test deterministic sampling
    obs = jnp.zeros(4)
    action1, _ = actor.deterministic_sample(obs, context)
    action2, _ = actor.deterministic_sample(obs, context)
    assert jnp.array_equal(action1, action2), "Deterministic samples should be equal"

    print("test_discrete_lstm_actor_multidiscrete passed")


def test_discrete_lstm_actor_per_dim_randomness():
    """Test that DiscreteLSTMActor samples independently per dimension."""
    from jaxtrl.rl.ppo.actor.discrete_lstm import DiscreteLSTMActor

    obs_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.MultiDiscrete([10, 10, 10])
    key = jax.random.key(42)

    actor = DiscreteLSTMActor(obs_space, action_space, key)
    context = actor.init_context(None)

    # Collect samples
    samples = []
    for i in range(1000):
        key, subkey = jax.random.split(key)
        obs = jnp.zeros(4)
        action, context = actor(obs, context, subkey)
        samples.append(np.array(action))

    samples = np.array(samples)

    # Check correlation between dimensions - should be low if independent
    for i in range(3):
        for j in range(i + 1, 3):
            corr = np.corrcoef(samples[:, i], samples[:, j])[0, 1]
            assert (
                abs(corr) < 0.15
            ), f"Dims {i} and {j} are too correlated: {corr:.4f}"

    print("test_discrete_lstm_actor_per_dim_randomness passed")


if __name__ == "__main__":
    # Test samplers directly
    test_categorical_sampler()
    test_multicategorical_sampler()
    test_per_dimension_randomness()

    # Test full actor integration
    test_discrete_lstm_actor_discrete()
    test_discrete_lstm_actor_multidiscrete()
    test_discrete_lstm_actor_per_dim_randomness()

    print("\nAll tests passed!")
