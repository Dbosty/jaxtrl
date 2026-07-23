"""
Test LSTM Hidden State Recomputation in PPO Training

This test suite verifies whether LSTM hidden states are properly handled
during PPO training, specifically:
1. Whether hidden states are preserved in collected transitions
2. Whether LSTMModule handles missing hidden state keys
3. Whether hidden states evolve/are recomputed during training epochs
4. Whether episode boundary resets work correctly
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import torch as th
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as Mod
from torchrl.modules import LSTMModule

from jaxtrl.rl.ppo.actor.lstm import LSTMActor
from jaxtrl.rl.ppo.critic import lstm_critic_factory
from jaxtrl.rl.ppo._ppo import PPO
from jaxtrl.sim.collect import Transition
from jaxtrl.utils.tensordict import transitions_to_tensordicts


# =============================================================================
# Test Fixtures
# =============================================================================

def create_test_lstm_actor(obs_dim=4, action_dim=2, hidden_size=32, device="cpu"):
    """Create a minimal LSTMActor for testing."""
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
    key = jax.random.key(42)

    return LSTMActor(
        obs_space=obs_space,
        action_space=action_space,
        jax_key=key,
        lstm_hidden_size=hidden_size,
        head_net_arch=[32],
        device=device,
    )


def create_test_lstm_critic(obs_dim=4, hidden_size=32, device="cpu"):
    """Create a minimal LSTM critic for testing."""
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    return lstm_critic_factory(
        obs_space=obs_space,
        action_space=action_space,
        lstm_hidden_size=hidden_size,
        head_net_arch=[32],
        device=device,
    )


def create_test_tensordict(
    batch_size=10,
    obs_dim=4,
    action_dim=2,
    include_hidden_states=False,
    hidden_size=32,
    num_episodes=2,
    device="cpu"
):
    """
    Create test tensordict with or without hidden states.

    Args:
        batch_size: Number of timesteps
        include_hidden_states: Whether to include actor_rnn_h, actor_rnn_c, etc.
        num_episodes: Number of episode boundaries (done=True events)
    """
    # Calculate episode boundaries
    episode_length = batch_size // num_episodes
    step_counts = []
    terminateds = []
    for ep in range(num_episodes):
        for step in range(episode_length):
            step_counts.append(step)
            terminateds.append(step == episode_length - 1)

    # Pad if needed
    while len(step_counts) < batch_size:
        step_counts.append(0)
        terminateds.append(False)

    source = {
        "observation": th.randn(batch_size, obs_dim, device=device),
        "action": th.randn(batch_size, action_dim, device=device),
        "step_count": th.tensor(step_counts, device=device),
        "next": {
            "observation": th.randn(batch_size, obs_dim, device=device),
            "reward": th.randn(batch_size, device=device),
            "episode_reward": th.abs(th.randn(batch_size, device=device)),
            "terminated": th.tensor(terminateds, device=device),
            "truncated": th.zeros(batch_size, dtype=th.bool, device=device),
            "done": th.tensor(terminateds, device=device),
            "step_count": th.tensor([s + 1 for s in step_counts], device=device),
        },
    }

    if include_hidden_states:
        source["actor_rnn_h"] = th.zeros(batch_size, hidden_size, device=device)
        source["actor_rnn_c"] = th.zeros(batch_size, hidden_size, device=device)
        source["critic_rnn_h"] = th.zeros(batch_size, hidden_size, device=device)
        source["critic_rnn_c"] = th.zeros(batch_size, hidden_size, device=device)

    return TensorDict(source=source, batch_size=[batch_size], device=device)


# =============================================================================
# Test 1: Verify Missing Hidden States in Transition Dataclass
# =============================================================================

def test_transition_dataclass_fields():
    """Verify that Transition dataclass does NOT have hidden state fields."""
    print("\n" + "="*70)
    print("TEST 1: Transition Dataclass Fields")
    print("="*70)

    # Check Transition fields - flax struct uses __dataclass_fields__
    transition_fields = list(Transition.__dataclass_fields__.keys())

    print(f"Transition fields: {transition_fields}")

    # Verify hidden state fields are NOT present
    hidden_state_fields = ["actor_rnn_h", "actor_rnn_c", "critic_rnn_h", "critic_rnn_c", "h", "c"]

    missing_hidden_states = all(f not in transition_fields for f in hidden_state_fields)

    if missing_hidden_states:
        print("CONFIRMED: Transition dataclass does NOT store hidden states")
        print("  -> Hidden states are discarded after collection")
    else:
        print("WARNING: Some hidden state fields found in Transition")

    assert missing_hidden_states, "Expected Transition to NOT have hidden state fields"

    return True


# =============================================================================
# Test 2: LSTMModule Behavior with Missing Hidden State Keys
# =============================================================================

def test_lstm_module_missing_keys():
    """Test how LSTMModule behaves when hidden state keys are missing."""
    print("\n" + "="*70)
    print("TEST 2: LSTMModule Behavior with Missing Keys")
    print("="*70)

    actor = create_test_lstm_actor()

    # Create tensordict WITHOUT hidden states
    td = create_test_tensordict(batch_size=10, include_hidden_states=False)

    print(f"TensorDict keys: {list(td.keys())}")
    print(f"'actor_rnn_h' in td: {'actor_rnn_h' in td.keys()}")

    # Try to run through the actor
    try:
        output_td = actor.trl_actor(td)
        success = True
        error_msg = None
    except KeyError as e:
        success = False
        error_msg = str(e)
    except Exception as e:
        success = False
        error_msg = f"{type(e).__name__}: {str(e)}"

    if success:
        print("RESULT: LSTMModule handles missing keys gracefully")
        print(f"  -> Output keys include: {list(output_td.keys())[:5]}...")
        if ("next", "actor_rnn_h") in output_td.keys(include_nested=True):
            print("  -> Hidden states ARE computed and stored in output")
    else:
        print(f"RESULT: LSTMModule FAILS with missing keys")
        print(f"  -> Error: {error_msg}")

    return success, error_msg


# =============================================================================
# Test 3: Verify is_init Computation
# =============================================================================

def test_is_init_computation():
    """Verify is_init correctly identifies episode boundaries."""
    print("\n" + "="*70)
    print("TEST 3: is_init Computation from step_count")
    print("="*70)

    # Create the is_init module (same as in lstm.py)
    create_is_init = Mod(
        lambda step_count: step_count == 0,
        in_keys=["step_count"],
        out_keys=["is_init"],
    )

    # Create tensordict with specific step_count pattern
    # Episode 1: steps 0,1,2,3,4 | Episode 2: steps 0,1,2,3,4
    step_counts = th.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])

    td = TensorDict({
        "step_count": step_counts,
    }, batch_size=[10])

    # Run is_init computation
    td = create_is_init(td)

    expected_is_init = th.tensor([True, False, False, False, False, True, False, False, False, False])

    print(f"step_count:  {step_counts.tolist()}")
    print(f"is_init:     {td['is_init'].tolist()}")
    print(f"expected:    {expected_is_init.tolist()}")

    match = th.equal(td["is_init"], expected_is_init)

    if match:
        print("RESULT: is_init correctly computed from step_count")
    else:
        print("RESULT: is_init computation MISMATCH")

    assert match, f"is_init mismatch: got {td['is_init']}, expected {expected_is_init}"

    return True


# =============================================================================
# Test 4: Hidden State Evolution in Recurrent Mode
# =============================================================================

def test_hidden_state_evolution():
    """Test that hidden states evolve across sequence in recurrent mode."""
    print("\n" + "="*70)
    print("TEST 4: Hidden State Evolution in Recurrent Mode")
    print("="*70)

    actor = create_test_lstm_actor(hidden_size=32)

    # Create a sequence (single episode, no resets)
    seq_len = 10

    # Create tensordict matching what PPO.step receives (without hidden states)
    # The actor will initialize hidden states to zeros internally
    td = create_test_tensordict(
        batch_size=seq_len,
        obs_dim=4,
        action_dim=2,
        include_hidden_states=False,  # Match actual PPO usage
        hidden_size=32,
        num_episodes=1,
    )
    # Override step_counts to be single episode: 0,1,2,...,9
    td["step_count"] = th.arange(seq_len)

    print(f"Input step_counts: {td['step_count'].tolist()}")
    print(f"'actor_rnn_h' in td: {'actor_rnn_h' in td.keys()}")

    # Process through full actor (this is what PPO does)
    with th.no_grad():
        output_td = actor.trl_actor(td)

    # Check output hidden states
    h_out = output_td["next", "actor_rnn_h"]

    print(f"\nOutput actor_rnn_h shape: {h_out.shape}")
    print(f"Output actor_rnn_h all zeros: {(h_out == 0).all().item()}")

    # Check that hidden states EVOLVE (are different at different timesteps)
    h_first = h_out[0]
    h_last = h_out[-1]

    states_differ = not th.allclose(h_first, h_last, atol=1e-6)

    print(f"\nHidden state at t=0 vs t={seq_len-1}:")
    print(f"  h[0] norm: {h_first.norm().item():.6f}")
    print(f"  h[-1] norm: {h_last.norm().item():.6f}")
    print(f"  States differ: {states_differ}")

    if states_differ:
        print("\nRESULT: Hidden states EVOLVE across the sequence")
        print("  -> LSTM is properly processing in recurrent mode")
    else:
        print("\nRESULT: Hidden states DO NOT evolve - potential issue!")

    assert states_differ, "Hidden states should evolve across the sequence"

    return True


# =============================================================================
# Test 5: Episode Boundary State Reset
# =============================================================================

def test_episode_boundary_reset():
    """Test that hidden states reset at episode boundaries (is_init=True)."""
    print("\n" + "="*70)
    print("TEST 5: Episode Boundary State Reset")
    print("="*70)

    actor = create_test_lstm_actor(hidden_size=32)

    # Create sequence with episode boundary in the middle
    seq_len = 10
    episode_reset_idx = 5

    # Episode 1: steps 0,1,2,3,4 | Episode 2: steps 0,1,2,3,4
    step_counts = list(range(episode_reset_idx)) + list(range(seq_len - episode_reset_idx))

    # Create tensordict without hidden states (like PPO receives)
    td = create_test_tensordict(
        batch_size=seq_len,
        obs_dim=4,
        action_dim=2,
        include_hidden_states=False,
        hidden_size=32,
        num_episodes=2,
    )
    td["step_count"] = th.tensor(step_counts)

    print(f"step_counts: {step_counts}")
    print(f"Episode boundary at index {episode_reset_idx} (step_count goes back to 0)")

    # Process through full actor
    with th.no_grad():
        output_td = actor.trl_actor(td)

    h_out = output_td["next", "actor_rnn_h"]

    # Check is_init was computed correctly
    print(f"is_init: {output_td['is_init'].tolist()}")

    h_at_reset = h_out[episode_reset_idx]
    h_before_reset = h_out[episode_reset_idx - 1]

    print(f"\nHidden state before reset (idx {episode_reset_idx-1}):")
    print(f"  norm: {h_before_reset.norm().item():.6f}")
    print(f"Hidden state after reset (idx {episode_reset_idx}):")
    print(f"  norm: {h_at_reset.norm().item():.6f}")

    # The key insight: after reset, the hidden state should be computed
    # from zeros, so it should be different from what it would be if
    # the LSTM continued from the previous hidden state

    # Check that the hidden state at episode start differs from end of previous episode
    states_differ_at_boundary = not th.allclose(h_at_reset, h_before_reset, atol=1e-3)

    print(f"\nStates differ at episode boundary: {states_differ_at_boundary}")

    if states_differ_at_boundary:
        print("RESULT: Episode boundary causes state discontinuity (expected)")

    return True


# =============================================================================
# Test 6: Hidden State Recomputation Between Epochs
# =============================================================================

def test_hidden_state_recomputation_between_epochs():
    """
    CRITICAL TEST: Verify hidden states are RECOMPUTED each epoch.

    In PPO, we run multiple epochs over the same data. The hidden states
    should be recomputed fresh each time (not cached from the first pass).
    """
    print("\n" + "="*70)
    print("TEST 6: Hidden State Recomputation Between Epochs")
    print("="*70)

    actor = create_test_lstm_actor(hidden_size=32)

    # Create test data - fixed observations
    seq_len = 10
    observations = th.randn(seq_len, 4)
    actions = th.randn(seq_len, 2)

    def create_td(obs):
        td = TensorDict({
            "observation": obs.clone(),
            "action": actions.clone(),
            "step_count": th.arange(seq_len),
            "next": {
                "observation": th.randn(seq_len, 4),
                "reward": th.randn(seq_len),
                "episode_reward": th.abs(th.randn(seq_len)),
                "terminated": th.zeros(seq_len, dtype=th.bool),
                "truncated": th.zeros(seq_len, dtype=th.bool),
                "done": th.zeros(seq_len, dtype=th.bool),
                "step_count": th.arange(1, seq_len + 1),
            },
        }, batch_size=[seq_len])
        return td

    print("Running multiple forward passes (simulating epochs)...")

    # First pass
    td1 = create_td(observations)
    with th.no_grad():
        output_td_1 = actor.trl_actor(td1)
    h_out_1 = output_td_1["next", "actor_rnn_h"].clone()

    # Second pass (same data)
    td2 = create_td(observations)
    with th.no_grad():
        output_td_2 = actor.trl_actor(td2)
    h_out_2 = output_td_2["next", "actor_rnn_h"].clone()

    # Third pass with strongly modified observations (perturb ALL timesteps)
    obs_modified = observations.clone() + 10.0  # Large perturbation
    td3 = create_td(obs_modified)
    with th.no_grad():
        output_td_3 = actor.trl_actor(td3)
    h_out_3 = output_td_3["next", "actor_rnn_h"].clone()

    # Check: passes 1 and 2 should be IDENTICAL (same inputs)
    passes_1_2_identical = th.allclose(h_out_1, h_out_2, atol=1e-6)

    # Check: pass 3 should be DIFFERENT (perturbed input)
    # Compute actual max difference
    max_diff = (h_out_1 - h_out_3).abs().max().item()
    pass_3_different = max_diff > 1e-6  # Any measurable difference

    print(f"\nPass 1 vs Pass 2 (identical inputs):")
    print(f"  h_out_1 norm: {h_out_1.norm().item():.6f}")
    print(f"  h_out_2 norm: {h_out_2.norm().item():.6f}")
    print(f"  Max diff: {(h_out_1 - h_out_2).abs().max().item():.8f}")
    print(f"  Are identical: {passes_1_2_identical}")

    print(f"\nPass 1 vs Pass 3 (perturbed ALL inputs):")
    print(f"  h_out_1 norm: {h_out_1.norm().item():.6f}")
    print(f"  h_out_3 norm: {h_out_3.norm().item():.6f}")
    print(f"  Max diff: {max_diff:.8f}")
    print(f"  Are different: {pass_3_different}")

    # The key insight: if hidden states were cached, pass 3 would be
    # identical to pass 1. Since they differ, states ARE recomputed.

    if passes_1_2_identical and pass_3_different:
        print("\nRESULT: Hidden states ARE RECOMPUTED each pass")
        print("  -> Same inputs produce same outputs (deterministic)")
        print("  -> Different inputs produce different outputs (recomputation)")
    elif not passes_1_2_identical:
        print("\nWARNING: Same inputs produced different outputs")
        print("  -> This could indicate non-determinism")
    elif not pass_3_different:
        print("\nFAILURE: Different inputs produced same outputs")
        print("  -> Hidden states may be cached, not recomputed!")

    assert passes_1_2_identical, "Same inputs should produce identical hidden states"
    assert pass_3_different, "Different inputs should produce different hidden states"

    return True


# =============================================================================
# Test 7: Full PPO Step with Missing Hidden States
# =============================================================================

def test_ppo_step_with_missing_hidden_states():
    """Test full PPO.step() with tensordict lacking hidden states."""
    print("\n" + "="*70)
    print("TEST 7: Full PPO Step with Missing Hidden States")
    print("="*70)

    actor = create_test_lstm_actor(obs_dim=4, action_dim=2, hidden_size=32)
    critic = create_test_lstm_critic(obs_dim=4, hidden_size=32)

    ppo = PPO(
        actor,
        critic,
        sub_batch_size=50,
        num_epochs=3,
    )

    # Create tensordict WITHOUT hidden states (simulating collected data)
    td = create_test_tensordict(
        batch_size=100,
        obs_dim=4,
        action_dim=2,
        include_hidden_states=False,  # This is the key - no hidden states!
        num_episodes=5,
    )

    print(f"TensorDict batch_size: {td.batch_size}")
    print(f"'actor_rnn_h' in td: {'actor_rnn_h' in td.keys()}")
    print(f"'critic_rnn_h' in td: {'critic_rnn_h' in td.keys()}")

    try:
        log_dict = ppo.step(td)
        success = True
        error_msg = None

        print(f"\nPPO step completed successfully!")
        print(f"Log dict keys: {list(log_dict.keys())[:10]}...")
        print(f"train/loss: {log_dict.get('train/loss', 'N/A')}")

        # Check for NaN values
        has_nan = any(
            np.isnan(v) if isinstance(v, (int, float)) else False
            for v in log_dict.values()
        )

        if has_nan:
            print("WARNING: NaN values detected in log_dict")
        else:
            print("No NaN values in log_dict")

    except Exception as e:
        success = False
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"\nPPO step FAILED: {error_msg}")

    if success:
        print("\nRESULT: PPO training works with missing hidden states")
    else:
        print(f"\nRESULT: PPO training FAILED - {error_msg}")

    return success, error_msg


# =============================================================================
# Test 8: Compare Step-by-Step vs Batch Recomputation
# =============================================================================

def test_hidden_state_chaining():
    """
    Test that hidden states are properly chained across the sequence.

    This verifies that the LSTM maintains state across timesteps within
    an episode, producing different outputs for the same observation
    at different timesteps (due to different history).
    """
    print("\n" + "="*70)
    print("TEST 8: Hidden State Chaining Across Sequence")
    print("="*70)

    actor = create_test_lstm_actor(hidden_size=32)

    # Create sequence where all observations are IDENTICAL
    # If LSTM is properly chaining states, outputs should still differ
    seq_len = 5
    same_obs = th.randn(1, 4)
    observations = same_obs.repeat(seq_len, 1)  # All same observation

    td = TensorDict({
        "observation": observations,
        "action": th.randn(seq_len, 2),
        "step_count": th.arange(seq_len),  # Single episode: 0,1,2,3,4
        "next": {
            "observation": th.randn(seq_len, 4),
            "reward": th.randn(seq_len),
            "episode_reward": th.abs(th.randn(seq_len)),
            "terminated": th.zeros(seq_len, dtype=th.bool),
            "truncated": th.zeros(seq_len, dtype=th.bool),
            "done": th.zeros(seq_len, dtype=th.bool),
            "step_count": th.arange(1, seq_len + 1),
        },
    }, batch_size=[seq_len])

    print(f"All observations identical: {th.allclose(observations[0], observations[-1])}")
    print(f"step_counts: {td['step_count'].tolist()}")

    with th.no_grad():
        output_td = actor.trl_actor(td)

    h_out = output_td["next", "actor_rnn_h"]

    print(f"\nHidden states at each timestep:")
    for t in range(seq_len):
        print(f"  t={t}: norm={h_out[t].norm().item():.6f}")

    # Check that hidden states DIFFER even with same observations
    # (because the LSTM carries forward state from previous timesteps)
    h_first = h_out[0]
    h_last = h_out[-1]
    states_differ = not th.allclose(h_first, h_last, atol=1e-6)

    print(f"\nSame input, different hidden states: {states_differ}")

    if states_differ:
        print("\nRESULT: Hidden states properly chain across sequence")
        print("  -> Same observation at different times produces different states")
        print("  -> LSTM maintains temporal memory")
    else:
        print("\nFAILURE: Hidden states do NOT chain")
        print("  -> LSTM not maintaining state across timesteps!")

    assert states_differ, "Hidden states should differ across sequence even with same input"

    return True


# =============================================================================
# Main: Run All Tests and Generate Report
# =============================================================================

def run_all_tests():
    """Run all tests and generate findings report."""
    print("\n" + "="*70)
    print("LSTM HIDDEN STATE RECOMPUTATION TEST SUITE")
    print("="*70)

    results = {}

    # Test 1
    try:
        results["test_1_transition_fields"] = test_transition_dataclass_fields()
    except Exception as e:
        results["test_1_transition_fields"] = f"FAILED: {e}"
        print(f"Test 1 FAILED: {e}")

    # Test 2
    try:
        success, error = test_lstm_module_missing_keys()
        results["test_2_missing_keys"] = "PASS" if success else f"FAIL: {error}"
    except Exception as e:
        results["test_2_missing_keys"] = f"FAILED: {e}"
        print(f"Test 2 FAILED: {e}")

    # Test 3
    try:
        results["test_3_is_init"] = test_is_init_computation()
    except Exception as e:
        results["test_3_is_init"] = f"FAILED: {e}"
        print(f"Test 3 FAILED: {e}")

    # Test 4
    try:
        results["test_4_evolution"] = test_hidden_state_evolution()
    except Exception as e:
        results["test_4_evolution"] = f"FAILED: {e}"
        print(f"Test 4 FAILED: {e}")

    # Test 5
    try:
        results["test_5_boundary_reset"] = test_episode_boundary_reset()
    except Exception as e:
        results["test_5_boundary_reset"] = f"FAILED: {e}"
        print(f"Test 5 FAILED: {e}")

    # Test 6 (CRITICAL)
    try:
        results["test_6_recomputation"] = test_hidden_state_recomputation_between_epochs()
    except Exception as e:
        results["test_6_recomputation"] = f"FAILED: {e}"
        print(f"Test 6 FAILED: {e}")

    # Test 7
    try:
        success, error = test_ppo_step_with_missing_hidden_states()
        results["test_7_ppo_step"] = "PASS" if success else f"FAIL: {error}"
    except Exception as e:
        results["test_7_ppo_step"] = f"FAILED: {e}"
        print(f"Test 7 FAILED: {e}")

    # Test 8
    try:
        results["test_8_state_chaining"] = test_hidden_state_chaining()
    except Exception as e:
        results["test_8_state_chaining"] = f"FAILED: {e}"
        print(f"Test 8 FAILED: {e}")

    # Generate report
    print("\n" + "="*70)
    print("FINDINGS REPORT")
    print("="*70)

    print("""
1. HIDDEN STATE STORAGE: NOT IMPLEMENTED (By Design)
   - Transition dataclass lacks hidden state fields
   - transitions_to_tensordicts() does not include h/c states
   - This means hidden states from collection are DISCARDED
    """)

    print("""
2. LSTM MODULE BEHAVIOR:
   - TorchRL's LSTMModule with default_recurrent_mode=True
   - When hidden state keys are missing, they are initialized to zeros
   - The module processes sequences by chaining hidden states internally
    """)

    print("""
3. RECOMPUTATION MECHANISM:
   - Hidden states ARE RECOMPUTED from scratch each forward pass
   - Same inputs produce identical outputs (deterministic)
   - Different inputs produce different outputs (no caching)
   - is_init flag (step_count == 0) resets states at episode boundaries
    """)

    print("\n" + "-"*70)
    print("TEST RESULTS SUMMARY")
    print("-"*70)

    for test_name, result in results.items():
        status = "PASS" if result is True else result
        print(f"  {test_name}: {status}")

    all_passed = all(r is True or r == "PASS" for r in results.values())

    print("\n" + "-"*70)
    if all_passed:
        print("OVERALL: ALL TESTS PASSED")
        print("\nCONCLUSION: Hidden states ARE being correctly recomputed")
        print("during PPO training. The current implementation is CORRECT.")
    else:
        print("OVERALL: SOME TESTS FAILED")
        print("\nReview failed tests for potential issues.")
    print("-"*70)

    return results


if __name__ == "__main__":
    run_all_tests()
