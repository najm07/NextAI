"""
Test SwapObjects correctness and invariance.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from env.track1_env import Track1CausalEnv


def test_swap_objects_correctness():
    """Test that SwapObjects correctly exchanges object data."""
    env = Track1CausalEnv()
    obs = env.reset(seed=42)
    
    # Record initial state
    X_before = obs["X"].copy()
    Rel_before = obs["Rel"].copy()
    
    # Swap objects 0 and 1
    i, j = 0, 1
    env.step(env.SWAP_OBJECTS, {"i": i, "j": j})
    
    X_after = env.X.copy()
    Rel_after = env.Rel.copy()
    
    # Check attribute swap (accounting for noise)
    # Object 0's attrs should now be close to what object 1 had
    # Object 1's attrs should now be close to what object 0 had
    noise_tolerance = 0.02  # Allow for natural noise
    
    attr_diff_0 = np.abs(X_after[0] - X_before[1])
    attr_diff_1 = np.abs(X_after[1] - X_before[0])
    
    assert np.all(attr_diff_0 < noise_tolerance), f"Object 0 attrs not swapped correctly: {attr_diff_0}"
    assert np.all(attr_diff_1 < noise_tolerance), f"Object 1 attrs not swapped correctly: {attr_diff_1}"
    
    # Check relation swap
    # Rel[0, 2, :] should now equal Rel_before[1, 2, :]
    # Rel[1, 2, :] should now equal Rel_before[0, 2, :]
    for r in range(env.R):
        # Row swap
        np.testing.assert_array_equal(Rel_after[0, 2, r], Rel_before[1, 2, r])
        np.testing.assert_array_equal(Rel_after[1, 2, r], Rel_before[0, 2, r])
        # Column swap
        np.testing.assert_array_equal(Rel_after[2, 0, r], Rel_before[2, 1, r])
        np.testing.assert_array_equal(Rel_after[2, 1, r], Rel_before[2, 0, r])
    
    print("[PASS] SwapObjects correctness test passed")


def test_double_swap_identity():
    """Test that swapping twice returns to (approximately) original state."""
    env = Track1CausalEnv()
    
    # Disable noise and relation effects for this test
    original_noise = env.config.noise_attr
    original_rel_effects = env.config.enable_relation_effects
    env.config.noise_attr = 0.0
    env.config.enable_relation_effects = False
    
    obs = env.reset(seed=42)
    X_original = obs["X"].copy()
    Rel_original = obs["Rel"].copy()
    
    # Swap 0 and 1
    env.step(env.SWAP_OBJECTS, {"i": 0, "j": 1})
    
    # Swap again
    env.step(env.SWAP_OBJECTS, {"i": 0, "j": 1})
    
    # Should be back to original (with very small tolerance for float errors)
    np.testing.assert_array_almost_equal(env.X, X_original, decimal=5)
    np.testing.assert_array_equal(env.Rel, Rel_original)
    
    # Restore settings
    env.config.noise_attr = original_noise
    env.config.enable_relation_effects = original_rel_effects
    
    print("[PASS] Double swap identity test passed")


def test_swap_preserves_properties():
    """Test that swap preserves aggregate properties."""
    env = Track1CausalEnv()
    
    # Disable noise and relation effects
    env.config.noise_attr = 0.0
    env.config.enable_relation_effects = False
    
    obs = env.reset(seed=42)
    
    # Compute aggregate properties before
    total_attrs_before = np.sum(obs["X"])
    total_rels_before = np.sum(obs["Rel"])
    attr_mean_before = np.mean(obs["X"], axis=0)  # Mean per attribute across objects
    
    # Swap
    env.step(env.SWAP_OBJECTS, {"i": 0, "j": 2})
    
    # Compute after
    total_attrs_after = np.sum(env.X)
    total_rels_after = np.sum(env.Rel)
    attr_mean_after = np.mean(env.X, axis=0)
    
    # Totals should be preserved
    np.testing.assert_almost_equal(total_attrs_before, total_attrs_after, decimal=5)
    np.testing.assert_equal(total_rels_before, total_rels_after)
    
    # Mean per attribute should be preserved
    np.testing.assert_array_almost_equal(attr_mean_before, attr_mean_after, decimal=5)
    
    print("[PASS] Swap preserves properties test passed")


def test_swap_different_pairs():
    """Test swapping different pairs of objects."""
    env = Track1CausalEnv()
    
    # Disable noise and relation effects
    env.config.noise_attr = 0.0
    env.config.enable_relation_effects = False
    
    for i in range(env.N):
        for j in range(env.N):
            if i != j:
                obs = env.reset(seed=42)
                X_before = obs["X"].copy()
                Rel_before = obs["Rel"].copy()
                
                env.step(env.SWAP_OBJECTS, {"i": i, "j": j})
                
                # Verify row swap in X
                np.testing.assert_array_almost_equal(env.X[i], X_before[j], decimal=5)
                np.testing.assert_array_almost_equal(env.X[j], X_before[i], decimal=5)
                
                # Other rows unchanged
                for k in range(env.N):
                    if k != i and k != j:
                        np.testing.assert_array_almost_equal(env.X[k], X_before[k], decimal=5)
    
    print("[PASS] Swap different pairs test passed")


def test_swap_self_rejection():
    """Test that swapping object with itself is rejected."""
    env = Track1CausalEnv()
    env.reset(seed=42)
    
    try:
        env.step(env.SWAP_OBJECTS, {"i": 0, "j": 0})
        assert False, "Should have raised assertion error"
    except AssertionError as e:
        assert "itself" in str(e).lower() or "swap" in str(e).lower() or "i != j" in str(e).lower() or "cannot" in str(e).lower()
    
    print("[PASS] Swap self rejection test passed")


def run_all_tests():
    """Run all swap invariance tests."""
    print("\n=== SwapObjects Invariance Tests ===\n")
    test_swap_objects_correctness()
    test_double_swap_identity()
    test_swap_preserves_properties()
    test_swap_different_pairs()
    test_swap_self_rejection()
    print("\n[PASS] All swap invariance tests passed!\n")


if __name__ == "__main__":
    run_all_tests()

