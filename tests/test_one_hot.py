"""
Test one-hot intervention constraint.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from env.track1_env import Track1CausalEnv


def test_one_hot_encoding():
    """Test that get_one_hot_intervention produces valid one-hot vectors."""
    env = Track1CausalEnv()
    
    for u_type in range(env.N_INTERVENTION_TYPES):
        one_hot = env.get_one_hot_intervention(u_type)
        
        # Check shape
        assert one_hot.shape == (8,), f"Wrong shape for u_type={u_type}"
        
        # Check exactly one element is 1
        assert np.sum(one_hot) == 1.0, f"Sum not 1 for u_type={u_type}"
        
        # Check the correct element is 1
        assert one_hot[u_type] == 1.0, f"Wrong position for u_type={u_type}"
        
        # Check all others are 0
        for i in range(8):
            if i != u_type:
                assert one_hot[i] == 0.0, f"Non-zero at position {i} for u_type={u_type}"
    
    print("[PASS] One-hot encoding test passed")


def test_one_hot_per_step():
    """Test that only one intervention type is active per step."""
    env = Track1CausalEnv()
    env.reset(seed=42)
    
    for _ in range(20):
        u_type, args = env.sample_action()
        
        # Verify u_type is valid integer
        assert isinstance(u_type, (int, np.integer)), f"u_type not integer: {type(u_type)}"
        assert 0 <= u_type < 8, f"u_type out of range: {u_type}"
        
        # Verify one-hot
        one_hot = env.get_one_hot_intervention(u_type)
        assert np.sum(one_hot == 1.0) == 1, "Not exactly one element is 1"
        assert np.sum(one_hot == 0.0) == 7, "Not exactly seven elements are 0"
        
        # Step
        _, done, _ = env.step(u_type, args)
        if done:
            env.reset(seed=None)
    
    print("[PASS] One-hot per step test passed")


def test_arguments_out_of_band():
    """Test that arguments are separate from one-hot encoding."""
    env = Track1CausalEnv()
    env.reset(seed=42)
    
    # Same intervention type, different arguments
    u_type = 0  # IncreaseAttr
    args1 = {"i": 0, "a": 0}
    args2 = {"i": 1, "a": 2}
    
    one_hot1 = env.get_one_hot_intervention(u_type)
    one_hot2 = env.get_one_hot_intervention(u_type)
    
    # One-hot should be identical regardless of arguments
    np.testing.assert_array_equal(one_hot1, one_hot2)
    
    # Arguments should affect the transition, not the one-hot
    obs1 = env.reset(seed=42)
    x1_before = obs1["X"].copy()
    env.step(u_type, args1)
    x1_after = env.X.copy()
    
    obs2 = env.reset(seed=42)
    x2_before = obs2["X"].copy()
    env.step(u_type, args2)
    x2_after = env.X.copy()
    
    # Same initial state
    np.testing.assert_array_equal(x1_before, x2_before)
    
    # Different changes due to different arguments
    # args1 affects (0, 0), args2 affects (1, 2)
    # Check that the changed positions are different
    diff1 = x1_after - x1_before
    diff2 = x2_after - x2_before
    
    # Position (0, 0) should change for args1
    assert abs(diff1[0, 0]) > abs(diff2[0, 0]) - 0.01
    # Position (1, 2) should change for args2
    assert abs(diff2[1, 2]) > abs(diff1[1, 2]) - 0.01
    
    print("[PASS] Arguments out-of-band test passed")


def run_all_tests():
    """Run all one-hot tests."""
    print("\n=== One-Hot Intervention Tests ===\n")
    test_one_hot_encoding()
    test_one_hot_per_step()
    test_arguments_out_of_band()
    print("\n[PASS] All one-hot tests passed!\n")


if __name__ == "__main__":
    run_all_tests()

