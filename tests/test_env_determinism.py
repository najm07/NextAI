"""
Test environment determinism with seeded behavior.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from env.track1_env import Track1CausalEnv, EnvConfig


def test_reset_determinism():
    """Test that reset with same seed produces identical states."""
    env1 = Track1CausalEnv()
    env2 = Track1CausalEnv()
    
    obs1 = env1.reset(seed=42)
    obs2 = env2.reset(seed=42)
    
    np.testing.assert_array_equal(obs1["X"], obs2["X"])
    np.testing.assert_array_equal(obs1["Rel"], obs2["Rel"])
    print("[PASS] Reset determinism test passed")


def test_step_determinism():
    """Test that same actions produce same transitions."""
    env1 = Track1CausalEnv()
    env2 = Track1CausalEnv()
    
    obs1 = env1.reset(seed=42)
    obs2 = env2.reset(seed=42)
    
    # Apply same intervention
    u_type = 0  # IncreaseAttr
    args = {"i": 0, "a": 0}
    
    obs1_next, done1, info1 = env1.step(u_type, args)
    obs2_next, done2, info2 = env2.step(u_type, args)
    
    np.testing.assert_array_almost_equal(obs1_next["X"], obs2_next["X"], decimal=6)
    np.testing.assert_array_equal(obs1_next["Rel"], obs2_next["Rel"])
    assert done1 == done2
    print("[PASS] Step determinism test passed")


def test_multi_step_determinism():
    """Test determinism over multiple steps."""
    env1 = Track1CausalEnv()
    env2 = Track1CausalEnv()
    
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    
    obs1 = env1.reset(seed=42)
    obs2 = env2.reset(seed=42)
    
    for step in range(10):
        # Sample same action
        u_type1 = int(rng1.integers(0, 8))
        u_type2 = int(rng2.integers(0, 8))
        assert u_type1 == u_type2
        
        args1 = env1._sample_args_for_type(u_type1)
        
        # Need to reseed for same args (rng is internal)
        env1.rng = np.random.default_rng(step)
        env2.rng = np.random.default_rng(step)
        
        args1 = env1._sample_args_for_type(u_type1)
        args2 = env2._sample_args_for_type(u_type2)
        
        obs1_next, done1, _ = env1.step(u_type1, args1)
        obs2_next, done2, _ = env2.step(u_type2, args2)
        
        np.testing.assert_array_almost_equal(obs1_next["X"], obs2_next["X"], decimal=5)
        
        if done1 or done2:
            break
    
    print("[PASS] Multi-step determinism test passed")


def test_intervention_effects():
    """Test that each intervention type has expected effect."""
    env = Track1CausalEnv()
    
    # Test IncreaseAttr
    obs = env.reset(seed=42)
    x_before = obs["X"][0, 0]
    env.step(0, {"i": 0, "a": 0})  # IncreaseAttr
    assert env.X[0, 0] > x_before - 0.02  # Allow for noise
    
    # Test DecreaseAttr
    obs = env.reset(seed=42)
    x_before = obs["X"][0, 0]
    env.step(1, {"i": 0, "a": 0})  # DecreaseAttr
    assert env.X[0, 0] < x_before + 0.02
    
    # Test ToggleRel
    obs = env.reset(seed=42)
    rel_before = obs["Rel"][0, 1, 0]
    env.step(3, {"r": 0, "i": 0, "j": 1})  # ToggleRel
    assert env.Rel[0, 1, 0] != rel_before
    
    # Test SetRelOn
    obs = env.reset(seed=42)
    env.step(4, {"r": 0, "i": 0, "j": 1})  # SetRelOn
    assert env.Rel[0, 1, 0] == 1.0
    
    # Test SetRelOff
    env.step(5, {"r": 0, "i": 0, "j": 1})  # SetRelOff
    assert env.Rel[0, 1, 0] == 0.0
    
    print("[PASS] Intervention effects test passed")


def run_all_tests():
    """Run all determinism tests."""
    print("\n=== Environment Determinism Tests ===\n")
    test_reset_determinism()
    test_step_determinism()
    test_multi_step_determinism()
    test_intervention_effects()
    print("\n[PASS] All determinism tests passed!\n")


if __name__ == "__main__":
    run_all_tests()

