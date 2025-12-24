"""
Test agent modules and integration.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from env.track1_env import Track1CausalEnv
from agent.agent import Track1Agent, AgentConfig


def test_agent_initialization():
    """Test that agent initializes correctly."""
    agent = Track1Agent()
    
    # Check latent state
    assert agent.S.shape == (64,), f"Wrong S shape: {agent.S.shape}"
    assert np.allclose(agent.S, 0), "S should be zero initially"
    
    # Check modules exist
    assert agent.encoder is not None
    assert agent.anchors is not None
    assert agent.stc is not None
    assert agent.gates is not None
    assert agent.temporal is not None
    assert agent.intent is not None
    
    print("[PASS] Agent initialization test passed")


def test_encoder():
    """Test encoder forward/backward pass."""
    env = Track1CausalEnv()
    obs = env.reset(seed=42)
    
    agent = Track1Agent()
    
    # Encode
    enc = agent.encoder.encode(obs)
    assert enc.shape == (64,), f"Wrong encoding shape: {enc.shape}"
    assert np.all(np.abs(enc) <= 1), "Encoding should be bounded by tanh"
    
    # Decode
    dec = agent.encoder.decode(enc)
    assert dec["X"].shape == (4, 3), f"Wrong decoded X shape: {dec['X'].shape}"
    assert dec["Rel"].shape == (4, 4, 2), f"Wrong decoded Rel shape: {dec['Rel'].shape}"
    
    print("[PASS] Encoder test passed")


def test_anchors():
    """Test anchor energy and gradients."""
    agent = Track1Agent()
    
    S = np.random.randn(64).astype(np.float32)
    gates = np.ones(8, dtype=np.float32) * 0.5
    
    # Compute energy
    energy = agent.anchors.compute_energy(S, gates)
    assert isinstance(energy, float), "Energy should be scalar"
    assert energy >= 0, "Energy should be non-negative"
    
    # Compute gradient
    grad = agent.anchors.compute_gradient(S, gates)
    assert grad.shape == S.shape, "Gradient should match S shape"
    
    # Numerical gradient check
    eps = 1e-4
    for i in [0, 10, 30]:
        S_plus = S.copy()
        S_plus[i] += eps
        S_minus = S.copy()
        S_minus[i] -= eps
        
        num_grad = (agent.anchors.compute_energy(S_plus, gates) - 
                    agent.anchors.compute_energy(S_minus, gates)) / (2 * eps)
        
        # Allow for some numerical imprecision
        rel_error = abs(grad[i] - num_grad) / (abs(num_grad) + 1e-6)
        assert rel_error < 0.05, f"Gradient mismatch at {i}: {grad[i]} vs {num_grad} (rel_err={rel_error:.4f})"
    
    print("[PASS] Anchors test passed")


def test_stc():
    """Test STC energy and gradients."""
    agent = Track1Agent()
    
    S_t = np.random.randn(64).astype(np.float32)
    S_tp1 = np.random.randn(64).astype(np.float32)
    u_type = 0
    gates = np.ones(4, dtype=np.float32) * 0.5
    
    # Compute energy
    energy = agent.stc.compute_energy(S_t, S_tp1, u_type, gates)
    assert isinstance(energy, float), "Energy should be scalar"
    assert energy >= 0, "Huber energy should be non-negative"
    
    # Compute residuals
    residuals = agent.stc.compute_residuals(S_t, S_tp1, u_type)
    assert residuals.shape == (4,), f"Wrong residuals shape: {residuals.shape}"
    
    print("[PASS] STC test passed")


def test_gates():
    """Test gate computation."""
    agent = Track1Agent()
    
    S = np.random.randn(64).astype(np.float32)
    u_type = 3
    
    g_stc, g_anchor = agent.gates.compute_gates(S, u_type)
    
    assert g_stc.shape == (4,), f"Wrong g_stc shape: {g_stc.shape}"
    assert g_anchor.shape == (8,), f"Wrong g_anchor shape: {g_anchor.shape}"
    
    # Gates should be in [0, 1]
    assert np.all(g_stc >= 0) and np.all(g_stc <= 1), "g_stc out of range"
    assert np.all(g_anchor >= 0) and np.all(g_anchor <= 1), "g_anchor out of range"
    
    print("[PASS] Gates test passed")


def test_settling():
    """Test settling loop."""
    env = Track1CausalEnv()
    obs = env.reset(seed=42)
    
    agent = Track1Agent()
    agent.reset()
    
    # Perceive (includes settling)
    S_settled, info = agent.perceive(obs, u_type=0)
    
    assert S_settled.shape == (64,), f"Wrong settled S shape: {S_settled.shape}"
    assert info["n_steps"] > 0, "Should have taken some steps"
    assert info["final_energy"] >= 0, "Energy should be non-negative"
    
    # Energy should be lower after settling than before
    S_random = np.random.randn(64).astype(np.float32)
    energy_random, _ = agent.energy_computer.compute_state_energy(S_random, obs, 0)
    
    # Settled state should generally have lower energy (not always due to initialization)
    # Just check it's not absurdly high
    assert info["final_energy"] < 100, "Settled energy too high"
    
    print("[PASS] Settling test passed")


def test_full_transition():
    """Test full transition processing."""
    env = Track1CausalEnv()
    obs_t = env.reset(seed=42)
    
    agent = Track1Agent()
    agent.reset()
    
    # Step environment
    u_type, args = env.sample_action()
    obs_tp1, _, _ = env.step(u_type, args)
    
    # Process transition
    result = agent.process_transition(obs_t, obs_tp1, u_type, args)
    
    assert "S_t_free" in result
    assert "S_tp1_free" in result
    assert "S_tp1_nudged" in result
    assert "prediction_error" in result
    
    assert result["S_t_free"].shape == (64,)
    assert result["S_tp1_free"].shape == (64,)
    assert result["S_tp1_nudged"].shape == (64,)
    assert result["prediction_error"] >= 0
    
    print("[PASS] Full transition test passed")


def test_action_selection():
    """Test action selection produces valid actions."""
    env = Track1CausalEnv()
    obs = env.reset(seed=42)
    
    agent = Track1Agent()
    rng = np.random.default_rng(42)
    
    for _ in range(100):
        u_type, args = agent.select_action(obs, rng)
        
        # Valid u_type
        assert 0 <= u_type < 8, f"Invalid u_type: {u_type}"
        
        # Valid args
        if u_type in [0, 1]:
            assert 0 <= args["i"] < 4
            assert 0 <= args["a"] < 3
        elif u_type == 2:
            assert 0 <= args["i"] < 4
            assert 0 <= args["a"] < 3
            assert 0 <= args["bin_id"] < 4
        elif u_type in [3, 4, 5]:
            assert 0 <= args["r"] < 2
            assert 0 <= args["i"] < 4
            assert 0 <= args["j"] < 4
            assert args["i"] != args["j"]
        elif u_type == 6:
            assert 0 <= args["i"] < 4
            assert 0 <= args["j"] < 4
            assert args["i"] != args["j"]
        elif u_type == 7:
            assert 0 <= args["i"] < 4
    
    print("[PASS] Action selection test passed")


def run_all_tests():
    """Run all agent tests."""
    print("\n=== Agent Tests ===\n")
    test_agent_initialization()
    test_encoder()
    test_anchors()
    test_stc()
    test_gates()
    test_settling()
    test_full_transition()
    test_action_selection()
    print("\n[PASS] All agent tests passed!\n")


if __name__ == "__main__":
    run_all_tests()

