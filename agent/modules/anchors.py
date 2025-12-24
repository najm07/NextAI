"""
Memory Anchors Module

Maintains M anchor prototypes that attract the latent state.
Activation by proximity through distance-based energy.
"""

import numpy as np
from typing import Dict, Tuple


class MemoryAnchors:
    """
    Memory anchors provide stable attractors in latent space.
    
    Energy: E_anchor(S) = Σ_m g_anchor[m] * α_m * ||S - μ_m||^2
    
    Where:
        - μ_m: Anchor prototype vectors
        - α_m: Anchor strengths
        - g_anchor[m]: Gate activation for anchor m
    """
    
    def __init__(
        self,
        latent_dim: int,
        n_anchors: int,
        seed: int = 42
    ):
        """
        Initialize anchors.
        
        Args:
            latent_dim: Dimension of latent state (dS)
            n_anchors: Number of anchor prototypes (M)
            seed: Random seed
        """
        self.dS = latent_dim
        self.M = n_anchors
        
        self.rng = np.random.default_rng(seed)
        
        # Initialize anchor prototypes spread across latent space
        # Using orthogonal-ish initialization
        self.mu = self.rng.normal(0, 0.5, (n_anchors, latent_dim)).astype(np.float32)
        
        # Normalize to unit sphere initially
        norms = np.linalg.norm(self.mu, axis=1, keepdims=True)
        self.mu = self.mu / (norms + 1e-6)
        
        # Anchor strengths (learnable)
        self.alpha = np.ones(n_anchors, dtype=np.float32) * 0.1
    
    def compute_energy(
        self,
        S: np.ndarray,
        gates: np.ndarray
    ) -> float:
        """
        Compute anchor attraction energy.
        
        E_anchor(S) = Σ_m g_anchor[m] * α_m * ||S - μ_m||^2
        
        Args:
            S: Latent state vector (dS,)
            gates: Gate activations for anchors (M,)
            
        Returns:
            Scalar energy value
        """
        # Compute squared distances to all anchors
        diff = S[np.newaxis, :] - self.mu  # (M, dS)
        sq_dist = np.sum(diff ** 2, axis=1)  # (M,)
        
        # Weighted sum
        energy = np.sum(gates * self.alpha * sq_dist)
        
        return float(energy)
    
    def compute_gradient(
        self,
        S: np.ndarray,
        gates: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient of anchor energy w.r.t. S.
        
        ∂E/∂S = 2 * Σ_m g_anchor[m] * α_m * (S - μ_m)
        
        Args:
            S: Latent state vector (dS,)
            gates: Gate activations for anchors (M,)
            
        Returns:
            Gradient vector (dS,)
        """
        diff = S[np.newaxis, :] - self.mu  # (M, dS)
        weights = (gates * self.alpha)[:, np.newaxis]  # (M, 1)
        
        grad = 2.0 * np.sum(weights * diff, axis=0)
        
        return grad.astype(np.float32)
    
    def get_activations(self, S: np.ndarray) -> np.ndarray:
        """
        Compute anchor activations based on proximity.
        
        Higher activation = closer to anchor (inverse distance).
        
        Args:
            S: Latent state vector
            
        Returns:
            Activation levels for each anchor (M,)
        """
        diff = S[np.newaxis, :] - self.mu
        sq_dist = np.sum(diff ** 2, axis=1)
        
        # Softmax-like activation based on negative distance
        activations = np.exp(-sq_dist)
        activations = activations / (np.sum(activations) + 1e-6)
        
        return activations.astype(np.float32)
    
    def get_top_k_anchors(self, S: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get indices and activations of top-k closest anchors.
        
        Args:
            S: Latent state vector
            k: Number of top anchors
            
        Returns:
            (indices, activations) tuple
        """
        activations = self.get_activations(S)
        indices = np.argsort(activations)[::-1][:k]
        
        return indices, activations[indices]
    
    def update_anchors(
        self,
        S_free: np.ndarray,
        S_nudged: np.ndarray,
        gates: np.ndarray,
        lr: float = 0.01
    ) -> Dict[str, float]:
        """
        Update anchor parameters using EP-style rule.
        
        The update moves anchors toward states where they are active.
        
        Args:
            S_free: Free-phase equilibrium state
            S_nudged: Nudged-phase equilibrium state
            gates: Gate activations
            lr: Learning rate
            
        Returns:
            Dict with update statistics
        """
        # Compute activations at both equilibria
        act_free = self.get_activations(S_free)
        act_nudged = self.get_activations(S_nudged)
        
        # Update prototypes toward nudged state weighted by activation difference
        delta_act = act_nudged - act_free
        
        for m in range(self.M):
            if gates[m] > 0.1:  # Only update active anchors
                # Move prototype toward S_nudged if it increased activation
                self.mu[m] += lr * delta_act[m] * (S_nudged - self.mu[m])
        
        # Update strengths based on how often anchors are active
        mean_act = (act_free + act_nudged) / 2
        self.alpha = 0.99 * self.alpha + 0.01 * mean_act * 0.2
        
        return {
            "mu_update_norm": float(np.linalg.norm(lr * delta_act[:, np.newaxis] * (S_nudged - self.mu))),
            "alpha_mean": float(np.mean(self.alpha)),
        }
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Return trainable parameters."""
        return {
            "mu": self.mu,
            "alpha": self.alpha,
        }
    
    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """Set parameters from dict."""
        if "mu" in params:
            self.mu = params["mu"]
        if "alpha" in params:
            self.alpha = params["alpha"]

