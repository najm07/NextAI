"""
Intent Module

Intent vectors provide goal-directed bias to the energy landscape.
For Track 1 curiosity-only training, intent represents "minimize surprise".
"""

import numpy as np
from typing import Dict


class IntentModule:
    """
    Intent vectors for goal-directed behavior.
    
    Energy contribution: E_intent(S) = -I^T W_I S
    
    This biases the energy landscape toward states aligned with the intent.
    For Track 1, intent is primarily for "minimize surprise" / exploration.
    """
    
    def __init__(
        self,
        latent_dim: int,
        intent_dim: int = 16,
        seed: int = 42
    ):
        """
        Initialize intent module.
        
        Args:
            latent_dim: Latent state dimension (dS)
            intent_dim: Intent vector dimension (dI)
            seed: Random seed
        """
        self.dS = latent_dim
        self.dI = intent_dim
        
        self.rng = np.random.default_rng(seed)
        
        # Intent vector (learnable)
        # Initialized to represent "minimize surprise" implicitly
        self.I = self.rng.normal(0, 0.1, intent_dim).astype(np.float32)
        
        # Projection matrix from intent to latent space
        scale = np.sqrt(2.0 / (intent_dim + latent_dim))
        self.W_I = self.rng.normal(0, scale, (intent_dim, latent_dim)).astype(np.float32)
    
    def compute_energy(self, S: np.ndarray) -> float:
        """
        Compute intent energy contribution.
        
        E_intent(S) = -I^T W_I S
        
        Negative sign means: states aligned with intent have lower energy.
        
        Args:
            S: Latent state (dS,)
            
        Returns:
            Scalar energy contribution
        """
        projected = self.I @ self.W_I  # (dS,)
        energy = -np.dot(projected, S)
        return float(energy)
    
    def compute_gradient(self, S: np.ndarray) -> np.ndarray:
        """
        Compute gradient of E_intent w.r.t. S.
        
        ∂E_intent/∂S = -W_I^T I
        
        Args:
            S: Latent state (dS,)
            
        Returns:
            Gradient vector (dS,)
        """
        grad = -self.W_I.T @ self.I
        return grad.astype(np.float32)
    
    def get_intent_projection(self) -> np.ndarray:
        """
        Get the intent projection into latent space.
        
        Returns:
            Projected intent vector (dS,)
        """
        return (self.I @ self.W_I).astype(np.float32)
    
    def update_intent(
        self,
        surprise_signal: float,
        S: np.ndarray,
        lr: float = 0.0001
    ) -> Dict[str, float]:
        """
        Update intent based on surprise/curiosity signal.
        
        For Track 1: intent should learn to represent states that
        minimize prediction error / maximize learning.
        
        Args:
            surprise_signal: Scalar indicating surprise (prediction error)
            S: Current latent state
            lr: Learning rate
            
        Returns:
            Update statistics
        """
        # Heuristic: update intent to point toward low-surprise states
        # If surprise is high, move intent away from current projected direction
        projected = self.I @ self.W_I
        alignment = np.dot(projected, S) / (np.linalg.norm(projected) * np.linalg.norm(S) + 1e-6)
        
        # If high surprise and aligned: move intent away
        # If low surprise and aligned: reinforce
        signal = -surprise_signal * alignment
        
        # Update I to change projection
        grad_I = signal * (self.W_I @ S)
        self.I += lr * grad_I.astype(np.float32)
        
        # Normalize to prevent explosion
        norm = np.linalg.norm(self.I)
        if norm > 1.0:
            self.I = self.I / norm
        
        return {
            "intent_norm": float(np.linalg.norm(self.I)),
            "intent_alignment": float(alignment),
        }
    
    def update_projection(
        self,
        S_free: np.ndarray,
        S_nudged: np.ndarray,
        lr: float = 0.0001
    ) -> Dict[str, float]:
        """
        Update projection matrix using EP-style rule.
        
        Args:
            S_free: Free-phase equilibrium
            S_nudged: Nudged-phase equilibrium
            lr: Learning rate
            
        Returns:
            Update statistics
        """
        # EP-style: update based on difference
        delta_S = S_nudged - S_free
        
        # Gradient for W_I: want to reduce energy at nudged state relative to free
        # ∂E/∂W_I = -I ⊗ S
        # Update: W_I += lr * I ⊗ delta_S (to lower energy at nudged)
        grad_W = np.outer(self.I, delta_S)
        self.W_I += lr * grad_W.astype(np.float32)
        
        return {
            "W_I_update_norm": float(np.linalg.norm(grad_W)),
        }
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Return trainable parameters."""
        return {
            "I": self.I,
            "W_I": self.W_I,
        }
    
    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """Set parameters from dict."""
        if "I" in params:
            self.I = params["I"]
        if "W_I" in params:
            self.W_I = params["W_I"]

