"""
Temporal Dynamics Modulators Module

Implements per-dimension time constants/inertia for settling dynamics.
"""

import numpy as np
from typing import Dict


class TemporalModulators:
    """
    Temporal modulators control the speed of settling per dimension.
    
    Update rule: S <- S - η * diag(z) * ∇_S E_total
    
    Where z ∈ (0, 1]^{dS} modulates step size per dimension.
    Dimensions with lower z values change more slowly (higher inertia).
    """
    
    def __init__(
        self,
        latent_dim: int,
        seed: int = 42
    ):
        """
        Initialize temporal modulators.
        
        Args:
            latent_dim: Dimension of latent state (dS)
            seed: Random seed
        """
        self.dS = latent_dim
        self.rng = np.random.default_rng(seed)
        
        # Time constants: initialized to 1 (no modulation)
        # Range: (0, 1] - lower = slower dynamics
        self.z = np.ones(latent_dim, dtype=np.float32)
        
        # Small network to produce z from state (optional, context-dependent)
        # For v0.1, use learned per-dimension constants
        self.use_dynamic_z = False
        
        # If using dynamic z, parameters for small network
        self.W_z = self.rng.normal(0, 0.1, (latent_dim, latent_dim)).astype(np.float32)
        self.b_z = np.zeros(latent_dim, dtype=np.float32)
    
    def get_modulation(self, S: np.ndarray = None) -> np.ndarray:
        """
        Get time constant modulation vector.
        
        Args:
            S: Current latent state (optional, for dynamic z)
            
        Returns:
            Modulation vector z ∈ (0, 1]^{dS}
        """
        if not self.use_dynamic_z or S is None:
            return self.z
        
        # Dynamic: produce z from S
        logits = S @ self.W_z + self.b_z
        z_dynamic = 0.1 + 0.9 * (1.0 / (1.0 + np.exp(-np.clip(logits, -10, 10))))
        
        return z_dynamic.astype(np.float32)
    
    def modulate_gradient(
        self,
        grad: np.ndarray,
        S: np.ndarray = None
    ) -> np.ndarray:
        """
        Apply temporal modulation to gradient.
        
        Args:
            grad: Original gradient (dS,)
            S: Current state (for dynamic z)
            
        Returns:
            Modulated gradient
        """
        z = self.get_modulation(S)
        return (z * grad).astype(np.float32)
    
    def update_parameters(
        self,
        S_history: np.ndarray,
        grad_history: np.ndarray,
        lr: float = 0.001
    ) -> Dict[str, float]:
        """
        Update temporal parameters based on settling behavior.
        
        Heuristic: dimensions that change a lot during settling should have
        moderate z (not too fast, not too slow).
        
        Args:
            S_history: History of S during settling (n_steps, dS)
            grad_history: History of gradients (n_steps, dS)
            lr: Learning rate
            
        Returns:
            Update statistics
        """
        if len(S_history) < 2:
            return {"z_mean": float(np.mean(self.z))}
        
        # Compute variance of each dimension during settling
        S_var = np.var(S_history, axis=0)
        
        # Target: z should be lower where variance is high (slow down fast-changing dims)
        # and higher where variance is low (speed up slow dims)
        target_z = 1.0 / (1.0 + S_var)
        target_z = 0.1 + 0.9 * target_z  # Keep in (0.1, 1]
        
        # Smooth update
        self.z = (1 - lr) * self.z + lr * target_z.astype(np.float32)
        
        return {
            "z_mean": float(np.mean(self.z)),
            "z_std": float(np.std(self.z)),
            "z_min": float(np.min(self.z)),
            "z_max": float(np.max(self.z)),
        }
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Return trainable parameters."""
        params = {"z": self.z}
        if self.use_dynamic_z:
            params["W_z"] = self.W_z
            params["b_z"] = self.b_z
        return params
    
    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """Set parameters from dict."""
        if "z" in params:
            self.z = params["z"]
        if "W_z" in params:
            self.W_z = params["W_z"]
        if "b_z" in params:
            self.b_z = params["b_z"]

