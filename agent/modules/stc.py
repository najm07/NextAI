"""
Soft Transition Constraints (STCs) Module

Implements intervention-channel gated transition constraints.
Each intervention type activates specific constraint factors.
"""

import numpy as np
from typing import Dict, List, Tuple


class TransitionSTCs:
    """
    Transition Soft Tensor Constraints.
    
    Energy: E_tr = Σ_{k} 1[u_type==k] * Σ_{ℓ ∈ L_k} g_stc[k,ℓ] * φ(r_{k,ℓ})
    
    Where:
        - k: Intervention type (0-7)
        - L_k: Constraint factors for intervention k
        - r_{k,ℓ} = w_t^T S_t[sub] + w_tp1^T S_{t+1}[sub] + b
        - φ: Huber loss or squared
    
    Each factor uses a sparse subset of latent dimensions.
    """
    
    def __init__(
        self,
        latent_dim: int,
        n_intervention_types: int = 8,
        factors_per_type: int = 4,
        subset_size: int = 8,
        seed: int = 42
    ):
        """
        Initialize STCs.
        
        Args:
            latent_dim: Dimension of latent state (dS)
            n_intervention_types: Number of intervention types (K)
            factors_per_type: Factors per intervention type (L_k)
            subset_size: Dimensions per factor
            seed: Random seed
        """
        self.dS = latent_dim
        self.K = n_intervention_types
        self.L = factors_per_type
        self.subset_size = subset_size
        
        self.rng = np.random.default_rng(seed)
        
        # For each intervention type k, store L_k constraint factors
        # Each factor has: indices for S_t, indices for S_{t+1}, weights, bias
        
        self.factors: List[List[Dict]] = []
        
        for k in range(self.K):
            type_factors = []
            for ell in range(self.L):
                # Sample random subset of dimensions
                idx_t = self.rng.choice(latent_dim, subset_size, replace=False)
                idx_tp1 = self.rng.choice(latent_dim, subset_size, replace=False)
                
                # Initialize weights (small values)
                w_t = self.rng.normal(0, 0.1, subset_size).astype(np.float32)
                w_tp1 = self.rng.normal(0, 0.1, subset_size).astype(np.float32)
                b = np.float32(0.0)
                
                type_factors.append({
                    "idx_t": idx_t,
                    "idx_tp1": idx_tp1,
                    "w_t": w_t,
                    "w_tp1": w_tp1,
                    "b": b,
                })
            self.factors.append(type_factors)
        
        # Huber loss parameter
        self.huber_delta = 1.0
    
    def _huber(self, r: float) -> float:
        """Huber loss function."""
        abs_r = abs(r)
        if abs_r <= self.huber_delta:
            return 0.5 * r ** 2
        else:
            return self.huber_delta * (abs_r - 0.5 * self.huber_delta)
    
    def _huber_grad(self, r: float) -> float:
        """Gradient of Huber loss."""
        if abs(r) <= self.huber_delta:
            return r
        else:
            return self.huber_delta * np.sign(r)
    
    def compute_residuals(
        self,
        S_t: np.ndarray,
        S_tp1: np.ndarray,
        u_type: int
    ) -> np.ndarray:
        """
        Compute residuals for all factors of the active intervention type.
        
        Args:
            S_t: Current latent state
            S_tp1: Next latent state
            u_type: Active intervention type
            
        Returns:
            Array of residuals for each factor (L,)
        """
        residuals = np.zeros(self.L, dtype=np.float32)
        
        for ell, factor in enumerate(self.factors[u_type]):
            r = (
                np.dot(factor["w_t"], S_t[factor["idx_t"]]) +
                np.dot(factor["w_tp1"], S_tp1[factor["idx_tp1"]]) +
                factor["b"]
            )
            residuals[ell] = r
        
        return residuals
    
    def compute_energy(
        self,
        S_t: np.ndarray,
        S_tp1: np.ndarray,
        u_type: int,
        gates: np.ndarray
    ) -> float:
        """
        Compute transition constraint energy.
        
        E_tr = Σ_ℓ g_stc[u_type, ℓ] * φ(r_ℓ)
        
        Args:
            S_t: Current latent state
            S_tp1: Next latent state
            u_type: Active intervention type
            gates: Gate values for this intervention type (L,)
            
        Returns:
            Scalar energy value
        """
        residuals = self.compute_residuals(S_t, S_tp1, u_type)
        
        energy = 0.0
        for ell in range(self.L):
            energy += gates[ell] * self._huber(residuals[ell])
        
        return float(energy)
    
    def compute_gradient_S_tp1(
        self,
        S_t: np.ndarray,
        S_tp1: np.ndarray,
        u_type: int,
        gates: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient of E_tr w.r.t. S_{t+1}.
        
        Used during settling to find next state.
        
        Args:
            S_t: Current latent state
            S_tp1: Next latent state
            u_type: Active intervention type
            gates: Gate values (L,)
            
        Returns:
            Gradient vector (dS,)
        """
        grad = np.zeros(self.dS, dtype=np.float32)
        residuals = self.compute_residuals(S_t, S_tp1, u_type)
        
        for ell, factor in enumerate(self.factors[u_type]):
            if gates[ell] > 1e-6:
                d_phi = self._huber_grad(residuals[ell])
                grad[factor["idx_tp1"]] += gates[ell] * d_phi * factor["w_tp1"]
        
        return grad
    
    def compute_gradient_S_t(
        self,
        S_t: np.ndarray,
        S_tp1: np.ndarray,
        u_type: int,
        gates: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient of E_tr w.r.t. S_t.
        
        Args:
            S_t: Current latent state
            S_tp1: Next latent state
            u_type: Active intervention type
            gates: Gate values (L,)
            
        Returns:
            Gradient vector (dS,)
        """
        grad = np.zeros(self.dS, dtype=np.float32)
        residuals = self.compute_residuals(S_t, S_tp1, u_type)
        
        for ell, factor in enumerate(self.factors[u_type]):
            if gates[ell] > 1e-6:
                d_phi = self._huber_grad(residuals[ell])
                grad[factor["idx_t"]] += gates[ell] * d_phi * factor["w_t"]
        
        return grad
    
    def get_violation_magnitudes(
        self,
        S_t: np.ndarray,
        S_tp1: np.ndarray,
        u_type: int
    ) -> np.ndarray:
        """
        Get magnitude of constraint violations for logging.
        
        Args:
            S_t: Current latent state
            S_tp1: Next latent state
            u_type: Active intervention type
            
        Returns:
            Absolute residuals for each factor
        """
        residuals = self.compute_residuals(S_t, S_tp1, u_type)
        return np.abs(residuals)
    
    def update_parameters(
        self,
        S_t: np.ndarray,
        S_tp1_free: np.ndarray,
        S_tp1_nudged: np.ndarray,
        u_type: int,
        gates: np.ndarray,
        lr: float = 0.001
    ) -> Dict[str, float]:
        """
        Update STC parameters using EP-style rule.
        
        The idea: reduce residuals that differ between free and nudged phases.
        
        Args:
            S_t: Current state
            S_tp1_free: Free-phase equilibrium
            S_tp1_nudged: Nudged-phase equilibrium
            u_type: Active intervention type
            gates: Gate values
            lr: Learning rate
            
        Returns:
            Update statistics
        """
        res_free = self.compute_residuals(S_t, S_tp1_free, u_type)
        res_nudged = self.compute_residuals(S_t, S_tp1_nudged, u_type)
        
        total_update = 0.0
        
        for ell, factor in enumerate(self.factors[u_type]):
            if gates[ell] > 0.1:
                # EP-style: update proportional to difference
                delta = res_nudged[ell] - res_free[ell]
                
                # Update weights to reduce residual in nudged direction
                factor["w_t"] -= lr * delta * S_t[factor["idx_t"]]
                factor["w_tp1"] -= lr * delta * S_tp1_nudged[factor["idx_tp1"]]
                factor["b"] -= lr * delta
                
                total_update += abs(delta)
        
        return {
            "stc_update_magnitude": float(total_update),
            "mean_residual_free": float(np.mean(np.abs(res_free))),
            "mean_residual_nudged": float(np.mean(np.abs(res_nudged))),
        }
    
    def get_parameters(self) -> Dict[str, List]:
        """Return all parameters."""
        return {"factors": self.factors}
    
    def set_parameters(self, params: Dict[str, List]) -> None:
        """Set parameters from dict."""
        if "factors" in params:
            self.factors = params["factors"]

