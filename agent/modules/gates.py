"""
Context Gates Module

Gates control the influence of different modules (STCs, Anchors).
Uses multiplicative gating (not content-based).
"""

import numpy as np
from typing import Dict, Tuple


class ContextGates:
    """
    Context gates that modulate module contributions.
    
    Outputs:
        - g_stc[k, ℓ] ∈ [0, 1]: Gates for STC factors
        - g_anchor[m] ∈ [0, 1]: Gates for anchors
    
    Gates are computed based on latent state and intervention context.
    """
    
    def __init__(
        self,
        latent_dim: int,
        n_intervention_types: int = 8,
        stc_factors_per_type: int = 4,
        n_anchors: int = 8,
        hidden_dim: int = 32,
        seed: int = 42
    ):
        """
        Initialize gates.
        
        Args:
            latent_dim: Latent state dimension (dS)
            n_intervention_types: Number of intervention types (K)
            stc_factors_per_type: STC factors per type (L)
            n_anchors: Number of anchors (M)
            hidden_dim: Hidden layer dimension
            seed: Random seed
        """
        self.dS = latent_dim
        self.K = n_intervention_types
        self.L = stc_factors_per_type
        self.M = n_anchors
        self.hidden_dim = hidden_dim
        
        self.rng = np.random.default_rng(seed)
        
        # Input: S (dS) + intervention one-hot (K)
        input_dim = latent_dim + n_intervention_types
        
        # Shared hidden layer
        scale1 = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hidden = self.rng.normal(0, scale1, (input_dim, hidden_dim)).astype(np.float32)
        self.b_hidden = np.zeros(hidden_dim, dtype=np.float32)
        
        # STC gate output: K * L gates total, but we only use L for active intervention
        # For simplicity, output L gates based on context
        scale2 = np.sqrt(2.0 / (hidden_dim + stc_factors_per_type))
        self.W_stc = self.rng.normal(0, scale2, (hidden_dim, stc_factors_per_type)).astype(np.float32)
        self.b_stc = np.zeros(stc_factors_per_type, dtype=np.float32)
        
        # Anchor gate output: M gates
        scale3 = np.sqrt(2.0 / (hidden_dim + n_anchors))
        self.W_anchor = self.rng.normal(0, scale3, (hidden_dim, n_anchors)).astype(np.float32)
        self.b_anchor = np.zeros(n_anchors, dtype=np.float32)
        
        # Initialize biases to produce moderate gate values (0.5-ish)
        self.b_stc += 0.5
        self.b_anchor += 0.5
    
    def compute_gates(
        self,
        S: np.ndarray,
        u_type: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gate activations.
        
        Args:
            S: Latent state (dS,)
            u_type: Active intervention type
            
        Returns:
            (g_stc, g_anchor) tuple
            - g_stc: (L,) gates for STC factors
            - g_anchor: (M,) gates for anchors
        """
        # Build input
        one_hot = np.zeros(self.K, dtype=np.float32)
        one_hot[u_type] = 1.0
        inp = np.concatenate([S, one_hot])
        
        # Hidden layer with ReLU
        h = np.maximum(0, inp @ self.W_hidden + self.b_hidden)
        
        # STC gates with sigmoid
        g_stc_logits = h @ self.W_stc + self.b_stc
        g_stc = 1.0 / (1.0 + np.exp(-np.clip(g_stc_logits, -20, 20)))
        
        # Anchor gates with sigmoid
        g_anchor_logits = h @ self.W_anchor + self.b_anchor
        g_anchor = 1.0 / (1.0 + np.exp(-np.clip(g_anchor_logits, -20, 20)))
        
        return g_stc.astype(np.float32), g_anchor.astype(np.float32)
    
    def compute_gradient(
        self,
        S: np.ndarray,
        u_type: int,
        d_g_stc: np.ndarray,
        d_g_anchor: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient of gate outputs w.r.t. S.
        
        Used when gates influence energy computation.
        
        Args:
            S: Latent state
            u_type: Active intervention type
            d_g_stc: Gradient from STC energy w.r.t. g_stc
            d_g_anchor: Gradient from anchor energy w.r.t. g_anchor
            
        Returns:
            Gradient w.r.t. S (dS,)
        """
        # Forward pass
        one_hot = np.zeros(self.K, dtype=np.float32)
        one_hot[u_type] = 1.0
        inp = np.concatenate([S, one_hot])
        
        z_h = inp @ self.W_hidden + self.b_hidden
        h = np.maximum(0, z_h)
        
        g_stc_logits = h @ self.W_stc + self.b_stc
        g_stc = 1.0 / (1.0 + np.exp(-np.clip(g_stc_logits, -20, 20)))
        
        g_anchor_logits = h @ self.W_anchor + self.b_anchor
        g_anchor = 1.0 / (1.0 + np.exp(-np.clip(g_anchor_logits, -20, 20)))
        
        # Backprop through sigmoid
        d_stc_logits = d_g_stc * g_stc * (1 - g_stc)
        d_anchor_logits = d_g_anchor * g_anchor * (1 - g_anchor)
        
        # Backprop to hidden
        d_h = d_stc_logits @ self.W_stc.T + d_anchor_logits @ self.W_anchor.T
        
        # Backprop through ReLU
        d_z_h = d_h * (z_h > 0)
        
        # Backprop to input
        d_inp = d_z_h @ self.W_hidden.T
        
        # Return only gradient w.r.t. S (first dS dimensions)
        return d_inp[:self.dS].astype(np.float32)
    
    def update_parameters(
        self,
        S_free: np.ndarray,
        S_nudged: np.ndarray,
        u_type: int,
        target_g_stc: np.ndarray,
        target_g_anchor: np.ndarray,
        lr: float = 0.001
    ) -> Dict[str, float]:
        """
        Update gate parameters.
        
        Args:
            S_free: Free-phase state
            S_nudged: Nudged-phase state
            u_type: Active intervention type
            target_g_stc: Target STC gates (based on EP signal)
            target_g_anchor: Target anchor gates
            lr: Learning rate
            
        Returns:
            Update statistics
        """
        # Use nudged state to compute current gates
        g_stc, g_anchor = self.compute_gates(S_nudged, u_type)
        
        # Simple gradient: move toward target
        one_hot = np.zeros(self.K, dtype=np.float32)
        one_hot[u_type] = 1.0
        inp = np.concatenate([S_nudged, one_hot])
        h = np.maximum(0, inp @ self.W_hidden + self.b_hidden)
        
        # STC gate update
        stc_error = target_g_stc - g_stc
        d_stc_logits = stc_error * g_stc * (1 - g_stc)
        self.W_stc += lr * np.outer(h, d_stc_logits)
        self.b_stc += lr * d_stc_logits
        
        # Anchor gate update
        anchor_error = target_g_anchor - g_anchor
        d_anchor_logits = anchor_error * g_anchor * (1 - g_anchor)
        self.W_anchor += lr * np.outer(h, d_anchor_logits)
        self.b_anchor += lr * d_anchor_logits
        
        return {
            "stc_gate_error": float(np.mean(np.abs(stc_error))),
            "anchor_gate_error": float(np.mean(np.abs(anchor_error))),
        }
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Return trainable parameters."""
        return {
            "W_hidden": self.W_hidden,
            "b_hidden": self.b_hidden,
            "W_stc": self.W_stc,
            "b_stc": self.b_stc,
            "W_anchor": self.W_anchor,
            "b_anchor": self.b_anchor,
        }
    
    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """Set parameters from dict."""
        for name, value in params.items():
            setattr(self, name, value)

