"""
Observation Encoder Module

Encodes observations (X, Rel) into latent space for state clamping.
"""

import numpy as np
from typing import Dict, Optional


class ObservationEncoder:
    """
    Encodes environment observations into latent state space.
    
    Maps (X, Rel, args_summary) -> enc_obs ∈ R^{dS}
    Used for observation consistency energy E_obs(S) = ||S - enc_obs||^2
    """
    
    def __init__(
        self,
        n_objects: int,
        n_attributes: int,
        n_relations: int,
        latent_dim: int,
        hidden_dim: int = 128,
        seed: int = 42
    ):
        """
        Initialize encoder.
        
        Args:
            n_objects: Number of objects (N)
            n_attributes: Attributes per object (A)
            n_relations: Relation types (R)
            latent_dim: Output dimension (dS)
            hidden_dim: Hidden layer dimension
            seed: Random seed
        """
        self.N = n_objects
        self.A = n_attributes
        self.R = n_relations
        self.dS = latent_dim
        self.hidden_dim = hidden_dim
        
        self.rng = np.random.default_rng(seed)
        
        # Input dimension: flattened X + flattened Rel
        self.input_dim = n_objects * n_attributes + n_objects * n_objects * n_relations
        
        # Two-layer MLP: input -> hidden -> output
        # Xavier initialization
        scale1 = np.sqrt(2.0 / (self.input_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + latent_dim))
        
        self.W1 = self.rng.normal(0, scale1, (self.input_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = self.rng.normal(0, scale2, (hidden_dim, latent_dim)).astype(np.float32)
        self.b2 = np.zeros(latent_dim, dtype=np.float32)
        
        # Decoder for prediction (symmetric structure)
        self.W_dec1 = self.rng.normal(0, scale2, (latent_dim, hidden_dim)).astype(np.float32)
        self.b_dec1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W_dec2 = self.rng.normal(0, scale1, (hidden_dim, self.input_dim)).astype(np.float32)
        self.b_dec2 = np.zeros(self.input_dim, dtype=np.float32)
    
    def encode(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Encode observation to latent space.
        
        Args:
            obs: Observation dict with 'X' and 'Rel'
            
        Returns:
            Encoded observation in R^{dS}
        """
        x_flat = obs["X"].flatten()
        rel_flat = obs["Rel"].flatten()
        inp = np.concatenate([x_flat, rel_flat])
        
        # Forward pass with ReLU activation
        h = np.maximum(0, inp @ self.W1 + self.b1)  # ReLU
        out = np.tanh(h @ self.W2 + self.b2)  # Bounded output
        
        return out.astype(np.float32)
    
    def decode(self, z: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Decode latent state back to observation space.
        
        Args:
            z: Latent state in R^{dS}
            
        Returns:
            Reconstructed observation dict
        """
        h = np.maximum(0, z @ self.W_dec1 + self.b_dec1)
        logits = h @ self.W_dec2 + self.b_dec2
        out = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))  # sigmoid
        
        # Split into X and Rel
        x_size = self.N * self.A
        X_rec = out[:x_size].reshape(self.N, self.A).astype(np.float32)
        Rel_rec = out[x_size:].reshape(self.N, self.N, self.R).astype(np.float32)
        
        return {"X": X_rec, "Rel": Rel_rec, "mask": None}
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Return all trainable parameters."""
        return {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "W_dec1": self.W_dec1,
            "b_dec1": self.b_dec1,
            "W_dec2": self.W_dec2,
            "b_dec2": self.b_dec2,
        }
    
    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """Set parameters from dict."""
        for name, value in params.items():
            setattr(self, name, value)
    
    def compute_gradients(
        self,
        obs: Dict[str, np.ndarray],
        S: np.ndarray,
        weight: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Compute gradients for encoder parameters given E_obs = ||S - enc_obs||^2.
        
        Uses backpropagation through the encoder.
        
        Args:
            obs: Current observation
            S: Current latent state
            weight: Weight for E_obs
            
        Returns:
            Gradients dict
        """
        # Forward pass with caching
        x_flat = obs["X"].flatten()
        rel_flat = obs["Rel"].flatten()
        inp = np.concatenate([x_flat, rel_flat])
        
        z1 = inp @ self.W1 + self.b1
        h1 = np.maximum(0, z1)  # ReLU
        z2 = h1 @ self.W2 + self.b2
        enc_obs = np.tanh(z2)
        
        # Gradient of E_obs = weight * ||S - enc_obs||^2 w.r.t. enc_obs
        # dE/d(enc_obs) = -2 * weight * (S - enc_obs)
        diff = S - enc_obs
        d_enc = -2.0 * weight * diff
        
        # Backprop through tanh
        d_z2 = d_enc * (1 - enc_obs ** 2)
        
        # Gradients for W2, b2
        grad_W2 = np.outer(h1, d_z2)
        grad_b2 = d_z2
        
        # Backprop through h1
        d_h1 = d_z2 @ self.W2.T
        
        # Backprop through ReLU
        d_z1 = d_h1 * (z1 > 0)
        
        # Gradients for W1, b1
        grad_W1 = np.outer(inp, d_z1)
        grad_b1 = d_z1
        
        return {
            "W1": grad_W1.astype(np.float32),
            "b1": grad_b1.astype(np.float32),
            "W2": grad_W2.astype(np.float32),
            "b2": grad_b2.astype(np.float32),
        }

