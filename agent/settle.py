"""
Settling Loop Module

Implements energy-based settling (iterative relaxation to low-energy fixed point).
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from .modules.energy import EnergyComputer
from .modules.temporal import TemporalModulators


@dataclass
class SettleConfig:
    """Configuration for settling."""
    n_steps: int = 20
    learning_rate: float = 0.1
    tolerance: float = 1e-4
    momentum: float = 0.9
    clip_grad: float = 10.0


class SettlingLoop:
    """
    Energy-based settling loop.
    
    Runs gradient descent on S to minimize total energy,
    modulated by temporal dynamics.
    """
    
    def __init__(
        self,
        energy_computer: EnergyComputer,
        temporal: TemporalModulators,
        config: Optional[SettleConfig] = None
    ):
        """
        Initialize settling loop.
        
        Args:
            energy_computer: Energy computation module
            temporal: Temporal modulators
            config: Settling configuration
        """
        self.energy = energy_computer
        self.temporal = temporal
        self.config = config or SettleConfig()
    
    def settle_state(
        self,
        S_init: np.ndarray,
        obs: Dict[str, np.ndarray],
        u_type: int,
        return_history: bool = False
    ) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Settle state S to minimize state energy.
        
        Args:
            S_init: Initial state
            obs: Current observation
            u_type: Active intervention type
            return_history: Whether to return full history
            
        Returns:
            (S_final, info) tuple
        """
        S = S_init.copy()
        velocity = np.zeros_like(S)
        
        history = {"S": [], "energy": [], "grad_norm": []}
        
        for step in range(self.config.n_steps):
            # Compute energy and gradient
            energy, components = self.energy.compute_state_energy(S, obs, u_type)
            grad = self.energy.compute_state_gradient(S, obs, u_type)
            
            # Clip gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > self.config.clip_grad:
                grad = grad * (self.config.clip_grad / grad_norm)
                grad_norm = self.config.clip_grad
            
            # Apply temporal modulation
            grad_modulated = self.temporal.modulate_gradient(grad, S)
            
            # Momentum update
            velocity = self.config.momentum * velocity - self.config.learning_rate * grad_modulated
            S = S + velocity
            
            # Record history
            if return_history:
                history["S"].append(S.copy())
                history["energy"].append(energy)
                history["grad_norm"].append(float(grad_norm))
            
            # Check convergence
            if grad_norm < self.config.tolerance:
                break
        
        info = {
            "n_steps": step + 1,
            "final_energy": float(energy),
            "final_grad_norm": float(grad_norm),
            "converged": grad_norm < self.config.tolerance,
            "energy_components": components,
        }
        
        if return_history:
            info["history"] = {
                "S": np.array(history["S"]),
                "energy": np.array(history["energy"]),
                "grad_norm": np.array(history["grad_norm"]),
            }
        
        return S.astype(np.float32), info
    
    def settle_transition(
        self,
        S_t: np.ndarray,
        S_tp1_init: np.ndarray,
        u_type: int,
        obs_tp1: Dict[str, np.ndarray],
        return_history: bool = False
    ) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Settle S_{t+1} given fixed S_t for transition.
        
        Args:
            S_t: Current state (fixed)
            S_tp1_init: Initial next state
            u_type: Active intervention type
            obs_tp1: Next observation
            return_history: Whether to return full history
            
        Returns:
            (S_tp1_final, info) tuple
        """
        S_tp1 = S_tp1_init.copy()
        velocity = np.zeros_like(S_tp1)
        
        history = {"S": [], "energy": [], "grad_norm": []}
        
        for step in range(self.config.n_steps):
            # Compute gradient w.r.t. S_{t+1}
            grad = self.energy.compute_transition_gradient_S_tp1(
                S_t, S_tp1, u_type, obs_tp1
            )
            
            # Clip gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > self.config.clip_grad:
                grad = grad * (self.config.clip_grad / grad_norm)
                grad_norm = self.config.clip_grad
            
            # Apply temporal modulation
            grad_modulated = self.temporal.modulate_gradient(grad, S_tp1)
            
            # Momentum update
            velocity = self.config.momentum * velocity - self.config.learning_rate * grad_modulated
            S_tp1 = S_tp1 + velocity
            
            # Record history
            if return_history:
                history["S"].append(S_tp1.copy())
                history["grad_norm"].append(float(grad_norm))
            
            # Check convergence
            if grad_norm < self.config.tolerance:
                break
        
        # Compute final energy
        _, obs_t_dummy = {}, {}  # Not needed for final energy
        energy, components = self.energy.compute_state_energy(S_tp1, obs_tp1, u_type)
        
        info = {
            "n_steps": step + 1,
            "final_energy": float(energy),
            "final_grad_norm": float(grad_norm),
            "converged": grad_norm < self.config.tolerance,
        }
        
        if return_history:
            info["history"] = {
                "S": np.array(history["S"]),
                "grad_norm": np.array(history["grad_norm"]),
            }
        
        return S_tp1.astype(np.float32), info
    
    def settle_with_nudge(
        self,
        S_t: np.ndarray,
        S_tp1_init: np.ndarray,
        u_type: int,
        obs_tp1: Dict[str, np.ndarray],
        target_enc: np.ndarray,
        nudge_strength: float = 0.5
    ) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Settle with nudge toward target encoding (for EP nudged phase).
        
        Args:
            S_t: Current state
            S_tp1_init: Initial next state
            u_type: Active intervention type
            obs_tp1: Next observation
            target_enc: Target encoding to nudge toward
            nudge_strength: Strength of nudge (β)
            
        Returns:
            (S_tp1_nudged, info) tuple
        """
        S_tp1 = S_tp1_init.copy()
        velocity = np.zeros_like(S_tp1)
        
        for step in range(self.config.n_steps):
            # Standard transition gradient
            grad = self.energy.compute_transition_gradient_S_tp1(
                S_t, S_tp1, u_type, obs_tp1
            )
            
            # Add nudge gradient: encourage S_{t+1} toward target_enc
            nudge_grad = 2.0 * nudge_strength * (S_tp1 - target_enc)
            grad = grad + nudge_grad
            
            # Clip
            grad_norm = np.linalg.norm(grad)
            if grad_norm > self.config.clip_grad:
                grad = grad * (self.config.clip_grad / grad_norm)
            
            # Modulate and update
            grad_modulated = self.temporal.modulate_gradient(grad, S_tp1)
            velocity = self.config.momentum * velocity - self.config.learning_rate * grad_modulated
            S_tp1 = S_tp1 + velocity
            
            if np.linalg.norm(grad) < self.config.tolerance:
                break
        
        info = {
            "n_steps": step + 1,
            "final_grad_norm": float(grad_norm),
        }
        
        return S_tp1.astype(np.float32), info

