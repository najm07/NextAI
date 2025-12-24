"""
Energy Computer Module

Computes total energy and gradients from all contributing modules.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from .encoder import ObservationEncoder
from .anchors import MemoryAnchors
from .stc import TransitionSTCs
from .gates import ContextGates
from .intent import IntentModule


@dataclass
class EnergyWeights:
    """Weights for energy components."""
    w_obs: float = 1.0
    w_anchor: float = 0.5
    w_state_stc: float = 0.1
    w_intent: float = 0.05
    w_transition: float = 1.0


class EnergyComputer:
    """
    Computes total energy and gradients for settling.
    
    Total energy at time t:
    E_total(S) = w_obs * E_obs(S) 
               + w_anchor * E_anchor(S)
               + w_state_stc * E_state_stc(S)
               + w_intent * E_intent(S)
    
    For transitions, includes E_tr(S_t, S_{t+1}, u_type).
    """
    
    def __init__(
        self,
        encoder: ObservationEncoder,
        anchors: MemoryAnchors,
        stc: TransitionSTCs,
        gates: ContextGates,
        intent: IntentModule,
        weights: Optional[EnergyWeights] = None
    ):
        """
        Initialize energy computer.
        
        Args:
            encoder: Observation encoder
            anchors: Memory anchors
            stc: Transition STCs
            gates: Context gates
            intent: Intent module
            weights: Energy component weights
        """
        self.encoder = encoder
        self.anchors = anchors
        self.stc = stc
        self.gates = gates
        self.intent = intent
        self.weights = weights or EnergyWeights()
    
    def compute_state_energy(
        self,
        S: np.ndarray,
        obs: Dict[str, np.ndarray],
        u_type: int
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute energy for a single state (not transition).
        
        Args:
            S: Latent state
            obs: Current observation
            u_type: Active intervention type
            
        Returns:
            (total_energy, component_dict) tuple
        """
        components = {}
        
        # Observation consistency energy
        enc_obs = self.encoder.encode(obs)
        E_obs = np.sum((S - enc_obs) ** 2)
        components["E_obs"] = float(E_obs)
        
        # Anchor energy
        g_stc, g_anchor = self.gates.compute_gates(S, u_type)
        E_anchor = self.anchors.compute_energy(S, g_anchor)
        components["E_anchor"] = float(E_anchor)
        
        # State STC energy (minimal in v0.1)
        E_state_stc = 0.0  # Placeholder for future state constraints
        components["E_state_stc"] = float(E_state_stc)
        
        # Intent energy
        E_intent = self.intent.compute_energy(S)
        components["E_intent"] = float(E_intent)
        
        # Weighted sum
        total = (
            self.weights.w_obs * E_obs +
            self.weights.w_anchor * E_anchor +
            self.weights.w_state_stc * E_state_stc +
            self.weights.w_intent * E_intent
        )
        components["E_total"] = float(total)
        
        return float(total), components
    
    def compute_state_gradient(
        self,
        S: np.ndarray,
        obs: Dict[str, np.ndarray],
        u_type: int
    ) -> np.ndarray:
        """
        Compute gradient of state energy w.r.t. S.
        
        Args:
            S: Latent state
            obs: Current observation
            u_type: Active intervention type
            
        Returns:
            Gradient vector (dS,)
        """
        # Observation gradient: ∂E_obs/∂S = 2(S - enc_obs)
        enc_obs = self.encoder.encode(obs)
        grad_obs = 2.0 * (S - enc_obs)
        
        # Anchor gradient
        g_stc, g_anchor = self.gates.compute_gates(S, u_type)
        grad_anchor = self.anchors.compute_gradient(S, g_anchor)
        
        # State STC gradient (none in v0.1)
        grad_state_stc = np.zeros_like(S)
        
        # Intent gradient
        grad_intent = self.intent.compute_gradient(S)
        
        # Weighted sum
        total_grad = (
            self.weights.w_obs * grad_obs +
            self.weights.w_anchor * grad_anchor +
            self.weights.w_state_stc * grad_state_stc +
            self.weights.w_intent * grad_intent
        )
        
        return total_grad.astype(np.float32)
    
    def compute_transition_energy(
        self,
        S_t: np.ndarray,
        S_tp1: np.ndarray,
        u_type: int,
        obs_t: Dict[str, np.ndarray],
        obs_tp1: Dict[str, np.ndarray]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute paired energy for transition.
        
        Args:
            S_t: Current state
            S_tp1: Next state
            u_type: Active intervention type
            obs_t: Current observation
            obs_tp1: Next observation
            
        Returns:
            (total_energy, component_dict) tuple
        """
        components = {}
        
        # State energies at both timesteps
        E_t, comp_t = self.compute_state_energy(S_t, obs_t, u_type)
        E_tp1, comp_tp1 = self.compute_state_energy(S_tp1, obs_tp1, u_type)
        
        components["E_state_t"] = E_t
        components["E_state_tp1"] = E_tp1
        
        # Transition STC energy
        g_stc, _ = self.gates.compute_gates(S_t, u_type)
        E_tr = self.stc.compute_energy(S_t, S_tp1, u_type, g_stc)
        components["E_transition"] = float(E_tr)
        
        # Total
        total = E_t + E_tp1 + self.weights.w_transition * E_tr
        components["E_total"] = float(total)
        
        return float(total), components
    
    def compute_transition_gradient_S_tp1(
        self,
        S_t: np.ndarray,
        S_tp1: np.ndarray,
        u_type: int,
        obs_tp1: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute gradient w.r.t. S_{t+1} for transition settling.
        
        Args:
            S_t: Current state (fixed during settling)
            S_tp1: Next state (being settled)
            u_type: Active intervention type
            obs_tp1: Next observation
            
        Returns:
            Gradient vector (dS,)
        """
        # State gradient at t+1
        grad_state = self.compute_state_gradient(S_tp1, obs_tp1, u_type)
        
        # Transition STC gradient
        g_stc, _ = self.gates.compute_gates(S_t, u_type)
        grad_tr = self.stc.compute_gradient_S_tp1(S_t, S_tp1, u_type, g_stc)
        
        total_grad = grad_state + self.weights.w_transition * grad_tr
        
        return total_grad.astype(np.float32)
    
    def get_gate_values(
        self,
        S: np.ndarray,
        u_type: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get current gate values for logging."""
        return self.gates.compute_gates(S, u_type)
    
    def get_anchor_activations(self, S: np.ndarray) -> np.ndarray:
        """Get anchor activations for logging."""
        return self.anchors.get_activations(S)
    
    def get_stc_violations(
        self,
        S_t: np.ndarray,
        S_tp1: np.ndarray,
        u_type: int
    ) -> np.ndarray:
        """Get STC violation magnitudes for logging."""
        return self.stc.get_violation_magnitudes(S_t, S_tp1, u_type)

