"""
Track 1 Agent

Main agent class that integrates all modules for energy-based cognitive processing.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

from .modules.encoder import ObservationEncoder
from .modules.anchors import MemoryAnchors
from .modules.stc import TransitionSTCs
from .modules.gates import ContextGates
from .modules.temporal import TemporalModulators
from .modules.intent import IntentModule
from .modules.energy import EnergyComputer, EnergyWeights
from .settle import SettlingLoop, SettleConfig


@dataclass
class AgentConfig:
    """Agent configuration."""
    # Environment
    n_objects: int = 4
    n_attributes: int = 3
    n_relations: int = 2
    
    # Architecture
    latent_dim: int = 64
    intent_dim: int = 16
    n_anchors: int = 8
    n_intervention_types: int = 8
    stc_factors_per_type: int = 4
    stc_subset_size: int = 8
    encoder_hidden: int = 128
    gate_hidden: int = 32
    
    # Settling
    n_settle_steps: int = 20
    settle_lr: float = 0.1
    settle_tolerance: float = 1e-4
    
    # Energy weights
    w_obs: float = 1.0
    w_anchor: float = 0.5
    w_state_stc: float = 0.1
    w_intent: float = 0.05
    w_transition: float = 1.0
    
    # Learning
    nudge_strength: float = 0.5
    
    # Seed
    seed: int = 42


class Track1Agent:
    """
    Track 1 Agent with energy-based settling and EP-style learning.
    
    Maintains:
        - Latent state S that evolves through settling
        - Modules: encoder, anchors, STCs, gates, temporal, intent
        - Energy computer for total energy and gradients
        - Settling loop for finding low-energy states
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize agent with all modules.
        
        Args:
            config: Agent configuration
        """
        self.config = config or AgentConfig()
        self.dS = self.config.latent_dim
        
        # Initialize latent state
        self.S = np.zeros(self.dS, dtype=np.float32)
        
        # Initialize modules
        self._init_modules()
        
        # Build energy computer and settling loop
        self._build_energy_system()
        
        # Episode state
        self.step_count = 0
        self.episode_count = 0
        
        # Logging buffers
        self.logs = {
            "energy_components": [],
            "gradient_norms": [],
            "gate_values": [],
            "anchor_activations": [],
            "stc_violations": [],
        }
    
    def _init_modules(self) -> None:
        """Initialize all neural modules."""
        cfg = self.config
        
        self.encoder = ObservationEncoder(
            n_objects=cfg.n_objects,
            n_attributes=cfg.n_attributes,
            n_relations=cfg.n_relations,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.encoder_hidden,
            seed=cfg.seed,
        )
        
        self.anchors = MemoryAnchors(
            latent_dim=cfg.latent_dim,
            n_anchors=cfg.n_anchors,
            seed=cfg.seed + 1,
        )
        
        self.stc = TransitionSTCs(
            latent_dim=cfg.latent_dim,
            n_intervention_types=cfg.n_intervention_types,
            factors_per_type=cfg.stc_factors_per_type,
            subset_size=cfg.stc_subset_size,
            seed=cfg.seed + 2,
        )
        
        self.gates = ContextGates(
            latent_dim=cfg.latent_dim,
            n_intervention_types=cfg.n_intervention_types,
            stc_factors_per_type=cfg.stc_factors_per_type,
            n_anchors=cfg.n_anchors,
            hidden_dim=cfg.gate_hidden,
            seed=cfg.seed + 3,
        )
        
        self.temporal = TemporalModulators(
            latent_dim=cfg.latent_dim,
            seed=cfg.seed + 4,
        )
        
        self.intent = IntentModule(
            latent_dim=cfg.latent_dim,
            intent_dim=cfg.intent_dim,
            seed=cfg.seed + 5,
        )
    
    def _build_energy_system(self) -> None:
        """Build energy computer and settling loop."""
        cfg = self.config
        
        weights = EnergyWeights(
            w_obs=cfg.w_obs,
            w_anchor=cfg.w_anchor,
            w_state_stc=cfg.w_state_stc,
            w_intent=cfg.w_intent,
            w_transition=cfg.w_transition,
        )
        
        self.energy_computer = EnergyComputer(
            encoder=self.encoder,
            anchors=self.anchors,
            stc=self.stc,
            gates=self.gates,
            intent=self.intent,
            weights=weights,
        )
        
        settle_config = SettleConfig(
            n_steps=cfg.n_settle_steps,
            learning_rate=cfg.settle_lr,
            tolerance=cfg.settle_tolerance,
        )
        
        self.settling = SettlingLoop(
            energy_computer=self.energy_computer,
            temporal=self.temporal,
            config=settle_config,
        )
    
    def reset(self) -> None:
        """Reset agent state for new episode."""
        # Reset latent state (could use learned init in future)
        self.S = np.zeros(self.dS, dtype=np.float32)
        self.step_count = 0
        self.episode_count += 1
        
        # Clear per-episode logs
        self.logs = {k: [] for k in self.logs}
    
    def perceive(
        self,
        obs: Dict[str, np.ndarray],
        u_type: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process observation through settling.
        
        Args:
            obs: Environment observation
            u_type: Active intervention type
            
        Returns:
            (S_settled, info) tuple
        """
        # Encode observation
        enc_obs = self.encoder.encode(obs)
        
        # Initialize from current state biased toward encoding
        S_init = 0.7 * self.S + 0.3 * enc_obs
        
        # Settle to low-energy state
        S_settled, settle_info = self.settling.settle_state(
            S_init, obs, u_type, return_history=False
        )
        
        # Update state
        self.S = S_settled
        
        # Log
        self._log_step(obs, u_type, settle_info)
        
        return S_settled, settle_info
    
    def predict_next(
        self,
        obs: Dict[str, np.ndarray],
        u_type: int,
        args: Dict[str, int]
    ) -> np.ndarray:
        """
        Predict encoding of next observation given intervention.
        
        Simple prediction: settle a hypothetical next state.
        
        Args:
            obs: Current observation
            u_type: Intervention type
            args: Intervention arguments
            
        Returns:
            Predicted encoding of next state
        """
        # Current encoded state
        enc_current = self.encoder.encode(obs)
        
        # Initialize prediction from current state
        S_pred_init = self.S.copy()
        
        # Settle as if we had the next observation (but we don't yet)
        # Use current obs as proxy - the STCs should handle transition
        S_pred, _ = self.settling.settle_transition(
            self.S, S_pred_init, u_type, obs
        )
        
        return S_pred
    
    def process_transition(
        self,
        obs_t: Dict[str, np.ndarray],
        obs_tp1: Dict[str, np.ndarray],
        u_type: int,
        args: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Process a full transition for learning (EP two-phase).
        
        Args:
            obs_t: Current observation
            obs_tp1: Next observation
            u_type: Intervention type
            args: Intervention arguments
            
        Returns:
            Dict with free and nudged equilibria and statistics
        """
        # Encode observations
        enc_t = self.encoder.encode(obs_t)
        enc_tp1 = self.encoder.encode(obs_tp1)
        
        # === Free Phase ===
        # Settle S_t from current state
        S_t_free, _ = self.settling.settle_state(
            self.S, obs_t, u_type
        )
        
        # Settle S_{t+1} given S_t (free, no nudge)
        S_tp1_init = S_t_free.copy()  # Initialize from S_t
        S_tp1_free, _ = self.settling.settle_transition(
            S_t_free, S_tp1_init, u_type, obs_tp1
        )
        
        # === Nudged Phase ===
        # Nudge toward actual next encoding
        S_tp1_nudged, _ = self.settling.settle_with_nudge(
            S_t_free, S_tp1_init, u_type, obs_tp1,
            target_enc=enc_tp1,
            nudge_strength=self.config.nudge_strength
        )
        
        # Compute prediction error (surprise)
        pred_error = np.mean((S_tp1_free - enc_tp1) ** 2)
        
        # Update internal state
        self.S = S_tp1_nudged
        self.step_count += 1
        
        return {
            "S_t_free": S_t_free,
            "S_tp1_free": S_tp1_free,
            "S_tp1_nudged": S_tp1_nudged,
            "enc_t": enc_t,
            "enc_tp1": enc_tp1,
            "prediction_error": float(pred_error),
        }
    
    def select_action(
        self,
        obs: Dict[str, np.ndarray],
        rng: np.random.Generator
    ) -> Tuple[int, Dict[str, int]]:
        """
        Select intervention (simple random policy for v0.1).
        
        Args:
            obs: Current observation
            rng: Random generator
            
        Returns:
            (u_type, args) tuple
        """
        n_obj = self.config.n_objects
        n_attr = self.config.n_attributes
        n_rel = self.config.n_relations
        
        # Uniform random intervention type
        u_type = int(rng.integers(0, self.config.n_intervention_types))
        
        # Sample valid arguments
        args = {}
        
        if u_type in [0, 1]:  # IncreaseAttr, DecreaseAttr
            args["i"] = int(rng.integers(0, n_obj))
            args["a"] = int(rng.integers(0, n_attr))
            
        elif u_type == 2:  # SetAttrToward
            args["i"] = int(rng.integers(0, n_obj))
            args["a"] = int(rng.integers(0, n_attr))
            args["bin_id"] = int(rng.integers(0, 4))
            
        elif u_type in [3, 4, 5]:  # Relation interventions
            args["r"] = int(rng.integers(0, n_rel))
            i = int(rng.integers(0, n_obj))
            j = int(rng.integers(0, n_obj - 1))
            if j >= i:
                j += 1
            args["i"] = i
            args["j"] = j
            
        elif u_type == 6:  # SwapObjects
            i = int(rng.integers(0, n_obj))
            j = int(rng.integers(0, n_obj - 1))
            if j >= i:
                j += 1
            args["i"] = i
            args["j"] = j
            
        elif u_type == 7:  # NoiseBurstAttr
            args["i"] = int(rng.integers(0, n_obj))
        
        return u_type, args
    
    def _log_step(
        self,
        obs: Dict[str, np.ndarray],
        u_type: int,
        settle_info: Dict[str, Any]
    ) -> None:
        """Log step information."""
        if "energy_components" in settle_info:
            self.logs["energy_components"].append(settle_info["energy_components"])
        
        self.logs["gradient_norms"].append(settle_info.get("final_grad_norm", 0.0))
        
        # Gate values
        g_stc, g_anchor = self.gates.compute_gates(self.S, u_type)
        self.logs["gate_values"].append({
            "g_stc": g_stc.copy(),
            "g_anchor": g_anchor.copy(),
        })
        
        # Anchor activations
        activations = self.anchors.get_activations(self.S)
        self.logs["anchor_activations"].append(activations.copy())
    
    def get_state(self) -> np.ndarray:
        """Get current latent state."""
        return self.S.copy()
    
    def set_state(self, S: np.ndarray) -> None:
        """Set latent state."""
        self.S = S.astype(np.float32)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get all trainable parameters."""
        return {
            "encoder": self.encoder.get_parameters(),
            "anchors": self.anchors.get_parameters(),
            "stc": self.stc.get_parameters(),
            "gates": self.gates.get_parameters(),
            "temporal": self.temporal.get_parameters(),
            "intent": self.intent.get_parameters(),
        }
    
    def get_logs(self) -> Dict[str, Any]:
        """Get accumulated logs."""
        return self.logs

