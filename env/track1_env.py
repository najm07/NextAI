"""
Track 1 Causal Environment (v0.1-A)

Object-centric symbolic environment with explicit relations.
Implements 8 one-hot intervention types with out-of-band arguments.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Environment configuration."""
    n_objects: int = 4
    n_attributes: int = 3
    n_relations: int = 2
    n_bins: int = 4
    bin_centers: Tuple[float, ...] = (0.125, 0.375, 0.625, 0.875)
    delta_attr: float = 0.05
    noise_attr: float = 0.01
    noise_burst_std: float = 0.08
    min_steps: int = 15
    max_steps: int = 25
    enable_relation_effects: bool = True
    relation_alignment_strength: float = 0.02


class Track1CausalEnv:
    """
    Track 1 Causal Environment with one-hot interventions.
    
    Observation space:
        - X: (N, A) continuous attributes in [0, 1]
        - Rel: (N, N, R) binary relations
        - mask: reserved for future use
    
    Intervention types (K=8, one-hot):
        0: IncreaseAttr(i, a)
        1: DecreaseAttr(i, a)
        2: SetAttrToward(i, a, bin_id)
        3: ToggleRel(r, i, j)
        4: SetRelOn(r, i, j)
        5: SetRelOff(r, i, j)
        6: SwapObjects(i, j)
        7: NoiseBurstAttr(i)
    """
    
    # Intervention type constants
    INCREASE_ATTR = 0
    DECREASE_ATTR = 1
    SET_ATTR_TOWARD = 2
    TOGGLE_REL = 3
    SET_REL_ON = 4
    SET_REL_OFF = 5
    SWAP_OBJECTS = 6
    NOISE_BURST_ATTR = 7
    
    N_INTERVENTION_TYPES = 8
    
    def __init__(self, config: Optional[EnvConfig] = None):
        """Initialize environment with configuration."""
        self.config = config or EnvConfig()
        self.N = self.config.n_objects
        self.A = self.config.n_attributes
        self.R = self.config.n_relations
        self.B = self.config.n_bins
        self.bin_centers = np.array(self.config.bin_centers, dtype=np.float32)
        
        # State
        self.X: Optional[np.ndarray] = None
        self.Rel: Optional[np.ndarray] = None
        self.step_count: int = 0
        self.max_steps: int = 0
        self.rng: Optional[np.random.Generator] = None
        
    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Initial observation dictionary
        """
        self.rng = np.random.default_rng(seed)
        
        # Initialize attributes uniformly in [0.2, 0.8] for stable starting point
        self.X = self.rng.uniform(0.2, 0.8, size=(self.N, self.A)).astype(np.float32)
        
        # Initialize relations sparsely (mostly zeros)
        self.Rel = (self.rng.random((self.N, self.N, self.R)) < 0.2).astype(np.float32)
        # No self-relations
        for i in range(self.N):
            self.Rel[i, i, :] = 0.0
            
        # Episode length
        self.max_steps = self.rng.integers(self.config.min_steps, self.config.max_steps + 1)
        self.step_count = 0
        
        return self._get_obs()
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        return {
            "X": self.X.copy(),
            "Rel": self.Rel.copy(),
            "mask": None
        }
    
    def step(self, u_type: int, args: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], bool, Dict[str, Any]]:
        """
        Execute one-hot intervention and return next observation.
        
        Args:
            u_type: Intervention type index (0-7)
            args: Out-of-band arguments dict with keys like 'i', 'j', 'a', 'r', 'bin_id'
            
        Returns:
            (obs_next, done, info) tuple
        """
        assert 0 <= u_type < self.N_INTERVENTION_TYPES, f"Invalid u_type: {u_type}"
        assert self.X is not None, "Environment not initialized. Call reset() first."
        
        info = {"u_type": u_type, "args": args.copy()}
        
        # Apply intervention
        self._apply_intervention(u_type, args)
        
        # Apply natural dynamics (drift + noise + relation effects)
        self._apply_natural_dynamics()
        
        # Clip attributes to valid range
        self.X = np.clip(self.X, 0.0, 1.0)
        
        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        return self._get_obs(), done, info
    
    def _apply_intervention(self, u_type: int, args: Dict[str, int]) -> None:
        """Apply the specified intervention."""
        
        if u_type == self.INCREASE_ATTR:
            i, a = args["i"], args["a"]
            self._validate_object_attr(i, a)
            self.X[i, a] += self.config.delta_attr
            
        elif u_type == self.DECREASE_ATTR:
            i, a = args["i"], args["a"]
            self._validate_object_attr(i, a)
            self.X[i, a] -= self.config.delta_attr
            
        elif u_type == self.SET_ATTR_TOWARD:
            i, a, bin_id = args["i"], args["a"], args["bin_id"]
            self._validate_object_attr(i, a)
            assert 0 <= bin_id < self.B, f"Invalid bin_id: {bin_id}"
            target = self.bin_centers[bin_id]
            # Move toward target by delta
            diff = target - self.X[i, a]
            step = np.sign(diff) * min(abs(diff), self.config.delta_attr * 2)
            self.X[i, a] += step
            
        elif u_type == self.TOGGLE_REL:
            r, i, j = args["r"], args["i"], args["j"]
            self._validate_relation(r, i, j)
            self.Rel[i, j, r] = 1.0 - self.Rel[i, j, r]
            
        elif u_type == self.SET_REL_ON:
            r, i, j = args["r"], args["i"], args["j"]
            self._validate_relation(r, i, j)
            self.Rel[i, j, r] = 1.0
            
        elif u_type == self.SET_REL_OFF:
            r, i, j = args["r"], args["i"], args["j"]
            self._validate_relation(r, i, j)
            self.Rel[i, j, r] = 0.0
            
        elif u_type == self.SWAP_OBJECTS:
            i, j = args["i"], args["j"]
            self._validate_swap(i, j)
            # Swap attribute rows
            self.X[[i, j]] = self.X[[j, i]]
            # Swap relation rows and columns
            self.Rel[[i, j], :, :] = self.Rel[[j, i], :, :]
            self.Rel[:, [i, j], :] = self.Rel[:, [j, i], :]
            
        elif u_type == self.NOISE_BURST_ATTR:
            i = args["i"]
            assert 0 <= i < self.N, f"Invalid object index: {i}"
            noise = self.rng.normal(0, self.config.noise_burst_std, size=self.A)
            self.X[i] += noise.astype(np.float32)
            
        else:
            raise ValueError(f"Unknown intervention type: {u_type}")
    
    def _apply_natural_dynamics(self) -> None:
        """Apply natural drift, noise, and relation-based effects."""
        # Small noise on all attributes
        noise = self.rng.normal(0, self.config.noise_attr, size=self.X.shape)
        self.X += noise.astype(np.float32)
        
        # Relation effects: linked objects partially align
        if self.config.enable_relation_effects:
            alpha = self.config.relation_alignment_strength
            for r in range(self.R):
                for i in range(self.N):
                    for j in range(self.N):
                        if i != j and self.Rel[i, j, r] > 0.5:
                            # i is linked to j, so i drifts toward j
                            diff = self.X[j] - self.X[i]
                            self.X[i] += alpha * diff
    
    def _validate_object_attr(self, i: int, a: int) -> None:
        """Validate object and attribute indices."""
        assert 0 <= i < self.N, f"Invalid object index: {i}"
        assert 0 <= a < self.A, f"Invalid attribute index: {a}"
    
    def _validate_relation(self, r: int, i: int, j: int) -> None:
        """Validate relation indices."""
        assert 0 <= r < self.R, f"Invalid relation type: {r}"
        assert 0 <= i < self.N, f"Invalid object index i: {i}"
        assert 0 <= j < self.N, f"Invalid object index j: {j}"
        assert i != j, "Self-relations not allowed"
    
    def _validate_swap(self, i: int, j: int) -> None:
        """Validate swap indices."""
        assert 0 <= i < self.N, f"Invalid object index i: {i}"
        assert 0 <= j < self.N, f"Invalid object index j: {j}"
        assert i != j, "Cannot swap object with itself"
    
    def sample_action(self) -> Tuple[int, Dict[str, int]]:
        """
        Sample a random valid action (intervention type + args).
        
        Returns:
            (u_type, args) tuple
        """
        u_type = self.rng.integers(0, self.N_INTERVENTION_TYPES)
        args = self._sample_args_for_type(u_type)
        return u_type, args
    
    def _sample_args_for_type(self, u_type: int) -> Dict[str, int]:
        """Sample valid arguments for a given intervention type."""
        args = {}
        
        if u_type in [self.INCREASE_ATTR, self.DECREASE_ATTR]:
            args["i"] = int(self.rng.integers(0, self.N))
            args["a"] = int(self.rng.integers(0, self.A))
            
        elif u_type == self.SET_ATTR_TOWARD:
            args["i"] = int(self.rng.integers(0, self.N))
            args["a"] = int(self.rng.integers(0, self.A))
            args["bin_id"] = int(self.rng.integers(0, self.B))
            
        elif u_type in [self.TOGGLE_REL, self.SET_REL_ON, self.SET_REL_OFF]:
            args["r"] = int(self.rng.integers(0, self.R))
            i = int(self.rng.integers(0, self.N))
            j = int(self.rng.integers(0, self.N - 1))
            if j >= i:
                j += 1  # Ensure i != j
            args["i"] = i
            args["j"] = j
            
        elif u_type == self.SWAP_OBJECTS:
            i = int(self.rng.integers(0, self.N))
            j = int(self.rng.integers(0, self.N - 1))
            if j >= i:
                j += 1
            args["i"] = i
            args["j"] = j
            
        elif u_type == self.NOISE_BURST_ATTR:
            args["i"] = int(self.rng.integers(0, self.N))
            
        return args
    
    def get_one_hot_intervention(self, u_type: int) -> np.ndarray:
        """Convert intervention type to one-hot vector."""
        one_hot = np.zeros(self.N_INTERVENTION_TYPES, dtype=np.float32)
        one_hot[u_type] = 1.0
        return one_hot
    
    def simulate_step(self, u_type: int, args: Dict[str, int]) -> Dict[str, np.ndarray]:
        """
        Non-destructive forward simulation.
        Creates a deep copy of the environment and returns the predicted observation.
        
        Args:
            u_type: Intervention type index (0-7)
            args: Out-of-band arguments dict
            
        Returns:
            Predicted observation dictionary
        """
        from copy import deepcopy
        
        if self.X is None or self.Rel is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        
        # Create deep copy of environment
        env_copy = deepcopy(self)
        
        # Execute step on copy
        obs_pred, _, _ = env_copy.step(u_type, args)
        
        return obs_pred
    
    @property
    def obs_shape(self) -> Dict[str, tuple]:
        """Return observation shapes."""
        return {
            "X": (self.N, self.A),
            "Rel": (self.N, self.N, self.R)
        }

