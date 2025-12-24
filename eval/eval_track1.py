"""
Track 1 Evaluation (Enhanced)

Implements comprehensive evaluation metrics with diverse perturbations:
1. Paraphrase stability (multiple perturbation types)
2. Counterfactual coherence (interpretable intervention comparisons)
3. Compositional transfer
4. Graceful degradation
5. Causal locality tests
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

import sys
sys.path.append(str(Path(__file__).parent.parent))

from env.track1_env import Track1CausalEnv, EnvConfig
from agent.agent import Track1Agent, AgentConfig


class PerturbationType(Enum):
    """Types of paraphrase perturbations."""
    GAUSSIAN_NOISE = "gaussian_noise"
    UNIFORM_NOISE = "uniform_noise"
    OBJECT_PERMUTATION = "object_permutation"
    ATTRIBUTE_SCALING = "attribute_scaling"
    ATTRIBUTE_DROPOUT = "attribute_dropout"
    RELATION_FLIP = "relation_flip"
    QUANTIZATION = "quantization"


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    n_paraphrase_samples: int = 5
    noise_levels: Tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.3)
    change_threshold: float = 0.1
    n_eval_episodes: int = 10
    seed: int = 42
    
    # Enhanced paraphrase settings
    perturbation_types: Tuple[str, ...] = (
        "gaussian_noise", "uniform_noise", "object_permutation",
        "attribute_scaling", "attribute_dropout", "relation_flip"
    )
    perturbation_strengths: Tuple[float, ...] = (0.01, 0.05, 0.1)
    
    # Counterfactual settings
    n_counterfactual_samples: int = 10


class Evaluator:
    """
    Track 1 Evaluator with enhanced perturbations.
    """
    
    # Interpretable intervention pairs for counterfactual tests
    INTERVENTION_PAIRS = [
        # (name, base_type, cf_type, description)
        ("attr_direction", 0, 1, "IncreaseAttr vs DecreaseAttr"),
        ("attr_vs_noise", 0, 7, "IncreaseAttr vs NoiseBurst"),
        ("rel_toggle_vs_set", 3, 4, "ToggleRel vs SetRelOn"),
        ("rel_on_vs_off", 4, 5, "SetRelOn vs SetRelOff"),
        ("swap_vs_attr", 6, 0, "SwapObjects vs IncreaseAttr"),
        ("local_vs_global", 0, 6, "Local attr change vs global swap"),
    ]
    
    # Intervention semantic categories
    INTERVENTION_CATEGORIES = {
        "attribute_local": [0, 1, 2],      # Affect single attribute
        "attribute_global": [7],            # Affect multiple attributes
        "relation": [3, 4, 5],              # Affect relations
        "structural": [6],                  # Affect structure (swap)
    }
    
    def __init__(
        self,
        env: Track1CausalEnv,
        agent: Track1Agent,
        config: Optional[EvalConfig] = None
    ):
        """Initialize evaluator."""
        self.env = env
        self.agent = agent
        self.config = config or EvalConfig()
        self.rng = np.random.default_rng(self.config.seed)
    
    # =========================================================================
    # DIVERSE PARAPHRASE PERTURBATIONS
    # =========================================================================
    
    def _apply_perturbation(
        self,
        obs: Dict[str, np.ndarray],
        perturbation_type: str,
        strength: float
    ) -> Dict[str, np.ndarray]:
        """
        Apply a specific perturbation type to observation.
        
        Args:
            obs: Original observation
            perturbation_type: Type of perturbation
            strength: Perturbation strength/magnitude
            
        Returns:
            Perturbed observation
        """
        X = obs["X"].copy()
        Rel = obs["Rel"].copy()
        
        if perturbation_type == "gaussian_noise":
            # Standard Gaussian noise on attributes
            X = X + self.rng.normal(0, strength, X.shape).astype(np.float32)
            
        elif perturbation_type == "uniform_noise":
            # Uniform noise on attributes
            X = X + self.rng.uniform(-strength, strength, X.shape).astype(np.float32)
            
        elif perturbation_type == "object_permutation":
            # Randomly permute object order (tests permutation invariance)
            if self.rng.random() < strength * 10:  # Probability of permuting
                perm = self.rng.permutation(X.shape[0])
                X = X[perm]
                Rel = Rel[perm][:, perm]
                
        elif perturbation_type == "attribute_scaling":
            # Scale attributes by random factor
            scale = 1.0 + self.rng.uniform(-strength, strength, (1, X.shape[1]))
            X = X * scale.astype(np.float32)
            
        elif perturbation_type == "attribute_dropout":
            # Zero out random attributes
            mask = self.rng.random(X.shape) > strength
            X = X * mask.astype(np.float32)
            
        elif perturbation_type == "relation_flip":
            # Flip random relations
            flip_mask = self.rng.random(Rel.shape) < strength
            Rel = np.where(flip_mask, 1 - Rel, Rel).astype(np.float32)
            # Ensure no self-relations
            for i in range(Rel.shape[0]):
                Rel[i, i, :] = 0.0
                
        elif perturbation_type == "quantization":
            # Quantize attributes to fewer levels
            n_levels = max(2, int(1.0 / (strength + 0.01)))
            X = np.round(X * n_levels) / n_levels
        
        # Clip to valid range
        X = np.clip(X, 0, 1).astype(np.float32)
        Rel = np.clip(Rel, 0, 1).astype(np.float32)
        
        return {"X": X, "Rel": Rel, "mask": None}
    
    def evaluate_paraphrase_stability_diverse(
        self,
        obs: Dict[str, np.ndarray],
        u_type: int,
        n_samples: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate paraphrase stability with diverse perturbation types.
        
        Tests whether semantically similar observations produce similar states.
        
        Args:
            obs: Base observation
            u_type: Intervention type
            n_samples: Samples per perturbation type
            
        Returns:
            Comprehensive stability metrics per perturbation type
        """
        # Get base settled state
        self.agent.reset()
        S_base, _ = self.agent.perceive(obs, u_type)
        top_anchors_base, _ = self.agent.anchors.get_top_k_anchors(S_base, k=3)
        enc_base = self.agent.encoder.encode(obs)
        
        results = {}
        
        for pert_type in self.config.perturbation_types:
            pert_results = {
                "cosine_distances": [],
                "anchor_overlaps": [],
                "encoding_distances": [],
                "state_l2_distances": [],
            }
            
            for strength in self.config.perturbation_strengths:
                for _ in range(n_samples):
                    # Apply perturbation
                    obs_pert = self._apply_perturbation(obs, pert_type, strength)
                    
                    # Reset and perceive
                    self.agent.reset()
                    S_pert, _ = self.agent.perceive(obs_pert, u_type)
                    
                    # Cosine distance
                    norm_base = np.linalg.norm(S_base)
                    norm_pert = np.linalg.norm(S_pert)
                    if norm_base > 1e-6 and norm_pert > 1e-6:
                        cos_sim = np.dot(S_base, S_pert) / (norm_base * norm_pert)
                        pert_results["cosine_distances"].append(1 - cos_sim)
                    
                    # Anchor overlap
                    top_anchors_pert, _ = self.agent.anchors.get_top_k_anchors(S_pert, k=3)
                    overlap = len(set(top_anchors_base) & set(top_anchors_pert)) / 3
                    pert_results["anchor_overlaps"].append(overlap)
                    
                    # Encoding distance
                    enc_pert = self.agent.encoder.encode(obs_pert)
                    pert_results["encoding_distances"].append(
                        float(np.linalg.norm(enc_base - enc_pert))
                    )
                    
                    # State L2 distance
                    pert_results["state_l2_distances"].append(
                        float(np.linalg.norm(S_base - S_pert))
                    )
            
            # Aggregate statistics
            results[pert_type] = {
                "mean_cosine_distance": float(np.mean(pert_results["cosine_distances"])),
                "std_cosine_distance": float(np.std(pert_results["cosine_distances"])),
                "mean_anchor_overlap": float(np.mean(pert_results["anchor_overlaps"])),
                "mean_encoding_distance": float(np.mean(pert_results["encoding_distances"])),
                "mean_state_l2_distance": float(np.mean(pert_results["state_l2_distances"])),
            }
        
        # Overall summary
        all_cosine = [r["mean_cosine_distance"] for r in results.values()]
        all_anchor = [r["mean_anchor_overlap"] for r in results.values()]
        
        results["_summary"] = {
            "overall_mean_cosine_distance": float(np.mean(all_cosine)),
            "overall_mean_anchor_overlap": float(np.mean(all_anchor)),
            "most_stable_perturbation": min(results.keys() - {"_summary"}, 
                                            key=lambda k: results[k]["mean_cosine_distance"]),
            "least_stable_perturbation": max(results.keys() - {"_summary"},
                                             key=lambda k: results[k]["mean_cosine_distance"]),
        }
        
        return results
    
    # =========================================================================
    # INTERPRETABLE COUNTERFACTUAL TESTS  
    # =========================================================================
    
    def evaluate_counterfactual_suite(
        self,
        obs: Dict[str, np.ndarray],
        n_samples: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate counterfactual coherence with interpretable intervention pairs.
        
        Tests whether related interventions produce predictably related states.
        
        Returns:
            Detailed counterfactual analysis
        """
        results = {}
        
        for name, base_type, cf_type, description in self.INTERVENTION_PAIRS:
            pair_results = {
                "description": description,
                "locality_scores": [],
                "direction_consistency": [],
                "magnitude_ratios": [],
                "settle_time_diffs": [],
            }
            
            for _ in range(n_samples):
                # Generate valid args for both interventions
                args = self._generate_matching_args(base_type, cf_type)
                
                # Get base state
                self.agent.reset()
                S_base, info_base = self.agent.perceive(obs, base_type)
                
                # Get counterfactual state
                self.agent.reset()
                S_cf, info_cf = self.agent.perceive(obs, cf_type)
                
                # Compute metrics
                diff = S_cf - S_base
                abs_diff = np.abs(diff)
                
                # Locality: fraction of dimensions with significant change
                locality = float(np.mean(abs_diff > self.config.change_threshold))
                pair_results["locality_scores"].append(locality)
                
                # Direction consistency: are changes in consistent direction?
                # (useful for opposite interventions like Increase/Decrease)
                if base_type in [0, 1] and cf_type in [0, 1]:
                    # For attr interventions, check if state moved in expected direction
                    # IncreaseAttr should produce higher values than DecreaseAttr
                    direction = np.mean(diff)  # Positive if CF > base
                    expected_sign = 1 if cf_type == 0 else -1  # Increase=0 should be positive
                    consistency = 1.0 if np.sign(direction) == expected_sign else 0.0
                    pair_results["direction_consistency"].append(consistency)
                
                # Magnitude ratio: how much did CF change vs base?
                base_energy, _ = self.agent.energy_computer.compute_state_energy(S_base, obs, base_type)
                cf_energy, _ = self.agent.energy_computer.compute_state_energy(S_cf, obs, cf_type)
                ratio = cf_energy / (base_energy + 1e-6)
                pair_results["magnitude_ratios"].append(float(ratio))
                
                # Settle time difference
                settle_diff = info_cf.get("n_steps", 0) - info_base.get("n_steps", 0)
                pair_results["settle_time_diffs"].append(settle_diff)
            
            # Aggregate
            results[name] = {
                "description": description,
                "mean_locality": float(np.mean(pair_results["locality_scores"])),
                "std_locality": float(np.std(pair_results["locality_scores"])),
                "mean_energy_ratio": float(np.mean(pair_results["magnitude_ratios"])),
                "mean_settle_time_diff": float(np.mean(pair_results["settle_time_diffs"])),
            }
            
            if pair_results["direction_consistency"]:
                results[name]["direction_consistency"] = float(
                    np.mean(pair_results["direction_consistency"])
                )
        
        return results
    
    def evaluate_causal_locality(
        self,
        n_episodes: int = 5
    ) -> Dict[str, Any]:
        """
        Test causal locality: interventions should have local effects.
        
        For each intervention, measure which objects/attributes change.
        
        Returns:
            Causal locality analysis per intervention type
        """
        results = {}
        
        for u_type in range(8):
            type_results = {
                "affected_objects": [],      # Which objects changed
                "affected_attributes": [],   # Which attributes changed  
                "spillover_ratio": [],       # Unintended changes
            }
            
            for ep in range(n_episodes):
                obs = self.env.reset(seed=self.rng.integers(0, 2**31))
                self.agent.reset()
                
                # Get pre-intervention state
                S_pre, _ = self.agent.perceive(obs, u_type)
                X_pre = obs["X"].copy()
                Rel_pre = obs["Rel"].copy()
                
                # Apply intervention
                args = self.env._sample_args_for_type(u_type)
                obs_next, _, _ = self.env.step(u_type, args)
                
                # Get post-intervention state
                S_post, _ = self.agent.perceive(obs_next, u_type)
                
                # Analyze changes in observation
                X_diff = np.abs(obs_next["X"] - X_pre)
                Rel_diff = np.abs(obs_next["Rel"] - Rel_pre)
                
                # Which objects were affected?
                object_changes = np.max(X_diff, axis=1)
                affected_objs = np.sum(object_changes > 0.01)
                type_results["affected_objects"].append(int(affected_objs))
                
                # Which attributes were affected?
                attr_changes = np.max(X_diff, axis=0)
                affected_attrs = np.sum(attr_changes > 0.01)
                type_results["affected_attributes"].append(int(affected_attrs))
                
                # Spillover: changes to non-target objects/attributes
                target_i = args.get("i", 0)
                target_a = args.get("a", 0)
                
                if u_type in [0, 1, 2]:  # Attribute interventions
                    # Expected: only target object's target attribute changes
                    intended_change = X_diff[target_i, target_a]
                    unintended_mask = np.ones_like(X_diff, dtype=bool)
                    unintended_mask[target_i, target_a] = False
                    unintended_change = np.sum(X_diff[unintended_mask])
                    spillover = unintended_change / (intended_change + unintended_change + 1e-6)
                elif u_type in [3, 4, 5]:  # Relation interventions
                    target_j = args.get("j", 1)
                    target_r = args.get("r", 0)
                    # Expected: only target relation changes
                    intended_change = Rel_diff[target_i, target_j, target_r]
                    unintended_mask = np.ones_like(Rel_diff, dtype=bool)
                    unintended_mask[target_i, target_j, target_r] = False
                    unintended_change = np.sum(Rel_diff[unintended_mask])
                    spillover = unintended_change / (intended_change + unintended_change + 1e-6)
                elif u_type == 6:  # Swap
                    target_j = args.get("j", 1)
                    # Expected: two objects swap, others unchanged
                    unchanged_mask = np.ones(X_diff.shape[0], dtype=bool)
                    unchanged_mask[target_i] = False
                    unchanged_mask[target_j] = False
                    spillover = np.mean(object_changes[unchanged_mask])
                else:  # NoiseBurst
                    # Expected: only target object changes
                    unintended_mask = np.ones(X_diff.shape[0], dtype=bool)
                    unintended_mask[target_i] = False
                    spillover = np.mean(object_changes[unintended_mask]) / (np.mean(object_changes) + 1e-6)
                
                type_results["spillover_ratio"].append(float(spillover))
            
            # Get intervention name
            int_names = ["IncreaseAttr", "DecreaseAttr", "SetAttrToward", 
                        "ToggleRel", "SetRelOn", "SetRelOff", 
                        "SwapObjects", "NoiseBurstAttr"]
            
            results[int_names[u_type]] = {
                "mean_affected_objects": float(np.mean(type_results["affected_objects"])),
                "mean_affected_attributes": float(np.mean(type_results["affected_attributes"])),
                "mean_spillover_ratio": float(np.mean(type_results["spillover_ratio"])),
                "causal_precision": 1.0 - float(np.mean(type_results["spillover_ratio"])),
            }
        
        return results
    
    def evaluate_intervention_reversibility(
        self,
        n_episodes: int = 5
    ) -> Dict[str, Any]:
        """
        Test if opposite interventions can reverse state changes.
        
        Apply intervention A, then its opposite, measure return to original.
        
        Returns:
            Reversibility metrics
        """
        # Opposite intervention pairs
        opposites = [
            (0, 1, "IncreaseAttr/DecreaseAttr"),
            (4, 5, "SetRelOn/SetRelOff"),
        ]
        
        results = {}
        
        for fwd_type, bwd_type, name in opposites:
            pair_results = {
                "state_recovery": [],
                "energy_recovery": [],
            }
            
            for ep in range(n_episodes):
                obs = self.env.reset(seed=self.rng.integers(0, 2**31))
                self.agent.reset()
                
                # Initial state
                S_init, _ = self.agent.perceive(obs, fwd_type)
                energy_init, _ = self.agent.energy_computer.compute_state_energy(S_init, obs, fwd_type)
                
                # Forward intervention
                args = self.env._sample_args_for_type(fwd_type)
                obs_fwd, _, _ = self.env.step(fwd_type, args)
                S_fwd, _ = self.agent.perceive(obs_fwd, fwd_type)
                
                # Backward intervention (reverse)
                obs_bwd, _, _ = self.env.step(bwd_type, args)  # Same args
                S_bwd, _ = self.agent.perceive(obs_bwd, bwd_type)
                energy_bwd, _ = self.agent.energy_computer.compute_state_energy(S_bwd, obs_bwd, bwd_type)
                
                # How much did we recover?
                fwd_dist = np.linalg.norm(S_fwd - S_init)
                bwd_dist = np.linalg.norm(S_bwd - S_init)
                
                # Recovery = 1 - (final distance / forward distance)
                if fwd_dist > 1e-6:
                    recovery = 1.0 - (bwd_dist / fwd_dist)
                else:
                    recovery = 1.0
                pair_results["state_recovery"].append(float(max(0, recovery)))
                
                # Energy recovery
                energy_recovery = 1.0 - abs(energy_bwd - energy_init) / (energy_init + 1e-6)
                pair_results["energy_recovery"].append(float(max(0, energy_recovery)))
            
            results[name] = {
                "mean_state_recovery": float(np.mean(pair_results["state_recovery"])),
                "mean_energy_recovery": float(np.mean(pair_results["energy_recovery"])),
            }
        
        return results
    
    def _generate_matching_args(self, type1: int, type2: int) -> Dict[str, int]:
        """Generate arguments valid for both intervention types."""
        args = {}
        N, A, R = self.env.N, self.env.A, self.env.R
        
        # Common object index
        args["i"] = int(self.rng.integers(0, N))
        
        # Second object for relations/swap
        j = int(self.rng.integers(0, N - 1))
        if j >= args["i"]:
            j += 1
        args["j"] = j
        
        # Attribute index
        args["a"] = int(self.rng.integers(0, A))
        
        # Relation type
        args["r"] = int(self.rng.integers(0, R))
        
        # Bin for SetAttrToward
        args["bin_id"] = int(self.rng.integers(0, 4))
        
        return args
    
    # =========================================================================
    # EXISTING METHODS (Enhanced)
    # =========================================================================
    
    def evaluate_paraphrase_stability(
        self,
        obs: Dict[str, np.ndarray],
        u_type: int,
        n_samples: int = 5
    ) -> Dict[str, float]:
        """Original simple paraphrase stability (for backward compatibility)."""
        self.agent.reset()
        S_base, _ = self.agent.perceive(obs, u_type)
        top_anchors_base, _ = self.agent.anchors.get_top_k_anchors(S_base, k=3)
        
        cosine_distances = []
        anchor_overlaps = []
        
        for _ in range(n_samples):
            obs_noisy = {
                "X": obs["X"] + self.rng.normal(0, 0.01, obs["X"].shape).astype(np.float32),
                "Rel": obs["Rel"],
                "mask": None,
            }
            obs_noisy["X"] = np.clip(obs_noisy["X"], 0, 1)
            
            self.agent.reset()
            S_noisy, _ = self.agent.perceive(obs_noisy, u_type)
            
            cos_sim = np.dot(S_base, S_noisy) / (
                np.linalg.norm(S_base) * np.linalg.norm(S_noisy) + 1e-6
            )
            cosine_distances.append(1 - cos_sim)
            
            top_anchors_noisy, _ = self.agent.anchors.get_top_k_anchors(S_noisy, k=3)
            overlap = len(set(top_anchors_base) & set(top_anchors_noisy)) / 3
            anchor_overlaps.append(overlap)
        
        return {
            "mean_cosine_distance": float(np.mean(cosine_distances)),
            "std_cosine_distance": float(np.std(cosine_distances)),
            "mean_anchor_overlap": float(np.mean(anchor_overlaps)),
            "std_anchor_overlap": float(np.std(anchor_overlaps)),
        }
    
    def evaluate_counterfactual_coherence(
        self,
        obs: Dict[str, np.ndarray],
        u_type_base: int,
        u_type_counterfactual: int,
        args: Dict[str, int]
    ) -> Dict[str, float]:
        """Original counterfactual coherence (for backward compatibility)."""
        self.agent.reset()
        S_base, info_base = self.agent.perceive(obs, u_type_base)
        
        self.agent.reset()
        S_cf, info_cf = self.agent.perceive(obs, u_type_counterfactual)
        
        diff = np.abs(S_cf - S_base)
        locality = np.mean(diff > self.config.change_threshold)
        change_magnitude = float(np.mean(diff))
        
        return {
            "locality": float(locality),
            "change_magnitude": change_magnitude,
            "settle_steps_base": info_base.get("n_steps", 0),
            "settle_steps_cf": info_cf.get("n_steps", 0),
        }
    
    def evaluate_graceful_degradation(
        self,
        obs: Dict[str, np.ndarray],
        u_type: int,
        noise_levels: Optional[Tuple[float, ...]] = None
    ) -> Dict[str, List[float]]:
        """Evaluate graceful degradation under noise."""
        noise_levels = noise_levels or self.config.noise_levels
        
        self.agent.reset()
        S_base, _ = self.agent.perceive(obs, u_type)
        
        pred_errors = []
        energies = []
        state_distances = []
        
        for noise_std in noise_levels:
            obs_noisy = {
                "X": obs["X"] + self.rng.normal(0, noise_std, obs["X"].shape).astype(np.float32),
                "Rel": obs["Rel"],
                "mask": None,
            }
            obs_noisy["X"] = np.clip(obs_noisy["X"], 0, 1)
            
            self.agent.reset()
            S_noisy, _ = self.agent.perceive(obs_noisy, u_type)
            
            energy, _ = self.agent.energy_computer.compute_state_energy(S_noisy, obs_noisy, u_type)
            enc_clean = self.agent.encoder.encode(obs)
            pred_error = np.mean((S_noisy - enc_clean) ** 2)
            state_dist = np.linalg.norm(S_noisy - S_base)
            
            pred_errors.append(float(pred_error))
            energies.append(float(energy))
            state_distances.append(float(state_dist))
        
        return {
            "noise_levels": list(noise_levels),
            "prediction_errors": pred_errors,
            "energies": energies,
            "state_distances": state_distances,
        }
    
    def evaluate_swap_invariance(self, n_episodes: int = 5) -> Dict[str, float]:
        """Evaluate SwapObjects invariance."""
        errors_before_swap = []
        errors_after_swap = []
        state_correlations = []
        
        for ep in range(n_episodes):
            obs = self.env.reset(seed=self.rng.integers(0, 2**31))
            self.agent.reset()
            
            for _ in range(5):
                u_type, args = self.agent.select_action(obs, self.rng)
                obs_next, done, _ = self.env.step(u_type, args)
                self.agent.perceive(obs, u_type)
                obs = obs_next
                if done:
                    break
            
            if done:
                continue
            
            S_before = self.agent.get_state().copy()
            enc_before = self.agent.encoder.encode(obs)
            error_before = np.mean((S_before - enc_before) ** 2)
            errors_before_swap.append(float(error_before))
            
            i, j = 0, 1
            obs_next, _, _ = self.env.step(self.env.SWAP_OBJECTS, {"i": i, "j": j})
            
            S_after, _ = self.agent.perceive(obs_next, self.env.SWAP_OBJECTS)
            enc_after = self.agent.encoder.encode(obs_next)
            error_after = np.mean((S_after - enc_after) ** 2)
            errors_after_swap.append(float(error_after))
            
            corr = np.corrcoef(S_before, S_after)[0, 1]
            state_correlations.append(float(corr) if not np.isnan(corr) else 0.0)
        
        return {
            "mean_error_before_swap": float(np.mean(errors_before_swap)) if errors_before_swap else 0.0,
            "mean_error_after_swap": float(np.mean(errors_after_swap)) if errors_after_swap else 0.0,
            "error_ratio": float(np.mean(errors_after_swap) / (np.mean(errors_before_swap) + 1e-6)) if errors_before_swap else 1.0,
            "mean_state_correlation": float(np.mean(state_correlations)) if state_correlations else 0.0,
        }
    
    # =========================================================================
    # FULL EVALUATION
    # =========================================================================
    
    def run_full_evaluation(self, enhanced: bool = True) -> Dict[str, Any]:
        """
        Run full evaluation suite.
        
        Args:
            enhanced: If True, run enhanced diverse tests
            
        Returns:
            Complete evaluation results
        """
        results = {}
        
        obs = self.env.reset(seed=self.config.seed)
        u_type = 0
        
        # 1. Paraphrase stability
        print("Evaluating paraphrase stability...")
        if enhanced:
            results["paraphrase_stability_diverse"] = self.evaluate_paraphrase_stability_diverse(
                obs, u_type, self.config.n_paraphrase_samples
            )
        results["paraphrase_stability"] = self.evaluate_paraphrase_stability(
            obs, u_type, self.config.n_paraphrase_samples
        )
        
        # 2. Counterfactual coherence
        print("Evaluating counterfactual coherence...")
        if enhanced:
            results["counterfactual_suite"] = self.evaluate_counterfactual_suite(obs)
        results["counterfactual_coherence"] = self.evaluate_counterfactual_coherence(
            obs, u_type_base=0, u_type_counterfactual=1, args={"i": 0, "a": 0}
        )
        
        # 3. Causal locality (enhanced only)
        if enhanced:
            print("Evaluating causal locality...")
            results["causal_locality"] = self.evaluate_causal_locality()
        
        # 4. Intervention reversibility (enhanced only)
        if enhanced:
            print("Evaluating intervention reversibility...")
            results["reversibility"] = self.evaluate_intervention_reversibility()
        
        # 5. Graceful degradation
        print("Evaluating graceful degradation...")
        results["graceful_degradation"] = self.evaluate_graceful_degradation(obs, u_type)
        
        # 6. Swap invariance
        print("Evaluating swap invariance...")
        results["swap_invariance"] = self.evaluate_swap_invariance()
        
        return results
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Pretty print evaluation results."""
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS (Enhanced)")
        print("=" * 70)
        
        # 1. Paraphrase Stability
        print("\n1. PARAPHRASE STABILITY")
        print("-" * 40)
        
        ps = results.get("paraphrase_stability", {})
        print(f"   Basic: cosine_dist={ps.get('mean_cosine_distance', 0):.4f}, "
              f"anchor_overlap={ps.get('mean_anchor_overlap', 0):.4f}")
        
        if "paraphrase_stability_diverse" in results:
            psd = results["paraphrase_stability_diverse"]
            print("\n   By perturbation type:")
            for pert_type in self.config.perturbation_types:
                if pert_type in psd:
                    p = psd[pert_type]
                    print(f"     {pert_type:20s}: cos_dist={p['mean_cosine_distance']:.4f}, "
                          f"anchor={p['mean_anchor_overlap']:.4f}")
            
            if "_summary" in psd:
                s = psd["_summary"]
                print(f"\n   Most stable:  {s['most_stable_perturbation']}")
                print(f"   Least stable: {s['least_stable_perturbation']}")
        
        # 2. Counterfactual Coherence
        print("\n2. COUNTERFACTUAL COHERENCE")
        print("-" * 40)
        
        if "counterfactual_suite" in results:
            cs = results["counterfactual_suite"]
            print("\n   Intervention pair comparisons:")
            for name, data in cs.items():
                print(f"     {data['description']:30s}: "
                      f"locality={data['mean_locality']:.3f}, "
                      f"energy_ratio={data['mean_energy_ratio']:.3f}")
        
        # 3. Causal Locality
        if "causal_locality" in results:
            print("\n3. CAUSAL LOCALITY")
            print("-" * 40)
            cl = results["causal_locality"]
            print("\n   Intervention precision (1.0 = perfectly local):")
            for int_name, data in cl.items():
                print(f"     {int_name:15s}: precision={data['causal_precision']:.3f}, "
                      f"spillover={data['mean_spillover_ratio']:.3f}")
        
        # 4. Intervention Reversibility
        if "reversibility" in results:
            print("\n4. INTERVENTION REVERSIBILITY")
            print("-" * 40)
            rev = results["reversibility"]
            for pair_name, data in rev.items():
                print(f"   {pair_name}: state_recovery={data['mean_state_recovery']:.3f}, "
                      f"energy_recovery={data['mean_energy_recovery']:.3f}")
        
        # 5. Graceful Degradation
        print("\n5. GRACEFUL DEGRADATION")
        print("-" * 40)
        gd = results["graceful_degradation"]
        for i, noise in enumerate(gd["noise_levels"]):
            print(f"   Noise {noise:.2f}: pred_err={gd['prediction_errors'][i]:.4f}, "
                  f"energy={gd['energies'][i]:.4f}, state_dist={gd['state_distances'][i]:.4f}")
        
        # 6. Swap Invariance
        print("\n6. SWAP INVARIANCE")
        print("-" * 40)
        si = results["swap_invariance"]
        print(f"   Error before: {si['mean_error_before_swap']:.4f}")
        print(f"   Error after:  {si['mean_error_after_swap']:.4f}")
        print(f"   Error ratio:  {si['error_ratio']:.4f}")
        print(f"   State corr:   {si['mean_state_correlation']:.4f}")
        
        print("\n" + "=" * 70)


def main():
    """Main evaluation entry point."""
    import yaml
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Track 1 Agent")
    parser.add_argument("--config", type=str, default="configs/track1.yaml",
                        help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint")
    parser.add_argument("--basic", action="store_true",
                        help="Run basic evaluation only (skip enhanced tests)")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment
    env_cfg = EnvConfig(
        n_objects=config["environment"]["n_objects"],
        n_attributes=config["environment"]["n_attributes"],
        n_relations=config["environment"]["n_relations"],
    )
    env = Track1CausalEnv(env_cfg)
    
    # Create agent
    agent_cfg = AgentConfig(
        n_objects=config["environment"]["n_objects"],
        n_attributes=config["environment"]["n_attributes"],
        n_relations=config["environment"]["n_relations"],
        latent_dim=config["agent"]["latent_dim"],
        intent_dim=config["agent"]["intent_dim"],
        n_anchors=config["agent"]["n_anchors"],
        seed=config["seed"],
    )
    agent = Track1Agent(agent_cfg)
    
    # Create evaluator with enhanced config
    eval_cfg = EvalConfig(
        n_paraphrase_samples=config["evaluation"]["n_paraphrase_samples"],
        noise_levels=tuple(config["evaluation"]["noise_levels"]),
        change_threshold=config["evaluation"]["change_threshold"],
        seed=config["seed"],
    )
    evaluator = Evaluator(env, agent, eval_cfg)
    
    # Run evaluation
    print("Running Track 1 Evaluation (Enhanced)...")
    results = evaluator.run_full_evaluation(enhanced=not args.basic)
    
    # Print results
    evaluator.print_results(results)


if __name__ == "__main__":
    main()
