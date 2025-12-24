"""
Instruction Parser Module

Maps natural language instructions to interventions (u_type, args).
Enables language-grounded action selection.
"""

import numpy as np
import re
from typing import Dict, List, Optional, Tuple
from .language_encoder import LanguageEmbedder


class InstructionParser:
    """
    Parses natural language instructions to intervention specifications.
    
    Supports both:
    1. Pattern matching (fast, interpretable)
    2. Neural classification (learned, flexible)
    """
    
    # Intervention type names
    INT_NAMES = [
        "IncreaseAttr", "DecreaseAttr", "SetAttrToward",
        "ToggleRel", "SetRelOn", "SetRelOff",
        "SwapObjects", "NoiseBurstAttr"
    ]
    
    # Keyword patterns for each intervention type
    # Canonical patterns first (match exact corpus format), then variations
    PATTERNS = {
        0: [  # IncreaseAttr
            # Canonical: "increase object {i} attr {a}"
            r"increase\s+object\s+(\d+)\s+attr\s+(\d+)",
            r"increase\s+(?:object\s+)?(\d+)(?:\s+(?:attr(?:ibute)?\s+)?(\d+))?",
            r"make\s+(?:object\s+)?(\d+)\s+(?:larger|bigger|higher)",
            r"raise\s+(?:object\s+)?(\d+)",
        ],
        1: [  # DecreaseAttr
            # Canonical: "decrease object {i} attr {a}"
            r"decrease\s+object\s+(\d+)\s+attr\s+(\d+)",
            r"decrease\s+(?:object\s+)?(\d+)(?:\s+(?:attr(?:ibute)?\s+)?(\d+))?",
            r"make\s+(?:object\s+)?(\d+)\s+(?:smaller|lower)",
            r"reduce\s+(?:object\s+)?(\d+)",
        ],
        2: [  # SetAttrToward
            # Canonical: "set object {i} attr {a} to bin {bin_id}"
            r"set\s+object\s+(\d+)\s+attr\s+(\d+)\s+to\s+bin\s+(\d+)",
            r"set\s+(?:object\s+)?(\d+)(?:\s+(?:attr(?:ibute)?\s+)?(\d+))?\s+(?:to(?:ward)?|=)\s+(?:bin\s+)?(\d+)",
        ],
        3: [  # ToggleRel
            # Canonical: "toggle relation {r} between object {i} and object {j}"
            r"toggle\s+relation\s+(\d+)\s+between\s+object\s+(\d+)\s+and\s+object\s+(\d+)",
            r"flip\s+relation\s+(\d+)\s+between\s+object\s+(\d+)\s+and\s+object\s+(\d+)",
            r"toggle\s+(?:rel(?:ation)?\s+)?(\d+)?\s*(?:between\s+)?(?:object\s+)?(\d+)\s+(?:and|to)\s+(?:object\s+)?(\d+)",
        ],
        4: [  # SetRelOn
            # Canonical: "set relation {r} between object {i} and object {j} on"
            r"set\s+relation\s+(\d+)\s+between\s+object\s+(\d+)\s+and\s+object\s+(\d+)\s+on",
            r"connect\s+(?:object\s+)?(\d+)\s+(?:and|to|with)\s+(?:object\s+)?(\d+)",
            r"link\s+(?:object\s+)?(\d+)\s+(?:and|to|with)\s+(?:object\s+)?(\d+)",
            r"make\s+(?:object\s+)?(\d+)\s+support\s+(?:object\s+)?(\d+)",
        ],
        5: [  # SetRelOff
            # Canonical: "set relation {r} between object {i} and object {j} off"
            r"set\s+relation\s+(\d+)\s+between\s+object\s+(\d+)\s+and\s+object\s+(\d+)\s+off",
            r"disconnect\s+(?:object\s+)?(\d+)\s+(?:and|from|with)\s+(?:object\s+)?(\d+)",
            r"unlink\s+(?:object\s+)?(\d+)\s+(?:and|from|with)\s+(?:object\s+)?(\d+)",
        ],
        6: [  # SwapObjects
            # Canonical: "swap object {i} and object {j}"
            r"swap\s+object\s+(\d+)\s+(?:and|with)\s+object\s+(\d+)",
            r"exchange\s+(?:object\s+)?(\d+)\s+(?:and|with)\s+(?:object\s+)?(\d+)",
        ],
        7: [  # NoiseBurstAttr
            # Canonical: "add noise to object {i}"
            r"add\s+noise\s+to\s+object\s+(\d+)",
            r"noise\s+object\s+(\d+)",
            r"perturb\s+(?:object\s+)?(\d+)",
        ],
    }
    
    def __init__(
        self,
        n_objects: int = 4,
        n_attributes: int = 3,
        n_relations: int = 2,
        latent_dim: int = 64,
        use_neural: bool = True,
        seed: int = 42
    ):
        """
        Initialize instruction parser.
        
        Args:
            n_objects: Number of objects
            n_attributes: Number of attributes
            n_relations: Number of relation types
            latent_dim: Latent dimension for neural classifier
            use_neural: Whether to use neural classifier
            seed: Random seed
        """
        self.N = n_objects
        self.A = n_attributes
        self.R = n_relations
        self.use_neural = use_neural
        
        self.rng = np.random.default_rng(seed)
        
        if use_neural:
            # Language embedder
            self.lang_embedder = LanguageEmbedder(
                output_dim=latent_dim,
                seed=seed
            )
            
            # Intervention type classifier
            scale = np.sqrt(2.0 / (latent_dim + 8))
            self.W_type = self.rng.normal(0, scale, (latent_dim, 8)).astype(np.float32)
            self.b_type = np.zeros(8, dtype=np.float32)
            
            # Argument extractors (per argument type)
            self.W_obj_i = self.rng.normal(0, scale, (latent_dim, n_objects)).astype(np.float32)
            self.W_obj_j = self.rng.normal(0, scale, (latent_dim, n_objects)).astype(np.float32)
            self.W_attr = self.rng.normal(0, scale, (latent_dim, n_attributes)).astype(np.float32)
            self.W_rel = self.rng.normal(0, scale, (latent_dim, n_relations)).astype(np.float32)
            self.W_bin = self.rng.normal(0, scale, (latent_dim, 4)).astype(np.float32)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / (np.sum(exp_x) + 1e-9)
    
    def parse_pattern(self, instruction: str) -> Tuple[Optional[int], Dict[str, int], float]:
        """
        Parse instruction using pattern matching.
        
        Args:
            instruction: Natural language instruction
            
        Returns:
            (u_type, args, confidence) tuple
        """
        instruction = instruction.lower().strip()
        
        for u_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, instruction)
                if match:
                    args = self._extract_args_from_match(u_type, match)
                    return u_type, args, 1.0
        
        return None, {}, 0.0
    
    def _extract_args_from_match(self, u_type: int, match: re.Match) -> Dict[str, int]:
        """Extract arguments from regex match."""
        groups = match.groups()
        args = {}
        
        if u_type in [0, 1]:  # IncreaseAttr, DecreaseAttr
            args["i"] = int(groups[0]) if groups[0] else 0
            args["a"] = int(groups[1]) if len(groups) > 1 and groups[1] else 0
            
        elif u_type == 2:  # SetAttrToward
            args["i"] = int(groups[0]) if groups[0] else 0
            args["a"] = int(groups[1]) if len(groups) > 1 and groups[1] else 0
            args["bin_id"] = int(groups[2]) if len(groups) > 2 and groups[2] else 0
            
        elif u_type in [3, 4, 5]:  # Relation interventions
            if len(groups) >= 3:
                args["r"] = int(groups[0]) if groups[0] else 0
                args["i"] = int(groups[1]) if groups[1] else 0
                args["j"] = int(groups[2]) if groups[2] else 1
            elif len(groups) >= 2:
                args["r"] = 0
                args["i"] = int(groups[0]) if groups[0] else 0
                args["j"] = int(groups[1]) if groups[1] else 1
                
        elif u_type == 6:  # SwapObjects
            args["i"] = int(groups[0]) if groups[0] else 0
            args["j"] = int(groups[1]) if len(groups) > 1 and groups[1] else 1
            
        elif u_type == 7:  # NoiseBurstAttr
            args["i"] = int(groups[0]) if groups[0] else 0
        
        # Validate and clamp
        args = self._validate_args(u_type, args)
        
        return args
    
    def _validate_args(self, u_type: int, args: Dict[str, int]) -> Dict[str, int]:
        """Validate and clamp argument values."""
        if "i" in args:
            args["i"] = max(0, min(args["i"], self.N - 1))
        if "j" in args:
            args["j"] = max(0, min(args["j"], self.N - 1))
            if args.get("i") == args.get("j"):
                args["j"] = (args["i"] + 1) % self.N
        if "a" in args:
            args["a"] = max(0, min(args["a"], self.A - 1))
        if "r" in args:
            args["r"] = max(0, min(args["r"], self.R - 1))
        if "bin_id" in args:
            args["bin_id"] = max(0, min(args["bin_id"], 3))
        
        return args
    
    def parse_neural(self, instruction: str) -> Tuple[int, Dict[str, int], np.ndarray]:
        """
        Parse instruction using neural classifier.
        
        Args:
            instruction: Natural language instruction
            
        Returns:
            (u_type, args, type_probs) tuple
        """
        # Embed instruction
        emb = self.lang_embedder.embed(instruction)
        
        # Classify intervention type
        type_logits = emb @ self.W_type + self.b_type
        type_probs = self._softmax(type_logits)
        u_type = int(np.argmax(type_probs))
        
        # Extract arguments
        obj_i_probs = self._softmax(emb @ self.W_obj_i)
        obj_j_probs = self._softmax(emb @ self.W_obj_j)
        attr_probs = self._softmax(emb @ self.W_attr)
        rel_probs = self._softmax(emb @ self.W_rel)
        bin_probs = self._softmax(emb @ self.W_bin)
        
        args = {
            "i": int(np.argmax(obj_i_probs)),
            "j": int(np.argmax(obj_j_probs)),
            "a": int(np.argmax(attr_probs)),
            "r": int(np.argmax(rel_probs)),
            "bin_id": int(np.argmax(bin_probs)),
        }
        
        # Ensure i != j
        if args["i"] == args["j"]:
            args["j"] = (args["i"] + 1) % self.N
        
        return u_type, args, type_probs
    
    def parse(
        self,
        instruction: str,
        prefer_pattern: bool = True
    ) -> Tuple[int, Dict[str, int], float]:
        """
        Parse instruction to intervention.
        
        Args:
            instruction: Natural language instruction
            prefer_pattern: Try pattern matching first
            
        Returns:
            (u_type, args, confidence) tuple
        """
        if prefer_pattern:
            u_type, args, conf = self.parse_pattern(instruction)
            if u_type is not None:
                return u_type, args, conf
        
        if self.use_neural:
            u_type, args, probs = self.parse_neural(instruction)
            confidence = float(probs[u_type])
            return u_type, args, confidence
        
        # Fallback: random
        u_type = int(self.rng.integers(0, 8))
        args = {"i": 0, "j": 1, "a": 0, "r": 0, "bin_id": 0}
        return u_type, args, 0.0
    
    def train_step(
        self,
        instruction: str,
        target_u_type: int,
        target_args: Dict[str, int],
        lr: float = 0.001
    ) -> Dict[str, float]:
        """
        Train neural parser on (instruction, intervention) pair.
        
        Uses slot-filling supervision for all arguments.
        
        Args:
            instruction: Input instruction
            target_u_type: Target intervention type
            target_args: Target arguments
            lr: Learning rate
            
        Returns:
            Training metrics
        """
        if not self.use_neural:
            return {"loss": 0.0}
        
        # Forward pass
        emb = self.lang_embedder.embed(instruction)
        
        type_logits = emb @ self.W_type + self.b_type
        type_probs = self._softmax(type_logits)
        
        # Cross-entropy loss for type
        type_loss = -np.log(type_probs[target_u_type] + 1e-9)
        
        # Gradient: d_loss/d_logits = probs - one_hot
        d_type = type_probs.copy()
        d_type[target_u_type] -= 1.0
        
        # Update type classifier
        self.W_type -= lr * np.outer(emb, d_type)
        self.b_type -= lr * d_type
        
        # Train ALL argument extractors with slot-filling supervision
        arg_loss = 0.0
        
        # Object i
        if "i" in target_args:
            target_i = min(target_args["i"], self.N - 1)
            obj_i_probs = self._softmax(emb @ self.W_obj_i)
            arg_loss -= np.log(obj_i_probs[target_i] + 1e-9)
            d_i = obj_i_probs.copy()
            d_i[target_i] -= 1.0
            self.W_obj_i -= lr * np.outer(emb, d_i)
        
        # Object j
        if "j" in target_args:
            target_j = min(target_args["j"], self.N - 1)
            obj_j_probs = self._softmax(emb @ self.W_obj_j)
            arg_loss -= np.log(obj_j_probs[target_j] + 1e-9)
            d_j = obj_j_probs.copy()
            d_j[target_j] -= 1.0
            self.W_obj_j -= lr * np.outer(emb, d_j)
        
        # Attribute
        if "a" in target_args:
            target_a = min(target_args["a"], self.A - 1)
            attr_probs = self._softmax(emb @ self.W_attr)
            arg_loss -= np.log(attr_probs[target_a] + 1e-9)
            d_a = attr_probs.copy()
            d_a[target_a] -= 1.0
            self.W_attr -= lr * np.outer(emb, d_a)
        
        # Relation
        if "r" in target_args:
            target_r = min(target_args["r"], self.R - 1)
            rel_probs = self._softmax(emb @ self.W_rel)
            arg_loss -= np.log(rel_probs[target_r] + 1e-9)
            d_r = rel_probs.copy()
            d_r[target_r] -= 1.0
            self.W_rel -= lr * np.outer(emb, d_r)
        
        # Bin
        if "bin_id" in target_args:
            target_bin = min(target_args["bin_id"], 3)
            bin_probs = self._softmax(emb @ self.W_bin)
            arg_loss -= np.log(bin_probs[target_bin] + 1e-9)
            d_bin = bin_probs.copy()
            d_bin[target_bin] -= 1.0
            self.W_bin -= lr * np.outer(emb, d_bin)
        
        return {
            "type_loss": float(type_loss),
            "arg_loss": float(arg_loss),
            "total_loss": float(type_loss + arg_loss),
        }
    
    def train_epoch(
        self,
        dataset: list,
        lr: float = 0.01,
        shuffle: bool = True
    ) -> Dict[str, float]:
        """
        Train for one epoch on dataset.
        
        Args:
            dataset: List of (instruction, u_type, args) tuples
            lr: Learning rate
            shuffle: Whether to shuffle
            
        Returns:
            Epoch metrics
        """
        if shuffle:
            indices = self.rng.permutation(len(dataset))
        else:
            indices = np.arange(len(dataset))
        
        total_type_loss = 0.0
        total_arg_loss = 0.0
        n_correct_type = 0
        n_correct_i = 0
        n_correct_j = 0
        
        for idx in indices:
            instruction, target_type, target_args = dataset[idx]
            
            # Evaluate before update
            u_type, args, _ = self.parse(instruction, prefer_pattern=False)
            if u_type == target_type:
                n_correct_type += 1
            if args.get("i") == target_args.get("i"):
                n_correct_i += 1
            if args.get("j") == target_args.get("j"):
                n_correct_j += 1
            
            # Train step
            metrics = self.train_step(instruction, target_type, target_args, lr)
            total_type_loss += metrics["type_loss"]
            total_arg_loss += metrics["arg_loss"]
        
        n = len(dataset)
        return {
            "type_loss": total_type_loss / n,
            "arg_loss": total_arg_loss / n,
            "type_accuracy": n_correct_type / n,
            "arg_i_accuracy": n_correct_i / n,
            "arg_j_accuracy": n_correct_j / n,
        }
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Return trainable parameters."""
        if not self.use_neural:
            return {}
        
        params = {
            "W_type": self.W_type,
            "b_type": self.b_type,
            "W_obj_i": self.W_obj_i,
            "W_obj_j": self.W_obj_j,
            "W_attr": self.W_attr,
            "W_rel": self.W_rel,
            "W_bin": self.W_bin,
        }
        params.update({f"lang_{k}": v for k, v in self.lang_embedder.get_parameters().items()})
        return params
    
    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """Set parameters from dict."""
        for name, value in params.items():
            if name.startswith("lang_"):
                lang_name = name[5:]
                setattr(self.lang_embedder, lang_name, value)
            elif hasattr(self, name):
                setattr(self, name, value)

