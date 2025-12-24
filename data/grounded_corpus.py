"""
Grounded Corpus Generator

Generates synthetic grounded language data from Track 1 trajectories.
No need for internet-scale data - grounding makes it hyper-efficient.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))

from env.track1_env import Track1CausalEnv, EnvConfig


@dataclass
class GroundedExample:
    """A single grounded language example."""
    # Observation context
    obs_X: np.ndarray
    obs_Rel: np.ndarray
    
    # Language
    description: str
    instruction: str
    consequence: str
    
    # Intervention
    u_type: int
    args: Dict[str, int]
    
    # Post-intervention observation
    obs_X_after: np.ndarray
    obs_Rel_after: np.ndarray


class GroundedCorpusGenerator:
    """
    Generates grounded language corpus from environment trajectories.
    
    Creates aligned (observation, language, intervention) triplets that
    maintain perfect causal grounding.
    """
    
    # Attribute names for descriptions
    ATTR_NAMES = ["size", "color", "intensity"]
    
    # Relation names
    REL_NAMES = ["supports", "connected"]
    
    # Intervention templates - CANONICAL (for high accuracy) + variations
    # First template in each list is the canonical form
    INSTRUCTION_TEMPLATES = {
        0: [  # IncreaseAttr - canonical first
            "increase object {i} attr {a}",
            "increase object {i} {attr}",
            "make object {i} larger",
        ],
        1: [  # DecreaseAttr
            "decrease object {i} attr {a}",
            "decrease object {i} {attr}",
            "make object {i} smaller",
        ],
        2: [  # SetAttrToward
            "set object {i} attr {a} to bin {bin_id}",
            "set object {i} {attr} toward {bin_level}",
        ],
        3: [  # ToggleRel
            "toggle relation {r} between object {i} and object {j}",
            "flip relation {r} between object {i} and object {j}",
        ],
        4: [  # SetRelOn
            "set relation {r} between object {i} and object {j} on",
            "connect object {i} to object {j}",
            "make object {i} support object {j}",
        ],
        5: [  # SetRelOff
            "set relation {r} between object {i} and object {j} off",
            "disconnect object {i} from object {j}",
        ],
        6: [  # SwapObjects
            "swap object {i} and object {j}",
            "exchange object {i} with object {j}",
        ],
        7: [  # NoiseBurstAttr
            "add noise to object {i}",
            "noise object {i}",
            "perturb object {i}",
        ],
    }
    
    # Canonical-only templates for high-accuracy training
    CANONICAL_TEMPLATES = {
        0: "increase object {i} attr {a}",
        1: "decrease object {i} attr {a}",
        2: "set object {i} attr {a} to bin {bin_id}",
        3: "toggle relation {r} between object {i} and object {j}",
        4: "set relation {r} between object {i} and object {j} on",
        5: "set relation {r} between object {i} and object {j} off",
        6: "swap object {i} and object {j}",
        7: "add noise to object {i}",
    }
    
    # State description templates
    STATE_TEMPLATES = [
        "object {i} has {level} {attr}",
        "the {attr} of object {i} is {level}",
        "object {i} {attr} level is {level}",
    ]
    
    RELATION_TEMPLATES = [
        "object {i} {rel} object {j}",
        "there is a {rel} link from object {i} to object {j}",
        "{i} is connected to {j} via {rel}",
    ]
    
    # Consequence templates
    CONSEQUENCE_TEMPLATES = {
        0: [
            "object {i} {attr} increased",
            "object {i} now has higher {attr}",
            "the {attr} of object {i} went up",
        ],
        1: [
            "object {i} {attr} decreased",
            "object {i} now has lower {attr}",
            "the {attr} of object {i} went down",
        ],
        2: [
            "object {i} {attr} moved toward {bin_level}",
            "object {i} {attr} is now {bin_level}",
        ],
        3: [
            "the {rel} relation between object {i} and {j} changed",
            "{rel} link between {i} and {j} toggled",
        ],
        4: [
            "object {i} now {rel} object {j}",
            "{rel} relation established between {i} and {j}",
        ],
        5: [
            "object {i} no longer {rel} object {j}",
            "{rel} relation removed between {i} and {j}",
        ],
        6: [
            "object {i} and object {j} swapped positions",
            "objects {i} and {j} exchanged",
        ],
        7: [
            "object {i} attributes perturbed by noise",
            "object {i} received random perturbation",
        ],
    }
    
    # Causal consequence templates (when relations affect attributes)
    CAUSAL_TEMPLATES = [
        "because object {i} {rel} object {j}, their {attr} values aligned",
        "due to the {rel} relation, object {j} {attr} changed",
        "the {rel} link caused object {j} to become more similar to object {i}",
    ]
    
    BIN_LEVELS = ["very low", "low", "medium", "high"]
    ATTR_LEVELS = ["low", "medium", "high"]
    
    def __init__(
        self,
        env: Optional[Track1CausalEnv] = None,
        seed: int = 42
    ):
        """
        Initialize corpus generator.
        
        Args:
            env: Environment (created if not provided)
            seed: Random seed
        """
        self.env = env or Track1CausalEnv()
        self.rng = np.random.default_rng(seed)
    
    def _get_attr_level(self, value: float) -> str:
        """Convert attribute value to level name."""
        if value < 0.33:
            return "low"
        elif value < 0.67:
            return "medium"
        else:
            return "high"
    
    def _describe_observation(self, obs: Dict[str, np.ndarray]) -> str:
        """Generate natural language description of observation."""
        descriptions = []
        X = obs["X"]
        Rel = obs["Rel"]
        
        # Describe objects
        for i in range(min(X.shape[0], 3)):  # Limit to 3 objects for brevity
            attr_idx = self.rng.integers(0, X.shape[1])
            attr_name = self.ATTR_NAMES[attr_idx] if attr_idx < len(self.ATTR_NAMES) else f"attr{attr_idx}"
            level = self._get_attr_level(X[i, attr_idx])
            
            template = self.rng.choice(self.STATE_TEMPLATES)
            desc = template.format(i=i, attr=attr_name, level=level)
            descriptions.append(desc)
        
        # Describe active relations
        for r in range(Rel.shape[2]):
            for i in range(Rel.shape[0]):
                for j in range(Rel.shape[1]):
                    if i != j and Rel[i, j, r] > 0.5:
                        rel_name = self.REL_NAMES[r] if r < len(self.REL_NAMES) else f"rel{r}"
                        template = self.rng.choice(self.RELATION_TEMPLATES)
                        desc = template.format(i=i, j=j, rel=rel_name)
                        descriptions.append(desc)
                        break  # Limit relations
        
        return ". ".join(descriptions[:4])  # Limit length
    
    def _generate_instruction(
        self, 
        u_type: int, 
        args: Dict[str, int],
        canonical: bool = True
    ) -> str:
        """
        Generate instruction for intervention.
        
        Args:
            u_type: Intervention type
            args: Intervention arguments
            canonical: If True, use canonical templates for higher accuracy
        """
        if canonical:
            # Use canonical template for consistent training
            template = self.CANONICAL_TEMPLATES.get(u_type, "apply intervention {u_type}")
        else:
            # Use varied templates
            templates = self.INSTRUCTION_TEMPLATES.get(u_type, ["perform intervention"])
            template = self.rng.choice(templates)
        
        # Build format args - use numeric indices for canonical
        format_args = {**args}
        format_args["u_type"] = u_type
        
        # Add attribute name (for non-canonical)
        attr_idx = args.get("a", 0)
        format_args["attr"] = self.ATTR_NAMES[attr_idx] if attr_idx < len(self.ATTR_NAMES) else f"attr{attr_idx}"
        
        # Add relation name (for non-canonical)
        rel_idx = args.get("r", 0)
        format_args["rel"] = self.REL_NAMES[rel_idx] if rel_idx < len(self.REL_NAMES) else f"rel{rel_idx}"
        
        # Add bin level (for non-canonical)
        bin_idx = args.get("bin_id", 0)
        format_args["bin_level"] = self.BIN_LEVELS[bin_idx] if bin_idx < len(self.BIN_LEVELS) else f"level{bin_idx}"
        
        try:
            instruction = template.format(**format_args)
        except KeyError:
            instruction = f"apply intervention {u_type}"
        
        return instruction
    
    def _generate_consequence(
        self,
        u_type: int,
        args: Dict[str, int],
        obs_before: Dict[str, np.ndarray],
        obs_after: Dict[str, np.ndarray]
    ) -> str:
        """Generate consequence description."""
        templates = self.CONSEQUENCE_TEMPLATES.get(u_type, ["intervention applied"])
        template = self.rng.choice(templates)
        
        # Build format args
        format_args = {**args}
        
        attr_idx = args.get("a", 0)
        format_args["attr"] = self.ATTR_NAMES[attr_idx] if attr_idx < len(self.ATTR_NAMES) else f"attr{attr_idx}"
        
        rel_idx = args.get("r", 0)
        format_args["rel"] = self.REL_NAMES[rel_idx] if rel_idx < len(self.REL_NAMES) else f"rel{rel_idx}"
        
        bin_idx = args.get("bin_id", 0)
        format_args["bin_level"] = self.BIN_LEVELS[bin_idx] if bin_idx < len(self.BIN_LEVELS) else f"level{bin_idx}"
        
        try:
            consequence = template.format(**format_args)
        except KeyError:
            consequence = "state changed"
        
        # Check for causal spillover effects (relation-induced changes)
        if self.env.config.enable_relation_effects:
            X_diff = np.abs(obs_after["X"] - obs_before["X"])
            target_i = args.get("i", 0)
            
            # Check if other objects changed significantly
            for j in range(X_diff.shape[0]):
                if j != target_i and np.max(X_diff[j]) > 0.02:
                    # Check if there's a relation
                    for r in range(obs_before["Rel"].shape[2]):
                        if obs_before["Rel"][target_i, j, r] > 0.5:
                            rel_name = self.REL_NAMES[r] if r < len(self.REL_NAMES) else f"rel{r}"
                            causal_template = self.rng.choice(self.CAUSAL_TEMPLATES)
                            attr_name = self.ATTR_NAMES[0] if self.ATTR_NAMES else "attribute"
                            causal_desc = causal_template.format(
                                i=target_i, j=j, rel=rel_name, attr=attr_name
                            )
                            consequence += ". " + causal_desc
                            break
        
        return consequence
    
    def generate_example(self) -> GroundedExample:
        """
        Generate a single grounded example.
        
        Returns:
            GroundedExample with aligned language and intervention
        """
        # Reset environment
        obs = self.env.reset(seed=self.rng.integers(0, 2**31))
        
        # Run a few random steps to get interesting state
        for _ in range(self.rng.integers(1, 5)):
            u_type, args = self.env.sample_action()
            obs, done, _ = self.env.step(u_type, args)
            if done:
                obs = self.env.reset()
        
        # Record pre-intervention observation
        obs_before = {
            "X": obs["X"].copy(),
            "Rel": obs["Rel"].copy(),
        }
        
        # Generate description of current state
        description = self._describe_observation(obs_before)
        
        # Sample intervention
        u_type, args = self.env.sample_action()
        
        # Generate instruction
        instruction = self._generate_instruction(u_type, args)
        
        # Apply intervention
        obs_after, _, _ = self.env.step(u_type, args)
        
        # Generate consequence description
        consequence = self._generate_consequence(u_type, args, obs_before, obs_after)
        
        return GroundedExample(
            obs_X=obs_before["X"],
            obs_Rel=obs_before["Rel"],
            description=description,
            instruction=instruction,
            consequence=consequence,
            u_type=u_type,
            args=args,
            obs_X_after=obs_after["X"].copy(),
            obs_Rel_after=obs_after["Rel"].copy(),
        )
    
    def generate_trajectory(self, n_steps: int = 10) -> List[GroundedExample]:
        """
        Generate a coherent trajectory of examples.
        
        Args:
            n_steps: Number of steps in trajectory
            
        Returns:
            List of GroundedExamples forming a trajectory
        """
        examples = []
        
        obs = self.env.reset(seed=self.rng.integers(0, 2**31))
        
        for _ in range(n_steps):
            obs_before = {"X": obs["X"].copy(), "Rel": obs["Rel"].copy()}
            
            description = self._describe_observation(obs_before)
            u_type, args = self.env.sample_action()
            instruction = self._generate_instruction(u_type, args)
            
            obs_after, done, _ = self.env.step(u_type, args)
            consequence = self._generate_consequence(u_type, args, obs_before, obs_after)
            
            examples.append(GroundedExample(
                obs_X=obs_before["X"],
                obs_Rel=obs_before["Rel"],
                description=description,
                instruction=instruction,
                consequence=consequence,
                u_type=u_type,
                args=args,
                obs_X_after=obs_after["X"].copy(),
                obs_Rel_after=obs_after["Rel"].copy(),
            ))
            
            if done:
                obs = self.env.reset()
            else:
                obs = obs_after
        
        return examples
    
    def generate_corpus(
        self,
        n_examples: int = 10000,
        save_path: Optional[str] = None
    ) -> List[GroundedExample]:
        """
        Generate a full grounded corpus.
        
        Args:
            n_examples: Number of examples to generate
            save_path: Optional path to save corpus
            
        Returns:
            List of GroundedExamples
        """
        print(f"Generating {n_examples} grounded examples...")
        
        examples = []
        n_trajectories = n_examples // 10
        
        for t in range(n_trajectories):
            trajectory = self.generate_trajectory(10)
            examples.extend(trajectory)
            
            if (t + 1) % 100 == 0:
                print(f"  Generated {len(examples)} examples...")
        
        # Fill remaining
        while len(examples) < n_examples:
            examples.append(self.generate_example())
        
        examples = examples[:n_examples]
        
        if save_path:
            self.save_corpus(examples, save_path)
        
        print(f"Generated {len(examples)} grounded examples")
        return examples
    
    def save_corpus(self, examples: List[GroundedExample], path: str) -> None:
        """Save corpus to file."""
        data = []
        for ex in examples:
            # Convert numpy int64 to Python int in args
            args_clean = {k: int(v) for k, v in ex.args.items()}
            data.append({
                "obs_X": ex.obs_X.tolist(),
                "obs_Rel": ex.obs_Rel.tolist(),
                "description": ex.description,
                "instruction": ex.instruction,
                "consequence": ex.consequence,
                "u_type": int(ex.u_type),
                "args": args_clean,
                "obs_X_after": ex.obs_X_after.tolist(),
                "obs_Rel_after": ex.obs_Rel_after.tolist(),
            })
        
        with open(path, 'w') as f:
            json.dump(data, f)
        
        print(f"Saved corpus to {path}")
    
    @staticmethod
    def load_corpus(path: str) -> List[GroundedExample]:
        """Load corpus from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        examples = []
        for d in data:
            examples.append(GroundedExample(
                obs_X=np.array(d["obs_X"], dtype=np.float32),
                obs_Rel=np.array(d["obs_Rel"], dtype=np.float32),
                description=d["description"],
                instruction=d["instruction"],
                consequence=d["consequence"],
                u_type=d["u_type"],
                args=d["args"],
                obs_X_after=np.array(d["obs_X_after"], dtype=np.float32),
                obs_Rel_after=np.array(d["obs_Rel_after"], dtype=np.float32),
            ))
        
        return examples


class GroundedDataset:
    """
    Dataset wrapper for training on grounded corpus.
    """
    
    def __init__(
        self,
        examples: List[GroundedExample],
        seed: int = 42
    ):
        """
        Initialize dataset.
        
        Args:
            examples: List of grounded examples
            seed: Random seed for shuffling
        """
        self.examples = examples
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(examples))
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> GroundedExample:
        return self.examples[idx]
    
    def shuffle(self) -> None:
        """Shuffle examples."""
        self.rng.shuffle(self.indices)
    
    def get_batch(self, batch_size: int) -> List[GroundedExample]:
        """Get a random batch of examples."""
        indices = self.rng.choice(len(self.examples), batch_size, replace=False)
        return [self.examples[i] for i in indices]
    
    def iter_batches(self, batch_size: int, shuffle: bool = True):
        """Iterate over batches."""
        if shuffle:
            self.shuffle()
        
        for i in range(0, len(self.examples), batch_size):
            batch_indices = self.indices[i:i + batch_size]
            yield [self.examples[j] for j in batch_indices]
    
    def get_instruction_dataset(self) -> List[Tuple[str, int, Dict[str, int]]]:
        """Get dataset for instruction parsing training."""
        return [(ex.instruction, ex.u_type, ex.args) for ex in self.examples]
    
    def get_description_dataset(self) -> List[Tuple[Dict[str, np.ndarray], str]]:
        """Get dataset for description generation training."""
        return [
            ({"X": ex.obs_X, "Rel": ex.obs_Rel}, ex.description)
            for ex in self.examples
        ]
    
    def get_consequence_dataset(self) -> List[Tuple[Dict[str, np.ndarray], str]]:
        """Get dataset for consequence generation training."""
        return [
            ({"X": ex.obs_X_after, "Rel": ex.obs_Rel_after}, ex.consequence)
            for ex in self.examples
        ]


def main():
    """Generate a sample corpus."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate grounded corpus")
    parser.add_argument("--n_examples", type=int, default=1000,
                        help="Number of examples to generate")
    parser.add_argument("--output", type=str, default="data/grounded_corpus.json",
                        help="Output file path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Create generator
    generator = GroundedCorpusGenerator(seed=args.seed)
    
    # Generate corpus
    examples = generator.generate_corpus(args.n_examples, args.output)
    
    # Print samples
    print("\nSample examples:")
    print("-" * 60)
    for i, ex in enumerate(examples[:3]):
        print(f"\nExample {i+1}:")
        print(f"  Description: {ex.description}")
        print(f"  Instruction: {ex.instruction}")
        print(f"  Intervention: type={ex.u_type}, args={ex.args}")
        print(f"  Consequence: {ex.consequence}")


if __name__ == "__main__":
    main()

