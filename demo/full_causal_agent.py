"""
Full Interactive Causal Reasoning Agent

A complete interactive demo showcasing:
- Natural language state descriptions
- Language-grounded interventions
- Energy-based reasoning
- Causal consequence tracking
"""

import numpy as np
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from env.track1_env import Track1CausalEnv, EnvConfig
from agent.language_agent import LanguageAgent, LanguageAgentConfig
from train.train_parser import train_parser_to_target_accuracy


class InteractiveCausalAgent:
    """
    Interactive causal reasoning agent with language interface.
    
    Supports:
    - Natural language instructions
    - State descriptions
    - Energy monitoring
    - Counterfactual queries
    - Causal explanations
    """
    
    # Intervention type names
    INT_NAMES = [
        "IncreaseAttr", "DecreaseAttr", "SetAttrToward",
        "ToggleRel", "SetRelOn", "SetRelOff",
        "SwapObjects", "NoiseBurstAttr"
    ]
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        train_parser: bool = True,
        seed: int = 42
    ):
        """
        Initialize interactive agent.
        
        Args:
            checkpoint_path: Path to saved checkpoint (optional)
            train_parser: Whether to train parser for high accuracy
            seed: Random seed
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Create environment
        self.env_config = EnvConfig(
            n_objects=4,
            n_attributes=3,
            n_relations=2,
            enable_relation_effects=True,
        )
        self.env = Track1CausalEnv(self.env_config)
        
        # Create agent
        self.agent_config = LanguageAgentConfig(
            n_objects=4,
            n_attributes=3,
            n_relations=2,
            latent_dim=64,
            intent_dim=16,
            n_anchors=8,
            lang_dim=64,
            use_language_perception=True,
            use_language_generation=True,
            use_instruction_parsing=True,
            seed=seed,
        )
        self.agent = LanguageAgent(self.agent_config)
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")
        
        # Train parser for high accuracy
        if train_parser:
            print("Training instruction parser for high accuracy...")
            self._train_parser()
        
        # State tracking
        self.obs = None
        self.obs_history = []
        self.action_history = []
        self.done = False
        self.step_count = 0
    
    def _train_parser(self, target_accuracy: float = 0.90):
        """Train the instruction parser to target accuracy."""
        from train.train_parser import generate_canonical_dataset
        
        # Generate training data
        train_data = generate_canonical_dataset(3000, self.seed)
        
        # Train for a few epochs
        for epoch in range(20):
            metrics = self.agent.instruction_parser.train_epoch(
                train_data, lr=0.05, shuffle=True
            )
            if metrics["type_accuracy"] >= target_accuracy:
                print(f"  Parser trained to {metrics['type_accuracy']*100:.1f}% accuracy")
                break
    
    def _load_checkpoint(self, path: str) -> None:
        """Load agent state from checkpoint."""
        try:
            checkpoint = np.load(path, allow_pickle=True).item()
            # Could restore parameters here
            print(f"Loaded checkpoint with {len(checkpoint)} keys")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    
    def save_checkpoint(self, path: str) -> None:
        """Save agent state to checkpoint."""
        checkpoint = {
            "agent_params": self.agent.get_parameters(),
            "step_count": self.step_count,
        }
        np.save(path, checkpoint, allow_pickle=True)
        print(f"Saved checkpoint to {path}")
    
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment and agent."""
        self.obs = self.env.reset(seed=self.rng.integers(0, 2**31))
        self.agent.reset()
        self.obs_history = [self.obs.copy()]
        self.action_history = []
        self.done = False
        self.step_count = 0
        
        # Initial perception
        self.agent.perceive_multimodal(self.obs, u_type=0)
        
        return self.obs
    
    def describe_state(self) -> str:
        """Get natural language description of current state."""
        if self.obs is None:
            return "No observation available. Call reset() first."
        return self.agent.describe_state(self.obs, self.agent.S)
    
    def get_energy(self) -> Dict[str, float]:
        """Get current energy components."""
        if self.obs is None:
            return {}
        
        energy, components = self.agent.energy_computer.compute_state_energy(
            self.agent.S, self.obs, u_type=0
        )
        return components
    
    # Keywords that indicate valid instructions
    VALID_KEYWORDS = [
        "increase", "decrease", "set", "toggle", "swap", "exchange",
        "connect", "disconnect", "link", "unlink", "add", "noise",
        "make", "raise", "lower", "reduce", "flip", "perturb",
        "object", "attr", "relation", "bin",
    ]
    
    def _is_valid_instruction(self, instruction: str) -> bool:
        """Check if instruction contains valid keywords."""
        instruction_lower = instruction.lower()
        return any(kw in instruction_lower for kw in self.VALID_KEYWORDS)
    
    def execute_instruction(self, instruction: str) -> Dict[str, Any]:
        """
        Execute a natural language instruction.
        
        Args:
            instruction: Natural language instruction
            
        Returns:
            Result dict with action, observation, description, energy
        """
        if self.obs is None:
            return {"error": "No observation. Call reset() first."}
        
        if self.done:
            return {"error": "Episode done. Call reset() to start new episode."}
        
        # Validate instruction contains expected keywords
        if not self._is_valid_instruction(instruction):
            return {
                "error": f"Unrecognized instruction: '{instruction}'\n"
                         f"Try something like: 'increase object 0 attr 1' or 'swap object 0 and object 1'\n"
                         f"Type /help for more examples."
            }
        
        # Parse instruction
        u_type, args, confidence = self.agent.parse_instruction(instruction)
        
        # Check if parsing failed
        if u_type is None:
            return {
                "error": f"Could not parse instruction: '{instruction}'\n"
                         f"Please use clearer phrasing like: 'increase object X attr Y'"
            }
        
        # Check confidence threshold
        if confidence < 0.5:
            return {
                "error": f"Low confidence ({confidence:.2f}) parsing: '{instruction}'\n"
                         f"Please use clearer phrasing like: 'increase object X attr Y'"
            }
        
        # Store pre-action state
        obs_before = {k: v.copy() if isinstance(v, np.ndarray) else v 
                      for k, v in self.obs.items()}
        S_before = self.agent.S.copy()
        energy_before, _ = self.agent.energy_computer.compute_state_energy(
            S_before, obs_before, u_type
        )
        
        # Execute action
        obs_after, done, info = self.env.step(u_type, args)
        
        # Update agent state
        result = self.agent.process_transition_with_language(
            obs_before, obs_after, u_type, args, instruction
        )
        
        # Update tracking
        self.obs = obs_after
        self.obs_history.append(obs_after.copy())
        self.action_history.append({
            "instruction": instruction,
            "u_type": u_type,
            "args": args,
            "confidence": confidence,
        })
        self.done = done
        self.step_count += 1
        
        # Get post-action energy
        energy_after, components = self.agent.energy_computer.compute_state_energy(
            self.agent.S, obs_after, u_type
        )
        
        return {
            "action": {
                "type": self.INT_NAMES[u_type],
                "type_id": u_type,
                "args": args,
                "confidence": confidence,
            },
            "description": result.get("intervention_description", ""),
            "state_description": result.get("state_description", self.describe_state()),
            "energy": {
                "before": energy_before,
                "after": energy_after,
                "delta": energy_after - energy_before,
                "components": components,
            },
            "prediction_error": result.get("prediction_error", 0.0),
            "done": done,
            "step": self.step_count,
        }
    
    def ask_counterfactual(self, instruction: str) -> Dict[str, Any]:
        """
        Ask a counterfactual question without executing.
        
        "What would happen if I..."
        
        Args:
            instruction: Hypothetical instruction
            
        Returns:
            Predicted effects
        """
        if self.obs is None:
            return {"error": "No observation. Call reset() first."}
        
        # Parse but don't execute
        u_type, args, confidence = self.agent.parse_instruction(instruction)
        
        # Get current state
        S_current = self.agent.S.copy()
        energy_current, _ = self.agent.energy_computer.compute_state_energy(
            S_current, self.obs, u_type
        )
        
        # Predict next state without stepping environment
        S_predicted = self.agent.predict_next(self.obs, u_type, args)
        
        # Compute predicted energy
        energy_predicted, _ = self.agent.energy_computer.compute_state_energy(
            S_predicted, self.obs, u_type
        )
        
        # Estimate change magnitude
        state_change = np.linalg.norm(S_predicted - S_current)
        
        return {
            "instruction": instruction,
            "parsed_action": {
                "type": self.INT_NAMES[u_type],
                "args": args,
            },
            "predicted_energy_change": energy_predicted - energy_current,
            "predicted_state_change": float(state_change),
            "confidence": confidence,
        }
    
    def get_history(self) -> list:
        """Get action history."""
        return self.action_history
    
    def undo(self) -> bool:
        """Undo last action (if possible via reset + replay)."""
        if len(self.action_history) <= 1:
            print("Nothing to undo.")
            return False
        
        # Get all but last action
        actions_to_replay = self.action_history[:-1]
        
        # Reset and replay
        self.reset()
        for action in actions_to_replay:
            self.execute_instruction(action["instruction"])
        
        print(f"Undid last action. Now at step {self.step_count}")
        return True


def print_header():
    """Print welcome header."""
    print("\n" + "=" * 70)
    print("    INTERACTIVE CAUSAL REASONING AGENT")
    print("    Track 1 (v0.1-A) with Language Extension")
    print("=" * 70)
    print()
    print("Commands:")
    print("  [instruction]  - Natural language instruction to execute")
    print("  /describe      - Describe current state")
    print("  /energy        - Show energy components")
    print("  /history       - Show action history")
    print("  /what if [X]   - Counterfactual query")
    print("  /undo          - Undo last action")
    print("  /reset         - Reset environment")
    print("  /help          - Show this help")
    print("  /quit          - Exit")
    print()
    print("Example instructions:")
    print('  "increase object 0 attr 1"')
    print('  "swap object 1 and object 2"')
    print('  "set relation 0 between object 0 and object 1 on"')
    print("=" * 70)


def interactive_causal_reasoning():
    """
    Main interactive loop for causal reasoning.
    """
    print_header()
    
    # Initialize agent
    print("\nInitializing agent...")
    agent = InteractiveCausalAgent(train_parser=True)
    
    # Reset environment
    obs = agent.reset()
    
    print("\n" + "-" * 70)
    print("INITIAL STATE")
    print("-" * 70)
    print(f"World: {agent.describe_state()}")
    energy = agent.get_energy()
    print(f"Energy: {energy.get('E_total', 0):.4f}")
    print("-" * 70)
    
    # Main loop
    while True:
        try:
            user_input = input("\nYour instruction: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.lower() in ["/quit", "/exit", "quit", "exit"]:
            print("Goodbye!")
            break
        
        elif user_input.lower() == "/help":
            print_header()
            continue
        
        elif user_input.lower() == "/describe":
            print(f"\nState: {agent.describe_state()}")
            continue
        
        elif user_input.lower() == "/energy":
            energy = agent.get_energy()
            print("\nEnergy Components:")
            for k, v in energy.items():
                print(f"  {k}: {v:.4f}")
            continue
        
        elif user_input.lower() == "/history":
            history = agent.get_history()
            print(f"\nAction History ({len(history)} actions):")
            for i, action in enumerate(history):
                print(f"  {i+1}. {action['instruction']} -> {agent.INT_NAMES[action['u_type']]}")
            continue
        
        elif user_input.lower().startswith("/what if "):
            query = user_input[9:].strip()
            result = agent.ask_counterfactual(query)
            print("\nCounterfactual Analysis:")
            print(f"  If you did: {result['parsed_action']['type']}({result['parsed_action']['args']})")
            print(f"  Predicted energy change: {result['predicted_energy_change']:.4f}")
            print(f"  Predicted state change: {result['predicted_state_change']:.4f}")
            continue
        
        elif user_input.lower() == "/undo":
            agent.undo()
            print(f"State: {agent.describe_state()}")
            continue
        
        elif user_input.lower() == "/reset":
            agent.reset()
            print("\n--- Environment Reset ---")
            print(f"State: {agent.describe_state()}")
            continue
        
        elif user_input.startswith("/"):
            print(f"Unknown command: {user_input}")
            print("Type /help for available commands")
            continue
        
        # Execute instruction
        result = agent.execute_instruction(user_input)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            continue
        
        # Display result
        print()
        print("-" * 70)
        print(f"Action: {result['action']['type']}({result['action']['args']}) "
              f"[confidence: {result['action']['confidence']:.2f}]")
        print(f"Result: {result['description']}")
        print(f"State: {result['state_description']}")
        print(f"Energy: {result['energy']['after']:.4f} "
              f"(delta: {result['energy']['delta']:+.4f})")
        print(f"Prediction Error: {result['prediction_error']:.4f}")
        print("-" * 70)
        
        if result["done"]:
            print("\n*** Episode complete! Type /reset to start a new episode ***")


def batch_demo():
    """Run a batch demo without interaction."""
    print("=" * 70)
    print("BATCH CAUSAL REASONING DEMO")
    print("=" * 70)
    
    agent = InteractiveCausalAgent(train_parser=True)
    agent.reset()
    
    print(f"\nInitial State: {agent.describe_state()}")
    print(f"Initial Energy: {agent.get_energy().get('E_total', 0):.4f}")
    
    # Demo instructions
    demo_instructions = [
        "increase object 0 attr 0",
        "set relation 0 between object 0 and object 1 on",
        "swap object 1 and object 2",
        "add noise to object 3",
        "decrease object 2 attr 1",
    ]
    
    print("\n" + "-" * 70)
    print("Executing demo instructions...")
    print("-" * 70)
    
    for instruction in demo_instructions:
        result = agent.execute_instruction(instruction)
        
        print(f"\n> {instruction}")
        
        if "error" in result:
            print(f"  Error: {result['error']}")
            continue
        
        print(f"  Action: {result['action']['type']}({result['action']['args']})")
        print(f"  Result: {result['description']}")
        print(f"  Energy: {result['energy']['after']:.4f} (delta: {result['energy']['delta']:+.4f})")
        
        if result["done"]:
            print("\n*** Episode ended ***")
            break
    
    print("\n" + "-" * 70)
    print("Demo complete!")
    print("-" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive Causal Reasoning Agent")
    parser.add_argument("--batch", action="store_true", help="Run batch demo instead of interactive")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    args = parser.parse_args()
    
    if args.batch:
        batch_demo()
    else:
        interactive_causal_reasoning()

