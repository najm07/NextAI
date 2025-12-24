"""
Multi-Turn Conversation Corpus Generator

Generates conversation episodes with:
- Multiple turns with pronoun references
- Causal reasoning chains
- Natural language instructions and responses
"""

import numpy as np
import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from env.track1_env import Track1CausalEnv, EnvConfig
from data.grounded_corpus import GroundedCorpusGenerator


class MultiTurnConversation:
    """Represents a multi-turn conversation episode."""
    
    def __init__(
        self,
        episode_id: int,
        initial_state: Dict[str, np.ndarray],
        turns: List[Dict[str, Any]]
    ):
        self.episode_id = episode_id
        self.initial_state = initial_state
        self.turns = turns  # List of (instruction, action, consequence, reasoning)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        # Helper to convert numpy types to native Python types
        def convert_value(v):
            if isinstance(v, (np.integer, np.int64, np.int32)):
                return int(v)
            elif isinstance(v, (np.floating, np.float64, np.float32)):
                return float(v)
            elif isinstance(v, np.ndarray):
                return v.tolist()
            elif isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            elif isinstance(v, list):
                return [convert_value(item) for item in v]
            else:
                return v
        
        return {
            "episode_id": convert_value(self.episode_id),
            "initial_state": {
                "X": convert_value(self.initial_state["X"]),
                "Rel": convert_value(self.initial_state["Rel"]),
            },
            "turns": [
                {
                    "instruction": turn["instruction"],
                    "u_type": convert_value(turn["u_type"]),
                    "args": convert_value(turn["args"]),
                    "consequence": turn["consequence"],
                    "reasoning": turn.get("reasoning", ""),
                    "energy_delta": convert_value(turn.get("energy_delta", 0.0)),
                }
                for turn in self.turns
            ]
        }


class MultiTurnCorpusGenerator:
    """
    Generates multi-turn conversation corpus.
    
    Each conversation:
    1. Starts with initial state
    2. Has 3-8 turns with natural language instructions
    3. Includes pronoun references ("it", "them")
    4. Tracks causal consequences
    """
    
    def __init__(self, seed: int = 42):
        """Initialize corpus generator."""
        self.rng = np.random.default_rng(seed)
        self.corpus_gen = GroundedCorpusGenerator(seed=seed)
        
        # Instruction templates with pronouns
        self.templates = {
            # EXISTING: Imperative actions
            "increase": [
                "increase object {i}",
                "make object {i} larger",
                "grow object {i}",
                "raise object {i}",
            ],
            "decrease": [
                "decrease object {i}",
                "make object {i} smaller",
                "shrink object {i}",
                "lower object {i}",
            ],
            "connect": [
                "connect object {i} to object {j}",
                "link object {i} and object {j}",
                "connect it to object {j}",  # Pronoun reference
                "link them together",
                "make object {i} support object {j}",
            ],
            "swap": [
                "swap object {i} and object {j}",
                "exchange object {i} with object {j}",
                "swap them",
                "switch object {i} and object {j}",
            ],
            "noise": [
                "add noise to object {i}",
                "perturb object {i}",
                "randomize object {i}",
            ],
            
            # NEW: Why/Explain reasoning
            "why": [
                "why did object {i} change?",
                "what caused object {i} to change?",
                "explain why object {i} changed",
                "why did the energy {direction}?",
                "what caused the alignment?",
                "explain the energy change",
                "why did object {i} become {state}?",
            ],
            
            # NEW: Queries about state
            "query": [
                "what is object {i}?",
                "describe object {i}",
                "what is the state of object {i}?",
                "describe the world",
                "current status",
                "show current state",
                "what are the connections?",
                "how many objects are connected?",
                "list all relations",
                "what is the energy?",
                "describe the current configuration",
            ],
            
            # NEW: Counterfactuals
            "counterfactual": [
                "what if I swapped object {i} and object {j}?",
                "what would happen if I connected object {i}?",
                "predict what happens if I increase object {i}",
                "what if I had swapped them first?",
                "imagine I connect object {i} to object {j}",
                "hypothetically, what if I decrease object {i}?",
            ],
            
            # NEW: Planning/Strategy
            "planning": [
                "connect all objects",
                "grow the largest object",
                "make object {i} the largest",
                "clear all relations",
                "disconnect everything",
                "maximize object {i}",
                "minimize object {i}",
                "balance all objects",
                "create a chain of connections",
            ],
            
            # NEW: Error recovery
            "recovery": [
                "undo last action",
                "reset",
                "go back",
                "fix the mistake",
                "revert",
                "start over",
            ],
            
            # NEW: Conditional reasoning
            "conditional": [
                "if object {i} is large, connect it to object {j}",
                "only if object {i} is connected, increase it",
                "when object {i} is high, swap it",
                "if connected, then increase",
            ],
            
            # NEW: Meta-reasoning
            "meta": [
                "what do you know?",
                "what are your beliefs?",
                "what is the energy status?",
                "describe your understanding",
                "what have you learned?",
                "summarize the changes",
            ],
        }
    
    def generate_conversation(self, n_turns: Optional[int] = None) -> MultiTurnConversation:
        """
        Generate a single multi-turn conversation.
        
        Args:
            n_turns: Number of turns (random 3-8 if None)
            
        Returns:
            MultiTurnConversation instance
        """
        if n_turns is None:
            n_turns = self.rng.integers(3, 9)
        
        # Create environment
        env = Track1CausalEnv()
        obs = env.reset(seed=self.rng.integers(0, 2**31))
        initial_state = {
            "X": obs["X"].copy(),
            "Rel": obs["Rel"].copy(),
        }
        
        turns = []
        last_changed_object = None
        action_history = []  # Track actions for recovery/undo
        
        for turn_idx in range(n_turns):
            # Initialize done flag for all code paths
            done = False
            
            # Determine instruction type based on turn position and context
            turn_type = self._select_turn_type(turn_idx, n_turns, last_changed_object, len(action_history))
            
            if turn_type == "query":
                # Query: map to QueryState (8)
                instruction = self._generate_query_instruction(env, obs)
                u_type = 8  # QueryState
                # Extract object if mentioned in instruction
                import re
                obj_match = re.search(r'object\s+(\d+)', instruction.lower())
                args = {"i": int(obj_match.group(1))} if obj_match else {}
                obs_after = obs  # No change
                consequence = "Query: " + self._describe_current_state(obs)
                
            elif turn_type == "why":
                # Why question: map to QueryWhy (9)
                instruction = self._generate_why_instruction(last_changed_object, action_history)
                u_type = 9  # QueryWhy
                args = {"i": last_changed_object} if last_changed_object is not None else {}
                obs_after = obs  # No change
                consequence = "Explanation: " + self._explain_last_change(action_history[-1] if action_history else None)
                
            elif turn_type == "counterfactual":
                # Counterfactual: map to Counterfactual (10)
                instruction, cf_u_type, cf_args = self._generate_counterfactual_instruction(env, last_changed_object)
                u_type = 10  # Counterfactual
                args = cf_args.copy() if cf_args else {}
                # Simulate but don't execute
                obs_after = obs  # No actual change
                consequence = "Hypothetical: " + self._predict_consequence(cf_u_type, cf_args, obs, env)
                
            elif turn_type == "planning":
                # Planning: map to planning action types (11-13)
                instruction, plan_u_type, plan_args = self._generate_planning_instruction(env, obs)
                # Map planning instruction to extended action code
                if "all" in instruction.lower() and "connect" in instruction.lower():
                    u_type = 11  # PlanConnectAll
                elif "largest" in instruction.lower() or "max" in instruction.lower():
                    u_type = 12  # PlanGrowMax
                elif "clear" in instruction.lower() or "disconnect" in instruction.lower():
                    u_type = 13  # PlanClearAll
                else:
                    u_type = 11  # Default to PlanConnectAll
                
                # For planning, we still execute the underlying action
                if plan_u_type is not None and plan_u_type < 8:  # Valid environment action
                    obs_after, done, info = env.step(plan_u_type, plan_args)
                    consequence = self.corpus_gen._generate_consequence(plan_u_type, plan_args, obs, obs_after)
                    action_history.append((plan_u_type, plan_args))
                    args = plan_args
                else:
                    obs_after = obs
                    consequence = "Planning instruction"
                    args = plan_args if plan_args else {}
                
            elif turn_type == "recovery":
                # Recovery: map to UndoLast (14) or ResetWorld (15)
                instruction = self._generate_recovery_instruction()
                if "reset" in instruction.lower() or "start over" in instruction.lower():
                    u_type = 15  # ResetWorld
                else:
                    u_type = 14  # UndoLast
                args = {}
                # For undo, would need to replay history - simplified for corpus
                obs_after = obs  # Placeholder
                consequence = "Recovery: action reverted"
                
            elif turn_type == "conditional":
                # Conditional: map to ConditionalIf (16)
                instruction, cond_u_type, cond_args = self._generate_conditional_instruction(env, obs)
                u_type = 16  # ConditionalIf
                if cond_u_type is not None:  # Condition met
                    obs_after, done, info = env.step(cond_u_type, cond_args)
                    consequence = self.corpus_gen._generate_consequence(cond_u_type, cond_args, obs, obs_after)
                    action_history.append((cond_u_type, cond_args))
                    args = cond_args
                else:  # Condition not met
                    obs_after = obs
                    consequence = "Condition not met: no action taken"
                    args = cond_args if cond_args else {}
                    
            elif turn_type == "meta":
                # Meta-reasoning: map to meta types (17-19)
                instruction = self._generate_meta_instruction()
                # Determine which meta type
                if "energy" in instruction.lower():
                    u_type = 17  # MetaEnergy
                elif "belief" in instruction.lower() or "know" in instruction.lower():
                    u_type = 18  # MetaBeliefs
                else:
                    u_type = 19  # MetaSummary
                args = {}
                obs_after = obs
                consequence = "Meta: " + self._summarize_beliefs(action_history, obs)
                
            else:
                # Regular action (imperative)
                if turn_idx == 0:
                    instruction, u_type, args = self._generate_first_instruction(env)
                else:
                    use_pronoun = self.rng.random() < 0.4
                    if use_pronoun and last_changed_object is not None:
                        instruction, u_type, args = self._generate_pronoun_instruction(
                            last_changed_object, env
                        )
                    else:
                        instruction, u_type, args = self._generate_instruction(env)
                
                obs_after, done, info = env.step(u_type, args)
                consequence = self.corpus_gen._generate_consequence(u_type, args, obs, obs_after)
                action_history.append((u_type, args))
            
            # Track most changed object for actions
            if u_type is not None and u_type in [0, 1, 2, 7]:  # Attr interventions
                last_changed_object = args.get("i", 0)
            elif u_type is not None and u_type in [3, 4, 5]:  # Relation interventions
                last_changed_object = args.get("i", 0)
            elif u_type is not None and u_type == 6:  # Swap
                last_changed_object = args.get("i", 0)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(instruction, consequence)
            
            turns.append({
                "instruction": instruction,
                "u_type": int(u_type),  # Now always has a valid u_type (0-19)
                "args": {k: int(v) for k, v in args.items()} if args else {},
                "consequence": consequence,
                "reasoning": reasoning,
                "energy_delta": 0.0,  # Placeholder
            })
            
            # Update observation for next turn
            obs = obs_after
            
            if done and u_type is not None:
                break
        
        return MultiTurnConversation(
            episode_id=self.rng.integers(0, 2**31),
            initial_state=initial_state,
            turns=turns
        )
    
    def _generate_first_instruction(self, env: Track1CausalEnv) -> tuple:
        """Generate first instruction with explicit object reference."""
        u_type, args = env.sample_action()
        
        # Convert to natural language
        if u_type == 0:  # IncreaseAttr
            instruction = f"increase object {args['i']}"
        elif u_type == 1:  # DecreaseAttr
            instruction = f"decrease object {args['i']}"
        elif u_type in [3, 4]:  # ToggleRel, SetRelOn
            instruction = f"connect object {args['i']} to object {args['j']}"
        elif u_type == 6:  # SwapObjects
            instruction = f"swap object {args['i']} and object {args['j']}"
        else:
            instruction = f"apply action {u_type} to object {args.get('i', 0)}"
        
        return instruction, u_type, args
    
    def _generate_pronoun_instruction(
        self,
        last_object: int,
        env: Track1CausalEnv
    ) -> tuple:
        """Generate instruction with pronoun reference."""
        # Generate action that uses the last changed object
        u_type = self.rng.integers(0, 8)
        
        if u_type in [0, 1, 2, 7]:  # Attr interventions
            args = {"i": last_object, "a": self.rng.integers(0, env.A)}
            if u_type == 2:  # SetAttrToward
                args["bin_id"] = self.rng.integers(0, 4)
            instruction = f"{'increase' if u_type == 0 else 'decrease'} it"
            
        elif u_type in [3, 4, 5]:  # Relation interventions
            other_obj = self.rng.integers(0, env.N)
            while other_obj == last_object:
                other_obj = self.rng.integers(0, env.N)
            args = {
                "r": self.rng.integers(0, env.R),
                "i": last_object,
                "j": other_obj
            }
            instruction = f"connect it to object {other_obj}"
            
        elif u_type == 6:  # SwapObjects
            other_obj = self.rng.integers(0, env.N)
            while other_obj == last_object:
                other_obj = self.rng.integers(0, env.N)
            args = {"i": last_object, "j": other_obj}
            instruction = f"swap it with object {other_obj}"
        
        else:
            args = {"i": last_object}
            instruction = f"modify it"
        
        return instruction, u_type, args
    
    def _generate_instruction(self, env: Track1CausalEnv) -> tuple:
        """Generate regular instruction."""
        return self._generate_first_instruction(env)
    
    def _select_turn_type(
        self,
        turn_idx: int,
        n_turns: int,
        last_changed_object: Optional[int],
        action_count: int
    ) -> str:
        """
        Select instruction type based on context.
        
        Distribution:
        - Turn 0: 30% query, 70% action
        - Turn 1-2: 50% action, 20% why, 15% query, 10% counterfactual, 5% meta
        - Turn 3+: 40% action, 20% why, 15% query, 10% counterfactual, 5% planning, 5% conditional, 5% meta
        """
        rand = self.rng.random()
        
        if turn_idx == 0:
            # First turn: mostly queries or actions
            if rand < 0.3:
                return "query"
            else:
                return "action"
        elif turn_idx < 3:
            # Early turns: actions + why + queries
            if rand < 0.5:
                return "action"
            elif rand < 0.7:
                return "why" if last_changed_object is not None else "action"
            elif rand < 0.85:
                return "query"
            elif rand < 0.95:
                return "counterfactual"
            else:
                return "meta"
        else:
            # Later turns: more variety
            if rand < 0.40:
                return "action"
            elif rand < 0.60:
                return "why" if last_changed_object is not None else "action"
            elif rand < 0.75:
                return "query"
            elif rand < 0.85:
                return "counterfactual"
            elif rand < 0.90:
                return "planning"
            elif rand < 0.95:
                return "conditional"
            elif rand < 0.98:
                return "recovery" if action_count > 0 else "action"
            else:
                return "meta"
    
    def _generate_query_instruction(self, env: Track1CausalEnv, obs: Dict[str, np.ndarray]) -> str:
        """Generate query instruction."""
        template = self.rng.choice(self.templates["query"])
        
        # Fill in object references if needed
        if "{i}" in template:
            obj = self.rng.integers(0, env.N)
            instruction = template.format(i=obj)
        else:
            instruction = template
        
        return instruction
    
    def _generate_why_instruction(
        self,
        last_changed_object: Optional[int],
        action_history: List[tuple]
    ) -> str:
        """Generate why question about recent changes."""
        if last_changed_object is None or not action_history:
            # Fallback to generic why
            return self.rng.choice([
                "why did that happen?",
                "explain the change",
                "what caused that?",
            ])
        
        template = self.rng.choice(self.templates["why"])
        
        # Fill in all placeholders
        format_args = {}
        
        if "{i}" in template:
            format_args["i"] = last_changed_object
        
        if "{direction}" in template:
            format_args["direction"] = self.rng.choice(["increase", "decrease", "change"])
        
        if "{state}" in template:
            format_args["state"] = self.rng.choice(["large", "small", "high", "low"])
            # Also need object reference if not already provided
            if "i" not in format_args:
                format_args["i"] = last_changed_object
        
        # Format template with all available arguments
        try:
            instruction = template.format(**format_args)
        except KeyError:
            # If template has unexpected placeholders, use a safe fallback
            instruction = f"why did object {last_changed_object} change?"
        
        return instruction
    
    def _generate_counterfactual_instruction(
        self,
        env: Track1CausalEnv,
        last_changed_object: Optional[int]
    ) -> tuple:
        """Generate counterfactual question."""
        template = self.rng.choice(self.templates["counterfactual"])
        
        # Generate hypothetical action
        u_type, args = env.sample_action()
        
        # Fill template
        if "{i}" in template and "{j}" in template:
            instruction = template.format(i=args.get("i", 0), j=args.get("j", 1))
        elif "{i}" in template:
            instruction = template.format(i=args.get("i", 0))
        else:
            instruction = template
        
        return instruction, u_type, args
    
    def _generate_planning_instruction(
        self,
        env: Track1CausalEnv,
        obs: Dict[str, np.ndarray]
    ) -> tuple:
        """Generate strategic planning instruction."""
        template = self.rng.choice(self.templates["planning"])
        
        if "all" in template.lower() or "everything" in template.lower():
            # Multi-object actions (simplified to single action)
            if "connect" in template.lower():
                u_type = 4  # SetRelOn
                i = self.rng.integers(0, env.N)
                j = self.rng.integers(0, env.N - 1)
                if j >= i:
                    j += 1
                args = {"r": 0, "i": i, "j": j}
                instruction = f"connect object {i} to object {j}"  # Simplified
            elif "clear" in template.lower() or "disconnect" in template.lower():
                u_type = 5  # SetRelOff
                i = self.rng.integers(0, env.N)
                j = self.rng.integers(0, env.N - 1)
                if j >= i:
                    j += 1
                args = {"r": 0, "i": i, "j": j}
                instruction = f"disconnect object {i} from object {j}"
            else:
                u_type, args = env.sample_action()
                instruction = template
        elif "largest" in template.lower():
            # Find largest object
            largest_obj = np.argmax(obs["X"].sum(axis=1))
            u_type = 0  # IncreaseAttr
            args = {"i": int(largest_obj), "a": 0}
            instruction = f"grow the largest object ({largest_obj})"
        elif "{i}" in template:
            obj = self.rng.integers(0, env.N)
            u_type = 0  # IncreaseAttr
            args = {"i": obj, "a": 0}
            instruction = template.format(i=obj)
        else:
            u_type, args = env.sample_action()
            instruction = template
        
        return instruction, u_type, args
    
    def _generate_recovery_instruction(self) -> str:
        """Generate recovery/undo instruction."""
        return self.rng.choice(self.templates["recovery"])
    
    def _generate_conditional_instruction(
        self,
        env: Track1CausalEnv,
        obs: Dict[str, np.ndarray]
    ) -> tuple:
        """Generate conditional instruction."""
        template = self.rng.choice(self.templates["conditional"])
        
        # Check condition (simplified: check if object is "large" = high attribute)
        obj = self.rng.integers(0, env.N)
        is_large = obs["X"][obj, 0] > 0.6  # Threshold for "large"
        
        if "if" in template.lower() and is_large:
            # Condition met: execute action
            if "connect" in template.lower():
                other_obj = self.rng.integers(0, env.N - 1)
                if other_obj >= obj:
                    other_obj += 1
                u_type = 4  # SetRelOn
                args = {"r": 0, "i": obj, "j": other_obj}
                instruction = template.format(i=obj, j=other_obj)
            elif "increase" in template.lower():
                u_type = 0  # IncreaseAttr
                args = {"i": obj, "a": 0}
                instruction = template.format(i=obj)
            else:
                u_type, args = env.sample_action()
                instruction = template
        else:
            # Condition not met
            u_type, args = None, {}
            instruction = template.format(i=obj, j=1) if "{j}" in template else template.format(i=obj)
        
        return instruction, u_type, args
    
    def _generate_meta_instruction(self) -> str:
        """Generate meta-reasoning query."""
        return self.rng.choice(self.templates["meta"])
    
    def _describe_current_state(self, obs: Dict[str, np.ndarray]) -> str:
        """Describe current world state."""
        # Simple description
        n_connected = int(obs["Rel"].sum())
        return f"World has {obs['X'].shape[0]} objects, {n_connected} connections"
    
    def _explain_last_change(self, last_action: Optional[tuple]) -> str:
        """Explain the last action's effect."""
        if last_action is None:
            return "No previous action to explain"
        u_type, args = last_action
        action_names = ["IncreaseAttr", "DecreaseAttr", "SetAttrToward", "ToggleRel",
                       "SetRelOn", "SetRelOff", "SwapObjects", "NoiseBurstAttr"]
        return f"Last action was {action_names[u_type]} on object {args.get('i', 'unknown')}"
    
    def _predict_consequence(
        self,
        u_type: int,
        args: Dict[str, int],
        obs: Dict[str, np.ndarray],
        env: Track1CausalEnv
    ) -> str:
        """Predict consequence of hypothetical action."""
        try:
            # Simulate step
            obs_pred = env.simulate_step(u_type, args)
            # Simple description
            return f"Action would modify object {args.get('i', 'unknown')}"
        except:
            return "Cannot predict consequence"
    
    def _summarize_beliefs(
        self,
        action_history: List[tuple],
        obs: Dict[str, np.ndarray]
    ) -> str:
        """Summarize agent's beliefs/understanding."""
        n_actions = len(action_history)
        n_connected = int(obs["Rel"].sum())
        return f"Performed {n_actions} actions, {n_connected} connections exist"
    
    def _generate_reasoning(
        self,
        instruction: str,
        consequence: str
    ) -> str:
        """Generate simple reasoning from instruction and consequence."""
        return f"Executed: {instruction}. Result: {consequence}"
    
    def generate_corpus(
        self,
        n_episodes: int = 10000,
        save_path: Optional[str] = None
    ) -> List[MultiTurnConversation]:
        """
        Generate corpus of multi-turn conversations.
        
        Args:
            n_episodes: Number of conversation episodes
            save_path: Optional path to save corpus
            
        Returns:
            List of MultiTurnConversation instances
        """
        print(f"Generating {n_episodes} multi-turn conversation episodes...")
        
        conversations = []
        for i in range(n_episodes):
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{n_episodes} episodes...")
            
            conv = self.generate_conversation()
            conversations.append(conv)
        
        print(f"Generated {len(conversations)} conversations")
        
        # Save if path provided
        if save_path:
            self.save_corpus(conversations, save_path)
        
        return conversations
    
    def save_corpus(
        self,
        conversations: List[MultiTurnConversation],
        path: str
    ) -> None:
        """Save corpus to JSON file."""
        data = [conv.to_dict() for conv in conversations]
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved corpus to {path} ({len(conversations)} conversations)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate Multi-Turn Conversation Corpus")
    parser.add_argument("--size", type=int, default=10000, help="Number of conversation episodes")
    parser.add_argument("--output", type=str, default="data/multi_turn.json", help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Create generator
    generator = MultiTurnCorpusGenerator(seed=args.seed)
    
    # Generate corpus
    conversations = generator.generate_corpus(n_episodes=args.size, save_path=args.output)
    
    print(f"\nCorpus generation complete!")
    print(f"  Episodes: {len(conversations)}")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()

