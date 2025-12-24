"""
Language-Aware Track 1 Agent

Extends Track1Agent with language perception, generation, and instruction following.
Non-disruptive extension that maintains backward compatibility.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass

from .agent import Track1Agent, AgentConfig
from .modules.language_encoder import MultimodalEncoder, LanguageEmbedder
from .modules.language_decoder import GroundedLanguageDecoder
from .modules.instruction_parser import InstructionParser
from .modules.energy import EnergyComputer, EnergyWeights
from .settle import SettlingLoop, SettleConfig


@dataclass
class LanguageAgentConfig(AgentConfig):
    """Extended configuration with language settings."""
    # Language dimensions
    lang_dim: int = 64
    lang_hidden: int = 128
    lang_max_len: int = 32
    
    # Feature flags
    use_language_perception: bool = True
    use_language_generation: bool = True
    use_instruction_parsing: bool = True
    
    # Generation settings
    generation_temperature: float = 1.0
    generation_greedy: bool = True


class LanguageAgent(Track1Agent):
    """
    Language-aware agent that extends Track1Agent.
    
    New capabilities:
    1. Multimodal perception: encode language + symbolic observations
    2. Grounded generation: describe states and interventions
    3. Instruction following: parse language to interventions
    """
    
    def __init__(self, config: Optional[LanguageAgentConfig] = None):
        """
        Initialize language-aware agent.
        
        Args:
            config: Language agent configuration
        """
        # Use default LanguageAgentConfig if not provided
        if config is None:
            config = LanguageAgentConfig()
        
        # Initialize base agent
        super().__init__(config)
        
        self.lang_config = config
        
        # Initialize language modules
        self._init_language_modules()
    
    def _init_language_modules(self) -> None:
        """Initialize language-specific modules."""
        cfg = self.lang_config
        
        if cfg.use_language_perception:
            # Multimodal encoder (replaces or augments base encoder)
            self.multimodal_encoder = MultimodalEncoder(
                n_objects=cfg.n_objects,
                n_attributes=cfg.n_attributes,
                n_relations=cfg.n_relations,
                latent_dim=cfg.latent_dim,
                lang_dim=cfg.lang_dim,
                hidden_dim=cfg.lang_hidden,
                seed=cfg.seed + 100,
            )
        
        if cfg.use_language_generation:
            # Grounded decoder for language output
            self.language_decoder = GroundedLanguageDecoder(
                latent_dim=cfg.latent_dim,
                lang_dim=cfg.lang_dim,
                hidden_dim=cfg.lang_hidden,
                max_len=cfg.lang_max_len,
                seed=cfg.seed + 101,
            )
        
        if cfg.use_instruction_parsing:
            # Instruction parser for language-to-action
            self.instruction_parser = InstructionParser(
                n_objects=cfg.n_objects,
                n_attributes=cfg.n_attributes,
                n_relations=cfg.n_relations,
                latent_dim=cfg.latent_dim,
                use_neural=True,
                seed=cfg.seed + 102,
            )
    
    def perceive_multimodal(
        self,
        obs: Dict[str, np.ndarray],
        u_type: int,
        language: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process observation with optional language context.
        
        Args:
            obs: Environment observation
            u_type: Active intervention type
            language: Optional language description
            
        Returns:
            (S_settled, info) tuple
        """
        # Add language to observation if provided
        obs_with_lang = {**obs}
        if language is not None:
            obs_with_lang["language"] = language
        
        # Use multimodal encoder if available
        if hasattr(self, 'multimodal_encoder') and self.lang_config.use_language_perception:
            enc_obs = self.multimodal_encoder.encode(obs_with_lang)
        else:
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
    
    def describe_state(
        self,
        obs: Optional[Dict[str, np.ndarray]] = None,
        S: Optional[np.ndarray] = None
    ) -> str:
        """
        Generate natural language description of current state.
        
        Args:
            obs: Optional observation (for template-based description)
            S: Optional latent state (uses self.S if not provided)
            
        Returns:
            Description string
        """
        if not hasattr(self, 'language_decoder'):
            return "Language generation not enabled"
        
        S_use = S if S is not None else self.S
        
        if obs is not None:
            # Template-based description
            return self.language_decoder.generate_state_description(obs, S_use)
        else:
            # Neural generation from settled state
            return self.language_decoder.generate(
                S_use,
                temperature=self.lang_config.generation_temperature,
                greedy=self.lang_config.generation_greedy
            )
    
    def describe_intervention(
        self,
        u_type: int,
        args: Dict[str, int],
        obs_before: Optional[Dict[str, np.ndarray]] = None,
        obs_after: Optional[Dict[str, np.ndarray]] = None
    ) -> str:
        """
        Generate description of an intervention and its effects.
        
        Args:
            u_type: Intervention type
            args: Intervention arguments
            obs_before: Observation before intervention
            obs_after: Observation after intervention
            
        Returns:
            Description string
        """
        if not hasattr(self, 'language_decoder'):
            return "Language generation not enabled"
        
        return self.language_decoder.generate_intervention_description(
            u_type, args, obs_before, obs_after
        )
    
    def parse_instruction(
        self,
        instruction: str
    ) -> Tuple[int, Dict[str, int], float]:
        """
        Parse natural language instruction to intervention.
        
        Args:
            instruction: Natural language instruction
            
        Returns:
            (u_type, args, confidence) tuple
        """
        if not hasattr(self, 'instruction_parser'):
            # Fallback to random action
            return 0, {"i": 0, "a": 0}, 0.0
        
        return self.instruction_parser.parse(instruction)
    
    def follow_instruction(
        self,
        instruction: str,
        obs: Dict[str, np.ndarray]
    ) -> Tuple[int, Dict[str, int], float]:
        """
        Follow a natural language instruction.
        
        First parses the instruction, then validates against current state.
        
        Args:
            instruction: Natural language instruction
            obs: Current observation
            
        Returns:
            (u_type, args, confidence) tuple
        """
        # Parse instruction
        u_type, args, confidence = self.parse_instruction(instruction)
        
        # Perceive with instruction context
        if self.lang_config.use_language_perception:
            self.perceive_multimodal(obs, u_type, language=instruction)
        
        return u_type, args, confidence
    
    def select_action_with_language(
        self,
        obs: Dict[str, np.ndarray],
        rng: np.random.Generator,
        instruction: Optional[str] = None
    ) -> Tuple[int, Dict[str, int], str]:
        """
        Select action, optionally guided by language instruction.
        
        Args:
            obs: Current observation
            rng: Random generator
            instruction: Optional language instruction
            
        Returns:
            (u_type, args, explanation) tuple
        """
        if instruction is not None:
            u_type, args, conf = self.follow_instruction(instruction, obs)
            explanation = f"Following instruction with confidence {conf:.2f}"
        else:
            # Default random action
            u_type, args = self.select_action(obs, rng)
            explanation = "Random exploration"
        
        return u_type, args, explanation
    
    def process_transition_with_language(
        self,
        obs_t: Dict[str, np.ndarray],
        obs_tp1: Dict[str, np.ndarray],
        u_type: int,
        args: Dict[str, int],
        instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process transition with language context.
        
        Args:
            obs_t: Current observation
            obs_tp1: Next observation
            u_type: Intervention type
            args: Intervention arguments
            instruction: Optional instruction context
            
        Returns:
            Extended result dict with language descriptions
        """
        # Add language to observations if provided
        obs_t_lang = {**obs_t}
        obs_tp1_lang = {**obs_tp1}
        
        if instruction is not None:
            obs_t_lang["language"] = instruction
        
        # Process transition (base method)
        result = self.process_transition(obs_t, obs_tp1, u_type, args)
        
        # Add language outputs
        if hasattr(self, 'language_decoder'):
            result["state_description"] = self.describe_state(obs_tp1, result["S_tp1_nudged"])
            result["intervention_description"] = self.describe_intervention(
                u_type, args, obs_t, obs_tp1
            )
        
        return result
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get all trainable parameters including language modules."""
        params = super().get_parameters()
        
        if hasattr(self, 'multimodal_encoder'):
            params["multimodal_encoder"] = self.multimodal_encoder.get_parameters()
        
        if hasattr(self, 'language_decoder'):
            params["language_decoder"] = self.language_decoder.get_parameters()
        
        if hasattr(self, 'instruction_parser'):
            params["instruction_parser"] = self.instruction_parser.get_parameters()
        
        return params
    
    def get_language_stats(self) -> Dict[str, Any]:
        """Get language module statistics for logging."""
        stats = {}
        
        if hasattr(self, 'multimodal_encoder'):
            stats["lang_encoder_active"] = True
        
        if hasattr(self, 'language_decoder'):
            stats["lang_decoder_active"] = True
        
        if hasattr(self, 'instruction_parser'):
            stats["instruction_parser_active"] = True
        
        return stats


def demo_language_agent():
    """Demonstrate language agent capabilities."""
    from env.track1_env import Track1CausalEnv
    
    print("=" * 60)
    print("Language Agent Demo")
    print("=" * 60)
    
    # Create environment and agent
    env = Track1CausalEnv()
    config = LanguageAgentConfig()
    agent = LanguageAgent(config)
    
    # Reset
    obs = env.reset(seed=42)
    agent.reset()
    
    rng = np.random.default_rng(42)
    
    # 1. Multimodal perception
    print("\n1. MULTIMODAL PERCEPTION")
    print("-" * 40)
    language_context = "object 0 is large and red"
    S, info = agent.perceive_multimodal(obs, u_type=0, language=language_context)
    print(f"Input language: '{language_context}'")
    print(f"Settled state norm: {np.linalg.norm(S):.4f}")
    
    # 2. State description
    print("\n2. STATE DESCRIPTION")
    print("-" * 40)
    description = agent.describe_state(obs)
    print(f"Generated: '{description}'")
    
    # 3. Instruction parsing
    print("\n3. INSTRUCTION PARSING")
    print("-" * 40)
    instructions = [
        "increase object 0 size",
        "swap object 1 and object 2",
        "make object 0 support object 1",
        "add noise to object 3",
    ]
    
    for inst in instructions:
        u_type, args, conf = agent.parse_instruction(inst)
        int_names = ["IncreaseAttr", "DecreaseAttr", "SetAttrToward",
                    "ToggleRel", "SetRelOn", "SetRelOff",
                    "SwapObjects", "NoiseBurstAttr"]
        print(f"  '{inst}'")
        print(f"    -> {int_names[u_type]}({args}) [conf={conf:.2f}]")
    
    # 4. Follow instruction
    print("\n4. FOLLOW INSTRUCTION")
    print("-" * 40)
    instruction = "increase object 1 color"
    u_type, args, conf = agent.follow_instruction(instruction, obs)
    print(f"Instruction: '{instruction}'")
    print(f"Action: type={u_type}, args={args}, conf={conf:.2f}")
    
    # 5. Full transition with language
    print("\n5. TRANSITION WITH LANGUAGE")
    print("-" * 40)
    obs_before = obs
    obs_after, _, _ = env.step(u_type, args)
    
    result = agent.process_transition_with_language(
        obs_before, obs_after, u_type, args, instruction
    )
    
    print(f"Prediction error: {result['prediction_error']:.4f}")
    if "intervention_description" in result:
        print(f"Intervention description: '{result['intervention_description']}'")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo_language_agent()

