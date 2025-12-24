"""
Language-Aware Training Loop

Mixed training curriculum that integrates language with Track 1 core:
- Stage 1: Symbolic-only (existing Track 1)
- Stage 2: Symbolic + language perception
- Stage 3: Symbolic + language readout
- Stage 4: Full language-perception-action loop
"""

import numpy as np
import yaml
import time
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from env.track1_env import Track1CausalEnv, EnvConfig
from agent.language_agent import LanguageAgent, LanguageAgentConfig
from data.grounded_corpus import GroundedCorpusGenerator, GroundedDataset, GroundedExample


@dataclass
class LanguageTrainConfig:
    """Language training configuration."""
    # Core learning rates
    lr_encoder: float = 0.001
    lr_anchors: float = 0.001
    lr_stc: float = 0.0005
    lr_gates: float = 0.0005
    lr_temporal: float = 0.0005
    lr_intent: float = 0.0001
    
    # Language learning rates
    lr_lang_encoder: float = 0.001
    lr_lang_decoder: float = 0.001
    lr_instruction_parser: float = 0.001
    
    # Staged curriculum
    stage1_steps: int = 2000    # Symbolic only
    stage2_steps: int = 4000    # + Language perception
    stage3_steps: int = 6000    # + Language readout
    # Stage 4: Full language loop
    
    # Training
    total_steps: int = 10000
    batch_size: int = 16
    log_interval: int = 100
    eval_interval: int = 500
    
    # Corpus settings
    corpus_size: int = 10000
    
    # EP parameters
    nudge_strength: float = 0.5
    
    # Seed
    seed: int = 42


class LanguageTrainer:
    """
    Language-aware trainer with mixed curriculum.
    """
    
    def __init__(
        self,
        env: Track1CausalEnv,
        agent: LanguageAgent,
        config: Optional[LanguageTrainConfig] = None
    ):
        """Initialize trainer."""
        self.env = env
        self.agent = agent
        self.config = config or LanguageTrainConfig()
        
        self.rng = np.random.default_rng(self.config.seed)
        
        # Training state
        self.global_step = 0
        self.episode = 0
        
        # Generate grounded corpus
        print("Generating grounded corpus...")
        self.corpus_generator = GroundedCorpusGenerator(env, seed=self.config.seed)
        corpus = self.corpus_generator.generate_corpus(self.config.corpus_size)
        self.dataset = GroundedDataset(corpus, seed=self.config.seed)
        print(f"Corpus size: {len(self.dataset)} examples")
        
        # Metrics
        self.running_pred_error = 0.0
        self.running_energy = 0.0
        self.running_lang_loss = 0.0
    
    def get_current_stage(self) -> int:
        """Get current training stage."""
        if self.global_step < self.config.stage1_steps:
            return 1
        elif self.global_step < self.config.stage2_steps:
            return 2
        elif self.global_step < self.config.stage3_steps:
            return 3
        else:
            return 4
    
    def get_stage_name(self) -> str:
        """Get human-readable stage name."""
        stage = self.get_current_stage()
        names = {
            1: "Symbolic",
            2: "Symbolic+LangPerception",
            3: "Symbolic+LangReadout",
            4: "Full Language Loop"
        }
        return names.get(stage, "Unknown")
    
    def train_step_symbolic(
        self,
        obs_t: Dict[str, np.ndarray],
        obs_tp1: Dict[str, np.ndarray],
        u_type: int,
        args: Dict[str, int]
    ) -> Dict[str, float]:
        """Train step using only symbolic observations (Stage 1)."""
        # Process transition
        result = self.agent.process_transition(obs_t, obs_tp1, u_type, args)
        
        pred_error = result["prediction_error"]
        
        # Update encoder
        enc_grads = self.agent.encoder.compute_gradients(
            obs_tp1, result["S_tp1_nudged"], weight=1.0
        )
        for name, grad in enc_grads.items():
            param = getattr(self.agent.encoder, name)
            param -= self.config.lr_encoder * grad
        
        # Update anchors
        g_stc, g_anchor = self.agent.gates.compute_gates(
            result["S_tp1_free"], u_type
        )
        self.agent.anchors.update_anchors(
            result["S_tp1_free"], result["S_tp1_nudged"], g_anchor,
            lr=self.config.lr_anchors
        )
        
        self.global_step += 1
        
        return {"prediction_error": pred_error}
    
    def train_step_with_perception(
        self,
        example: GroundedExample
    ) -> Dict[str, float]:
        """Train step with language perception (Stage 2)."""
        obs_t = {"X": example.obs_X, "Rel": example.obs_Rel, "mask": None}
        obs_tp1 = {"X": example.obs_X_after, "Rel": example.obs_Rel_after, "mask": None}
        
        # Add language context
        obs_t["language"] = example.description
        obs_tp1["language"] = example.consequence
        
        # Perceive with language
        self.agent.perceive_multimodal(obs_t, example.u_type, example.description)
        
        # Process transition
        result = self.agent.process_transition(obs_t, obs_tp1, example.u_type, example.args)
        
        pred_error = result["prediction_error"]
        
        # Update multimodal encoder
        if hasattr(self.agent, 'multimodal_encoder'):
            enc_grads = self.agent.multimodal_encoder.compute_gradients(
                obs_tp1, result["S_tp1_nudged"], weight=1.0
            )
            for name, grad in enc_grads.items():
                if hasattr(self.agent.multimodal_encoder, name):
                    param = getattr(self.agent.multimodal_encoder, name)
                    param -= self.config.lr_lang_encoder * grad
        
        self.global_step += 1
        
        return {"prediction_error": pred_error, "stage": 2}
    
    def train_step_with_readout(
        self,
        example: GroundedExample
    ) -> Dict[str, float]:
        """Train step with language readout (Stage 3)."""
        # First do perception training
        metrics = self.train_step_with_perception(example)
        
        # Then train language decoder
        if hasattr(self.agent, 'language_decoder'):
            # Train on consequence description
            decoder_metrics = self.agent.language_decoder.train_step(
                self.agent.S,  # Current settled state
                example.consequence,
                lr=self.config.lr_lang_decoder
            )
            metrics["decoder_loss"] = decoder_metrics["loss"]
        
        metrics["stage"] = 3
        
        return metrics
    
    def train_step_full_language(
        self,
        example: GroundedExample
    ) -> Dict[str, float]:
        """Train step with full language loop (Stage 4)."""
        # Train readout
        metrics = self.train_step_with_readout(example)
        
        # Train instruction parser
        if hasattr(self.agent, 'instruction_parser'):
            parser_metrics = self.agent.instruction_parser.train_step(
                example.instruction,
                example.u_type,
                example.args,
                lr=self.config.lr_instruction_parser
            )
            metrics["parser_loss"] = parser_metrics["total_loss"]
        
        metrics["stage"] = 4
        
        return metrics
    
    def run_episode(self) -> Dict[str, float]:
        """Run one training episode."""
        obs = self.env.reset(seed=self.rng.integers(0, 2**31))
        self.agent.reset()
        
        episode_metrics = []
        done = False
        stage = self.get_current_stage()
        
        while not done:
            if stage == 1:
                # Pure symbolic
                u_type, args = self.agent.select_action(obs, self.rng)
                obs_next, done, _ = self.env.step(u_type, args)
                metrics = self.train_step_symbolic(obs, obs_next, u_type, args)
            else:
                # Get grounded example for language training
                batch = self.dataset.get_batch(1)
                example = batch[0]
                
                if stage == 2:
                    metrics = self.train_step_with_perception(example)
                elif stage == 3:
                    metrics = self.train_step_with_readout(example)
                else:
                    metrics = self.train_step_full_language(example)
                
                # Still step environment for episode structure
                u_type, args = self.agent.select_action(obs, self.rng)
                obs_next, done, _ = self.env.step(u_type, args)
            
            episode_metrics.append(metrics)
            obs = obs_next
            stage = self.get_current_stage()
        
        self.episode += 1
        
        # Aggregate
        avg_pred_error = np.mean([m.get("prediction_error", 0) for m in episode_metrics])
        
        return {
            "mean_pred_error": float(avg_pred_error),
            "episode_length": len(episode_metrics),
            "stage": stage,
        }
    
    def train_on_corpus(self, n_batches: int = 100) -> Dict[str, float]:
        """Train directly on grounded corpus."""
        stage = self.get_current_stage()
        
        total_loss = 0.0
        n_examples = 0
        
        for batch in self.dataset.iter_batches(self.config.batch_size):
            for example in batch:
                self.agent.reset()
                
                if stage == 2:
                    metrics = self.train_step_with_perception(example)
                elif stage == 3:
                    metrics = self.train_step_with_readout(example)
                else:
                    metrics = self.train_step_full_language(example)
                
                total_loss += metrics.get("prediction_error", 0)
                n_examples += 1
                
                if n_examples >= n_batches * self.config.batch_size:
                    break
            
            if n_examples >= n_batches * self.config.batch_size:
                break
        
        return {
            "avg_loss": total_loss / max(n_examples, 1),
            "n_examples": n_examples,
        }
    
    def train(self) -> Dict[str, List[float]]:
        """Run full training loop."""
        history = {
            "pred_error": [],
            "stage": [],
        }
        
        start_time = time.time()
        
        while self.global_step < self.config.total_steps:
            stage = self.get_current_stage()
            
            if stage == 1:
                # Run episode-based training
                episode_stats = self.run_episode()
                history["pred_error"].append(episode_stats["mean_pred_error"])
            else:
                # Mix corpus training with episodes
                corpus_metrics = self.train_on_corpus(n_batches=10)
                history["pred_error"].append(corpus_metrics["avg_loss"])
            
            history["stage"].append(stage)
            
            # Update running averages
            alpha = 0.01
            self.running_pred_error = (1 - alpha) * self.running_pred_error + alpha * history["pred_error"][-1]
            
            # Logging
            if self.global_step % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"Step {self.global_step:6d} | "
                    f"Stage {stage} ({self.get_stage_name():20s}) | "
                    f"PredErr: {self.running_pred_error:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )
        
        return history
    
    def evaluate_language(self) -> Dict[str, Any]:
        """Evaluate language capabilities."""
        results = {}
        
        # Sample test examples
        test_examples = self.dataset.get_batch(20)
        
        # 1. Instruction parsing accuracy
        if hasattr(self.agent, 'instruction_parser'):
            correct_type = 0
            correct_args = 0
            
            for ex in test_examples:
                u_type, args, conf = self.agent.parse_instruction(ex.instruction)
                
                if u_type == ex.u_type:
                    correct_type += 1
                    if args.get("i") == ex.args.get("i"):
                        correct_args += 1
            
            results["instruction_type_accuracy"] = correct_type / len(test_examples)
            results["instruction_arg_accuracy"] = correct_args / len(test_examples)
        
        # 2. Description quality (basic check)
        if hasattr(self.agent, 'language_decoder'):
            obs = {"X": test_examples[0].obs_X, "Rel": test_examples[0].obs_Rel, "mask": None}
            self.agent.reset()
            self.agent.perceive_multimodal(obs, 0)
            
            description = self.agent.describe_state(obs)
            results["sample_description"] = description
            results["description_length"] = len(description.split())
        
        return results


def main():
    """Main training entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Language-Aware Track 1 Agent")
    parser.add_argument("--config", type=str, default="configs/track1.yaml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--corpus_size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
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
    
    # Create language-aware agent
    agent_cfg = LanguageAgentConfig(
        n_objects=config["environment"]["n_objects"],
        n_attributes=config["environment"]["n_attributes"],
        n_relations=config["environment"]["n_relations"],
        latent_dim=config["agent"]["latent_dim"],
        intent_dim=config["agent"]["intent_dim"],
        n_anchors=config["agent"]["n_anchors"],
        seed=args.seed,
        use_language_perception=True,
        use_language_generation=True,
        use_instruction_parsing=True,
    )
    agent = LanguageAgent(agent_cfg)
    
    # Create trainer
    train_cfg = LanguageTrainConfig(
        total_steps=args.steps or 10000,
        corpus_size=args.corpus_size,
        seed=args.seed,
    )
    trainer = LanguageTrainer(env, agent, train_cfg)
    
    # Train
    print("=" * 60)
    print("Language-Aware Track 1 Training")
    print("=" * 60)
    print(f"Stages: Symbolic -> +Perception -> +Readout -> Full Loop")
    print(f"Corpus: {train_cfg.corpus_size} grounded examples")
    print("=" * 60)
    
    history = trainer.train()
    
    # Evaluate language capabilities
    print("\nEvaluating language capabilities...")
    eval_results = trainer.evaluate_language()
    
    print("\nLanguage Evaluation Results:")
    for k, v in eval_results.items():
        print(f"  {k}: {v}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

