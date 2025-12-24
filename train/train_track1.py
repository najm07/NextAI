"""
Track 1 Training Loop

Implements EP-style two-phase learning with staged plasticity.
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
from agent.agent import Track1Agent, AgentConfig


@dataclass
class TrainConfig:
    """Training configuration."""
    # Learning rates
    lr_encoder: float = 1e-3
    lr_anchors: float = 1e-3
    lr_stc: float = 5e-4
    lr_gates: float = 5e-4
    lr_temporal: float = 5e-4
    lr_intent: float = 1e-4
    
    # Staged plasticity thresholds
    stage1_steps: int = 2000   # Anchors + encoder only
    stage2_steps: int = 5000   # + STCs
    stage3_steps: int = 8000   # + Gates + Temporal
    # Stage 4: all including Intent
    
    # Training
    total_steps: int = 10000
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: int = 2000
    
    # EP parameters
    nudge_strength: float = 0.5
    
    # Seed
    seed: int = 42


class Trainer:
    """
    Track 1 Trainer with EP-style learning and staged plasticity.
    """
    
    def __init__(
        self,
        env: Track1CausalEnv,
        agent: Track1Agent,
        config: Optional[TrainConfig] = None
    ):
        """
        Initialize trainer.
        
        Args:
            env: Track 1 environment
            agent: Track 1 agent
            config: Training configuration
        """
        self.env = env
        self.agent = agent
        self.config = config or TrainConfig()
        
        self.rng = np.random.default_rng(self.config.seed)
        
        # Training state
        self.global_step = 0
        self.episode = 0
        
        # Metrics
        self.metrics = {
            "prediction_error": [],
            "energy": [],
            "surprise": [],
            "stc_violation": [],
        }
        
        # Running averages for logging
        self.running_pred_error = 0.0
        self.running_energy = 0.0
    
    def get_current_stage(self) -> int:
        """Get current training stage based on step count."""
        if self.global_step < self.config.stage1_steps:
            return 1
        elif self.global_step < self.config.stage2_steps:
            return 2
        elif self.global_step < self.config.stage3_steps:
            return 3
        else:
            return 4
    
    def get_active_modules(self) -> Dict[str, bool]:
        """Get which modules are trainable in current stage."""
        stage = self.get_current_stage()
        
        return {
            "encoder": True,  # Always train
            "anchors": True,  # Always train
            "stc": stage >= 2,
            "gates": stage >= 3,
            "temporal": stage >= 3,
            "intent": stage >= 4,
        }
    
    def train_step(
        self,
        obs_t: Dict[str, np.ndarray],
        obs_tp1: Dict[str, np.ndarray],
        u_type: int,
        args: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Execute one training step with EP-style learning.
        
        Args:
            obs_t: Current observation
            obs_tp1: Next observation
            u_type: Intervention type
            args: Intervention arguments
            
        Returns:
            Dict with training metrics
        """
        # Get free and nudged equilibria
        result = self.agent.process_transition(obs_t, obs_tp1, u_type, args)
        
        S_t_free = result["S_t_free"]
        S_tp1_free = result["S_tp1_free"]
        S_tp1_nudged = result["S_tp1_nudged"]
        enc_tp1 = result["enc_tp1"]
        pred_error = result["prediction_error"]
        
        # Get active modules
        active = self.get_active_modules()
        
        # === Parameter updates based on EP principle ===
        # Update proportional to difference between free and nudged equilibria
        
        # 1. Encoder update (always)
        if active["encoder"]:
            # Update encoder to reduce error at nudged state
            enc_grads = self.agent.encoder.compute_gradients(
                obs_tp1, S_tp1_nudged, weight=1.0
            )
            for name, grad in enc_grads.items():
                param = getattr(self.agent.encoder, name)
                param -= self.config.lr_encoder * grad
        
        # 2. Anchors update (always)
        if active["anchors"]:
            g_stc, g_anchor = self.agent.gates.compute_gates(S_tp1_free, u_type)
            self.agent.anchors.update_anchors(
                S_tp1_free, S_tp1_nudged, g_anchor,
                lr=self.config.lr_anchors
            )
        
        # 3. STC update (stage 2+)
        if active["stc"]:
            g_stc, _ = self.agent.gates.compute_gates(S_t_free, u_type)
            self.agent.stc.update_parameters(
                S_t_free, S_tp1_free, S_tp1_nudged, u_type, g_stc,
                lr=self.config.lr_stc
            )
        
        # 4. Gates update (stage 3+)
        if active["gates"]:
            # Target gates: ones that were useful for reducing error
            # Heuristic: gates that correspond to large free/nudged difference
            diff = np.abs(S_tp1_nudged - S_tp1_free)
            target_g_stc = np.ones(self.agent.config.stc_factors_per_type) * 0.5
            target_g_anchor = np.ones(self.agent.config.n_anchors) * 0.5
            
            self.agent.gates.update_parameters(
                S_tp1_free, S_tp1_nudged, u_type,
                target_g_stc, target_g_anchor,
                lr=self.config.lr_gates
            )
        
        # 5. Temporal update (stage 3+)
        if active["temporal"]:
            # Update based on settling dynamics (need history)
            # For now, simple update based on state variance
            pass
        
        # 6. Intent update (stage 4)
        if active["intent"]:
            self.agent.intent.update_intent(
                pred_error, S_tp1_nudged,
                lr=self.config.lr_intent
            )
            self.agent.intent.update_projection(
                S_tp1_free, S_tp1_nudged,
                lr=self.config.lr_intent
            )
        
        # Update running averages
        alpha = 0.01
        self.running_pred_error = (1 - alpha) * self.running_pred_error + alpha * pred_error
        
        # Compute energy for logging
        energy, _ = self.agent.energy_computer.compute_state_energy(
            S_tp1_nudged, obs_tp1, u_type
        )
        self.running_energy = (1 - alpha) * self.running_energy + alpha * energy
        
        self.global_step += 1
        
        return {
            "prediction_error": pred_error,
            "energy": energy,
            "stage": self.get_current_stage(),
        }
    
    def run_episode(self) -> Dict[str, float]:
        """
        Run one full episode of training.
        
        Returns:
            Episode statistics
        """
        obs = self.env.reset(seed=self.rng.integers(0, 2**31))
        self.agent.reset()
        
        episode_pred_error = []
        episode_energy = []
        done = False
        
        while not done:
            # Select action
            u_type, args = self.agent.select_action(obs, self.rng)
            
            # Step environment
            obs_next, done, info = self.env.step(u_type, args)
            
            # Train step
            metrics = self.train_step(obs, obs_next, u_type, args)
            
            episode_pred_error.append(metrics["prediction_error"])
            episode_energy.append(metrics["energy"])
            
            obs = obs_next
        
        self.episode += 1
        
        return {
            "mean_pred_error": float(np.mean(episode_pred_error)),
            "mean_energy": float(np.mean(episode_energy)),
            "episode_length": len(episode_pred_error),
        }
    
    def train(self, n_episodes: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Run full training loop.
        
        Args:
            n_episodes: Number of episodes (or run until total_steps)
            
        Returns:
            Training history
        """
        history = {
            "pred_error": [],
            "energy": [],
            "episode_length": [],
        }
        
        start_time = time.time()
        
        while self.global_step < self.config.total_steps:
            episode_stats = self.run_episode()
            
            history["pred_error"].append(episode_stats["mean_pred_error"])
            history["energy"].append(episode_stats["mean_energy"])
            history["episode_length"].append(episode_stats["episode_length"])
            
            # Logging
            if self.episode % 10 == 0:
                elapsed = time.time() - start_time
                print(
                    f"Episode {self.episode:4d} | "
                    f"Step {self.global_step:6d} | "
                    f"Stage {self.get_current_stage()} | "
                    f"PredErr: {self.running_pred_error:.4f} | "
                    f"Energy: {self.running_energy:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )
            
            if n_episodes and self.episode >= n_episodes:
                break
        
        return history
    
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "global_step": self.global_step,
            "episode": self.episode,
            "agent_params": self.agent.get_parameters(),
            "metrics": self.metrics,
        }
        np.save(path, checkpoint, allow_pickle=True)
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = np.load(path, allow_pickle=True).item()
        self.global_step = checkpoint["global_step"]
        self.episode = checkpoint["episode"]
        self.metrics = checkpoint["metrics"]
        # Note: would need to implement set_parameters for full restore


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main training entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Track 1 Agent")
    parser.add_argument("--config", type=str, default="configs/track1.yaml",
                        help="Path to config file")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override total training steps")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create environment
    env_cfg = EnvConfig(
        n_objects=config["environment"]["n_objects"],
        n_attributes=config["environment"]["n_attributes"],
        n_relations=config["environment"]["n_relations"],
        n_bins=config["environment"]["n_bins"],
        delta_attr=config["environment"]["delta_attr"],
        noise_attr=config["environment"]["noise_attr"],
        noise_burst_std=config["environment"]["noise_burst_std"],
        min_steps=config["environment"]["min_steps"],
        max_steps=config["environment"]["max_steps"],
        enable_relation_effects=config["environment"]["enable_relation_effects"],
        relation_alignment_strength=config["environment"]["relation_alignment_strength"],
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
        n_intervention_types=config["agent"]["n_intervention_types"],
        stc_factors_per_type=config["agent"]["stc_factors_per_type"],
        stc_subset_size=config["agent"]["stc_subset_size"],
        encoder_hidden=config["agent"]["encoder_hidden"],
        gate_hidden=config["agent"]["gate_hidden"],
        n_settle_steps=config["settling"]["n_steps"],
        settle_lr=config["settling"]["learning_rate"],
        settle_tolerance=config["settling"]["tolerance"],
        w_obs=config["settling"]["w_obs"],
        w_anchor=config["settling"]["w_anchor"],
        w_state_stc=config["settling"]["w_state_stc"],
        w_intent=config["settling"]["w_intent"],
        nudge_strength=config["learning"]["nudge_strength"],
        seed=args.seed or config["seed"],
    )
    agent = Track1Agent(agent_cfg)
    
    # Create trainer
    train_cfg = TrainConfig(
        lr_encoder=config["learning"]["lr_encoder"],
        lr_anchors=config["learning"]["lr_anchors"],
        lr_stc=config["learning"]["lr_stc"],
        lr_gates=config["learning"]["lr_gates"],
        lr_temporal=config["learning"]["lr_temporal"],
        lr_intent=config["learning"]["lr_intent"],
        stage1_steps=config["learning"]["stage1_steps"],
        stage2_steps=config["learning"]["stage2_steps"],
        stage3_steps=config["learning"]["stage3_steps"],
        total_steps=args.steps or config["learning"]["total_steps"],
        log_interval=config["learning"]["log_interval"],
        eval_interval=config["learning"]["eval_interval"],
        nudge_strength=config["learning"]["nudge_strength"],
        seed=args.seed or config["seed"],
    )
    trainer = Trainer(env, agent, train_cfg)
    
    # Train
    print("=" * 60)
    print("Track 1 (v0.1-A) Training")
    print("=" * 60)
    print(f"Environment: {env_cfg.n_objects} objects, {env_cfg.n_attributes} attrs, {env_cfg.n_relations} rels")
    print(f"Agent: dS={agent_cfg.latent_dim}, {agent_cfg.n_anchors} anchors")
    print(f"Training: {train_cfg.total_steps} steps, staged plasticity")
    print("=" * 60)
    
    history = trainer.train()
    
    print("\nTraining complete!")
    print(f"Final prediction error: {history['pred_error'][-1]:.4f}")
    print(f"Final energy: {history['energy'][-1]:.4f}")
    
    # Save checkpoint
    trainer.save_checkpoint("checkpoints/track1_final.npy")


if __name__ == "__main__":
    main()

