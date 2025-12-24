"""
Grounded Language Decoder Module

Generates natural language descriptions from settled latent states.
Trained on grounded data to maintain causal accuracy.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from .language_encoder import Vocabulary


class GroundedLanguageDecoder:
    """
    Decoder that generates language from settled states.
    
    Key principle: NOT trained end-to-end, but on grounded state-description pairs.
    This maintains causal grounding and prevents hallucination.
    
    Architecture:
        S_settled → readout MLP → language embedding → token generation
    """
    
    def __init__(
        self,
        latent_dim: int,
        lang_dim: int = 64,
        hidden_dim: int = 128,
        max_len: int = 32,
        seed: int = 42
    ):
        """
        Initialize decoder.
        
        Args:
            latent_dim: Latent state dimension (dS)
            lang_dim: Language embedding dimension
            hidden_dim: Hidden layer dimension
            max_len: Maximum output sequence length
            seed: Random seed
        """
        self.dS = latent_dim
        self.lang_dim = lang_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        
        self.rng = np.random.default_rng(seed)
        self.vocab = Vocabulary()
        
        # State readout: S → language embedding
        scale1 = np.sqrt(2.0 / (latent_dim + hidden_dim))
        self.W_readout1 = self.rng.normal(0, scale1, (latent_dim, hidden_dim)).astype(np.float32)
        self.b_readout1 = np.zeros(hidden_dim, dtype=np.float32)
        
        scale2 = np.sqrt(2.0 / (hidden_dim + lang_dim))
        self.W_readout2 = self.rng.normal(0, scale2, (hidden_dim, lang_dim)).astype(np.float32)
        self.b_readout2 = np.zeros(lang_dim, dtype=np.float32)
        
        # Token predictor: embedding + context → next token logits
        vocab_size = self.vocab.vocab_size
        self.token_embeddings = self.rng.normal(0, 0.1, (vocab_size, lang_dim)).astype(np.float32)
        
        # RNN-like hidden state update (simple GRU-like)
        scale_h = np.sqrt(2.0 / (lang_dim * 2 + lang_dim))
        self.W_h = self.rng.normal(0, scale_h, (lang_dim * 2, lang_dim)).astype(np.float32)
        self.b_h = np.zeros(lang_dim, dtype=np.float32)
        
        # Output logits
        scale_out = np.sqrt(2.0 / (lang_dim + vocab_size))
        self.W_out = self.rng.normal(0, scale_out, (lang_dim, vocab_size)).astype(np.float32)
        self.b_out = np.zeros(vocab_size, dtype=np.float32)
        
        # Temperature for generation
        self.temperature = 1.0
    
    def readout(self, S: np.ndarray) -> np.ndarray:
        """
        Convert settled state to language embedding.
        
        Args:
            S: Settled latent state (dS,)
            
        Returns:
            Language embedding (lang_dim,)
        """
        h = np.maximum(0, S @ self.W_readout1 + self.b_readout1)
        lang_emb = np.tanh(h @ self.W_readout2 + self.b_readout2)
        return lang_emb.astype(np.float32)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / (np.sum(exp_x) + 1e-9)
    
    def generate(
        self,
        S_settled: np.ndarray,
        max_len: Optional[int] = None,
        temperature: Optional[float] = None,
        greedy: bool = True
    ) -> str:
        """
        Generate language description from settled state.
        
        Args:
            S_settled: Settled latent state
            max_len: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            greedy: If True, use greedy decoding; else sample
            
        Returns:
            Generated text string
        """
        max_len = max_len or self.max_len
        temperature = temperature or self.temperature
        
        # Get context from state
        context = self.readout(S_settled)
        
        # Start with BOS token
        current_token = self.vocab.bos_idx
        hidden = context.copy()
        
        generated_tokens = []
        
        for _ in range(max_len):
            # Get token embedding
            token_emb = self.token_embeddings[current_token]
            
            # Update hidden state
            combined = np.concatenate([token_emb, hidden])
            hidden = np.tanh(combined @ self.W_h + self.b_h)
            
            # Compute next token logits
            logits = hidden @ self.W_out + self.b_out
            logits = logits / temperature
            
            # Get next token
            if greedy:
                next_token = int(np.argmax(logits))
            else:
                probs = self._softmax(logits)
                next_token = int(self.rng.choice(len(probs), p=probs))
            
            # Stop at EOS
            if next_token == self.vocab.eos_idx:
                break
            
            # Skip special tokens
            if next_token not in [self.vocab.pad_idx, self.vocab.bos_idx]:
                generated_tokens.append(next_token)
            
            current_token = next_token
        
        # Decode to text
        words = [self.vocab.idx2word.get(t, "<UNK>") for t in generated_tokens]
        return " ".join(words)
    
    def generate_intervention_description(
        self,
        u_type: int,
        args: Dict[str, int],
        obs_before: Dict[str, np.ndarray],
        obs_after: Dict[str, np.ndarray]
    ) -> str:
        """
        Generate template-based intervention description.
        
        This is a more controlled generation mode using templates.
        
        Args:
            u_type: Intervention type
            args: Intervention arguments
            obs_before: Observation before intervention
            obs_after: Observation after intervention
            
        Returns:
            Description string
        """
        INT_NAMES = [
            "IncreaseAttr", "DecreaseAttr", "SetAttrToward",
            "ToggleRel", "SetRelOn", "SetRelOff",
            "SwapObjects", "NoiseBurstAttr"
        ]
        
        int_name = INT_NAMES[u_type] if 0 <= u_type < 8 else "Unknown"
        
        # Templates based on intervention type
        templates = {
            0: "object {i} attribute {a} increased",
            1: "object {i} attribute {a} decreased",
            2: "object {i} attribute {a} set toward bin {bin_id}",
            3: "relation {r} between object {i} and object {j} toggled",
            4: "relation {r} between object {i} and object {j} turned on",
            5: "relation {r} between object {i} and object {j} turned off",
            6: "object {i} and object {j} swapped",
            7: "object {i} received noise burst",
        }
        
        template = templates.get(u_type, "unknown intervention")
        
        try:
            description = template.format(**args)
        except KeyError:
            description = f"{int_name} applied"
        
        # Add observation change summary
        if obs_before is not None and obs_after is not None:
            X_diff = obs_after["X"] - obs_before["X"]
            max_change_obj = int(np.argmax(np.abs(X_diff).sum(axis=1)))
            max_change_val = float(np.max(np.abs(X_diff)))
            
            if max_change_val > 0.01:
                description += f". object {max_change_obj} changed most"
        
        return description
    
    def generate_state_description(
        self,
        obs: Dict[str, np.ndarray],
        S_settled: np.ndarray
    ) -> str:
        """
        Generate a description of the current state.
        
        Args:
            obs: Current observation
            S_settled: Settled latent state
            
        Returns:
            State description
        """
        X = obs["X"]
        Rel = obs["Rel"]
        
        descriptions = []
        
        # Describe each object
        for i in range(X.shape[0]):
            attrs = X[i]
            # Classify attribute levels
            levels = []
            for a, val in enumerate(attrs):
                if val < 0.33:
                    levels.append("low")
                elif val < 0.67:
                    levels.append("medium")
                else:
                    levels.append("high")
            descriptions.append(f"object {i} has attr0 {levels[0]}")
        
        # Describe active relations
        for r in range(Rel.shape[2]):
            for i in range(Rel.shape[0]):
                for j in range(Rel.shape[1]):
                    if i != j and Rel[i, j, r] > 0.5:
                        descriptions.append(f"object {i} linked to object {j} via relation {r}")
        
        return ". ".join(descriptions[:5])  # Limit to 5 statements
    
    def train_step(
        self,
        S_settled: np.ndarray,
        target_text: str,
        lr: float = 0.001
    ) -> Dict[str, float]:
        """
        Train decoder on grounded (state, description) pair.
        
        Uses teacher forcing.
        
        Args:
            S_settled: Settled state
            target_text: Target description
            lr: Learning rate
            
        Returns:
            Training metrics
        """
        # Encode target
        target_tokens = self.vocab.encode(target_text, self.max_len)
        
        # Get context
        context = self.readout(S_settled)
        
        # Teacher forcing through sequence
        hidden = context.copy()
        total_loss = 0.0
        n_tokens = 0
        
        for t in range(len(target_tokens) - 1):
            current_token = target_tokens[t]
            next_token = target_tokens[t + 1]
            
            if next_token == self.vocab.pad_idx:
                break
            
            # Forward
            token_emb = self.token_embeddings[current_token]
            combined = np.concatenate([token_emb, hidden])
            hidden = np.tanh(combined @ self.W_h + self.b_h)
            logits = hidden @ self.W_out + self.b_out
            
            # Cross-entropy loss
            probs = self._softmax(logits)
            loss = -np.log(probs[next_token] + 1e-9)
            total_loss += loss
            n_tokens += 1
            
            # Gradient for this step (simplified)
            # d_loss/d_logits = probs - one_hot(next_token)
            d_logits = probs.copy()
            d_logits[next_token] -= 1.0
            
            # Update W_out, b_out
            self.W_out -= lr * np.outer(hidden, d_logits)
            self.b_out -= lr * d_logits
        
        avg_loss = total_loss / max(n_tokens, 1)
        
        return {
            "loss": float(avg_loss),
            "n_tokens": n_tokens,
        }
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Return trainable parameters."""
        return {
            "W_readout1": self.W_readout1,
            "b_readout1": self.b_readout1,
            "W_readout2": self.W_readout2,
            "b_readout2": self.b_readout2,
            "token_embeddings": self.token_embeddings,
            "W_h": self.W_h,
            "b_h": self.b_h,
            "W_out": self.W_out,
            "b_out": self.b_out,
        }
    
    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """Set parameters from dict."""
        for name, value in params.items():
            setattr(self, name, value)

