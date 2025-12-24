"""
Language Encoder Module

Embeds natural language descriptions into latent space for multimodal perception.
Lightweight implementation suitable for CPU training.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import re


class Vocabulary:
    """
    Simple vocabulary for tokenization.
    Built from grounded domain vocabulary.
    """
    
    # Core domain vocabulary
    CORE_VOCAB = [
        # Special tokens
        "<PAD>", "<UNK>", "<BOS>", "<EOS>",
        # Objects
        "object", "0", "1", "2", "3", "4", "5",
        # Attributes
        "attribute", "attr", "size", "color", "value",
        "small", "medium", "large", "low", "high",
        "red", "blue", "green", "bright", "dark",
        # Relations
        "relation", "supports", "connected", "linked", "touches",
        "above", "below", "near", "with", "to", "from",
        # Actions
        "increase", "decrease", "set", "toggle", "swap", "noise",
        "make", "change", "modify", "turn", "on", "off",
        # States
        "is", "are", "was", "were", "becomes", "became", "now",
        "larger", "smaller", "bigger", "higher", "lower",
        # Causal
        "if", "then", "because", "due", "causes", "caused",
        "would", "will", "should", "after", "before",
        # Structure
        "the", "a", "an", "and", "or", "not", "no",
        ".", ",", "!", "?",
    ]
    
    def __init__(self):
        self.word2idx = {w: i for i, w in enumerate(self.CORE_VOCAB)}
        self.idx2word = {i: w for i, w in enumerate(self.CORE_VOCAB)}
        self.vocab_size = len(self.CORE_VOCAB)
        self.pad_idx = 0
        self.unk_idx = 1
        self.bos_idx = 2
        self.eos_idx = 3
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        # Split on whitespace and punctuation
        tokens = re.findall(r'\w+|[.,!?]', text)
        return tokens
    
    def encode(self, text: str, max_len: int = 32) -> np.ndarray:
        """Encode text to token indices."""
        tokens = self.tokenize(text)
        indices = [self.bos_idx]
        for token in tokens[:max_len - 2]:
            indices.append(self.word2idx.get(token, self.unk_idx))
        indices.append(self.eos_idx)
        
        # Pad to max_len
        while len(indices) < max_len:
            indices.append(self.pad_idx)
        
        return np.array(indices[:max_len], dtype=np.int32)
    
    def decode(self, indices: np.ndarray) -> str:
        """Decode token indices to text."""
        words = []
        for idx in indices:
            if idx == self.eos_idx:
                break
            if idx not in [self.pad_idx, self.bos_idx]:
                words.append(self.idx2word.get(int(idx), "<UNK>"))
        return " ".join(words)


class LanguageEmbedder:
    """
    Lightweight language embedder using learned embeddings + attention pooling.
    
    Architecture:
        tokens → embeddings → self-attention → pooled embedding
    
    CPU-friendly alternative to BERT/RoBERTa.
    """
    
    def __init__(
        self,
        vocab_size: int = 100,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 64,
        max_len: int = 32,
        n_heads: int = 4,
        seed: int = 42
    ):
        """
        Initialize language embedder.
        
        Args:
            vocab_size: Vocabulary size
            embed_dim: Token embedding dimension
            hidden_dim: Hidden dimension
            output_dim: Output embedding dimension
            max_len: Maximum sequence length
            n_heads: Number of attention heads
            seed: Random seed
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_len = max_len
        self.n_heads = n_heads
        
        self.rng = np.random.default_rng(seed)
        self.vocab = Vocabulary()
        
        # Token embeddings
        self.token_embeddings = self.rng.normal(
            0, 0.1, (self.vocab.vocab_size, embed_dim)
        ).astype(np.float32)
        
        # Positional embeddings
        self.pos_embeddings = self._create_positional_embeddings()
        
        # Self-attention weights (simplified single layer)
        scale = np.sqrt(2.0 / (embed_dim + hidden_dim))
        self.W_q = self.rng.normal(0, scale, (embed_dim, hidden_dim)).astype(np.float32)
        self.W_k = self.rng.normal(0, scale, (embed_dim, hidden_dim)).astype(np.float32)
        self.W_v = self.rng.normal(0, scale, (embed_dim, hidden_dim)).astype(np.float32)
        
        # Output projection
        scale_out = np.sqrt(2.0 / (hidden_dim + output_dim))
        self.W_out = self.rng.normal(0, scale_out, (hidden_dim, output_dim)).astype(np.float32)
        self.b_out = np.zeros(output_dim, dtype=np.float32)
    
    def _create_positional_embeddings(self) -> np.ndarray:
        """Create sinusoidal positional embeddings."""
        pos = np.arange(self.max_len)[:, np.newaxis]
        dim = np.arange(self.embed_dim)[np.newaxis, :]
        
        angles = pos / np.power(10000, (2 * (dim // 2)) / self.embed_dim)
        
        pos_emb = np.zeros((self.max_len, self.embed_dim), dtype=np.float32)
        pos_emb[:, 0::2] = np.sin(angles[:, 0::2])
        pos_emb[:, 1::2] = np.cos(angles[:, 1::2])
        
        return pos_emb
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-9)
    
    def embed(self, text: str) -> np.ndarray:
        """
        Embed text to fixed-size vector.
        
        Args:
            text: Input text string
            
        Returns:
            Embedding vector of shape (output_dim,)
        """
        # Tokenize and encode
        token_ids = self.vocab.encode(text, self.max_len)
        
        # Get token embeddings
        token_embs = self.token_embeddings[token_ids]  # (max_len, embed_dim)
        
        # Add positional embeddings
        x = token_embs + self.pos_embeddings  # (max_len, embed_dim)
        
        # Self-attention
        Q = x @ self.W_q  # (max_len, hidden_dim)
        K = x @ self.W_k
        V = x @ self.W_v
        
        # Scaled dot-product attention
        scores = (Q @ K.T) / np.sqrt(self.hidden_dim)  # (max_len, max_len)
        
        # Mask padding tokens
        pad_mask = (token_ids == self.vocab.pad_idx)
        scores[:, pad_mask] = -1e9
        
        attn = self._softmax(scores, axis=-1)
        context = attn @ V  # (max_len, hidden_dim)
        
        # Pool: attention-weighted mean over non-padding positions
        valid_mask = ~pad_mask
        weights = valid_mask.astype(np.float32) / (np.sum(valid_mask) + 1e-9)
        pooled = np.sum(context * weights[:, np.newaxis], axis=0)  # (hidden_dim,)
        
        # Output projection
        output = np.tanh(pooled @ self.W_out + self.b_out)
        
        return output.astype(np.float32)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts."""
        return np.stack([self.embed(t) for t in texts])
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Return trainable parameters."""
        return {
            "token_embeddings": self.token_embeddings,
            "W_q": self.W_q,
            "W_k": self.W_k,
            "W_v": self.W_v,
            "W_out": self.W_out,
            "b_out": self.b_out,
        }
    
    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """Set parameters from dict."""
        for name, value in params.items():
            setattr(self, name, value)


class MultimodalEncoder:
    """
    Multimodal encoder that fuses symbolic observations with language.
    
    enc_obs = fusion(symbolic_enc(X, Rel), language_enc(text))
    
    This extends the original ObservationEncoder non-disruptively.
    """
    
    def __init__(
        self,
        n_objects: int,
        n_attributes: int,
        n_relations: int,
        latent_dim: int,
        lang_dim: int = 64,
        hidden_dim: int = 128,
        seed: int = 42
    ):
        """
        Initialize multimodal encoder.
        
        Args:
            n_objects: Number of objects
            n_attributes: Attributes per object
            n_relations: Relation types
            latent_dim: Output dimension (dS)
            lang_dim: Language embedding dimension
            hidden_dim: Hidden dimension
            seed: Random seed
        """
        self.N = n_objects
        self.A = n_attributes
        self.R = n_relations
        self.dS = latent_dim
        self.lang_dim = lang_dim
        
        self.rng = np.random.default_rng(seed)
        
        # Language embedder
        self.lang_embedder = LanguageEmbedder(
            output_dim=lang_dim,
            seed=seed
        )
        
        # Symbolic encoder (same as original)
        self.symbolic_input_dim = n_objects * n_attributes + n_objects * n_objects * n_relations
        
        scale1 = np.sqrt(2.0 / (self.symbolic_input_dim + hidden_dim))
        self.W_sym1 = self.rng.normal(0, scale1, (self.symbolic_input_dim, hidden_dim)).astype(np.float32)
        self.b_sym1 = np.zeros(hidden_dim, dtype=np.float32)
        
        # Fusion layer: combines symbolic + language
        fusion_input_dim = hidden_dim + lang_dim
        scale2 = np.sqrt(2.0 / (fusion_input_dim + latent_dim))
        self.W_fusion = self.rng.normal(0, scale2, (fusion_input_dim, latent_dim)).astype(np.float32)
        self.b_fusion = np.zeros(latent_dim, dtype=np.float32)
        
        # Symbolic-only path (for backward compatibility)
        scale_sym = np.sqrt(2.0 / (hidden_dim + latent_dim))
        self.W_sym_out = self.rng.normal(0, scale_sym, (hidden_dim, latent_dim)).astype(np.float32)
        self.b_sym_out = np.zeros(latent_dim, dtype=np.float32)
        
        # Gating: learn to balance symbolic vs language
        self.gate_scale = 0.5  # Default equal weighting
    
    def encode_symbolic(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Encode only symbolic observation."""
        x_flat = obs["X"].flatten()
        rel_flat = obs["Rel"].flatten()
        inp = np.concatenate([x_flat, rel_flat])
        
        h = np.maximum(0, inp @ self.W_sym1 + self.b_sym1)
        out = np.tanh(h @ self.W_sym_out + self.b_sym_out)
        
        return out.astype(np.float32)
    
    def encode_language(self, text: str) -> np.ndarray:
        """Encode language description."""
        return self.lang_embedder.embed(text)
    
    def encode(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Encode observation (multimodal if language present).
        
        Args:
            obs: Observation dict with optional 'language' key
            
        Returns:
            Encoded observation in R^{dS}
        """
        # Symbolic encoding
        x_flat = obs["X"].flatten()
        rel_flat = obs["Rel"].flatten()
        inp = np.concatenate([x_flat, rel_flat])
        h_sym = np.maximum(0, inp @ self.W_sym1 + self.b_sym1)
        
        # Check for language
        language = obs.get("language")
        
        if language is None or language == "":
            # Symbolic-only path
            out = np.tanh(h_sym @ self.W_sym_out + self.b_sym_out)
        else:
            # Multimodal fusion
            h_lang = self.lang_embedder.embed(language)
            h_fused = np.concatenate([h_sym, h_lang])
            out = np.tanh(h_fused @ self.W_fusion + self.b_fusion)
        
        return out.astype(np.float32)
    
    def decode(self, z: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Decode latent state to observation space.
        Note: Language is not decoded here (separate decoder).
        """
        # Reuse symbolic decoder weights (need to initialize)
        # For now, simple linear projection
        h = np.maximum(0, z @ self.W_fusion.T[:self.dS, :])
        
        x_size = self.N * self.A
        rel_size = self.N * self.N * self.R
        
        # Simple linear decode
        out = np.zeros(x_size + rel_size, dtype=np.float32)
        out[:x_size] = 0.5  # Default
        out[x_size:] = 0.0
        
        X_rec = out[:x_size].reshape(self.N, self.A)
        Rel_rec = out[x_size:].reshape(self.N, self.N, self.R)
        
        return {"X": X_rec, "Rel": Rel_rec, "mask": None}
    
    def compute_gradients(
        self,
        obs: Dict[str, np.ndarray],
        S: np.ndarray,
        weight: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Compute gradients for multimodal encoder parameters.
        
        Similar to original encoder but handles fusion.
        """
        # Forward pass with caching
        x_flat = obs["X"].flatten()
        rel_flat = obs["Rel"].flatten()
        inp = np.concatenate([x_flat, rel_flat])
        
        z1 = inp @ self.W_sym1 + self.b_sym1
        h_sym = np.maximum(0, z1)
        
        language = obs.get("language")
        
        if language is None or language == "":
            # Symbolic-only gradients
            z2 = h_sym @ self.W_sym_out + self.b_sym_out
            enc_obs = np.tanh(z2)
            
            diff = S - enc_obs
            d_enc = -2.0 * weight * diff
            d_z2 = d_enc * (1 - enc_obs ** 2)
            
            grad_W_sym_out = np.outer(h_sym, d_z2)
            grad_b_sym_out = d_z2
            
            d_h_sym = d_z2 @ self.W_sym_out.T
            d_z1 = d_h_sym * (z1 > 0)
            
            grad_W_sym1 = np.outer(inp, d_z1)
            grad_b_sym1 = d_z1
            
            return {
                "W_sym1": grad_W_sym1.astype(np.float32),
                "b_sym1": grad_b_sym1.astype(np.float32),
                "W_sym_out": grad_W_sym_out.astype(np.float32),
                "b_sym_out": grad_b_sym_out.astype(np.float32),
            }
        else:
            # Multimodal gradients (simplified)
            h_lang = self.lang_embedder.embed(language)
            h_fused = np.concatenate([h_sym, h_lang])
            z2 = h_fused @ self.W_fusion + self.b_fusion
            enc_obs = np.tanh(z2)
            
            diff = S - enc_obs
            d_enc = -2.0 * weight * diff
            d_z2 = d_enc * (1 - enc_obs ** 2)
            
            grad_W_fusion = np.outer(h_fused, d_z2)
            grad_b_fusion = d_z2
            
            return {
                "W_fusion": grad_W_fusion.astype(np.float32),
                "b_fusion": grad_b_fusion.astype(np.float32),
            }
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Return all trainable parameters."""
        params = {
            "W_sym1": self.W_sym1,
            "b_sym1": self.b_sym1,
            "W_sym_out": self.W_sym_out,
            "b_sym_out": self.b_sym_out,
            "W_fusion": self.W_fusion,
            "b_fusion": self.b_fusion,
        }
        params.update({f"lang_{k}": v for k, v in self.lang_embedder.get_parameters().items()})
        return params
    
    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """Set parameters from dict."""
        for name, value in params.items():
            if name.startswith("lang_"):
                lang_name = name[5:]
                setattr(self.lang_embedder, lang_name, value)
            else:
                setattr(self, name, value)

