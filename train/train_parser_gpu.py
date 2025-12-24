"""
GPU-Accelerated Instruction Parser Training

Trains the neural instruction parser using PyTorch for GPU acceleration.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

# Add project root to path (works in both local and Colab)
import os
project_root = Path(__file__).parent.parent
if not (project_root / 'data').exists():
    # Try current working directory (for Colab)
    cwd = Path.cwd()
    if (cwd / 'data').exists():
        project_root = cwd
    elif (cwd / 'NextAI' / 'data').exists():
        project_root = cwd / 'NextAI'
sys.path.insert(0, str(project_root))

from data.grounded_corpus import GroundedCorpusGenerator, GroundedDataset
from env.track1_env import Track1CausalEnv


class GPUInstructionParser(nn.Module):
    """
    GPU-accelerated instruction parser using PyTorch.
    """
    
    INT_NAMES = [
        "IncreaseAttr", "DecreaseAttr", "SetAttrToward",
        "ToggleRel", "SetRelOn", "SetRelOff",
        "SwapObjects", "NoiseBurstAttr",
        "QueryState", "QueryWhy", "Counterfactual",
        "PlanConnectAll", "PlanGrowMax", "PlanClearAll",
        "UndoLast", "ResetWorld", "ConditionalIf",
        "MetaEnergy", "MetaBeliefs", "MetaSummary"
    ]
    
    N_ACTION_TYPES = 20
    
    def __init__(
        self,
        n_objects: int = 4,
        n_attributes: int = 3,
        n_relations: int = 2,
        vocab_size: int = 100,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        max_len: int = 32,
        device: str = "cuda"
    ):
        super().__init__()
        self.n_objects = n_objects
        self.n_attributes = n_attributes
        self.n_relations = n_relations
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.device = device
        
        # Vocabulary (simple tokenizer)
        self.vocab = self._create_vocab()
        self.vocab_size = len(self.vocab)
        
        # Token embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, embed_dim)
        
        # Positional embeddings
        self.register_buffer('pos_embeddings', self._create_positional_embeddings())
        
        # Language encoder (self-attention)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.lang_proj = nn.Linear(embed_dim, latent_dim)
        
        # Intervention type classifier
        self.type_classifier = nn.Linear(latent_dim, self.N_ACTION_TYPES)
        
        # Argument extractors
        self.obj_i_classifier = nn.Linear(latent_dim, n_objects)
        self.obj_j_classifier = nn.Linear(latent_dim, n_objects)
        self.attr_classifier = nn.Linear(latent_dim, n_attributes)
        self.rel_classifier = nn.Linear(latent_dim, n_relations)
        self.bin_classifier = nn.Linear(latent_dim, 4)
        
        # Initialize weights
        self._init_weights()
        
    def _create_vocab(self) -> Dict[str, int]:
        """Create vocabulary from common words."""
        words = [
            '<pad>', '<unk>',
            'object', 'attr', 'attribute', 'relation', 'bin',
            'increase', 'decrease', 'set', 'toggle', 'swap', 'add', 'noise',
            'to', 'between', 'and', 'on', 'off',
            'what', 'is', 'the', 'of', 'describe', 'current', 'status',
            'why', 'did', 'change', 'caused', 'explain',
            'if', 'would', 'happen', 'predict', 'imagine', 'hypothetically',
            'connect', 'all', 'link', 'everything', 'grow', 'largest', 'maximize',
            'clear', 'disconnect', 'undo', 'last', 'action', 'go', 'back', 'revert',
            'reset', 'start', 'over', 'when', 'only', 'large', 'high',
            'energy', 'beliefs', 'understanding', 'learned', 'summarize', 'changes'
        ]
        # Add numbers 0-9
        words.extend([str(i) for i in range(10)])
        return {word: idx for idx, word in enumerate(words)}
    
    def _create_positional_embeddings(self) -> torch.Tensor:
        """Create sinusoidal positional embeddings."""
        pos = torch.arange(self.max_len, dtype=torch.float32).unsqueeze(1)
        dim = torch.arange(self.embed_dim, dtype=torch.float32).unsqueeze(0)
        
        angles = pos / (10000 ** (2 * (dim // 2) / self.embed_dim))
        
        pos_emb = torch.zeros(self.max_len, self.embed_dim)
        pos_emb[:, 0::2] = torch.sin(angles[:, 0::2])
        pos_emb[:, 1::2] = torch.cos(angles[:, 1::2])
        
        return pos_emb
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text to token IDs."""
        text = text.lower().strip()
        words = text.split()
        token_ids = []
        
        for word in words:
            if word in self.vocab:
                token_ids.append(self.vocab[word])
            else:
                token_ids.append(self.vocab.get('<unk>', 1))
        
        # Pad or truncate to max_len
        if len(token_ids) > self.max_len:
            token_ids = token_ids[:self.max_len]
        else:
            token_ids.extend([self.vocab['<pad>']] * (self.max_len - len(token_ids)))
        
        return torch.tensor(token_ids, dtype=torch.long, device=self.device)
    
    def forward(self, instructions: List[str]) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            instructions: List of instruction strings
            
        Returns:
            Dictionary with predictions
        """
        batch_size = len(instructions)
        
        # Tokenize
        token_ids = torch.stack([self._tokenize(inst) for inst in instructions])  # (B, L)
        
        # Embed tokens
        token_embs = self.token_embedding(token_ids)  # (B, L, embed_dim)
        
        # Add positional embeddings
        x = token_embs + self.pos_embeddings.unsqueeze(0)  # (B, L, embed_dim)
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)  # (B, L, embed_dim)
        
        # Pool: mean over sequence (masking padding)
        pad_mask = (token_ids == self.vocab['<pad>'])
        valid_mask = ~pad_mask  # (B, L)
        pooled = (attn_out * valid_mask.unsqueeze(-1).float()).sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).float()  # (B, embed_dim)
        
        # Project to latent
        latent = torch.tanh(self.lang_proj(pooled))  # (B, latent_dim)
        
        # Classify
        type_logits = self.type_classifier(latent)  # (B, N_ACTION_TYPES)
        obj_i_logits = self.obj_i_classifier(latent)  # (B, n_objects)
        obj_j_logits = self.obj_j_classifier(latent)  # (B, n_objects)
        attr_logits = self.attr_classifier(latent)  # (B, n_attributes)
        rel_logits = self.rel_classifier(latent)  # (B, n_relations)
        bin_logits = self.bin_classifier(latent)  # (B, 4)
        
        return {
            'type_logits': type_logits,
            'obj_i_logits': obj_i_logits,
            'obj_j_logits': obj_j_logits,
            'attr_logits': attr_logits,
            'rel_logits': rel_logits,
            'bin_logits': bin_logits,
            'latent': latent
        }
    
    def parse(self, instruction: str) -> Tuple[int, Dict[str, int], float]:
        """Parse single instruction."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward([instruction])
            
            type_probs = torch.softmax(outputs['type_logits'], dim=-1)
            u_type = int(torch.argmax(type_probs[0]))
            confidence = float(type_probs[0, u_type])
            
            obj_i = int(torch.argmax(outputs['obj_i_logits'][0]))
            obj_j = int(torch.argmax(outputs['obj_j_logits'][0]))
            attr = int(torch.argmax(outputs['attr_logits'][0]))
            rel = int(torch.argmax(outputs['rel_logits'][0]))
            bin_id = int(torch.argmax(outputs['bin_logits'][0]))
            
            args = {
                'i': obj_i,
                'j': obj_j if obj_j != obj_i else (obj_i + 1) % self.n_objects,
                'a': attr,
                'r': rel,
                'bin_id': bin_id
            }
            
            return u_type, args, confidence


def load_multi_turn_corpus(corpus_path: str) -> List[Tuple[str, int, Dict]]:
    """Load multi-turn conversation corpus."""
    with open(corpus_path, 'r') as f:
        conversations = json.load(f)
    
    dataset = []
    for conv in conversations:
        for turn in conv.get("turns", []):
            instruction = turn["instruction"]
            u_type = turn["u_type"]
            args = turn["args"]
            dataset.append((instruction, u_type, args))
    
    return dataset


class ParserDataset(Dataset):
    """PyTorch Dataset for instruction parser training."""
    
    def __init__(self, data: List[Tuple[str, int, Dict]], device: str = "cuda"):
        self.data = data
        self.device = device
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        instruction, u_type, args = self.data[idx]
        # Pre-compute target tensors
        target_i = args.get('i', 0)
        target_j = args.get('j', 0)
        target_a = args.get('a', 0)
        target_r = args.get('r', 0)
        target_bin = args.get('bin_id', 0)
        
        # Clamp values to valid ranges
        target_i = min(target_i, 3)
        target_j = min(target_j, 3)
        target_a = min(target_a, 2)
        target_r = min(target_r, 1)
        target_bin = min(target_bin, 3)
        
        return {
            'instruction': instruction,
            'u_type': u_type,
            'target_i': target_i,
            'target_j': target_j,
            'target_a': target_a,
            'target_r': target_r,
            'target_bin': target_bin,
            'has_i': 'i' in args,
            'has_j': 'j' in args,
            'has_a': 'a' in args,
            'has_r': 'r' in args,
            'has_bin': 'bin_id' in args,
        }


def train_parser_gpu(
    target_accuracy: float = 0.90,
    max_epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 256,  # Increased default batch size for better GPU utilization
    n_train: int = 5000,
    n_val: int = 500,
    seed: int = 42,
    corpus: Optional[str] = None,
    device: str = "cuda",
    use_amp: bool = True,  # Mixed precision training
    num_workers: int = 4,  # DataLoader workers
    pin_memory: bool = True  # Faster CPU-GPU transfer
) -> GPUInstructionParser:
    """
    Train GPU-accelerated parser.
    """
    print("=" * 60)
    print("GPU-Accelerated Instruction Parser Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Target accuracy: {target_accuracy*100:.0f}%")
    print(f"Training examples: {n_train}")
    print(f"Validation examples: {n_val}")
    print(f"Batch size: {batch_size}")
    print()
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create model
    model = GPUInstructionParser(
        n_objects=4,
        n_attributes=3,
        n_relations=2,
        device=device
    ).to(device)
    
    # Load or generate datasets
    if corpus:
        print(f"Loading multi-turn corpus from {corpus}...")
        all_data = load_multi_turn_corpus(corpus)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(all_data))
        n_train_actual = min(n_train, len(all_data) - n_val)
        train_indices = indices[:n_train_actual]
        val_indices = indices[n_train_actual:n_train_actual + n_val]
        train_data = [all_data[i] for i in train_indices]
        val_data = [all_data[i] for i in val_indices]
        print(f"Loaded {len(all_data)} examples from corpus")
    else:
        print("Error: corpus path required")
        return model
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Create datasets and dataloaders
    train_dataset = ParserDataset(train_data, device=device)
    val_dataset = ParserDataset(val_data, device=device)
    
    # Use DataLoader with optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers if device == "cuda" else 0,  # Workers only for GPU
        pin_memory=pin_memory and device == "cuda",
        persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers if device == "cuda" else 0,
        pin_memory=pin_memory and device == "cuda",
        persistent_workers=num_workers > 0
    )
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    type_criterion = nn.CrossEntropyLoss()
    arg_criterion = nn.CrossEntropyLoss()
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device == "cuda") else None
    
    best_accuracy = 0.0
    patience = 20
    patience_counter = 0
    
    print("Training...")
    if use_amp and device == "cuda":
        print("Using mixed precision (FP16) training")
    print(f"Batch size: {batch_size}, DataLoader workers: {num_workers}")
    print("-" * 60)
    
    for epoch in range(max_epochs):
        model.train()
        total_type_loss = 0.0
        total_arg_loss = 0.0
        n_batches = 0
        
        # Training loop with DataLoader
        for batch in train_loader:
            instructions = batch['instruction']
            target_types = torch.tensor(batch['u_type'], dtype=torch.long, device=device)
            
            # Prepare argument targets (batch them efficiently)
            target_i = torch.tensor(batch['target_i'], dtype=torch.long, device=device)
            target_j = torch.tensor(batch['target_j'], dtype=torch.long, device=device)
            target_a = torch.tensor(batch['target_a'], dtype=torch.long, device=device)
            target_r = torch.tensor(batch['target_r'], dtype=torch.long, device=device)
            target_bin = torch.tensor(batch['target_bin'], dtype=torch.long, device=device)
            
            has_i = torch.tensor(batch['has_i'], dtype=torch.bool, device=device)
            has_j = torch.tensor(batch['has_j'], dtype=torch.bool, device=device)
            has_a = torch.tensor(batch['has_a'], dtype=torch.bool, device=device)
            has_r = torch.tensor(batch['has_r'], dtype=torch.bool, device=device)
            has_bin = torch.tensor(batch['has_bin'], dtype=torch.bool, device=device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(instructions)
                    type_loss = type_criterion(outputs['type_logits'], target_types)
                    
                    # Batch argument losses efficiently
                    arg_losses = []
                    if has_i.any():
                        arg_losses.append(arg_criterion(outputs['obj_i_logits'][has_i], target_i[has_i]))
                    if has_j.any():
                        arg_losses.append(arg_criterion(outputs['obj_j_logits'][has_j], target_j[has_j]))
                    if has_a.any():
                        arg_losses.append(arg_criterion(outputs['attr_logits'][has_a], target_a[has_a]))
                    if has_r.any():
                        arg_losses.append(arg_criterion(outputs['rel_logits'][has_r], target_r[has_r]))
                    if has_bin.any():
                        arg_losses.append(arg_criterion(outputs['bin_logits'][has_bin], target_bin[has_bin]))
                    
                    arg_loss = sum(arg_losses) / max(len(arg_losses), 1) if arg_losses else torch.tensor(0.0, device=device)
                    total_loss = type_loss + arg_loss
                
                # Mixed precision backward pass
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision
                outputs = model(instructions)
                type_loss = type_criterion(outputs['type_logits'], target_types)
                
                arg_losses = []
                if has_i.any():
                    arg_losses.append(arg_criterion(outputs['obj_i_logits'][has_i], target_i[has_i]))
                if has_j.any():
                    arg_losses.append(arg_criterion(outputs['obj_j_logits'][has_j], target_j[has_j]))
                if has_a.any():
                    arg_losses.append(arg_criterion(outputs['attr_logits'][has_a], target_a[has_a]))
                if has_r.any():
                    arg_losses.append(arg_criterion(outputs['rel_logits'][has_r], target_r[has_r]))
                if has_bin.any():
                    arg_losses.append(arg_criterion(outputs['bin_logits'][has_bin], target_bin[has_bin]))
                
                arg_loss = sum(arg_losses) / max(len(arg_losses), 1) if arg_losses else torch.tensor(0.0, device=device)
                total_loss = type_loss + arg_loss
                
                total_loss.backward()
                optimizer.step()
            
            total_type_loss += type_loss.item()
            total_arg_loss += arg_loss.item()
            n_batches += 1
        
        avg_type_loss = total_type_loss / max(n_batches, 1)
        avg_arg_loss = total_arg_loss / max(n_batches, 1)
        
        # Validation
        model.eval()
        n_correct_type = 0
        n_correct_i = 0
        n_correct_full = 0
        
        with torch.no_grad():
            for batch in val_loader:
                instructions = batch['instruction']
                target_types = torch.tensor(batch['u_type'], dtype=torch.long, device=device)
                target_i = torch.tensor(batch['target_i'], dtype=torch.long, device=device)
                has_i = torch.tensor(batch['has_i'], dtype=torch.bool, device=device)
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(instructions)
                else:
                    outputs = model(instructions)
                
                type_probs = torch.softmax(outputs['type_logits'], dim=-1)
                type_preds = torch.argmax(type_probs, dim=-1)
                
                for idx in range(len(instructions)):
                    pred_type = int(type_preds[idx])
                    target_type = int(target_types[idx])
                    
                    if pred_type == target_type:
                        n_correct_type += 1
                    
                    if has_i[idx]:
                        pred_i = int(torch.argmax(outputs['obj_i_logits'][idx]))
                        if pred_i == int(target_i[idx]):
                            n_correct_i += 1
                        if pred_type == target_type and pred_i == int(target_i[idx]):
                            n_correct_full += 1
        
        val_type_acc = n_correct_type / len(val_data)
        val_i_acc = n_correct_i / len(val_data) if n_correct_i > 0 else 0.0
        val_full_acc = n_correct_full / len(val_data) if n_correct_full > 0 else 0.0
        
        # Print progress
        if epoch % 5 == 0 or val_type_acc >= target_accuracy:
            print(f"Epoch {epoch:3d} | "
                  f"Type: {val_type_acc*100:5.1f}% | "
                  f"Arg-i: {val_i_acc*100:5.1f}% | "
                  f"Full: {val_full_acc*100:5.1f}% | "
                  f"TypeLoss: {avg_type_loss:.4f} | "
                  f"ArgLoss: {avg_arg_loss:.4f}")
        
        # Check target
        if val_type_acc >= target_accuracy:
            print(f"\nTarget accuracy reached at epoch {epoch}!")
            break
        
        # Early stopping
        if val_type_acc > best_accuracy:
            best_accuracy = val_type_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # Learning rate decay
        if epoch > 0 and epoch % 20 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.8
    
    print("-" * 60)
    print(f"\nFinal validation accuracy:")
    print(f"  Type: {val_type_acc*100:.1f}%")
    print(f"  Arg-i: {val_i_acc*100:.1f}%")
    print(f"  Full: {val_full_acc*100:.1f}%")
    
    return model


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train GPU-accelerated instruction parser")
    parser.add_argument("--target", type=float, default=0.90, help="Target accuracy")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_train", type=int, default=5000)
    parser.add_argument("--n_val", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corpus", type=str, required=True, help="Path to multi-turn corpus JSON")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()
    
    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Train parser
    trained_model = train_parser_gpu(
        target_accuracy=args.target,
        max_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        n_train=args.n_train,
        n_val=args.n_val,
        seed=args.seed,
        corpus=args.corpus,
        device=args.device
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

