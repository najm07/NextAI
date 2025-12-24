"""
Focused Instruction Parser Training

Trains the neural instruction parser to 90%+ accuracy using canonical corpus.
"""

import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple, Dict

sys.path.append(str(Path(__file__).parent.parent))

from agent.modules.instruction_parser import InstructionParser
from data.grounded_corpus import GroundedCorpusGenerator, GroundedDataset
from env.track1_env import Track1CausalEnv


def generate_canonical_dataset(n_examples: int = 5000, seed: int = 42) -> List[Tuple[str, int, Dict]]:
    """
    Generate canonical instruction dataset for high-accuracy training.
    
    Uses fixed templates that match the parser patterns exactly.
    """
    rng = np.random.default_rng(seed)
    
    N = 4  # objects
    A = 3  # attributes
    R = 2  # relations
    
    # Canonical templates
    templates = {
        0: "increase object {i} attr {a}",
        1: "decrease object {i} attr {a}",
        2: "set object {i} attr {a} to bin {bin_id}",
        3: "toggle relation {r} between object {i} and object {j}",
        4: "set relation {r} between object {i} and object {j} on",
        5: "set relation {r} between object {i} and object {j} off",
        6: "swap object {i} and object {j}",
        7: "add noise to object {i}",
    }
    
    dataset = []
    
    for _ in range(n_examples):
        u_type = int(rng.integers(0, 8))
        
        # Generate valid args
        i = int(rng.integers(0, N))
        j = int(rng.integers(0, N - 1))
        if j >= i:
            j += 1
        a = int(rng.integers(0, A))
        r = int(rng.integers(0, R))
        bin_id = int(rng.integers(0, 4))
        
        args = {"i": i, "j": j, "a": a, "r": r, "bin_id": bin_id}
        
        # Generate instruction from template
        template = templates[u_type]
        instruction = template.format(**args)
        
        dataset.append((instruction, u_type, args))
    
    return dataset


def train_parser_to_target_accuracy(
    target_accuracy: float = 0.90,
    max_epochs: int = 100,
    lr: float = 0.05,
    n_train: int = 5000,
    n_val: int = 500,
    seed: int = 42
) -> InstructionParser:
    """
    Train instruction parser until target accuracy is reached.
    
    Args:
        target_accuracy: Target type accuracy (0.90 = 90%)
        max_epochs: Maximum training epochs
        lr: Learning rate
        n_train: Number of training examples
        n_val: Number of validation examples
        seed: Random seed
        
    Returns:
        Trained InstructionParser
    """
    print("=" * 60)
    print("Instruction Parser Training")
    print("=" * 60)
    print(f"Target accuracy: {target_accuracy*100:.0f}%")
    print(f"Training examples: {n_train}")
    print(f"Validation examples: {n_val}")
    print()
    
    # Create parser
    parser = InstructionParser(
        n_objects=4,
        n_attributes=3,
        n_relations=2,
        latent_dim=64,
        use_neural=True,
        seed=seed
    )
    
    # Generate datasets
    print("Generating canonical training data...")
    train_data = generate_canonical_dataset(n_train, seed)
    val_data = generate_canonical_dataset(n_val, seed + 1000)
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    print()
    
    # Training loop
    print("Training...")
    print("-" * 60)
    
    best_accuracy = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(max_epochs):
        # Train epoch
        metrics = parser.train_epoch(train_data, lr=lr, shuffle=True)
        
        # Evaluate on validation
        n_correct_type = 0
        n_correct_i = 0
        n_correct_full = 0
        
        for instruction, target_type, target_args in val_data:
            # Use neural parser only (no pattern fallback for fair eval)
            u_type, args, _ = parser.parse(instruction, prefer_pattern=False)
            
            type_ok = u_type == target_type
            i_ok = args.get("i") == target_args.get("i")
            
            if type_ok:
                n_correct_type += 1
            if i_ok:
                n_correct_i += 1
            if type_ok and i_ok:
                n_correct_full += 1
        
        val_type_acc = n_correct_type / len(val_data)
        val_i_acc = n_correct_i / len(val_data)
        val_full_acc = n_correct_full / len(val_data)
        
        # Print progress
        if epoch % 5 == 0 or val_type_acc >= target_accuracy:
            print(f"Epoch {epoch:3d} | "
                  f"Type: {val_type_acc*100:5.1f}% | "
                  f"Arg-i: {val_i_acc*100:5.1f}% | "
                  f"Full: {val_full_acc*100:5.1f}% | "
                  f"Loss: {metrics['type_loss']:.4f}")
        
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
            lr *= 0.8
    
    print("-" * 60)
    print(f"\nFinal validation accuracy:")
    print(f"  Type: {val_type_acc*100:.1f}%")
    print(f"  Arg-i: {val_i_acc*100:.1f}%")
    print(f"  Full: {val_full_acc*100:.1f}%")
    
    return parser


def evaluate_parser(parser: InstructionParser) -> None:
    """Evaluate parser on test cases."""
    print("\n" + "=" * 60)
    print("Parser Evaluation")
    print("=" * 60)
    
    INT_NAMES = [
        "IncreaseAttr", "DecreaseAttr", "SetAttrToward",
        "ToggleRel", "SetRelOn", "SetRelOff",
        "SwapObjects", "NoiseBurstAttr"
    ]
    
    # Test canonical patterns
    test_cases = [
        # Canonical format
        ("increase object 0 attr 1", 0, {"i": 0, "a": 1}),
        ("decrease object 2 attr 0", 1, {"i": 2, "a": 0}),
        ("set object 1 attr 2 to bin 3", 2, {"i": 1, "a": 2, "bin_id": 3}),
        ("toggle relation 0 between object 1 and object 3", 3, {"r": 0, "i": 1, "j": 3}),
        ("set relation 1 between object 0 and object 2 on", 4, {"r": 1, "i": 0, "j": 2}),
        ("set relation 0 between object 2 and object 3 off", 5, {"r": 0, "i": 2, "j": 3}),
        ("swap object 1 and object 3", 6, {"i": 1, "j": 3}),
        ("add noise to object 2", 7, {"i": 2}),
        # Variations
        ("make object 0 larger", 0, {"i": 0}),
        ("connect object 1 to object 2", 4, {"i": 1, "j": 2}),
    ]
    
    n_type_ok = 0
    n_args_ok = 0
    
    for instruction, expected_type, expected_args in test_cases:
        u_type, args, conf = parser.parse(instruction)
        
        type_ok = u_type == expected_type
        args_ok = args.get("i") == expected_args.get("i", args.get("i"))
        
        if type_ok:
            n_type_ok += 1
        if args_ok:
            n_args_ok += 1
        
        status = "[OK]  " if (type_ok and args_ok) else "[FAIL]"
        print(f'{status} "{instruction}"')
        print(f"       Got: {INT_NAMES[u_type]}, i={args.get('i')}, j={args.get('j')} (conf={conf:.2f})")
        print(f"       Exp: {INT_NAMES[expected_type]}, i={expected_args.get('i')}, j={expected_args.get('j')}")
        print()
    
    print("-" * 60)
    print(f"Type accuracy: {n_type_ok}/{len(test_cases)} = {100*n_type_ok/len(test_cases):.0f}%")
    print(f"Args accuracy: {n_args_ok}/{len(test_cases)} = {100*n_args_ok/len(test_cases):.0f}%")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train instruction parser")
    parser.add_argument("--target", type=float, default=0.90, help="Target accuracy")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--n_train", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Train parser
    trained_parser = train_parser_to_target_accuracy(
        target_accuracy=args.target,
        max_epochs=args.epochs,
        lr=args.lr,
        n_train=args.n_train,
        seed=args.seed
    )
    
    # Evaluate
    evaluate_parser(trained_parser)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

