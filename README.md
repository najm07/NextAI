# Track 1 (v0.1-A): Non-Transformer Cognitive System

A dynamical, state-centric, energy/conflict-based cognitive system that learns from interventions in a toy causal environment.

## Overview

This system implements:
- **Energy-based settling**: "Thinking" is iterative relaxation of latent state S to a low-conflict fixed point
- **One-hot interventions**: Exactly one intervention type active per step (do-style causal interventions)
- **Out-of-band arguments**: Intervention arguments are separate from the one-hot encoding
- **EP-style learning**: Two-phase free/nudged settling scheme (equilibrium-propagation style)
- **No direct parameter coupling**: All module influence flows through the latent state S

## Architecture

### Agent Modules

1. **Encoder** (`agent/modules/encoder.py`): Maps observations to latent space
2. **Memory Anchors** (`agent/modules/anchors.py`): Stable attractors in latent space  
3. **STCs** (`agent/modules/stc.py`): Soft Transition Constraints for intervention-gated dynamics
4. **Gates** (`agent/modules/gates.py`): Context-dependent modulation of STCs and Anchors
5. **Temporal Modulators** (`agent/modules/temporal.py`): Per-dimension settling dynamics
6. **Intent** (`agent/modules/intent.py`): Goal-directed energy bias

### Environment

Object-centric symbolic environment with:
- N objects (default: 4)
- A continuous attributes per object (default: 3)
- R binary relation types (default: 2)
- 8 intervention types (one-hot):
  - 0: IncreaseAttr(i, a)
  - 1: DecreaseAttr(i, a)
  - 2: SetAttrToward(i, a, bin_id)
  - 3: ToggleRel(r, i, j)
  - 4: SetRelOn(r, i, j)
  - 5: SetRelOff(r, i, j)
  - 6: SwapObjects(i, j)
  - 7: NoiseBurstAttr(i)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train/train_track1.py --config configs/track1.yaml
```

Options:
- `--steps N`: Override total training steps
- `--seed N`: Override random seed

### Evaluation

```bash
python eval/eval_track1.py --config configs/track1.yaml
```

### Running Tests

```bash
python tests/run_all_tests.py
```

Or run individual test modules:
```bash
python tests/test_env_determinism.py
python tests/test_one_hot.py
python tests/test_swap_invariance.py
python tests/test_agent.py
```

## Configuration

See `configs/track1.yaml` for all configurable parameters:
- Environment settings (objects, attributes, relations, noise)
- Agent architecture (latent dim, anchors, STC factors)
- Settling parameters (steps, learning rate, tolerance)
- Training settings (learning rates, staged plasticity)
- Evaluation metrics

## Staged Plasticity Schedule

Training proceeds in 4 stages:
1. **Stage 1** (0-2000 steps): Train Anchors + Encoder only
2. **Stage 2** (2000-5000 steps): Add STCs
3. **Stage 3** (5000-8000 steps): Add Gates + Temporal Modulators
4. **Stage 4** (8000+ steps): Add Intent

## Evaluation Metrics

1. **Paraphrase Stability**: Consistency under minor observation variations
2. **Counterfactual Coherence**: Locality of changes under different interventions
3. **Graceful Degradation**: Smooth performance decline under noise
4. **Swap Invariance**: Robustness to SwapObjects intervention

## Acceptance Criteria (v0.1-A)

1. ✓ Environment runs with consistent transitions for each intervention
2. ✓ Agent runs full loop for 10k steps on CPU without instability
3. ✓ Training reduces prediction error and average conflict energy
4. ✓ SwapObjects does not collapse performance
5. ✓ Metrics degrade smoothly under added observation noise

## Project Structure

```
├── configs/
│   └── track1.yaml         # Configuration file
├── env/
│   ├── __init__.py
│   └── track1_env.py       # Track1CausalEnv
├── agent/
│   ├── __init__.py
│   ├── agent.py            # Track1Agent main class
│   ├── settle.py           # Settling loop
│   └── modules/
│       ├── __init__.py
│       ├── encoder.py      # Observation encoder
│       ├── anchors.py      # Memory anchors
│       ├── stc.py          # Transition STCs
│       ├── gates.py        # Context gates
│       ├── temporal.py     # Temporal modulators
│       ├── intent.py       # Intent module
│       └── energy.py       # Energy computer
├── train/
│   ├── __init__.py
│   └── train_track1.py     # Training loop
├── eval/
│   ├── __init__.py
│   └── eval_track1.py      # Evaluation suite
├── tests/
│   ├── __init__.py
│   ├── test_env_determinism.py
│   ├── test_one_hot.py
│   ├── test_swap_invariance.py
│   ├── test_agent.py
│   └── run_all_tests.py
├── checkpoints/            # Saved model checkpoints
├── requirements.txt
└── README.md
```

## Language Extension (Non-Disruptive)

The system includes a modular language extension that adds:

### 1. Language as Perception Input
```python
obs = {
  "X": X_t,
  "Rel": Rel_t,
  "language": "object 0 is large and red"  # Optional language context
}
```

The multimodal encoder fuses symbolic and language representations:
```python
enc_obs = multimodal_encoder.encode(X, Rel, language)
```

### 2. Language as Output (Grounded Generation)
```python
description = agent.describe_state(obs)
# Output: "object 0 has attr0 high. object 1 supports object 2"
```

The decoder is trained on grounded data to maintain causal accuracy.

### 3. Language-Grounded Actions
```python
u_type, args, conf = agent.parse_instruction("make object 0 support object 1")
# Output: SetRelOn, {"i": 0, "j": 1, "r": 0}, confidence=1.0
```

### Language Training
```bash
# Generate grounded corpus and train
python train/train_language.py --steps 10000 --corpus_size 10000
```

### Language Modules
- `agent/modules/language_encoder.py` - Vocabulary, LanguageEmbedder, MultimodalEncoder
- `agent/modules/language_decoder.py` - GroundedLanguageDecoder
- `agent/modules/instruction_parser.py` - InstructionParser
- `agent/language_agent.py` - LanguageAgent (extends Track1Agent)
- `data/grounded_corpus.py` - GroundedCorpusGenerator, GroundedDataset

### Training Curriculum
The language training uses staged curriculum:
1. **Stage 1**: Symbolic only (Track 1 core)
2. **Stage 2**: + Language perception
3. **Stage 3**: + Language readout
4. **Stage 4**: Full language-perception-action loop

## Interactive Demo

### Full Causal Reasoning Agent
Run the interactive agent:
```bash
python demo/full_causal_agent.py
```

Or run the batch demo:
```bash
python demo/full_causal_agent.py --batch
```

### Interactive Commands
```
[instruction]  - Natural language instruction to execute
/describe      - Describe current state
/energy        - Show energy components
/history       - Show action history
/what if [X]   - Counterfactual query
/undo          - Undo last action
/reset         - Reset environment
/help          - Show help
/quit          - Exit
```

### Example Session
```
Your instruction: increase object 0 attr 0
----------------------------------------------------------------------
Action: IncreaseAttr({'i': 0, 'a': 0}) [confidence: 1.00]
Result: object 0 attribute 0 increased. object 0 changed most
State: object 0 has attr0 medium. object 1 has attr0 high...
Energy: 0.6954 (delta: -0.0595)
----------------------------------------------------------------------

Your instruction: /what if swap object 1 and object 2
Counterfactual Analysis:
  If you did: SwapObjects({'i': 1, 'j': 2})
  Predicted energy change: +0.3059
  Predicted state change: 2.145
```

## License

MIT

