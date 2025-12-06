# Architecture Documentation

## Overview

Atlas implements a decoder-only transformer architecture (GPT-style) built from scratch in PyTorch. This document provides a deep dive into the system architecture, design decisions, and implementation details.

## Table of Contents

- [System Architecture](#system-architecture)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Data Flow](#data-flow)
- [Design Decisions](#design-decisions)
- [Performance Considerations](#performance-considerations)

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         Atlas LLM                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │    Config    │  │     Data     │  │  Tokenizer   │       │
│  │  Management  │  │   Pipeline   │  │   (tiktoken) │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            │                                │
│                   ┌────────▼─────────┐                      │
│                   │   Model (AtlasLM)│                      │
│                   │  - Embeddings    │                      │
│                   │  - Transformer   │                      │
│                   │  - LM Head       │                      │
│                   └────────┬─────────┘                      │
│                            │                                │
│         ┌──────────────────┼──────────────────┐             │
│         │                  │                  │             │
│  ┌──────▼───────┐  ┌───────▼──────┐  ┌───────▼──────┐       │
│  │   Training   │  │  Inference   │  │    Export    │       │
│  │   Pipeline   │  │  Generation  │  │    (GGUF)    │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
atlas/
├── model/              # Core model components
│   ├── embeddings.py   # Token + positional embeddings
│   ├── attention.py    # Multi-head self-attention
│   ├── mlp.py          # Feed-forward networks
│   ├── transformer.py  # Transformer block
│   └── model.py        # AtlasLM (main model)
│
├── training/           # Training infrastructure
│   ├── trainer.py      # Training loop
│   ├── evaluator.py    # Evaluation logic
│   ├── optimizer.py    # Optimizer setup
│   ├── loss.py         # Loss functions
│   └── checkpoint.py   # Checkpoint management
│
├── tokenizer/          # Tokenization
│   └── tokenizer.py    # Wrapper for tiktoken
│
├── data/               # Data processing
│   ├── dataset.py      # PyTorch Dataset
│   ├── loader.py       # DataLoader utilities
│   └── preprocessing.py # Data cleaning
│
├── config/             # Configuration
│   ├── config.py       # Config dataclasses
│   ├── loader.py       # Config loading
│   └── cli.py          # CLI parsing
│
├── inference/          # Text generation
│   └── generation.py   # Sampling strategies
│
├── export/             # Model export
│   └── gguf.py         # GGUF format writer
│
└── utils/              # Utilities
    └── __init__.py     # Shared utilities
```

## Model Architecture

### Transformer Block

Each transformer block follows the pre-norm architecture:

```
Input
  │
  ├─────────────────┐
  │                 │
  │  LayerNorm      │
  │     │           │
  │  MultiHead      │
  │  Attention      │
  │     │           │
  └─────+───────────┘  (Residual Connection)
        │
        ├─────────────────┐
        │                 │
        │  LayerNorm      │
        │     │           │
        │    MLP          │
        │     │           │
        └─────+───────────┘  (Residual Connection)
              │
           Output
```

### Components Breakdown

#### 1. Token Embeddings (`TokenEmbedding`)

```python
# Input: Token IDs [batch_size, seq_len]
# Output: Embeddings [batch_size, seq_len, hidden_size]

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        self.embedding = nn.Embedding(vocab_size, hidden_size)
```

**Purpose**: Convert discrete token IDs to continuous vector representations.

**Key Features**:
- Learnable embedding lookup table
- Shared weights with output projection (weight tying)
- Vocabulary size: 50,257 (GPT-2 BPE tokenizer)

#### 2. Positional Embeddings (`PositionalEmbedding`)

```python
# Input: Sequence positions [seq_len]
# Output: Position embeddings [seq_len, hidden_size]

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, hidden_size):
        self.embedding = nn.Embedding(max_seq_len, hidden_size)
```

**Purpose**: Encode position information into embeddings.

**Design Choice**: Learned positional embeddings (vs. sinusoidal)
- More flexible for varied sequence lengths
- Better performance in practice
- No extrapolation beyond max_seq_len

#### 3. Combined Embeddings (`CombinedEmbedding`)

```python
# Combines token + position embeddings with dropout
output = dropout(token_emb + pos_emb)
```

#### 4. Multi-Head Attention (`MultiHeadAttention`)

```python
# Input: [batch_size, seq_len, hidden_size]
# Output: [batch_size, seq_len, hidden_size]

Q = input @ W_q  # Query projection
K = input @ W_k  # Key projection
V = input @ W_v  # Value projection

# Split into multiple heads
Q = Q.view(batch, seq_len, num_heads, head_dim)
K = K.view(batch, seq_len, num_heads, head_dim)
V = V.view(batch, seq_len, num_heads, head_dim)

# Scaled dot-product attention with causal mask
scores = (Q @ K.transpose(-2, -1)) / sqrt(head_dim)
scores = scores.masked_fill(causal_mask == 0, -inf)
attn = softmax(scores, dim=-1)
output = attn @ V

# Concatenate heads and project
output = output.view(batch, seq_len, hidden_size)
output = output @ W_o
```

**Key Features**:
- Causal masking for autoregressive generation
- Scaled dot-product attention
- Parallel attention heads
- Optional attention dropout

**Dimensions**:
- `hidden_size = num_heads × head_dim`
- Default: 12 heads × 64 dim = 768 hidden_size (SMALL)

#### 5. Feed-Forward Network (`MLP`)

```python
# Input: [batch_size, seq_len, hidden_size]
# Output: [batch_size, seq_len, hidden_size]

hidden = activation(input @ W1 + b1)  # Expand
output = dropout(hidden @ W2 + b2)    # Project back
```

**Structure**:
- Two linear layers with activation in between
- Expansion ratio: typically 4x (hidden → 4×hidden → hidden)
- Activation: GELU (default) or SiLU

**Purpose**: Add non-linearity and increase model capacity.

#### 6. Transformer Block (`TransformerBlock`)

```python
def forward(x, mask=None):
    # Attention sub-layer
    x = x + attention(layer_norm1(x), mask)
    
    # MLP sub-layer
    x = x + mlp(layer_norm2(x))
    
    return x
```

**Architecture**: Pre-norm (LayerNorm before sub-layers)
- More stable training than post-norm
- Better gradient flow
- Standard in modern transformers

#### 7. AtlasLM (Full Model)

```python
class AtlasLM(nn.Module):
    def __init__(self, config):
        self.embeddings = CombinedEmbedding(...)
        self.blocks = nn.ModuleList([
            TransformerBlock(...) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.embeddings.token_embedding.weight
```

**Forward Pass**:
```python
def forward(tokens, mask=None):
    # 1. Embed tokens
    x = self.embeddings(tokens)
    
    # 2. Apply transformer blocks
    for block in self.blocks:
        x = block(x, mask)
    
    # 3. Final layer norm
    x = self.ln_f(x)
    
    # 4. Project to vocabulary
    logits = self.lm_head(x)
    
    return logits
```

### Model Configurations

| Config | Params | Layers | Heads | Hidden | MLP | Context |
|--------|--------|--------|-------|--------|-----|---------|
| TINY | 40M | 6 | 8 | 512 | 2048 | 512 |
| SMALL | 124M | 12 | 12 | 768 | 3072 | 512 |
| DEFAULT | 350M | 24 | 16 | 1024 | 4096 | 512 |
| LARGE | 500M | 24 | 16 | 1280 | 5120 | 512 |

**Parameter Calculation**:
```
Embeddings: vocab_size × hidden_size × 2  (token + position)
Attention: hidden_size × hidden_size × 4 × num_layers  (Q,K,V,O projections)
MLP: hidden_size × mlp_size × 2 × num_layers  (up + down)
Output: vocab_size × hidden_size  (weight tied with embeddings)
```

## Training Pipeline

### Training Loop

```
┌─────────────────────────────────────────────────────────┐
│                     Training Epoch                      │
└─────────────────────────────────────────────────────────┘
                           │
                ┌──────────▼──────────┐
                │   Load Batch        │
                │  (DataLoader)       │
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │  Forward Pass       │
                │  - Embeddings       │
                │  - Transformer      │
                │  - Compute Loss     │
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │  Backward Pass      │
                │  - Compute Grads    │
                │  - Accumulate       │
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │  Optimizer Step     │
                │  - Clip Gradients   │
                │  - Update Weights   │
                │  - LR Schedule      │
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │  Logging            │
                │  - Loss, PPL        │
                │  - Throughput       │
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │  Checkpoint         │
                │  (if step % N == 0) │
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │  Validation         │
                │  (if epoch ends)    │
                └─────────────────────┘
```

### Gradient Accumulation

For memory-constrained training:

```python
effective_batch_size = batch_size × gradient_accumulation_steps

for step in range(gradient_accumulation_steps):
    # Forward pass (small batch)
    loss = model(batch) / gradient_accumulation_steps
    
    # Backward pass (accumulate gradients)
    loss.backward()
    
# Update weights (after accumulation)
optimizer.step()
optimizer.zero_grad()
```

**Benefits**:
- Larger effective batch sizes without OOM
- Better gradient estimates
- More stable training

### Learning Rate Schedule

**Warmup + Cosine Decay**:

```python
if step < warmup_steps:
    lr = base_lr * (step / warmup_steps)
else:
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(π * progress))
```

**Schedule Visualization**:
```
LR  │     ╱─────╲
    │    ╱       ╲
    │   ╱         ╲___
    │  ╱              ╲___
    │ ╱                   ╲___
    └─────────────────────────────> Steps
      Warmup    Cosine Decay
```

## Data Flow

### Training Data Flow

```
Raw Text Files (Wikipedia)
         │
         ▼
    Preprocessing
    - Clean text
    - Remove special chars
    - Normalize Unicode
         │
         ▼
    Tokenization (tiktoken)
    - BPE encoding
    - Add special tokens
         │
         ▼
    TextDataset
    - Chunk into sequences
    - Create overlapping windows
         │
         ▼
    DataLoader
    - Batch sequences
    - Shuffle (training)
    - Pin memory
         │
         ▼
    Model Training
```

### Sequence Creation

```
Text: "The quick brown fox jumps over the lazy dog"

Tokenized: [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290]

With stride (overlap):
Sequence 1: [464, 2068, 7586, 21831]  (tokens 0-3)
Sequence 2: [2068, 7586, 21831, 18045] (tokens 1-4)  <- overlap
Sequence 3: [7586, 21831, 18045, 625]  (tokens 2-5)  <- overlap
...
```

**Parameters**:
- `max_seq_len`: Maximum sequence length (512 default)
- `stride`: Step size between sequences (< max_seq_len for overlap)
- Overlap helps model learn context transitions

## Design Decisions

### 1. Pre-Norm vs Post-Norm

**Choice**: Pre-norm (LayerNorm before attention/MLP)

**Rationale**:
- More stable training gradients
- Better convergence in deep networks
- Standard in modern transformers (GPT-3, LLaMA)

### 2. Weight Tying

**Choice**: Tie input embeddings and output projection weights

**Rationale**:
- Reduces parameters (~50M for 50K vocab × 1K hidden)
- Improves performance (shared semantic space)
- Standard practice in language models

### 3. Learned vs Sinusoidal Positional Embeddings

**Choice**: Learned positional embeddings

**Rationale**:
- More flexible, learned from data
- Better empirical performance
- No extrapolation needed (fixed max_seq_len)

### 4. Activation Functions

**Choice**: GELU (default), with SiLU option

**Rationale**:
- GELU: Smooth, performs well in transformers (GPT, BERT)
- SiLU: Alternative with similar properties (LLaMA)
- Both outperform ReLU in practice

### 5. Attention Implementation

**Choice**: Standard scaled dot-product with causal masking

**Rationale**:
- Well-understood and proven
- Efficient implementation in PyTorch
- Compatible with future optimizations (FlashAttention)

### 6. Checkpoint Strategy

**Choice**: Multiple checkpoint types (step, epoch, best)

**Rationale**:
- Step checkpoints: Resume training easily
- Epoch checkpoints: Evaluation milestones
- Best checkpoint: Best model for inference
- Automatic cleanup prevents disk overflow

## Performance Considerations

### Memory Optimization

**Gradient Checkpointing**:
- Trade computation for memory
- Recompute activations during backward pass
- ~30% slower, ~40% less memory
- Essential for large models on limited VRAM

**Memory Breakdown** (DEFAULT config, batch=4, seq=512):
```
Model Parameters:     ~1.4 GB (350M params × 4 bytes)
Activations:          ~8 GB   (saved for backprop)
Gradients:            ~1.4 GB (same size as params)
Optimizer States:     ~2.8 GB (Adam: 2× param size)
                      ────────
Total:                ~13.6 GB

With Gradient Checkpointing:
Activations:          ~3 GB   (65% reduction)
Total:                ~8.6 GB
```

### Computational Efficiency

**Attention Complexity**: O(n² × d)
- n = sequence length
- d = hidden dimension
- Quadratic in sequence length (main bottleneck)

**Optimization Strategies**:
1. Smaller sequences (512 vs 1024) → 4× faster attention
2. Fewer layers → Linear speedup
3. Smaller hidden size → Quadratic speedup in attention
4. Batch processing → Better GPU utilization

### Throughput Optimization

**Factors Affecting Speed**:
1. **Batch Size**: Larger = better GPU utilization (up to memory limit)
2. **Sequence Length**: Shorter = faster (quadratic attention cost)
3. **Model Size**: Smaller = faster (linear)
4. **Data Loading**: Use `num_workers > 0`, `pin_memory=True`
5. **Mixed Precision**: FP16/BF16 (2× speedup, half memory)

## Future Enhancements

Planned architectural improvements:

1. **FlashAttention**: O(n) memory attention
2. **Rotary Position Embeddings (RoPE)**: Better length extrapolation
3. **Multi-Query Attention**: Faster inference
4. **Mixture of Experts (MoE)**: Conditional computation
5. **Grouped Query Attention (GQA)**: Balance between MHA and MQA

---

**Last Updated**: December 7, 2025
