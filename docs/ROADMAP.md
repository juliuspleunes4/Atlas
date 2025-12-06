# Atlas Roadmap

**Goal**: Build a from-scratch LLM that can be trained, evaluated, and exported to `.gguf` format for inference. The final model should produce coherent, useful responses—not garbage output—while being realistic in scope (smaller than GPT-scale but functionally competent).

---

## Target Specifications

### Model Architecture
- **Model Size**: ~100-350M parameters (minimum for useful conversation)
  - Reality check: GPT-2 Small (117M) can hold basic conversations
  - GPT-2 Medium (345M) is noticeably more coherent
  - Below 100M typically produces repetitive/nonsensical output
- **Architecture**: Decoder-only transformer (GPT-2 style)
- **Vocabulary Size**: 50,261 tokens (GPT-2 BPE tokenizer via tiktoken)
- **Context Length**: 1024 tokens (standard for conversation)
- **Hidden Size**: 768 (GPT-2 Small standard, good balance)
- **Number of Layers**: 12 (minimum for decent reasoning)
- **Attention Heads**: 12 (must divide hidden_size evenly)
- **MLP Ratio**: 4.0 (standard FFN expansion → 3072 intermediate size)
- **Activation**: GELU (modern standard)
- **Estimated Parameters**: ~117M (similar to GPT-2 Small)

### Training Target
- **Dataset**: Quality conversational/instruction data
  - Option 1: OpenWebText subset (~8GB, diverse web text)
  - Option 2: Combined WikiText + small instruction dataset
  - Option 3: TinyStories (simple but coherent) + dialogue data
- **Training Tokens**: 1-5B tokens (minimum for emergent abilities)
  - Note: GPT-2 was trained on 40GB (~10B tokens)
  - We can get decent results with 1-5B on focused data
- **Batch Size**: 16-32 sequences (memory constrained)
- **Training Time**: 1-3 days on modern GPU (RTX 3080+) or 1-2 weeks on CPU
- **Loss Target**: Perplexity < 30 on validation set
- **Hardware**: 16GB+ GPU recommended (or train on CPU with patience)

### Success Criteria
1. **Model trains successfully** without NaN losses or instability
2. **Generates coherent text** at inference time (not just random tokens)
3. **Responds meaningfully** to simple prompts (e.g., "Once upon a time")
4. **Exports cleanly** to `.gguf` format with correct metadata
5. **Inference works** in external tools (e.g., llama.cpp, if applicable)

### Constraints
- **Realistic scope**: Focus on quality over scale
- **Reproducible**: Clear configs and random seeds
- **Well-tested**: Every component has passing tests
- **Documented**: Code should be clear and commented

---

## Phase 0: Project Foundation ✅

### 0.1 Repository Structure ✅
- [x] Create initial directory structure
- [x] Set up `atlas/` package with `__init__.py`
- [x] Create subdirectories:
  - `atlas/model/` – model architecture components
  - `atlas/tokenizer/` – tokenization logic
  - `atlas/training/` – training loop and optimization
  - `atlas/data/` – dataset loading and processing
  - `atlas/config/` – configuration management
  - `atlas/inference/` – generation and inference
  - `atlas/utils/` – logging, checkpointing, metrics
  - `atlas/export/` – model export (GGUF, etc.)
  - `tests/` – all test modules
  - `scripts/` – CLI scripts for training, inference, export

### 0.2 Development Environment ✅
- [x] Create `requirements.txt` with core dependencies:
  - PyTorch
  - NumPy
  - tiktoken (tokenizer)
  - tqdm (progress bars)
  - pyyaml (config files)
  - pytest (testing)
  - gguf (GGUF export)
- [x] Create `setup.py` for package installation
- [x] Add `.gitignore` for Python, checkpoints, data, logs
- [ ] Set up linting/formatting (black, flake8, mypy)
- [ ] Configure pre-commit hooks (optional but recommended)

### 0.3 Documentation Setup ✅
- [x] Create `docs/ROADMAP.md`
- [x] Create `docs/CHANGELOG.md`
- [x] Update `README.md` with project overview, quickstart
- [ ] Add architecture documentation (optional: `docs/ARCHITECTURE.md`)

---

## Phase 1: Configuration System

### 1.1 Configuration Schema
- [ ] Design configuration dataclass/schema for:
  - Model architecture (layers, heads, hidden_size, vocab_size, max_seq_len, dropout, etc.)
  - Training parameters (batch_size, learning_rate, warmup_steps, max_steps, gradient_accumulation, etc.)
  - Data parameters (dataset paths, tokenizer path, sequence length)
  - Logging/checkpointing (log_interval, checkpoint_interval, checkpoint_dir)
  - Inference parameters (temperature, top_k, top_p, max_new_tokens)
- [ ] Implement config loading from YAML/JSON files
- [ ] Implement config CLI overrides (argparse or similar)
- [ ] Add validation for required fields and sensible defaults

### 1.2 Configuration Testing
- [ ] Write tests for config loading
- [ ] Write tests for config validation (invalid values, missing fields)
- [ ] Write tests for CLI override behavior

---

## Phase 2: Tokenizer Integration ✅

### 2.1 Tokenizer Implementation ✅
- [x] Choose tokenizer backend (tiktoken)
- [x] Implement `Tokenizer` class with methods:
  - `encode(text: str) -> List[int]`
  - `decode(tokens: List[int]) -> str`
  - `vocab_size() -> int`
  - Load from pretrained vocab (GPT-2)
- [x] Add special tokens (BOS, EOS, PAD, UNK)
- [x] Add batch encoding/decoding support

### 2.2 Tokenizer Testing ✅
- [x] Test encoding/decoding round-trip
- [x] Test special token handling
- [x] Test batch operations
- [x] Test edge cases (empty strings, very long strings, Unicode characters)

### 2.3 Vocabulary Preparation ✅
- [x] Using pretrained GPT-2 tokenizer (tiktoken)
- [x] Tokenizer can be configured via encoding_name parameter
- [x] Special tokens added to vocabulary automatically

---

## Phase 3: Model Architecture

### 3.1 Core Components

#### 3.1.1 Embeddings
- [ ] Implement `TokenEmbedding` layer
- [ ] Implement `PositionalEmbedding` layer (learned or sinusoidal)
- [ ] Combine token + position embeddings
- [ ] Add dropout after embeddings

#### 3.1.2 Attention Mechanism
- [ ] Implement multi-head self-attention:
  - Q, K, V projections
  - Scaled dot-product attention
  - Causal masking (for autoregressive LM)
  - Attention dropout
  - Output projection
- [ ] Add support for optional attention bias (ALiBi, RoPE, etc.) [optional for v1]
- [ ] Ensure efficient implementation (fused kernels if using Flash Attention) [optional for v1]

#### 3.1.3 Feed-Forward Network (MLP)
- [ ] Implement MLP block:
  - Linear → Activation (GELU, SiLU, etc.) → Linear
  - Dropout
- [ ] Make hidden dimension configurable (typically 4x hidden_size)

#### 3.1.4 Transformer Block
- [ ] Implement `TransformerBlock`:
  - Layer norm → Multi-head attention → Residual
  - Layer norm → MLP → Residual
- [ ] Support both pre-norm and post-norm architectures (prefer pre-norm)
- [ ] Add dropout as configured

#### 3.1.5 Output Head
- [ ] Implement final layer norm
- [ ] Implement language modeling head (linear projection to vocab_size)
- [ ] Optionally tie input embedding weights with output projection (weight tying)

### 3.2 Full Model Assembly
- [ ] Implement `AtlasLM` model class:
  - Embedding layer
  - Stack of N transformer blocks
  - Final layer norm
  - LM head
  - Forward pass returning logits
- [ ] Add parameter initialization (Xavier, normal, etc.)
- [ ] Add method to count parameters
- [ ] Add device handling (CPU/CUDA)

### 3.3 Model Testing
- [ ] Test forward pass with small model (1 layer, 64 hidden, 512 vocab)
- [ ] Verify output shapes (batch, seq_len, vocab_size)
- [ ] Test causal masking (future tokens are masked)
- [ ] Test gradient flow (backward pass runs without error)
- [ ] Test with different batch sizes, sequence lengths
- [ ] Test parameter initialization ranges
- [ ] Test device placement (CPU, CUDA if available)

---

## Phase 4: Data Pipeline ✅

### 4.1 Dataset Abstraction ✅
- [x] Implement `TextDataset` class (PyTorch Dataset):
  - Load raw text from file(s) - single or multiple
  - Tokenize text using Tokenizer
  - Split into fixed-length sequences (sliding window with configurable stride)
  - Proper handling of leftover tokens (no padding)
- [x] Implement `DataLoader` integration:
  - Batching with collate_batch()
  - Shuffling support
  - Efficient loading (num_workers, pin_memory)
  - Train/val/test splitting with reproducible seeds

### 4.2 Data Preprocessing ✅
- [x] Implement preprocessing utilities:
  - clean_text(): Text cleaning, normalization, Unicode handling
  - chunk_text(): Split long documents with overlap
  - load_text_file(): Load and clean text files
  - load_jsonl(): JSONL format support with custom fields
  - iterate_documents(): Memory-efficient iteration
  - count_tokens(): Token counting utility
  - filter_by_length(): Filter by token count
- [x] Support multiple data formats (txt, JSONL)
- [x] Data is loaded on-demand (no pre-tokenization needed)

### 4.3 Data Testing ✅
- [x] Test dataset loading with various text samples (27 tests)
- [x] Test batching and sequence length consistency (16 tests)
- [x] Test shuffling and splitting behavior
- [x] Test preprocessing functions (29 tests)
- [x] Test edge cases (empty files, single token sequences, Unicode, etc.)
- [x] All 72 data pipeline tests passing

---

## Phase 5: Training Loop

### 5.1 Loss and Optimization
- [ ] Implement cross-entropy loss for language modeling
- [ ] Implement optimizer setup (AdamW recommended)
- [ ] Implement learning rate scheduler:
  - Warmup phase (linear warmup)
  - Decay phase (cosine, linear, or constant)
- [ ] Implement gradient clipping
- [ ] Implement gradient accumulation (for large effective batch sizes)

### 5.2 Training Loop Implementation
- [ ] Implement main training loop:
  - Iterate over batches
  - Forward pass
  - Compute loss
  - Backward pass
  - Optimizer step (with gradient accumulation)
  - LR scheduler step
- [ ] Add training metrics:
  - Loss (per step, moving average)
  - Perplexity
  - Tokens per second
  - Learning rate tracking
- [ ] Add logging (console and/or file):
  - Log every N steps
  - Log to stdout with tqdm progress bar
- [ ] Add checkpointing:
  - Save model, optimizer, scheduler state
  - Save every N steps or at end of epoch
  - Resume from checkpoint
- [ ] Add early stopping (optional)

### 5.3 Evaluation Loop
- [ ] Implement evaluation loop:
  - Run model on validation set
  - Compute validation loss and perplexity
  - No gradient computation (eval mode)
- [ ] Run evaluation periodically during training
- [ ] Log evaluation metrics

### 5.4 Training Testing
- [ ] Test training loop runs without crashing (1-2 steps, tiny model, tiny data)
- [ ] Test loss decreases over a few steps (overfitting test on tiny data)
- [ ] Test checkpointing (save and resume)
- [ ] Test gradient accumulation correctness
- [ ] Test LR scheduler correctness (warmup → decay)
- [ ] Test evaluation loop

### 5.5 Training Script
- [ ] Create `scripts/train.py`:
  - Parse config file
  - Initialize model, tokenizer, dataset, optimizer
  - Run training loop
  - Handle interruptions gracefully (Ctrl+C)
- [ ] Add CLI arguments for config overrides

---

## Phase 6: Inference and Generation

### 6.1 Generation Implementation
- [ ] Implement autoregressive generation:
  - Start with prompt tokens
  - Generate one token at a time
  - Append to sequence, repeat
- [ ] Implement sampling strategies:
  - Greedy decoding (argmax)
  - Temperature sampling
  - Top-k sampling
  - Top-p (nucleus) sampling
- [ ] Implement generation stopping conditions:
  - Max length reached
  - EOS token generated
- [ ] Add support for batched generation (optional for v1)

### 6.2 Inference Testing
- [ ] Test generation with tiny model
- [ ] Test different sampling strategies produce different outputs
- [ ] Test prompt conditioning works correctly
- [ ] Test stopping conditions (max length, EOS)
- [ ] Test generation with various temperatures

### 6.3 Inference Script
- [ ] Create `scripts/infer.py`:
  - Load model from checkpoint
  - Load tokenizer
  - Accept prompt from CLI or stdin
  - Generate and print response
- [ ] Add CLI arguments for generation parameters (temperature, top_k, etc.)

---

## Phase 7: Model Export to GGUF

### 7.1 GGUF Format Understanding
- [ ] Research GGUF format specification
- [ ] Understand required metadata fields
- [ ] Understand tensor layout and quantization options

### 7.2 Export Implementation
- [ ] Implement export script to convert PyTorch model to GGUF:
  - Extract model weights
  - Map Atlas model structure to GGUF tensor names
  - Write metadata (architecture, hyperparameters, tokenizer info)
  - Write tensors in GGUF format
- [ ] Support float16 and float32 export (quantization optional for v1)
- [ ] Validate exported GGUF file (file integrity, loadable by llama.cpp)

### 7.3 Export Testing
- [ ] Test export script runs without error
- [ ] Test exported GGUF file has correct metadata
- [ ] Test exported GGUF file has correct tensor shapes
- [ ] Test exported GGUF file can be loaded by llama.cpp or another GGUF-compatible tool
- [ ] Compare outputs from PyTorch model vs. GGUF model (should match)

### 7.4 Export Script
- [ ] Create `scripts/export_gguf.py`:
  - Load trained model checkpoint
  - Convert to GGUF format
  - Save to output file
- [ ] Add CLI arguments for model path, output path, quantization options

---

## Phase 8: End-to-End Integration

### 8.1 Full Pipeline Test
- [ ] Create end-to-end test with tiny model and tiny dataset:
  - Train for a few steps
  - Save checkpoint
  - Run inference
  - Export to GGUF
  - Verify GGUF output
- [ ] Document the full workflow in README.md

### 8.2 Real Training Run
- [ ] Prepare real dataset (placeholder/TODO for dataset source)
- [ ] Train small model (e.g., 6 layers, 512 hidden, 8 heads) on real data
- [ ] Monitor training loss, validation loss, perplexity
- [ ] Generate samples periodically to qualitatively evaluate output
- [ ] Train until validation loss plateaus or reasonable performance achieved

### 8.3 Final GGUF Export
- [ ] Export trained model to GGUF
- [ ] Test GGUF inference with llama.cpp or similar tool
- [ ] Validate generated text quality (coherent, not garbage)

---

## Phase 9: Optimization and Refinement

### 9.1 Performance Optimization
- [ ] Profile training loop (identify bottlenecks)
- [ ] Optimize data loading (prefetching, caching)
- [ ] Optimize model forward pass (use of fused kernels, Flash Attention)
- [ ] Benchmark tokens/second, memory usage

### 9.2 Model Quality Improvements
- [ ] Experiment with hyperparameters:
  - Model size (layers, heads, hidden_size)
  - Learning rate, warmup steps, batch size
  - Dropout, weight decay
- [ ] Add more training data (if available)
- [ ] Implement additional training techniques:
  - Mixed precision training (FP16/BF16)
  - Gradient checkpointing (for larger models)
- [ ] Add validation metrics beyond perplexity (optional: BLEU, ROUGE, custom evals)

### 9.3 Quantization (Optional)
- [ ] Implement or integrate quantization for GGUF export:
  - Q4_0, Q5_0, Q8_0, etc.
- [ ] Test quantized model inference quality vs. speed tradeoff

### 9.4 Additional Export Formats (Optional)
- [ ] Export to ONNX
- [ ] Export to SafeTensors
- [ ] Export to TorchScript

---

## Phase 10: Documentation and Release

### 10.1 Documentation
- [ ] Complete README.md:
  - Project overview
  - Installation instructions
  - Quickstart guide (train, infer, export)
  - Example usage
- [ ] Document configuration file format (example configs)
- [ ] Document training best practices
- [ ] Document generation parameters and their effects
- [ ] Document export process and GGUF usage

### 10.2 Examples and Tutorials
- [ ] Add example config files for different model sizes (tiny, small, medium)
- [ ] Add example training script invocations
- [ ] Add example inference script invocations
- [ ] Add example export script invocations

### 10.3 Testing and CI
- [ ] Ensure all tests pass
- [ ] Add continuous integration (GitHub Actions, etc.):
  - Run tests on push
  - Lint and format checks
- [ ] Achieve reasonable test coverage (aim for >80%)

### 10.4 Release Preparation
- [ ] Tag release version (e.g., v0.1.0)
- [ ] Write release notes
- [ ] Update CHANGELOG.md with all changes
- [ ] Archive trained model checkpoint and GGUF file (optional: upload to Hugging Face)

---

## Appendix: Key Design Decisions

### Model Architecture
- **Architecture**: Decoder-only Transformer (GPT-style)
- **Attention**: Multi-head self-attention with causal masking
- **Normalization**: Layer normalization (pre-norm recommended)
- **Activation**: GELU or SiLU
- **Positional Encoding**: Learned embeddings (or sinusoidal/RoPE as alternative)

### Training Strategy
- **Optimizer**: AdamW (betas=[0.9, 0.95], weight_decay=0.1)
- **LR Schedule**: Linear warmup + cosine decay
- **Batch Size**: As large as GPU memory allows (use gradient accumulation)
- **Sequence Length**: 512–2048 tokens (context window)
- **Mixed Precision**: FP16 or BF16 for speed (optional but recommended)

### Data Requirements
- **Minimum**: A few million tokens for basic coherence
- **Recommended**: 1B+ tokens for decent quality
- **Source**: Public domain text, books, Wikipedia, code (as appropriate)

### Model Size Targets
- **Tiny** (for testing): 1-2 layers, 128-256 hidden, 1-2 heads (~1M params)
- **Small** (baseline): 6-12 layers, 512-768 hidden, 8-12 heads (~50-150M params)
- **Medium** (stretch goal): 24 layers, 1024 hidden, 16 heads (~350M params)

### Export Target
- **Format**: GGUF (compatible with llama.cpp)
- **Precision**: FP16 or quantized (Q4_0, Q8_0)
- **Use case**: Inference on CPU or consumer GPU

---

## Success Criteria

At the end of this roadmap, Atlas should:

1. **Train**: Successfully train a Transformer-based language model on a text corpus.
2. **Generate**: Produce coherent, contextually relevant text from prompts (not random garbage).
3. **Export**: Convert the trained model to GGUF format.
4. **Infer**: Run inference on the GGUF model using llama.cpp or compatible tools.
5. **Test**: Have comprehensive test coverage for all major components.
6. **Document**: Provide clear documentation for setup, training, inference, and export.

The final model doesn't need to rival GPT-4, but it should demonstrate that the entire pipeline—from raw text to production-ready GGUF inference—works correctly and produces useful outputs.
