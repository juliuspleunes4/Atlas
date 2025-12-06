# Changelog

All notable changes to Atlas will be documented in this file.

---

## 2025-12-06 - Fix DataLoader Batch Format for Training

- **Fixed batch collation** (`atlas/data/loader.py`):
  - `collate_batch()` now returns dictionary with `'input_ids'` key instead of plain tensor
  - Matches expected format in `Trainer.train_step()` which accesses `batch['input_ids']`
  - Updated docstrings and examples to reflect dictionary structure
- **Updated tests** (`tests/test_data.py`):
  - Fixed 6 data loader tests to expect dictionary structure
  - Updated assertions to check `batch['input_ids']` instead of direct tensor access
  - All 72 data tests passing
- **Training now fully functional**:
  - Complete pipeline from data loading through training loop works correctly
  - Model, tokenizer, data loading, and training all integrated properly

## 2025-12-06 - Complete Training Pipeline and GPU-Optimized Configs

- **Complete training pipeline** (`scripts/start_training.ps1` / `.sh`):
  - **Zero-friction onboarding**: Handles entire pipeline from clone to training
  - Interactive prompts for all decisions (GPU config, data prep, etc.)
  - Automatically checks and installs Atlas package (`pip install -e .`)
  - Creates virtual environment if missing
  - Checks for training data and prompts user if not found
  - Prepares data automatically with user confirmation
  - Lets user choose GPU configuration (small/default/large)
  - Comprehensive error handling for all edge cases
  - Perfect for new developers - just run one command!
- **GPU-optimized configuration presets**:
  - `configs/small.yaml`: ~124M params, 6-8GB VRAM, fastest training
  - `configs/default.yaml`: ~350M params, 12-14GB VRAM, balanced (recommended)
  - `configs/large.yaml`: ~500M params, 14-15GB VRAM, maximum quality
  - Optimized for RTX 5060 Ti 16GB but adaptable to other GPUs
  - Clear memory usage and parameter count documentation
- **Data preparation script** (`scripts/prepare_data.py`):
  - Automatic extraction of Wikipedia SimpleEnglish dataset from zip
  - Organizes text files with clean naming convention
  - Displays statistics (file count, total size)
  - `--list` flag to show available datasets
  - Cross-platform (Windows/Linux/Mac)
- **Setup automation scripts** (`scripts/setup_and_train.ps1` / `.sh`):
  - Automatically checks prerequisites (Python, venv, dependencies)
  - Auto-prepares data if not already done
  - Launches training with default config
  - Colored output and clear status messages
- **Documentation updates**:
  - Updated README.md with "Absolute Beginners" quick start
  - Added GPU configuration comparison table
  - Clear instructions for automated vs manual workflows
  - Updated data/README.md with usage examples
- **Developer experience highlights**:
  - **One command from zero to training**: `.\scripts\start_training.ps1`
  - Handles missing dependencies, data, configs automatically
  - Interactive prompts guide users through choices
  - Comprehensive error messages with next steps
  - Works on fresh clone with zero manual setup
- **Code quality**:
  - Removed emojis from all scripts to prevent mojibake issues
  - Replaced with ASCII tags: [INFO], [SUCCESS], [ERROR], [WARNING]
  - Robust error handling and input validation
  - Cross-platform compatibility (Windows PowerShell + Linux/Mac Bash)

## 2025-12-06 - Enhanced Training CLI and Logging

- **Comprehensive logging system** for training script:
  - Dual logging to console and persistent `training.log` file
  - Session-based logging with timestamps and separators
  - Resume-aware logging (appends to existing log with clear session markers)
  - Detailed progress tracking and metrics reporting
- **Rich console output** during training:
  - System information (PyTorch version, CUDA device, GPU memory)
  - Model statistics (parameters, size in fp32/fp16)
  - Dataset statistics (samples, tokens, loading time)
  - Training configuration and hyperparameters
  - Real-time progress indicators (percentage complete, ETA)
  - Per-epoch metrics (loss, perplexity, throughput, learning rate)
  - GPU memory usage monitoring
  - Validation results with improvement tracking
  - Best model indicators and checkpoint confirmations
- **Improved error handling and interruption**:
  - Graceful Ctrl+C handling with checkpoint saving
  - Emergency checkpoints on errors
  - Clear resume instructions on interruption
  - Detailed error logging
- **Training statistics**:
  - Tokens per second throughput
- **Test coverage**:
  - Added 4 comprehensive tests for logging functionality in `tests/test_train_script.py`
  - Tests cover: log file creation, resume mode, directory creation, dual logging
  - All 292 tests passing (288 existing + 4 new logging tests)
  - Estimated epochs and training time
  - Steps per epoch calculation
  - Total training time and per-step averages
  - Memory usage tracking (GPU allocated/reserved)
- **Better user experience**:
  - Clear section markers and visual separators
  - Progress indicators with emojis (🎉 success, ⚠️ interrupt, ❌ error, 🌟 new best)
  - Comprehensive session summaries
  - Easy-to-follow resume commands

## 2025-12-06 - GGUF export (Phase 7)

- Implemented complete GGUF export functionality for model deployment
- **GGUF writer** (`atlas/export/gguf.py`):
  - `GGUFWriter`: Complete GGUF file format writer
  - Support for GGUF v3 format specification
  - Metadata serialization (strings, integers, floats, booleans, arrays)
  - Tensor serialization with proper alignment
  - F32 and F16 quantization support
  - Automatic tensor name mapping from Atlas to GGUF format
  - Binary file writing with correct byte ordering
- **Export function** (`export_atlas_to_gguf`):
  - Convert PyTorch models to GGUF format
  - Model architecture metadata export
  - Tokenizer configuration export
  - State dict tensor mapping and conversion
  - Support for all transformer blocks, embeddings, output head
- **Export script** (`scripts/export_gguf.py`):
  - CLI interface for model export
  - Checkpoint loading with config inference
  - F32/F16 quantization options
  - File size reporting
  - Error handling and validation
- **Testing** (`tests/test_export.py`):
  - Added 17 comprehensive tests for GGUF export
  - Test writer functionality, metadata, tensors
  - Test file structure, magic numbers, headers
  - Test F16 creates smaller files than F32
  - Test full model export pipeline
  - All 288 tests passing (17 export tests)

## 2025-12-06 - Inference script (Phase 6.3)

- Implemented production-ready inference script for text generation
- **Inference script** (`scripts/infer.py`):
  - Complete command-line interface for text generation
  - Multiple input modes: single prompt, prompts file, interactive
  - Configurable sampling parameters (temperature, top-k, top-p)
  - Greedy vs sampling decoding modes
  - Checkpoint loading with config inference
  - Batch generation from file (one prompt per line)
  - Interactive mode with prompt loop
  - Output to file or stdout
  - Separator customization for multiple generations
  - Device selection (CUDA/CPU)
  - Graceful error handling
- **Testing** (`tests/test_infer_script.py`):
  - Added 12 comprehensive tests for inference script
  - Test prompts loading, checkpoint loading, config validation
  - Test output formatting, file I/O
  - All 271 tests passing (12 inference script tests)

## 2025-12-06 - Training script (Phase 5.5)

- Implemented production-ready training script for end-to-end model training
- **Training script** (`scripts/train.py`):
  - Complete command-line interface using argparse
  - YAML configuration file parsing
  - Model, tokenizer, dataset, optimizer initialization from config
  - Training loop with checkpointing and evaluation
  - Graceful interrupt handling (Ctrl+C saves checkpoint)
  - CLI argument overrides for key parameters
  - Progress tracking with logging
  - Automatic device selection (CUDA/CPU)
  - Support for multiple data files (comma-separated paths)
  - Emergency checkpoint on exceptions
  - Resume training from checkpoint with `--resume`
- **Configuration example** (`scripts/config_example.yaml`):
  - Sample YAML config with all training parameters
  - Model architecture settings (vocab size, layers, heads, etc.)
  - Training hyperparameters (LR, batch size, scheduler)
  - Data settings (max sequence length, num workers)
  - Checkpointing configuration
- **Testing** (`tests/test_train_script.py`):
  - Added 9 tests for training script functionality
  - Test config loading, model creation, dataset creation
  - Test CLI override behavior, device selection
  - All 259 tests passing (9 training script tests)

## 2025-12-06 - Inference and text generation (Phase 6)

- Implemented autoregressive text generation with multiple sampling strategies
- **Generation features** (`atlas/inference/generation.py`):
  - `TextGenerator`: Main generation class with autoregressive decoding
  - `GenerationConfig`: Configuration dataclass for generation parameters
  - `generate_text()`: Convenience function for quick text generation
  - **Sampling strategies**:
    - Greedy decoding (argmax, do_sample=False)
    - Temperature sampling (adjustable randomness)
    - Top-k sampling (sample from top k tokens)
    - Top-p (nucleus) sampling (sample from cumulative probability mass)
    - Combined top-k + top-p for fine-grained control
  - **Stopping conditions**:
    - Maximum token length (max_new_tokens)
    - EOS token detection (eos_token_id)
  - Batch generation support
  - Device-aware (CPU/GPU)
- **Testing** (`tests/test_inference.py`):
  - Added 21 comprehensive tests for generation
  - Test all sampling strategies (greedy, temperature, top-k, top-p)
  - Test configuration validation, EOS stopping, batch generation
  - Test determinism of greedy, randomness of sampling
  - All 250 tests passing (21 inference tests)

## 2025-12-06 - Checkpointing system (Phase 5, Task 4)

- Implemented `CheckpointManager` for model state persistence
- **Checkpointing features** (`atlas/training/checkpoint.py`):
  - `CheckpointManager`: Complete checkpoint management system
  - `CheckpointMetadata`: Dataclass for checkpoint metadata (step, epoch, loss, perplexity, LR)
  - `save_checkpoint()`: Save model, optimizer, scheduler state with metadata
  - `load_checkpoint()`: Restore training state from checkpoint
  - `save_best_checkpoint()`: Track and save best model based on validation loss
  - `load_best_checkpoint()`: Load the best performing model
  - `load_latest_checkpoint()`: Resume from most recent checkpoint
  - Automatic cleanup of old checkpoints (configurable retention)
  - JSON metadata files for easy inspection
  - `list_checkpoints()`: Query available checkpoints
- **Testing** (`tests/test_training.py`):
  - Added 10 comprehensive tests for CheckpointManager
  - Test save/load with model, optimizer, scheduler
  - Test best model tracking, latest checkpoint loading
  - Test automatic cleanup, metadata persistence
  - All 229 tests passing (56 training tests total)
- **Phase 5 (Training Loop) complete**: Loss, optimizer, training loop, evaluation, and checkpointing all implemented and tested

## 2025-12-06 - Evaluation loop implementation (Phase 5, Task 3)

- Implemented `Evaluator` class for validation without gradient computation
- **Evaluation features** (`atlas/training/evaluator.py`):
  - `Evaluator`: Evaluation loop with no gradient computation
  - `EvaluationMetrics`: Container for validation metrics (loss, perplexity, token count)
  - `evaluate_step()`: Single evaluation step
  - `evaluate()`: Full validation with progress bar and metric tracking
  - `evaluate_model()`: Convenience function for quick evaluation
  - Token-weighted loss computation for accurate metrics
- **Trainer integration**:
  - Added `Trainer.evaluate()` method for seamless validation during training
  - Integrated evaluator with existing training workflow
- **Testing** (`tests/test_training.py`):
  - Added 9 comprehensive tests for Evaluator class
  - Test no-gradient computation, max_batches, eval mode
  - Test metrics computation, convenience functions, trainer integration
  - All 219 tests passing (46 training tests total)

## 2025-12-06 - Training loop implementation (Phase 5, Task 2)

- Implemented `Trainer` class for training loop orchestration
- **Training features** (`atlas/training/trainer.py`):
  - `Trainer`: Main training loop with gradient accumulation
  - `TrainingMetrics`: Container for training metrics (loss, perplexity, LR, tokens/sec)
  - `train_step()`: Single training step with forward/backward pass
  - `train_epoch()`: Full epoch training with progress bar (tqdm)
  - Support for gradient accumulation (for large effective batch sizes)
  - Automatic gradient clipping integration
  - Learning rate scheduler integration
  - Progress tracking with real-time metrics display
- **Testing** (`tests/test_training.py`):
  - Added 10 comprehensive tests for Trainer class
  - Test gradient accumulation, scheduler integration, clipping
  - Test epoch training, max steps, metrics computation
  - Test loss decrease on overfitting (sanity check)
  - All 210 tests passing (37 training tests total)
- **GPU support**: Upgraded PyTorch to nightly build (2.10.0.dev20251206+cu128) for RTX 5060 Ti (sm_120) support

## 2025-12-06 - Training utilities (Phase 5, Task 1)

- Implemented loss functions and optimizer utilities for training
- **Loss functions** (`atlas/training/loss.py`):
  - `compute_lm_loss()`: Cross-entropy loss for language modeling
  - `compute_lm_loss_with_logits_shift()`: Automatic shifting for next-token prediction
  - `compute_perplexity()`: Convert loss to perplexity metric
  - Support for ignore_index (padding), multiple reduction modes
- **Optimizer utilities** (`atlas/training/optimizer.py`):
  - `create_optimizer()`: AdamW with selective weight decay
  - `create_scheduler()`: LR scheduler with warmup and decay (cosine/linear/constant)
  - `get_optimizer_state()`: Optimizer state inspection
  - `clip_gradients()`: Gradient clipping by global norm
  - Excludes biases and LayerNorm from weight decay (best practice)
- **Tests**: Added 27 comprehensive tests covering:
  - Loss computation with various reduction modes (11 tests)
  - Optimizer creation and parameter grouping (7 tests)
  - Scheduler warmup and decay schedules (9 tests)
- All 27 tests passing

## 2025-12-06 - Data Pipeline complete (Phase 4)

- Implemented complete data loading and preprocessing pipeline for training
- **TextDataset** (`atlas/data/dataset.py`):
  - Loads text from single or multiple files
  - Tokenizes using existing `Tokenizer` class
  - Creates fixed-length sequences with sliding window support
  - Configurable stride for overlapping or non-overlapping sequences
  - Efficient token buffering (loads and tokenizes all files once)
  - Proper handling of leftover tokens (no padding)
  - Statistics and vocab size utilities
- **DataLoader utilities** (`atlas/data/loader.py`):
  - `create_dataloader()`: Create PyTorch DataLoaders with proper batching
  - `collate_batch()`: Batch collation function
  - `get_dataloader_stats()`: DataLoader statistics
  - `split_dataset()`: Train/val/test splitting with reproducible seeds
- **Preprocessing** (`atlas/data/preprocessing.py`):
  - `clean_text()`: Text cleaning, normalization, Unicode handling
  - `chunk_text()`: Split long documents into chunks with overlap
  - `load_text_file()`: Load and optionally clean text files
  - `load_jsonl()`: Load JSONL format with custom field names
  - `iterate_documents()`: Memory-efficient document iteration
  - `count_tokens()`: Token counting utility
  - `filter_by_length()`: Filter texts by token count
- **Tests**: Added 72 comprehensive tests covering:
  - Dataset: loading, sequence creation, indexing, edge cases (27 tests)
  - DataLoader: batching, splitting, collation, integration (16 tests)
  - Preprocessing: cleaning, chunking, file loading, filtering (29 tests)
- All 72 tests passing

## 2025-12-06 - Model architecture implementation complete (Phase 3)

- Implemented complete decoder-only transformer architecture following GPT-2 design
- **Embeddings** (`atlas/model/embeddings.py`):
  - `TokenEmbedding`: Vocabulary embedding layer
  - `PositionalEmbedding`: Learned position embeddings
  - `CombinedEmbedding`: Combines token and positional embeddings with dropout
- **Attention** (`atlas/model/attention.py`):
  - `MultiHeadAttention`: Multi-head self-attention with causal masking
  - Efficient combined Q/K/V projection
  - Scaled dot-product attention with cached causal mask
- **MLP** (`atlas/model/mlp.py`):
  - Position-wise feed-forward network with configurable activation (GELU/SiLU/ReLU)
  - Configurable expansion ratio (default 4.0x)
- **TransformerBlock** (`atlas/model/transformer.py`):
  - Pre-norm architecture (LayerNorm before attention and MLP)
  - Residual connections around attention and MLP sub-layers
- **AtlasLM** (`atlas/model/model.py`):
  - Complete language model assembly
  - Weight tying between embeddings and LM head
  - Autoregressive generation with temperature, top-k sampling, and EOS support
  - Parameter counting utilities
- Added 42 comprehensive tests covering all model components - **ALL PASSING**
- Tests validate: initialization, forward pass, shape preservation, dropout behavior, causal masking, generation, different configurations

## 2025-12-06 - Tokenizer implementation complete (Phase 2)

- Implemented `Tokenizer` class using tiktoken (GPT-2 BPE) as backend
- Added support for special tokens: BOS, EOS, PAD, UNK
- Implemented batch encoding/decoding operations
- Added comprehensive error handling for edge cases
- Added 27 rigorous tests covering all tokenizer functionality - all passing
- Tests include: round-trip encoding/decoding, special tokens, batch operations, Unicode, edge cases

## 2025-12-06 - Enhanced README with badges and contribution guidelines

- Added status badges (Python version, license, code style, tests)
- Enhanced installation section with detailed venv setup for Windows/macOS/Linux
- Added comprehensive contribution guidelines covering code quality, testing, commits, and documentation
- Added development workflow section with detailed commands
- Added FAQ section for common questions
- Updated project status showing Phase 0 and Phase 1 completion

## 2025-12-06 - Configuration system complete (Phase 1)

- Implemented complete configuration schema with dataclasses: `ModelConfig`, `TrainingConfig`, `DataConfig`, `LoggingConfig`, `InferenceConfig`, `AtlasConfig`
- Added comprehensive validation in `__post_init__` methods for all config classes
- Implemented YAML config loading with merge functionality (`atlas/config/loader.py`)
- Implemented CLI argument parsing with override support (`atlas/config/cli.py`)
- Added 32 rigorous tests covering all config functionality - all passing
- Virtual environment created and dependencies installed

## 2025-12-06 - Project foundation complete (Phase 0)

- Set up complete package structure with `atlas/` and all subdirectories (model, tokenizer, training, data, config, inference, utils, export)
- Created `tests/` directory with test file stubs for all modules
- Created `scripts/` directory for CLI scripts
- Added `requirements.txt` with core dependencies (PyTorch, NumPy, tiktoken, tqdm, pyyaml, pytest, gguf)
- Updated `.gitignore` for Atlas-specific patterns (checkpoints, logs, .gguf files)
- Created `setup.py` for package installation
- Updated `README.md` with project overview, structure, and quickstart guide

## 2025-12-06 - Initial roadmap created

- Created comprehensive `docs/ROADMAP.md` breaking down the entire LLM development process
- Roadmap covers 10 phases from project foundation through documentation and release
- Includes detailed tasks for model architecture, training, inference, and GGUF export
- Defines success criteria: trained model → coherent text generation → working GGUF export
