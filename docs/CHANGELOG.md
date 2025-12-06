# Changelog

All notable changes to Atlas will be documented in this file.

---

## [v1.0.0] - 2025-12-07 - First Stable Release 🎉

**Major Milestone**: Atlas v1.0.0 represents the first complete, production-ready release of the from-scratch language model implementation.

### 🎯 Complete Features

**Core Architecture** (Phase 3):
- Full decoder-only transformer architecture (GPT-style)
- Multi-head self-attention with causal masking
- Feed-forward networks with multiple activation functions (GELU, SiLU, ReLU)
- Pre-norm architecture with residual connections
- Learned positional embeddings
- Weight tying between embeddings and output head
- Gradient checkpointing for memory efficiency
- 51 comprehensive model tests

**Training Infrastructure** (Phase 5):
- Complete training loop with gradient accumulation
- Learning rate scheduling (warmup + cosine decay)
- Checkpoint management (step-based, epoch-based, best model)
- Automatic checkpoint resumption with interactive prompts
- Progress tracking and logging
- Validation and evaluation
- 62 training tests including auto-resume

**Data Pipeline** (Phase 4):
- Text dataset with sliding window tokenization
- Multiple file format support (txt, JSONL)
- Preprocessing utilities (cleaning, chunking, filtering)
- Efficient data loading with PyTorch DataLoader
- Train/validation splitting
- 72 data pipeline tests

**Configuration System** (Phase 1):
- YAML-based configuration
- CLI override support
- Multiple pre-configured model sizes (TINY to ULTRA)
- Validation and type checking
- 32 configuration tests

**Tokenizer** (Phase 2):
- GPT-2 BPE tokenizer via tiktoken
- Batch encoding/decoding
- Special token handling
- 27 tokenizer tests

**Inference** (Phase 6):
- Text generation with sampling strategies
- Temperature, top-k, top-p sampling
- Interactive and batch modes
- 33 inference tests

**Model Export** (Phase 7):
- GGUF format export
- Float32 and Float16 quantization
- Metadata embedding
- 17 export tests

### 📊 Statistics

- **307 passing tests** across all components
- **6 model configurations** (40M to 500M parameters)
- **10 comprehensive documentation files**
- **Clean, modular codebase** with 94%+ coverage on core modules

### 🎁 Model Configurations

Six production-ready configurations:
- **TINY** (40M params): Testing and development
- **SMALL** (124M params): GPT-2 Small equivalent
- **DEFAULT** (350M params): Recommended, GPT-2 Medium equivalent
- **LARGE** (500M params): Maximum quality
- **XLARGE** (500M params): Memory-optimized
- **ULTRA** (500M params): Extreme low-temperature operation

### 📚 Documentation

Complete documentation suite:
- README.md - Project overview and quickstart
- ROADMAP.md - Development plan and progress
- CHANGELOG.md - This file
- ARCHITECTURE.md - Technical deep-dive
- CONTRIBUTING.md - Contribution guidelines
- CODE_OF_CONDUCT.md - Community standards
- SECURITY.md - Security policy
- LICENSE_GUIDE.md - Licensing information
- TESTING.md - Testing guide
- FAQ.md - Frequently asked questions

### 🚀 Getting Started

```bash
git clone https://github.com/juliuspleunes4/Atlas.git
cd Atlas
.\scripts\run_pipeline.ps1  # Windows
./scripts/run_pipeline.sh   # Linux/Mac
```

### 🙏 Acknowledgments

This release represents the culmination of comprehensive development work across all phases of the project. Special thanks to all contributors and users who provided feedback during development.

---

## 2025-12-06 - Automatic Checkpoint Resume with Interactive Prompts

**Added**:
- **Auto-resume functionality** for seamless training continuation
  - Automatically detects existing checkpoints when starting training
  - Interactive prompt asks user to resume or start fresh
  - Works in both `train.py` script and pipeline scripts
  - Displays checkpoint info (step, epoch, loss, perplexity) before prompting
  
- **New CheckpointManager methods**:
  - `find_latest_checkpoint()`: Finds most recent checkpoint (excludes best checkpoint)
  - `get_checkpoint_info(path)`: Retrieves metadata without loading full checkpoint
  - Both support step-based and epoch-based checkpoints
  
- **Interactive pipeline improvements**:
  - `run_pipeline.ps1` (PowerShell): Colored prompt with checkpoint details
  - `run_pipeline.sh` (Bash): Checkpoint detection with JSON metadata parsing
  - User-friendly y/n prompts with input validation

- **New tests for auto-resume** (6 new tests, total: 307 passing)
  - `test_find_latest_checkpoint_empty_dir`: Handles empty checkpoint directory
  - `test_find_latest_checkpoint`: Finds most recent by timestamp
  - `test_find_latest_checkpoint_excludes_best`: Correctly excludes best.pt
  - `test_get_checkpoint_info`: Retrieves metadata without loading model
  - `test_get_checkpoint_info_no_metadata`: Handles missing JSON gracefully
  - `test_find_latest_with_epoch_checkpoints`: Works with both step and epoch checkpoints

**How It Works**:
1. Start training: `python scripts/train.py --config configs/ultra.yaml --train-data data/processed/wikipedia`
2. If checkpoint exists, you see:
   ```
   ╔════════════════════════════════════════════════════════════════════════════╗
   ║                   EXISTING CHECKPOINT DETECTED                             ║
   ╚════════════════════════════════════════════════════════════════════════════╝
   
   Found checkpoint: atlas_step_2000.pt
     Step: 2000
     Epoch: 5
     Loss: 2.3456
     Perplexity: 10.23
   
   Resume from checkpoint? (y/n):
   ```
3. Choose `y` to resume or `n` for fresh start

**Documentation**:
- Added "Checkpoint Auto-Resume" section to README Quick Start
- Explains automatic detection, prompt workflow, and manual override with `--resume` flag
4. Training continues from where you left off (or starts fresh)

**Benefits**:
- No need to manually specify `--resume` flag and checkpoint path
- Prevents accidental overwrites of training progress
- User-friendly prompts with all relevant info displayed
- Works seamlessly with both interactive scripts and direct python command

---

## 2025-12-06 - ULTRA Config Optimized for Low GPU Temperature

**Changed**:
- **ULTRA config extreme optimization**: Designed for COOLEST running temperature while maintaining 500M parameters
  - Reduced sequence length from 512 → 256 tokens (75% less compute than 1024, 50% less than 512)
  - Doubled gradient accumulation from 32 → 64 to maintain effective batch size
  - Added gradient checkpointing support (trades recomputation for lower memory/temp)
  - Updated VRAM estimate: 3-5GB (down from 5-7GB)
  - Now uses single worker for minimal system load

**Added**:
- **Gradient checkpointing support** in `AtlasLM` model
  - Enabled via `gradient_checkpointing: true` in config
  - Trades compute for memory during training
  - Reduces GPU memory usage and temperature
  - Implemented using `torch.utils.checkpoint.checkpoint()` with `use_reentrant=False`
  - Only active during training mode, bypassed during eval/inference

- **Comprehensive gradient checkpointing tests** (9 new tests, total: 301 passing)
  - `test_gradient_checkpointing_disabled_by_default`: Verifies feature is opt-in
  - `test_gradient_checkpointing_can_be_enabled`: Tests config-based activation
  - `test_gradient_checkpointing_forward_pass_train_mode`: Validates train mode behavior
  - `test_gradient_checkpointing_forward_pass_eval_mode`: Ensures eval mode bypasses checkpointing
  - `test_gradient_checkpointing_backward_pass`: Confirms gradients flow correctly
  - `test_gradient_checkpointing_same_output_as_normal`: Verifies identical outputs
  - `test_gradient_checkpointing_produces_valid_gradients`: Tests gradient validity for training
  - `test_gradient_checkpointing_with_mask`: Validates attention mask compatibility
  - `test_gradient_checkpointing_multiple_layers`: Tests scalability across layer counts

**Removed**:
- Removed MEDIUM configuration (150M params) - created temporarily but not part of official configs

**Updated**:
- `README.md`: Updated ULTRA description and VRAM estimates
- `run_pipeline.ps1` and `run_pipeline.sh`: Updated ULTRA description to emphasize low temperature

**Why**:
- User reported high GPU temperatures (70°C) during training with loud fans
- ULTRA config now prioritizes lowest GPU load and temperature over training speed
- Gradient checkpointing provides additional memory savings with minimal speed impact
- Best choice for users wanting maximum parameters while keeping system cool and quiet
- Comprehensive tests ensure feature works correctly and doesn't break existing functionality

---

## 2025-12-06 - Enhanced Checkpoint Management System

**Improved**:
- **Epoch-based checkpoints**: Now saves checkpoint at the end of every epoch (keeps last 5)
- **Step-based checkpoints**: Continues to save at intervals (keeps last 3 for TINY/SMALL/DEFAULT, 2 for LARGE/XLARGE)
- **Best model**: Always kept separately, never deleted regardless of other cleanup rules
- **Auto-adjusted intervals**: Save interval automatically adjusts after first epoch to target ~10 minute intervals
- **Checkpoint overwriting**: Existing checkpoints are properly overwritten when resuming training
- Updated `CheckpointManager` class:
  - Added `keep_last_epochs` parameter (default: 5)
  - Added `is_epoch_end` parameter to `save_checkpoint()`
  - Separate cleanup logic for step-based vs epoch-based checkpoints
  - Best checkpoint now saves with metadata JSON

**Checkpoint naming convention**:
- Step checkpoints: `atlas_step_1000.pt`
- Epoch checkpoints: `atlas_epoch_5_step_2340.pt`
- Best model: `atlas_best.pt` (always kept)

**Example checkpoint directory after training**:
```
checkpoints/
├── atlas_best.pt                    # Best validation loss (always kept)
├── atlas_step_5000.pt              # Recent step checkpoint
├── atlas_step_5250.pt              # Recent step checkpoint
├── atlas_epoch_10_step_4680.pt     # End of epoch 10
├── atlas_epoch_11_step_5148.pt     # End of epoch 11
├── atlas_epoch_12_step_5616.pt     # End of epoch 12
└── ... (+ JSON metadata for each)
```

## 2025-12-06 - Added Memory-Optimized XLARGE Configuration

**Added**:
- New `configs/xlarge.yaml` configuration for memory-constrained training
  - Same 500M parameters as LARGE config
  - Reduced batch size (2 vs 8) with increased gradient accumulation (16 vs 4)
  - Uses ~8-10GB VRAM instead of 14-15GB (40% reduction)
  - Same effective batch size (32) maintains training dynamics
  - Perfect for maximizing model size while staying within GPU limits
- Updated `run_pipeline.ps1/.sh` to include XLARGE option (option 5)
- Updated README.md configuration comparison table with XLARGE details

**Why this matters**:
- Allows training largest possible models on memory-limited GPUs
- Gradient accumulation technique maintains training quality
- Provides safer memory margin for 16GB GPUs like RTX 5060 Ti

## 2025-12-06 - Renamed Training Scripts and Removed Obsolete Files

**Changed**:
- Renamed `start_training.ps1/.sh` → `run_pipeline.ps1/.sh`
  - Better reflects that it handles the complete pipeline, not just training start
  - Updated all references in README.md, data/README.md, scripts/README.md, and CHANGELOG.md
- Removed obsolete `scripts/config_example.yaml`
  - Superseded by proper configuration files in `configs/` directory (tiny, small, default, large)
  - Removed references from documentation

## 2025-12-06 - Codebase Cleanup and Documentation Updates

- **Removed redundant automation scripts**:
  - Deleted `scripts/setup_and_train.ps1` and `scripts/setup_and_train.sh`
  - These were redundant with the more comprehensive `run_pipeline.ps1/.sh` scripts
  - Cleaner script structure with single automation entry point
- **Updated documentation**:
  - Expanded `scripts/README.md` with comprehensive script documentation
    - Main Scripts: run_pipeline.ps1/.sh with interactive configuration menu
    - Core Scripts: train.py, infer.py, export_gguf.py, prepare_data.py
    - Usage examples and configuration guidance for each script
  - Updated `data/README.md`:
    - References corrected from `setup_and_train` to `run_pipeline`
    - Added configuration selection guidance (TINY/SMALL/DEFAULT/LARGE)
    - Shows VRAM requirements for each preset
    - Manual training command examples

## 2025-12-06 - Add Lightweight Training Config and Fix Batch Format

- **New TINY configuration** (`configs/tiny.yaml`):
  - Ultra-lightweight model: ~40M parameters
  - Memory usage: 4-6GB VRAM (perfect for low-end GPUs or testing)
  - Shorter sequences (512 tokens) and smaller batches (4)
  - 8 layers, 512 hidden size, 8 attention heads
  - Fast training for quick iteration and debugging
- **Updated training pipeline** (`scripts/run_pipeline.ps1`):
  - Added TINY option to GPU configuration menu (option 1)
  - Now offers 4 presets: TINY, SMALL, DEFAULT, LARGE
  - Users can choose based on available GPU memory
- **Updated README.md**:
  - Added comprehensive configuration comparison table
  - Shows parameters, layers, VRAM usage, training speed for all configs
  - Updated test count from 288 to 292 passing tests
  - Added configuration selection guidance
- **Fixed batch collation** (`atlas/data/loader.py`):
  - `collate_batch()` now returns dictionary with `'input_ids'` key instead of plain tensor
  - Matches expected format in `Trainer.train_step()` which accesses `batch['input_ids']`
  - Updated docstrings and examples to reflect dictionary structure
- **Updated tests** (`tests/test_data.py`):
  - Fixed 6 data loader tests to expect dictionary structure
  - Updated assertions to check `batch['input_ids']` instead of direct tensor access
  - All 292 tests passing ✅
- **Training now fully functional**:
  - Complete pipeline from data loading through training loop works correctly
  - Model, tokenizer, data loading, and training all integrated properly
  - Multiple GPU memory profiles available for different hardware

## 2025-12-06 - Complete Training Pipeline and GPU-Optimized Configs

- **Complete training pipeline** (`scripts/run_pipeline.ps1` / `.sh`):
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
  - **One command from zero to training**: `.\scripts\run_pipeline.ps1`
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
