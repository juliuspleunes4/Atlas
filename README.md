# 🌍 Atlas

[![Version](https://img.shields.io/badge/version-v1.1.0-blue.svg)](https://github.com/juliuspleunes4/Atlas/releases/tag/v1.1.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-324%20passing-brightgreen.svg)](tests/)

**A from-scratch language model implementation with GGUF export.**

Atlas is a complete pipeline for building, training, and deploying decoder-only transformer language models. The project focuses on clarity, modularity, and the ability to export trained models to GGUF format for efficient inference.

## ✨ Features

- **🎯 Clean Implementation**: Decoder-only transformer architecture built from scratch with PyTorch
- **🔄 Complete Pipeline**: Training, evaluation, inference, and export all in one place
- **💾 Memory Efficient**: 8-bit optimizer support (75% memory reduction) for large models on consumer GPUs
- **⚡ Reliable Checkpointing**: Built-in mid-epoch checkpoint saving every N steps
- **📦 GGUF Export**: Convert trained models to GGUF format for use with llama.cpp
- **🧩 Modular Design**: Well-organized codebase with clear separation of concerns
- **✅ Comprehensive Testing**: 324 tests covering all major components

## 📊 Project Status

🚧 **Under active development** 🚧

Atlas is currently in early development. See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the full development plan.

**Completed:**
- ✅ Phase 0: Project Foundation
- ✅ Phase 1: Configuration System (32 tests)
- ✅ Phase 2: Tokenizer Integration (27 tests)
- ✅ Phase 3: Model Architecture (51 tests - **+9 gradient checkpointing tests**)
- ✅ Phase 4: Data Pipeline (72 tests)
- ✅ Phase 5: Training Loop (62 tests - **+6 auto-resume tests**)
- ✅ Phase 5.5: Training Script (13 tests)
- ✅ Phase 6: Inference & Generation (21 tests)
- ✅ Phase 6.3: Inference Script (12 tests)
- ✅ Phase 7: GGUF Export (17 tests)

**Skipped (for now):**
- ⏭️ Phase 8: End-to-End Integration (individual components thoroughly tested)

**Upcoming:**
- Phase 9-10: Advanced features and optimization

**Total: 324 passing tests** ✨

## ⚙️ Model Configurations

Atlas provides multiple pre-configured model sizes optimized for different hardware capabilities:

| Config | Parameters | Layers | Hidden Size | Heads | Sequence Length | Batch Size | VRAM Usage | Best For |
|--------------|-----------|--------|-------------|-------|----------------|------------|------------|----------|
| **TINY** | ~40M | 8 | 512 | 8 | 512 | 4 (×4 accum) | 4-6 GB | Testing, debugging, low-end GPUs |
| **SMALL** | ~124M | 12 | 768 | 12 | 1024 | 24 (×2 accum) | 6-8 GB | Quick experiments, prototyping |
| **DEFAULT** | ~350M | 24 | 1024 | 16 | 1024 | 16 (×2 accum) | 12-14 GB | **Recommended** for most users |
| **LARGE** | ~500M | 30 | 1280 | 20 | 1024 | 8 (×4 accum) | 14-15 GB | Maximum quality, high-end GPUs |
| **XLARGE** | ~500M | 30 | 1280 | 20 | 1024 | 2 (×16 accum) | 8-10 GB | Max params with memory safety |
| **ULTRA** | ~650M | 30 | 1280 | 20 | 256 | 1 (×16 accum) | 3-5 GB | **Cool & Quiet** - Low GPU temp |

**Configuration Details:**

- **TINY** (`configs/tiny.yaml`): Ultra-lightweight for testing and development
  - Effective batch size: 16 (4 × 4 gradient accumulation)
  - Training time: ~1-2 days on RTX 3080 for 10K steps
  - Good for verifying pipeline, debugging, or running on older GPUs
  
- **SMALL** (`configs/small.yaml`): GPT-2 Small equivalent
  - Effective batch size: 48 (24 × 2 gradient accumulation)
  - Training time: ~3-5 days on RTX 3080 for 20K steps
  - Capable of basic text generation and learning patterns
  
- **DEFAULT** (`configs/default.yaml`): GPT-2 Medium equivalent (Recommended)
  - Effective batch size: 32 (16 × 2 gradient accumulation)
  - Training time: ~1-2 weeks on RTX 3080 for 50K steps
  - Optimized for RTX 5060 Ti 16GB with safe memory margin
  - Best balance of quality and training time
  
- **LARGE** (`configs/large.yaml`): Maximum quality
  - Effective batch size: 32 (8 × 4 gradient accumulation)
  - Training time: ~2-3 weeks on RTX 3080 for 80K steps
  - Close to 16GB VRAM limit - ensure good cooling
- **XLARGE** (`configs/xlarge.yaml`): Memory-optimized maximum size
  - Effective batch size: 32 (2 × 16 gradient accumulation)
  - Same 500M parameters as LARGE but uses 40% less VRAM
  - Training time: Similar to LARGE (~2-3 weeks for 80K steps)
  - **Best choice for maximizing model size while staying within GPU limits**

- **ULTRA** (`configs/ultra.yaml`): Extreme low-temperature optimization
  - Effective batch size: 64 (1 × 64 gradient accumulation)
  - Same 500M parameters, shorter sequences (256 tokens)
  - Uses absolute minimum VRAM (batch_size=1) + gradient checkpointing
  - Runs COOLEST of all configs - minimal GPU load/temperature
  - Training time: Slower than XLARGE due to extreme accumulation
  - **Best for: Maximum parameters while keeping GPU cool and quiet**

**Choosing a Configuration:**

Use the automated training script to select interactively:
```powershell
.\scripts\run_pipeline.ps1  # Windows
./scripts/run_pipeline.sh   # Linux/Mac
```

Or specify directly:
```bash
python scripts/train.py --config configs/tiny.yaml --train-data data/processed/wikipedia
```

## 🚀 Installation

### 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### 🔧 Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/juliuspleunes4/Atlas.git
cd Atlas
```

2. **Create a virtual environment**

**On Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Install Atlas as a package (development mode)**

```bash
pip install -e .
```

5. **Verify installation**

```bash
pytest tests/ -v
```

## 🎯 Quick Start (Recommended)

**The easiest way to get started - fully automated and interactive!**

### Step 1: Clone the repository
```bash
git clone https://github.com/juliuspleunes4/Atlas.git
cd Atlas
```

### Step 2: Download training data
Get the Wikipedia SimpleEnglish dataset from [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-simpleenglish) (171 MB zip file)

### Step 3: Place the zip file
```
Atlas/data/raw/archive.zip
```

### Step 4: Run the interactive training pipeline
```powershell
# Windows
.\scripts\run_pipeline.ps1

# Linux/Mac
chmod +x scripts/run_pipeline.sh
./scripts/run_pipeline.sh
```

**That's it!** 🎉 The script will:
- ✅ Check Python and create virtual environment
- ✅ Install all dependencies automatically
- ✅ Prepare the dataset (249K articles)
- ✅ Show you an interactive menu to choose model size
- ✅ Start training with your selected configuration

The script handles everything else automatically and guides you through each step!

### 🔄 Checkpoint Auto-Resume

Atlas automatically detects existing checkpoints and asks if you want to resume training:

```
┌──────────────────────────────────────────────────────┐
│  Checkpoint found: checkpoints/atlas_step_500.pt     │
│  Step: 500                                           │
│  Epoch: 1                                            │
│  Loss: 3.456                                         │
└──────────────────────────────────────────────────────┘

Resume from checkpoint? (y/n):
```

- **Choose "y"**: Continue training from the checkpoint (preserves optimizer state, learning rate, etc.)
- **Choose "n"**: Start a fresh training session (existing checkpoints remain untouched)

This works in:
- Interactive pipeline scripts (`run_pipeline.ps1`, `run_pipeline.sh`)
- Direct training script (`python scripts/train.py`)

To bypass the prompt and force resumption:
```bash
python scripts/train.py --config configs/default.yaml --resume checkpoints/atlas_step_500.pt
```

---

## 🔧 Advanced Usage

### 📦 Manual Data Preparation

If you need more control over data preparation:

```bash
# Basic usage
python scripts/prepare_data.py --input data/raw/archive.zip

# Custom output directory
python scripts/prepare_data.py --input data/raw/archive.zip --output data/processed/my_wiki

# List prepared datasets
python scripts/prepare_data.py --list
```

This extracts and organizes 249K articles (~400MB) into the processed directory.

### 🏋️ Manual Training

```bash
# Basic training with a configuration
python scripts/train.py --config configs/default.yaml --train-data data/processed/wikipedia

# Train with validation data
python scripts/train.py \
  --config configs/default.yaml \
  --train-data data/processed/wikipedia \
  --val-data data/processed/validation

# Resume training from a checkpoint
python scripts/train.py \
  --config configs/default.yaml \
  --train-data data/processed/wikipedia \
  --resume checkpoints/atlas_step_5000.pt

# Override config parameters from CLI
python scripts/train.py \
  --config configs/default.yaml \
  --train-data data/processed/wikipedia \
  --max-steps 100000 \
  --save-interval 500 \
  --eval-interval 1000
```

**Available training arguments:**
- `--config`: Path to YAML configuration file (required)
- `--train-data`: Path to training data directory (required)
- `--val-data`: Path to validation data (optional)
- `--output-dir`: Checkpoint directory (default: `./checkpoints`)
- `--resume`: Resume from checkpoint path
- `--max-steps`: Override max training steps
- `--save-interval`: Save checkpoint every N steps (default: 1000)
- `--eval-interval`: Evaluate every N steps (default: 1000)
- `--device`: Device to use (default: auto-detect CUDA)

### 💬 Inference

Generate text from a trained model:

```bash
# Single prompt
python scripts/infer.py \
  --checkpoint checkpoints/my_model/checkpoint_step_10000.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 100 \
  --temperature 0.8 \
  --top-k 40 \
  --do-sample
```

Interactive mode:

```bash
python scripts/infer.py \
  --checkpoint checkpoints/my_model/best_model.pt \
  --interactive
```

Batch generation from file:

```bash
python scripts/infer.py \
  --checkpoint checkpoints/my_model/best_model.pt \
  --prompts-file prompts.txt \
  --output-file generated.txt \
  --temperature 0.9 \
  --top-p 0.95 \
  --do-sample
```

### 📦 Export to GGUF

Export trained model to GGUF format:

```bash
# Export with float32 (default)
python scripts/export_gguf.py \
  --checkpoint checkpoints/my_model/best_model.pt \
  --output models/atlas_model.gguf \
  --quantization f32
```

Export with float16 for smaller file size:

```bash
python scripts/export_gguf.py \
  --checkpoint checkpoints/my_model/best_model.pt \
  --output models/atlas_model_f16.gguf \
  --quantization f16 \
  --tokenizer gpt2
```

## 📁 Project Structure

```
Atlas/
├── atlas/              # Main package
│   ├── model/         # Model architecture (embeddings, attention, transformer blocks)
│   ├── tokenizer/     # Tokenization and vocabulary
│   ├── training/      # Training loop and optimization
│   ├── data/          # Dataset loading and preprocessing
│   ├── config/        # Configuration management
│   ├── inference/     # Text generation and sampling
│   ├── utils/         # Logging, checkpointing, metrics
│   └── export/        # Model export (GGUF, etc.)
├── tests/             # Test suite
├── scripts/           # CLI scripts
├── docs/              # Documentation
│   ├── ROADMAP.md    # Development roadmap
│   └── CHANGELOG.md  # Change log
└── README.md          # This file
```

## 🏗️ Architecture

Atlas implements a **decoder-only transformer** architecture (GPT-style):

- **Multi-head self-attention** with causal masking
- **Feed-forward networks (MLP)** with configurable activation (GELU/SiLU)
- **Layer normalization** (pre-norm architecture)
- **Learned positional embeddings**
- **Weight tying** between input embeddings and output projection

## 🛠️ Development

### 🧪 Running Tests

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_config.py -v
```

Run with coverage:
```bash
pytest tests/ --cov=atlas --cov-report=html
```

### 💎 Code Quality

Format code:
```bash
black atlas/ tests/ scripts/
```

Lint code:
```bash
flake8 atlas/ tests/ scripts/
```

Type checking:
```bash
mypy atlas/
```

Run all quality checks:
```bash
black atlas/ tests/ scripts/ && flake8 atlas/ tests/ scripts/ && mypy atlas/ && pytest tests/ -v
```

## 🗺️ Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the complete development plan, which includes:

- Phase 0-1: Project foundation and configuration system
- Phase 2: Tokenizer integration
- Phase 3: Model architecture implementation
- Phase 4: Data pipeline
- Phase 5: Training loop
- Phase 6: Inference and generation
- Phase 7: GGUF export
- Phase 8: End-to-end integration
- Phase 9: Optimization and refinement
- Phase 10: Documentation and release

## 🤝 Contributing

We welcome contributions to Atlas! Whether you're fixing bugs, adding features, improving documentation, or writing tests, your help is appreciated.

### 🎬 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/juliuspleunes4/Atlas.git
   cd Atlas
   ```
3. **Create a virtual environment** and install dependencies (see Installation section)
4. **Create a new branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### 📝 Development Guidelines

#### 💎 Code Quality

- **Format your code** with black:
  ```bash
  black atlas/ tests/ scripts/
  ```
- **Lint your code** with flake8:
  ```bash
  flake8 atlas/ tests/ scripts/
  ```
- **Type check** with mypy:
  ```bash
  mypy atlas/
  ```

#### 🧪 Testing

- **Write tests** for all new features and bug fixes
- **Run the test suite** to ensure nothing breaks:
  ```bash
  pytest tests/ -v
  ```
- **Check test coverage**:
  ```bash
  pytest tests/ --cov=atlas --cov-report=html
  ```
- **Aim for >80% coverage** on new code

#### 📝 Commit Guidelines

- Write clear, descriptive commit messages
- Use conventional commit format:
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation changes
  - `test:` for test additions/changes
  - `refactor:` for code refactoring
  - `chore:` for maintenance tasks

#### 📚 Documentation

- Update `docs/CHANGELOG.md` with your changes
- Add docstrings to all public functions and classes
- Update README.md if adding user-facing features

### 🚀 Submitting Changes

1. **Ensure all tests pass** and code is formatted
2. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add amazing new feature"
   ```
3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
4. **Create a Pull Request** on GitHub
5. **Address review feedback** if requested

### 💡 What to Contribute

Check out the [roadmap](docs/ROADMAP.md) for areas that need work:

- 🔧 **Core Features**: Model architecture, training loop, inference
- 📝 **Documentation**: Tutorials, examples, API docs
- 🧪 **Tests**: Increase coverage, add edge cases
- 🐛 **Bug Fixes**: Check issues for known bugs
- ✨ **Optimizations**: Performance improvements, memory efficiency

### 🌟 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow

## 📄 License

See [LICENSE](LICENSE) for details.

## ❓ FAQ

### How do I activate the virtual environment?

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### How do I deactivate the virtual environment?

Simply run:
```bash
deactivate
```

### Tests are failing, what should I do?

1. Ensure you've activated the virtual environment
2. Make sure all dependencies are installed: `pip install -r requirements.txt`
3. Try running a specific test file to isolate the issue
4. Check if you're using Python 3.8+: `python --version`

### Where should I start contributing?

1. Check the [roadmap](docs/ROADMAP.md) for tasks marked as `[ ]` (not started)
2. Look for `TODO` comments in the codebase
3. Improve test coverage in existing modules
4. Add documentation and examples

## 🙏 Acknowledgments

- Inspired by modern transformer architectures (GPT, LLaMA)
- GGUF format from [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
- Built with [PyTorch](https://pytorch.org/)
