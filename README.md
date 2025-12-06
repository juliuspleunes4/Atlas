# 🌍 Atlas

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

**A from-scratch language model implementation with GGUF export.**

Atlas is a complete pipeline for building, training, and deploying decoder-only transformer language models. The project focuses on clarity, modularity, and the ability to export trained models to GGUF format for efficient inference.

## ✨ Features

- **🎯 Clean Implementation**: Decoder-only transformer architecture built from scratch with PyTorchrch
- **🔄 Complete Pipeline**: Training, evaluation, inference, and export all in one place
- **📦 GGUF Export**: Convert trained models to GGUF format for use with llama.cpp
- **🧩 Modular Design**: Well-organized codebase with clear separation of concerns
- **✅ Comprehensive Testing**: Test coverage for all major components

## 📊 Project Status

🚧 **Under active development** 🚧

Atlas is currently in early development. See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the full development plan.

**Completed:**
- ✅ Phase 0: Project Foundation
- ✅ Phase 1: Configuration System (32 tests)
- ✅ Phase 2: Tokenizer Integration (27 tests)
- ✅ Phase 3: Model Architecture (42 tests)
- ✅ Phase 4: Data Pipeline (72 tests)
- ✅ Phase 5: Training Loop (56 tests)
- ✅ Phase 5.5: Training Script (9 tests)
- ✅ Phase 6: Inference & Generation (21 tests)
- ✅ Phase 6.3: Inference Script (12 tests)
- ✅ Phase 7: GGUF Export (17 tests)

**Skipped (for now):**
- ⏭️ Phase 8: End-to-End Integration (individual components thoroughly tested)

**Upcoming:**
- Phase 9-10: Advanced features and optimization

**Total: 288 passing tests**

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

## 🎯 Quick Start

### 🎬 For Absolute Beginners

1. **Clone the repository**
   ```bash
   git clone https://github.com/juliuspleunes4/Atlas.git
   cd Atlas
   ```

2. **Download training data**: Get the Wikipedia SimpleEnglish dataset from [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-simpleenglish)

3. **Place the zip file** in `data/raw/archive.zip`

4. **Run the training pipeline**:
   ```powershell
   # Windows
   .\scripts\start_training.ps1
   
   # Linux/Mac
   chmod +x scripts/start_training.sh
   ./scripts/start_training.sh
   ```

That's it! The script handles everything else automatically.

---

### 📦 Manual Data Preparation (Optional)

If you prefer manual control:

1. **Download Wikipedia SimpleEnglish dataset** from [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-simpleenglish)

2. **Place the zip file** in `data/raw/`:
   ```
   Atlas/data/raw/archive.zip
   ```

3. **Prepare the data**:
   ```bash
   python scripts/prepare_data.py --input data/raw/archive.zip
   ```

   This extracts and organizes 249K articles (~400MB) into `data/processed/wikipedia/`.

### 🚀 Complete Training Pipeline (Recommended for New Users)

**The absolute easiest way - handles everything automatically:**

**Windows:**
```powershell
.\scripts\start_training.ps1
```

**Linux/Mac:**
```bash
chmod +x scripts/start_training.sh
./scripts/start_training.sh
```

This interactive script will:
1. ✅ Check Python and create virtual environment if needed
2. ✅ Install Atlas package (`pip install -e .`) if needed
3. ✅ Check for training data (prompts you if missing)
4. ✅ Prepare data automatically if not already done
5. ✅ Let you choose GPU config (small/default/large)
6. ✅ Start training with your chosen configuration
7. ✅ Handle all edge cases and errors gracefully

Perfect for developers who want **zero friction** from clone to training!

### 🏋️ Manual Training

Train a model step-by-step:

```bash
# 1. Prepare data (if not done already)
python scripts/prepare_data.py --input data/raw/archive.zip

# This will:
# - Extract archive.zip to data/processed/wikipedia/
# - Organize text files with clean naming (wiki_00000.txt, wiki_00001.txt, ...)
# - Display statistics (file count, total size)
# - Ready in seconds!

# Optional: List available datasets
python scripts/prepare_data.py --list

# Optional: Custom output directory
python scripts/prepare_data.py --input data/raw/archive.zip --output data/processed/my_wiki

# 2. Train the model
python scripts/train.py --config configs/default.yaml
```

Resume training from a checkpoint:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --resume checkpoints/checkpoint_step_5000.pt
```

Override config parameters from CLI:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --learning-rate 1e-3 \
  --batch-size 16 \
  --max-steps 50000
```

**Data Preparation Options:**

```bash
# Show help
python scripts/prepare_data.py --help

# List available raw and processed datasets
python scripts/prepare_data.py --list

# Prepare with default output (data/processed/wikipedia/)
python scripts/prepare_data.py --input data/raw/archive.zip

# Prepare with custom output directory
python scripts/prepare_data.py --input data/raw/archive.zip --output data/processed/custom_name
```

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
