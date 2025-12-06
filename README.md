# Atlas

**A from-scratch language model implementation with GGUF export.**

Atlas is a complete pipeline for building, training, and deploying decoder-only transformer language models. The project focuses on clarity, modularity, and the ability to export trained models to GGUF format for efficient inference.

## Features

- **Clean Implementation**: Decoder-only transformer architecture built from scratch with PyTorch
- **Complete Pipeline**: Training, evaluation, inference, and export all in one place
- **GGUF Export**: Convert trained models to GGUF format for use with llama.cpp
- **Modular Design**: Well-organized codebase with clear separation of concerns
- **Comprehensive Testing**: Test coverage for all major components

## Project Status

🚧 **Under active development** 🚧

Atlas is currently in early development. See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the full development plan.

## Installation

```bash
# Clone the repository
git clone https://github.com/juliuspleunes4/Atlas.git
cd Atlas

# Install dependencies
pip install -r requirements.txt

# Install Atlas as a package (development mode)
pip install -e .
```

## Quick Start

> **Note**: The following examples are placeholders for future functionality.

### Training

```bash
# Train a model (TODO: implementation pending)
python scripts/train.py --config configs/small.yaml
```

### Inference

```bash
# Generate text (TODO: implementation pending)
python scripts/infer.py --checkpoint checkpoints/model.pt --prompt "Once upon a time"
```

### Export to GGUF

```bash
# Export to GGUF format (TODO: implementation pending)
python scripts/export_gguf.py --checkpoint checkpoints/model.pt --output model.gguf
```

## Project Structure

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

## Architecture

Atlas implements a **decoder-only transformer** architecture (GPT-style):

- **Multi-head self-attention** with causal masking
- **Feed-forward networks (MLP)** with configurable activation (GELU/SiLU)
- **Layer normalization** (pre-norm architecture)
- **Learned positional embeddings**
- **Weight tying** between input embeddings and output projection

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black atlas/ tests/ scripts/

# Lint
flake8 atlas/ tests/ scripts/

# Type checking
mypy atlas/
```

## Roadmap

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

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass
2. Code follows the existing style (black, flake8, mypy)
3. New features include tests
4. Changes are documented in `docs/CHANGELOG.md`

## License

See [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by modern transformer architectures (GPT, LLaMA)
- GGUF format from [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
