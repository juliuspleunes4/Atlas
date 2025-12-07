# Contributing to Atlas

Thank you for your interest in contributing to Atlas! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Guidelines](#code-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- PyTorch 2.0+
- CUDA-capable GPU (optional but recommended)

### Setting Up Development Environment

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Atlas.git
   cd Atlas
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Run Tests**
   ```bash
   pytest tests/ -v
   ```

## Development Workflow

### Branching Strategy

- `main` - stable, production-ready code
- Feature branches - `feature/your-feature-name`
- Bug fixes - `fix/bug-description`
- Documentation - `docs/what-you-are-documenting`

### Creating a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### Making Changes

1. **Pick Small, Focused Tasks**: Work on one well-defined feature or fix at a time
2. **Write Tests First** (TDD approach encouraged)
3. **Implement Your Changes**: Follow code guidelines below
4. **Run Tests**: Ensure all tests pass
5. **Update Documentation**: Update relevant docs and changelog

### Committing Changes

Use clear, descriptive commit messages following conventional commits:

```bash
# Feature
git commit -m "feat: add gradient checkpointing support"

# Bug fix
git commit -m "fix: correct attention mask broadcasting"

# Documentation
git commit -m "docs: update training guide with new configs"

# Tests
git commit -m "test: add comprehensive tokenizer tests"

# Refactor
git commit -m "refactor: simplify dataset loading logic"

# Performance
git commit -m "perf: optimize attention computation"
```

## Code Guidelines

### General Principles

- **Build scalable, modular, well-organized code**
- **Small, focused tasks** over big multi-feature changes
- **No shortcuts**: correctness, robustness, and test coverage are paramount
- **Never assume requirements**: use clear TODO markers where needed
- Follow **clean code** and **industry best practices**

### Python Style

- Follow **PEP 8** style guide
- Use **type hints** everywhere:
  ```python
  def train_step(model: AtlasLM, batch: torch.Tensor) -> float:
      """Train for one step and return loss."""
      ...
  ```

- Write **clear docstrings**:
  ```python
  def create_attention_mask(seq_len: int) -> torch.Tensor:
      """
      Create causal attention mask for autoregressive generation.
      
      Args:
          seq_len: Sequence length
          
      Returns:
          Boolean mask of shape (seq_len, seq_len)
      """
      ...
  ```

- **Small, focused functions**: Each function should do one thing well
- **Avoid premature optimization**: Clarity first, then measured improvements

### Code Organization

Keep code modular with clear package boundaries:

- `atlas/model/` - Core model components
- `atlas/tokenizer/` - Tokenization
- `atlas/training/` - Training loop, losses, schedulers
- `atlas/config/` - Configuration schemas
- `atlas/inference/` - Generation utilities
- `atlas/utils/` - Logging, checkpointing, metrics
- `atlas/data/` - Dataset handling
- `atlas/export/` - Model export formats

### Configuration & Scalability

- **No hard-coded hyperparameters** in core logic
- Use configuration objects or arguments
- Design for scalability from tiny to large models
- Use placeholders + `TODO:` comments for incomplete features

### Data Handling

**CRITICAL**: Never introduce mock data or hard-coded examples

- **No fake datasets** or sample corpora in the codebase
- Use **abstractions** (dataset loaders, interfaces)
- Add `TODO:` comments for data dependencies
- Tests can use **minimal synthetic examples** (small tensors, short strings)

## Testing

### Testing Philosophy

After **every new feature**, you must:

1. **Design rigorous tests** covering all relevant aspects
2. **Cover every realistic use case** and edge case
3. **Validate both happy path and failure behavior**
4. **Be extremely thorough** - don't just test the normal flow

### Test Structure

```python
def test_feature_name():
    """Test description of what is being validated."""
    # Setup
    config = ModelConfig(...)
    model = AtlasLM(config)
    
    # Execute
    result = model.forward(input_data)
    
    # Validate
    assert result.shape == expected_shape
    assert torch.all(torch.isfinite(result))
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_model.py -v

# Specific test
pytest tests/test_model.py::test_attention_forward -v

# With coverage
pytest tests/ --cov=atlas --cov-report=html
```

### Test Requirements

- Every non-trivial module must have tests
- Tests must be deterministic and fast
- Use small tensors/minimal configs for unit tests
- No external network access or real datasets in tests

### Handling Failing Tests

When a test fails:

1. **Determine where the bug is**: test or implementation?
2. **If test is incorrect**: Fix test to reflect correct behavior
3. **If implementation is incorrect**: Fix implementation, refactor as needed
4. **Never** modify tests just to make them pass
5. **Never** weaken assertions or coverage to hide bugs

## Documentation

### Code Documentation

- **Docstrings**: All public functions, classes, and methods
- **Type hints**: Required for all function signatures
- **Comments**: Explain "why", not "what"
- **TODO markers**: For incomplete features or planned improvements

### Project Documentation

When making changes:

1. **Update `docs/CHANGELOG.md`** (mandatory for notable changes)
   ```markdown
   ## 2025-12-07 - Add feature X
   
   - Implemented Y in `atlas/module/file.py`
   - Added Z tests covering edge cases
   - Updated configuration schema to support W
   ```

2. **Update relevant docs** in `docs/` folder
3. **Update `README.md`** if needed (installation, quickstart, etc.)
4. **Never create new markdown files** for status reports or summaries

### Documentation Style

- Use clear, concise language
- Provide code examples where appropriate
- Include expected outputs
- Link to related documentation

## Pull Request Process

### Before Submitting

1. ✅ All tests pass locally
2. ✅ Code follows style guidelines
3. ✅ Documentation is updated
4. ✅ `docs/CHANGELOG.md` is updated
5. ✅ Commit messages are clear and descriptive
6. ✅ No hardcoded values or mock data
7. ✅ Branch is up to date with main

### Submitting PR

1. **Push your branch**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request** on GitHub with:
   - Clear title describing the change
   - Detailed description of what and why
   - Reference any related issues
   - List of changes made
   - Testing performed

3. **PR Template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Changes Made
   - Added X feature
   - Fixed Y bug
   - Updated Z documentation
   
   ## Testing
   - Added N new tests
   - All 319+ tests passing
   - Manual testing: [describe]
   
   ## Checklist
   - [ ] Tests pass
   - [ ] Documentation updated
   - [ ] CHANGELOG updated
   - [ ] No hardcoded values
   - [ ] Follows code guidelines
   ```

### Review Process

- Maintainers will review your PR
- Address feedback and make requested changes
- Keep discussions focused and professional
- Be patient - reviews take time

### After Merge

- Delete your feature branch (optional)
- Pull latest main:
  ```bash
  git checkout main
  git pull origin main
  ```

## Community

### Getting Help

- **Issues**: For bugs, feature requests, questions
- **Discussions**: For general questions and community chat
- **Documentation**: Check `docs/` folder first

### Areas Needing Contribution

- **Core Features**: Model architectures, training improvements
- **Testing**: More comprehensive test coverage
- **Documentation**: Tutorials, guides, examples
- **Performance**: Optimization and profiling
- **Tooling**: Development tools and utilities
- **Examples**: Sample projects and use cases

### Recognition

Contributors will be acknowledged in:
- Repository contributors list
- Release notes (for significant contributions)
- `docs/CHANGELOG.md`

## Questions?

If you have questions about contributing:
1. Check existing documentation
2. Search closed issues
3. Open a new issue with your question
4. Tag it with `question` label

---

**Thank you for contributing to Atlas!** 🚀

Every contribution, no matter how small, helps make Atlas better for everyone.

---

**Last Updated**: December 7, 2025
