# Testing Guide

## Overview

Atlas maintains a comprehensive test suite with **325+ tests** covering all major components. This guide explains our testing philosophy, how to run tests, and how to write new tests.

## Testing Philosophy

### Core Principles

1. **Thoroughness Over Speed**: Every feature must have comprehensive tests
2. **No Shortcuts**: Tests must validate real behavior, not convenience
3. **Test Everything**: Units, integration, edge cases, and error conditions
4. **Fail Correctly**: Tests should fail when behavior changes unexpectedly

### When to Write Tests

- **Before implementing** (Test-Driven Development encouraged)
- **After implementing** (if TDD not used)
- **When fixing bugs** (regression tests)
- **When refactoring** (ensure behavior unchanged)

### What to Test

✅ **Do Test**:
- Public APIs and interfaces
- Edge cases and boundary conditions
- Error handling and validation
- Integration between components
- Performance-critical paths
- Backward compatibility

❌ **Don't Test**:
- Private implementation details
- External library behavior (trust PyTorch, etc.)
- Obvious Python language features

## Test Structure

### Directory Layout

```
tests/
├── __init__.py
├── test_config.py          # Configuration system (32 tests)
├── test_data.py            # Data loading & preprocessing (78 tests)
├── test_export.py          # GGUF export (17 tests)
├── test_infer_script.py    # Inference script (12 tests)
├── test_inference.py       # Text generation (24 tests)
├── test_model.py           # Model architecture (51 tests)
├── test_tokenizer.py       # Tokenization (26 tests)
├── test_train_script.py    # Training script (11 tests)
├── test_training.py        # Training loop & checkpoint (62 tests)
└── test_utils.py           # Utilities (future)
```

### Test Organization

Tests are organized by module and functionality:

```python
# test_model.py

# Embeddings tests
def test_token_embedding_initialization(): ...
def test_token_embedding_forward(): ...
def test_positional_embedding_initialization(): ...

# Attention tests
def test_attention_initialization(): ...
def test_attention_forward(): ...
def test_attention_causal_mask(): ...

# MLP tests
def test_mlp_initialization(): ...
def test_mlp_forward(): ...
def test_mlp_activation_gelu(): ...

# Full model tests
def test_atlas_lm_initialization(): ...
def test_atlas_lm_forward(): ...
def test_atlas_lm_generate_basic(): ...
```

## Running Tests

### All Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=atlas --cov-report=html

# Parallel execution (faster)
pytest tests/ -n auto
```

### Specific Tests

```bash
# Single file
pytest tests/test_model.py -v

# Single test
pytest tests/test_model.py::test_attention_forward -v

# Tests matching pattern
pytest tests/ -k "attention" -v

# Tests in multiple files
pytest tests/test_model.py tests/test_training.py -v
```

### Test Output Control

```bash
# Show print statements
pytest tests/ -v -s

# Show only failures
pytest tests/ -v --tb=short

# Stop on first failure
pytest tests/ -v -x

# Show slowest tests
pytest tests/ --durations=10
```

### Test Markers

```bash
# Run only fast tests (< 1s)
pytest tests/ -v -m "not slow"

# Run integration tests
pytest tests/ -v -m "integration"

# Skip certain tests
pytest tests/ -v -m "not gpu_required"
```

## Writing Tests

### Basic Test Structure

```python
def test_feature_name():
    """
    Clear description of what is being tested.
    Include context if behavior is non-obvious.
    """
    # 1. Setup (Arrange)
    config = ModelConfig(
        hidden_size=64,
        num_layers=2,
        num_heads=4
    )
    model = AtlasLM(config)
    
    # 2. Execute (Act)
    input_tokens = torch.randint(0, 1000, (2, 10))  # batch=2, seq=10
    output = model(input_tokens)
    
    # 3. Validate (Assert)
    assert output.shape == (2, 10, config.vocab_size)
    assert torch.all(torch.isfinite(output))
```

### Test Requirements

Every test must:

1. **Be deterministic**: Same input → same output
2. **Be isolated**: No dependencies on other tests
3. **Be fast**: Use minimal configs and data
4. **Be clear**: Obvious what is being tested and why

### Assertion Best Practices

```python
# ✅ Good: Clear expectation
assert loss > 0, "Loss must be positive"
assert tokens.shape == (batch_size, seq_len)

# ✅ Good: Tolerance for floating point
assert abs(loss - expected_loss) < 1e-6

# ✅ Good: Check multiple properties
assert output.shape == expected_shape
assert output.dtype == torch.float32
assert not torch.any(torch.isnan(output))

# ❌ Bad: Vague assertion
assert result  # What are we checking?

# ❌ Bad: Too strict for floating point
assert loss == 2.5  # Use approximate equality

# ❌ Bad: Multiple unrelated assertions
assert loss > 0 and accuracy < 1 and model.training  # Split these
```

### Testing Edge Cases

Always test boundary conditions:

```python
def test_attention_single_token():
    """Test attention with sequence length 1."""
    attention = MultiHeadAttention(config)
    x = torch.randn(2, 1, config.hidden_size)  # seq_len=1
    output = attention(x)
    assert output.shape == x.shape

def test_attention_max_sequence():
    """Test attention at maximum sequence length."""
    attention = MultiHeadAttention(config)
    x = torch.randn(2, config.max_seq_len, config.hidden_size)
    output = attention(x)
    assert output.shape == x.shape

def test_dataset_empty_file():
    """Test dataset handles empty files gracefully."""
    with pytest.raises(ValueError, match="empty"):
        dataset = TextDataset(["empty.txt"], tokenizer)
```

### Testing Error Conditions

Validate that errors are raised correctly:

```python
def test_invalid_config_hidden_size():
    """Test that invalid hidden_size raises error."""
    with pytest.raises(ValueError, match="hidden_size.*divisible.*num_heads"):
        config = ModelConfig(
            hidden_size=100,  # Not divisible by num_heads
            num_heads=12
        )

def test_negative_learning_rate():
    """Test that negative learning rate raises error."""
    with pytest.raises(ValueError, match="learning_rate.*positive"):
        config = TrainingConfig(learning_rate=-0.001)
```

### Fixtures

Use pytest fixtures for reusable setup:

```python
import pytest

@pytest.fixture
def small_config():
    """Small config for fast tests."""
    return ModelConfig(
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        vocab_size=1000,
        max_seq_len=128
    )

@pytest.fixture
def sample_model(small_config):
    """Small model instance."""
    return AtlasLM(small_config)

def test_model_forward(sample_model, small_config):
    """Test using fixtures."""
    tokens = torch.randint(0, small_config.vocab_size, (2, 10))
    output = sample_model(tokens)
    assert output.shape == (2, 10, small_config.vocab_size)
```

## Test Categories

### 1. Unit Tests

Test individual components in isolation:

```python
def test_mlp_forward():
    """Test MLP forward pass with known input."""
    mlp = MLP(hidden_size=64, mlp_size=256)
    x = torch.randn(2, 10, 64)
    output = mlp(x)
    
    assert output.shape == (2, 10, 64)
    assert output.dtype == torch.float32
    assert torch.all(torch.isfinite(output))
```

### 2. Integration Tests

Test component interactions:

```python
def test_transformer_block_residual_connections():
    """Test that residual connections work correctly."""
    block = TransformerBlock(config)
    x = torch.randn(2, 10, config.hidden_size)
    
    # With residual, output should differ from input
    output = block(x)
    assert not torch.allclose(output, x)
    
    # But should be close (residual adds small changes)
    diff = (output - x).abs().mean()
    assert diff < 1.0  # Some change, but not huge
```

### 3. End-to-End Tests

Test complete workflows:

```python
def test_train_single_step_end_to_end():
    """Test complete training step."""
    # Setup
    config = small_config()
    model = AtlasLM(config)
    optimizer = torch.optim.Adam(model.parameters())
    batch = torch.randint(0, config.vocab_size, (4, 32))
    
    # Train one step
    optimizer.zero_grad()
    logits = model(batch)
    loss = compute_lm_loss(logits, batch)
    loss.backward()
    optimizer.step()
    
    # Validate
    assert loss.item() > 0
    assert all(p.grad is not None for p in model.parameters())
```

### 4. Regression Tests

Prevent bugs from returning:

```python
def test_checkpoint_resume_preserves_step():
    """
    Regression test for issue #123.
    Ensure resumed training continues from correct step.
    """
    manager = CheckpointManager("checkpoints/")
    
    # Save at step 100
    manager.save_checkpoint(model, optimizer, step=100, epoch=1, loss=2.5)
    
    # Load and check
    checkpoint = manager.load_checkpoint("checkpoints/atlas_step_100.pt")
    assert checkpoint["step"] == 100  # Was incorrectly 0 before fix
```

### 5. Performance Tests

Validate critical performance characteristics:

```python
import time

def test_attention_performance():
    """Test that attention completes in reasonable time."""
    attention = MultiHeadAttention(config)
    x = torch.randn(32, 512, config.hidden_size)
    
    start = time.time()
    output = attention(x)
    duration = time.time() - start
    
    assert duration < 1.0, f"Attention too slow: {duration}s"
```

## Common Test Patterns

### Testing Training Dynamics

```python
def test_training_reduces_loss():
    """Test that training actually reduces loss."""
    model = AtlasLM(small_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = torch.randint(0, small_config().vocab_size, (8, 32))
    
    # Initial loss
    initial_loss = compute_lm_loss(model(batch), batch).item()
    
    # Train for 10 steps
    for _ in range(10):
        optimizer.zero_grad()
        loss = compute_lm_loss(model(batch), batch)
        loss.backward()
        optimizer.step()
    
    # Final loss should be lower
    final_loss = compute_lm_loss(model(batch), batch).item()
    assert final_loss < initial_loss, "Training should reduce loss"
```

### Testing Gradient Flow

```python
def test_gradients_flow_through_model():
    """Test that gradients propagate to all parameters."""
    model = AtlasLM(small_config())
    batch = torch.randint(0, small_config().vocab_size, (4, 10))
    
    # Forward and backward
    loss = compute_lm_loss(model(batch), batch)
    loss.backward()
    
    # Check all parameters have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.all(torch.isfinite(param.grad)), f"Invalid gradient for {name}"
```

### Testing Determinism

```python
def test_generation_deterministic_with_same_seed():
    """Test that generation is deterministic given same seed."""
    model = AtlasLM(small_config())
    prompt = torch.tensor([[1, 2, 3]])
    
    # Generate twice with same seed
    torch.manual_seed(42)
    output1 = model.generate(prompt, max_new_tokens=10, temperature=0.0)
    
    torch.manual_seed(42)
    output2 = model.generate(prompt, max_new_tokens=10, temperature=0.0)
    
    assert torch.equal(output1, output2), "Generation should be deterministic"
```

## Test Coverage

### Current Coverage

```bash
pytest tests/ --cov=atlas --cov-report=term

# Output:
Name                          Stmts   Miss  Cover
-------------------------------------------------
atlas/__init__.py                 5      0   100%
atlas/config/__init__.py         10      0   100%
atlas/config/config.py          120      5    96%
atlas/model/model.py            180     12    93%
atlas/training/trainer.py       250     18    93%
...
-------------------------------------------------
TOTAL                          2500    150    94%
```

### Coverage Goals

- **Overall**: >90% line coverage
- **Core modules** (model, training): >95%
- **Critical paths**: 100%

### Viewing Coverage

```bash
# Generate HTML report
pytest tests/ --cov=atlas --cov-report=html

# Open in browser
open htmlcov/index.html  # Mac/Linux
start htmlcov/index.html  # Windows
```

## Continuous Integration

Tests run automatically on:
- Every push to GitHub
- Every pull request
- Nightly (full test suite with large configs)

### CI Requirements

All tests must pass before merging:
- ✅ All 326+ tests pass
- ✅ No warnings or errors
- ✅ Coverage remains above 90%
- ✅ No flaky tests (inconsistent pass/fail)

## Debugging Failed Tests

### Running Failed Test

```bash
# Run with verbose output
pytest tests/test_model.py::test_attention_forward -vv

# Show full traceback
pytest tests/test_model.py::test_attention_forward -vv --tb=long

# Drop into debugger on failure
pytest tests/test_model.py::test_attention_forward --pdb
```

### Common Failure Causes

1. **Shape mismatches**: Check tensor dimensions
2. **Type errors**: Ensure correct dtypes (float32 vs int64)
3. **Device mismatches**: CPU vs GPU tensors
4. **Random seed issues**: Set seed for deterministic tests
5. **Floating point precision**: Use `torch.allclose()` instead of `==`

### Adding Debug Output

```python
def test_with_debug_output(capsys):
    """Example test with debug output."""
    model = AtlasLM(small_config())
    output = model(torch.randint(0, 1000, (2, 10)))
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")
    
    # Run with -s to see output: pytest tests/ -v -s
    assert output.shape[0] == 2
```

## Best Practices Summary

✅ **Do**:
- Write tests for every new feature
- Test edge cases and error conditions
- Use descriptive test names and docstrings
- Keep tests fast (< 1 second each)
- Make tests deterministic (set random seeds)
- Test behavior, not implementation
- Use small configs for unit tests
- Run tests before committing

❌ **Don't**:
- Modify tests to make them pass (fix the code!)
- Skip tests without good reason
- Write tests that depend on external services
- Use large models/datasets in unit tests
- Test private implementation details
- Write non-deterministic tests
- Ignore test warnings
- Commit failing tests

---

**Questions?** See [CONTRIBUTING.md](CONTRIBUTING.md) or open an issue.

---

**Last Updated**: December 7, 2025
