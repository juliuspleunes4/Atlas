# Frequently Asked Questions (FAQ)

## Table of Contents

- [General Questions](#general-questions)
- [Installation & Setup](#installation--setup)
- [Training](#training)
- [Performance & Optimization](#performance--optimization)
- [Model Configuration](#model-configuration)
- [Checkpoints & Resumption](#checkpoints--resumption)
- [Inference & Generation](#inference--generation)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

## General Questions

### What is Atlas?

Atlas is a from-scratch Large Language Model (LLM) implementation built with PyTorch. It's designed to be educational, modular, and production-ready, implementing a decoder-only transformer architecture (GPT-style).

### What can I do with Atlas?

- Train custom language models from scratch
- Experiment with different model architectures and configurations
- Fine-tune models on specific domains
- Learn how modern LLMs work under the hood
- Export models to GGUF format for deployment

### What makes Atlas different?

- **Built from scratch**: No external model frameworks, pure PyTorch
- **Educational**: Clear, documented code for learning
- **Modular**: Easy to modify and extend
- **Production-ready**: Proper testing, checkpointing, and deployment tools
- **Flexible**: Multiple configurations from 40M to 500M+ parameters

### Is Atlas production-ready?

Atlas is actively developed and suitable for:
- ✅ Learning and experimentation
- ✅ Research projects
- ✅ Custom domain models
- ⚠️ Production use (with proper testing and validation)

## Installation & Setup

### What are the system requirements?

**Minimum**:
- Python 3.8+
- 8GB RAM
- CPU (training will be slow)

**Recommended**:
- Python 3.10+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM
- 100GB+ free disk space (for datasets and checkpoints)

### How do I install Atlas?

```bash
# Clone repository
git clone https://github.com/juliuspleunes4/Atlas.git
cd Atlas

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Can I use Atlas without a GPU?

Yes, but training will be significantly slower. For CPU-only usage:
- Use the TINY configuration (~40M params)
- Set `device: cpu` in config
- Reduce batch size and sequence length
- Training may take days instead of hours

### Which GPU do I need?

| GPU VRAM | Recommended Config | Parameters | Notes |
|----------|-------------------|------------|-------|
| 4-6GB    | TINY             | ~40M       | Basic experiments |
| 6-8GB    | SMALL            | ~124M      | Quick training |
| 8-12GB   | DEFAULT/XLARGE   | ~350-500M  | Best balance |
| 12-16GB  | LARGE/ULTRA      | ~500M      | Maximum capacity |
| 16GB+    | Custom           | 500M+      | High-end training |

## Training

### How do I start training?

**Quick Start**:
```bash
# Interactive pipeline (recommended)
./scripts/run_pipeline.ps1  # Windows
./scripts/run_pipeline.sh   # Linux/Mac

# Manual training
python scripts/train.py --config configs/default.yaml --train-data data/processed/wikipedia
```

### Where do I get training data?

Atlas uses the Wikipedia SimpleEnglish dataset:
1. Download from [Kaggle](https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-simpleenglish)
2. Place `archive.zip` in `data/raw/`
3. Run `python scripts/prepare_data.py --input data/raw/archive.zip`

### How long does training take?

Depends on configuration and GPU:

| Config  | GPU          | Speed     | Time to 1 Epoch (~249K articles) |
|---------|--------------|-----------|----------------------------------|
| TINY    | RTX 3060     | ~15 it/s  | ~2-3 hours                       |
| SMALL   | RTX 3060     | ~10 it/s  | ~4-5 hours                       |
| DEFAULT | RTX 4090     | ~8 it/s   | ~6-8 hours                       |
| LARGE   | RTX 4090     | ~5 it/s   | ~10-12 hours                     |
| ULTRA   | RTX 5060 Ti  | ~8 it/s   | ~6-8 hours                       |

### Can I stop training and resume later?

Yes! Atlas automatically detects checkpoints and prompts you to resume. You can also:
```bash
# Resume from specific checkpoint
python scripts/train.py --config configs/default.yaml --resume checkpoints/atlas_step_1000.pt

# Resume from best checkpoint
python scripts/train.py --config configs/default.yaml --resume checkpoints/best_model.pt
```

### How do I monitor training progress?

Training shows:
- **Loss**: Should decrease over time
- **Perplexity**: Lower is better (exp(loss))
- **Throughput**: Iterations per second (it/s)
- **Tokens/sec**: Processing speed

Logs are saved to:
- Console output (real-time)
- `training.log` (full log)
- Checkpoint metadata JSON files

## Performance & Optimization

### My GPU is running hot, is that normal?

Yes! GPUs are designed to run hot:
- **50-75°C**: Normal range during training
- **75-85°C**: High but acceptable
- **85°C+**: Consider improving cooling or reducing load

To reduce temperature:
- Use ULTRA configuration (gradient checkpointing)
- Reduce batch size
- Lower max_seq_len (512 → 256)
- Improve case airflow
- Clean GPU fans

### How can I speed up training?

1. **Use larger batch sizes** (if VRAM allows)
2. **Enable gradient checkpointing** (trades speed for memory)
3. **Use fp16/bf16 mixed precision** (if supported)
4. **Reduce sequence length** for faster iterations
5. **Use gradient accumulation** for larger effective batch sizes
6. **Use multiple GPUs** (future feature)

### What is gradient checkpointing?

Gradient checkpointing trades computation for memory:
- Recomputes activations during backward pass instead of storing them
- Reduces VRAM usage significantly
- Slightly slower training (~10-20%)
- Enabled in ULTRA config with `gradient_checkpointing: true`

### Training is slower than expected, why?

Common causes:
- **Data loading bottleneck**: Increase `num_workers` in config
- **Small batch size**: Use gradient accumulation for larger effective batches
- **CPU bottleneck**: Ensure GPU utilization is high (use `nvidia-smi`)
- **Disk I/O**: Move data to SSD instead of HDD
- **Old PyTorch version**: Update to latest stable version

## Model Configuration

### Which configuration should I choose?

| Use Case | Recommended Config | Why |
|----------|-------------------|-----|
| Learning/Testing | TINY or SMALL | Fast iteration, low resources |
| Experiments | DEFAULT | Good balance of quality and speed |
| Best Quality | LARGE | Maximum capacity with standard setup |
| Memory Limited | XLARGE or ULTRA | Same params as LARGE, less VRAM |
| Production | DEFAULT or LARGE | Proven, well-tested configurations |

### Can I create custom configurations?

Yes! Copy an existing config and modify:
```yaml
# configs/my_config.yaml
model:
  hidden_size: 768      # Model dimension
  num_layers: 12        # Transformer blocks
  num_heads: 12         # Attention heads
  vocab_size: 50257     # Vocabulary size
  max_seq_len: 512      # Context length
  
training:
  batch_size: 4
  learning_rate: 3e-4
  max_steps: 100000
```

Key considerations:
- `hidden_size` must be divisible by `num_heads`
- Larger models need lower learning rates
- Adjust batch size based on VRAM

### What do the hyperparameters mean?

- **hidden_size**: Model embedding dimension (larger = more capacity)
- **num_layers**: Number of transformer blocks (deeper = more capacity)
- **num_heads**: Parallel attention heads (more = richer representations)
- **vocab_size**: Size of tokenizer vocabulary (usually 50257 for GPT-2)
- **max_seq_len**: Maximum context length (longer = more memory)
- **learning_rate**: Step size for optimization (too high = unstable, too low = slow)
- **batch_size**: Samples per gradient update (larger = more stable)
- **gradient_accumulation_steps**: Accumulate gradients for larger effective batch

## Checkpoints & Resumption

### When are checkpoints saved?

Atlas saves four types of checkpoints:

1. **Step-based**: Every 100 global steps (keeps last 5, ~18.4GB total)
2. **Epoch-based**: At end of each epoch (keeps last 5)
3. **Best model**: When performance improves (always kept)
   - Uses validation loss when validation data available
   - Uses training loss otherwise
4. **Interrupt**: When pressing Ctrl+C during training (manual backup)

### Which checkpoint should I use?

- **Latest checkpoint** (`atlas_step_X.pt`): Most recent training state
- **Best checkpoint** (`atlas_best.pt`): Best performance (inference-ready)
- **Epoch checkpoint** (`atlas_epoch_X_step_Y.pt`): End of specific epoch
- **Interrupt checkpoint** (`atlas_interrupt_step_X.pt`): Emergency save from Ctrl+C

For inference, use **atlas_best.pt**. For resuming training, Atlas auto-detects the latest.

### How do I load a checkpoint?

```python
from atlas.training.checkpoint import CheckpointManager

# Load checkpoint
manager = CheckpointManager("checkpoints/")
checkpoint = manager.load_checkpoint("checkpoints/best_model.pt")

# Access components
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
step = checkpoint["step"]
loss = checkpoint["loss"]
```

### Can I transfer checkpoints between machines?

Yes! Checkpoints are portable between:
- ✅ Different machines with same Python/PyTorch versions
- ✅ CPU and GPU (checkpoint stores both)
- ✅ Different operating systems
- ⚠️ Different PyTorch versions (usually works, but test)
- ❌ Different model architectures (configs must match)

## Inference & Generation

### How do I generate text?

```bash
# Basic generation
python scripts/infer.py \
  --checkpoint checkpoints/best_model.pt \
  --prompt "Once upon a time"

# Advanced options
python scripts/infer.py \
  --checkpoint checkpoints/best_model.pt \
  --prompt "The meaning of life is" \
  --max-tokens 100 \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.95
```

### What do the generation parameters do?

- **temperature**: Controls randomness (0.1 = focused, 1.0 = balanced, 2.0 = creative)
- **top_k**: Only sample from top K tokens (lower = more focused)
- **top_p**: Nucleus sampling threshold (0.9 = sample from top 90% probability mass)
- **max_tokens**: Maximum length of generated text

### How can I improve generation quality?

1. **Train longer**: More epochs usually improve quality
2. **Use best checkpoint**: Lowest validation loss
3. **Adjust temperature**: Lower (0.7) for coherent, higher (1.0+) for creative
4. **Use top_p**: Nucleus sampling (0.9-0.95) often works well
5. **Better prompts**: Give context and clear instructions

### Can I export my model?

Yes! Export to GGUF format for use with llama.cpp:

```bash
python scripts/export_gguf.py \
  --checkpoint checkpoints/best_model.pt \
  --output atlas_model.gguf \
  --quantize f16  # or f32
```

Then use with llama.cpp for efficient inference.

## Troubleshooting

### Out of Memory (OOM) errors

Try these solutions in order:

1. **Reduce batch_size**: Start with 1
2. **Enable gradient_checkpointing**: Use ULTRA config
3. **Reduce max_seq_len**: 512 → 256 → 128
4. **Use gradient accumulation**: Smaller batches, accumulate gradients
5. **Use smaller model**: Switch to SMALL or TINY config

### Training loss is not decreasing

Common causes:

1. **Learning rate too high**: Reduce by 10x
2. **Learning rate too low**: Increase by 10x
3. **Batch size too small**: Increase or use gradient accumulation
4. **Data issues**: Check dataset quality and preprocessing
5. **Insufficient training**: Train for more steps/epochs

### CUDA out of memory despite small batch size

- Restart Python kernel to clear GPU memory
- Check for memory leaks: `nvidia-smi` to monitor VRAM
- Ensure no other processes using GPU
- Try `torch.cuda.empty_cache()` between runs
- Reboot if VRAM remains allocated

### Model generates repetitive text

- Lower temperature (try 0.7-0.8)
- Use top_p sampling (0.9-0.95)
- Increase top_k (40-50)
- Train longer for better model quality
- Check if training loss is still decreasing

### Tests failing after changes

1. **Read the error**: Understand what's failing
2. **Verify the test**: Is the test correct?
3. **Check your changes**: Did you break existing functionality?
4. **Run specific test**: `pytest tests/test_file.py::test_name -v`
5. **Fix root cause**: Don't just make tests pass

## Advanced Topics

### Can I use a custom tokenizer?

Yes! Implement the tokenizer interface:

```python
from atlas.tokenizer import Tokenizer

# Use your tokenizer
custom_tokenizer = YourTokenizer(...)
dataset = TextDataset(files, tokenizer=custom_tokenizer)
```

### How do I implement a custom loss?

```python
from atlas.training.loss import compute_lm_loss

def custom_loss(logits, targets, mask=None):
    """Your custom loss function."""
    base_loss = compute_lm_loss(logits, targets)
    # Add your modifications
    return modified_loss
```

### Can I train on multiple GPUs?

Multi-GPU support is planned but not yet implemented. For now:
- Use single GPU training
- Consider training multiple models in parallel on different GPUs
- Contribute multi-GPU support! (see `docs/CONTRIBUTING.md`)

### How do I fine-tune on custom data?

1. **Prepare your dataset**: Text files in `data/processed/your_data/`
2. **Train from checkpoint**:
   ```bash
   python scripts/train.py \
     --config configs/default.yaml \
     --train-data data/processed/your_data \
     --resume checkpoints/pretrained.pt \
     --learning-rate 1e-5  # Lower LR for fine-tuning
   ```

### How do I implement a new architecture?

1. Create new components in `atlas/model/`
2. Update `ModelConfig` in `atlas/config/config.py`
3. Add comprehensive tests in `tests/test_model.py`
4. Update documentation
5. Submit PR! (see `docs/CONTRIBUTING.md`)

---

## Still Have Questions?

- **Check Documentation**: See `docs/` folder
- **Search Issues**: Someone may have asked before
- **Open Issue**: Tag with `question` label
- **Read Code**: Atlas is designed to be readable!

---

**Last Updated**: December 7, 2025
