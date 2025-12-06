# Atlas Scripts

This directory contains command-line scripts and automation tools for Atlas.

## 🚀 Main Scripts

### Training Pipeline (Recommended)
- **`run_pipeline.ps1`** / **`run_pipeline.sh`**: Complete automated training pipeline
  - Handles everything from setup to training start
  - Interactive GPU configuration selection (tiny/small/default/large)
  - Automatic data preparation and dependency checks
  - Zero-friction onboarding for new users
  - **Use this for the easiest experience!**

### Core Scripts
- **`train.py`**: Model training script
  - Full training loop with checkpointing and logging
  - Supports resume from checkpoint
  - CLI parameter overrides
  
- **`infer.py`**: Text generation and inference
  - Single prompt, batch prompts, or interactive mode
  - Configurable sampling parameters (temperature, top-k, top-p)
  - Output to file or stdout

- **`export_gguf.py`**: Model export to GGUF format
  - Convert trained models for llama.cpp compatibility
  - F32/F16 quantization options

### Utility Scripts
- **`prepare_data.py`**: Dataset preparation
  - Extract and organize training data
  - Supports Wikipedia SimpleEnglish dataset
  - Custom output directories

## 📖 Usage Examples

### Quick Start (Automated)
```powershell
# Windows
.\scripts\run_pipeline.ps1

# Linux/Mac
chmod +x scripts/run_pipeline.sh
./scripts/run_pipeline.sh
```

### Manual Training
```bash
# Train with specific config
python scripts/train.py --config configs/default.yaml --train-data data/processed/wikipedia

# Resume from checkpoint
python scripts/train.py --config configs/default.yaml --train-data data/processed/wikipedia --resume checkpoints/checkpoint_step_5000.pt
```

### Inference
```bash
# Single prompt
python scripts/infer.py --checkpoint checkpoints/best_model.pt --prompt "Once upon a time"

# Interactive mode
python scripts/infer.py --checkpoint checkpoints/best_model.pt --interactive
```

### Export Model
```bash
# Export to GGUF format
python scripts/export_gguf.py --checkpoint checkpoints/best_model.pt --output atlas_model.gguf
```

## 📁 Configuration Files

All configuration files are in the `configs/` directory:
- **`tiny.yaml`**: Ultra-lightweight (40M params, 4-6GB VRAM)
- **`small.yaml`**: Small model (124M params, 6-8GB VRAM)
- **`default.yaml`**: Standard model (350M params, 12-14GB VRAM)
- **`large.yaml`**: Large model (500M params, 14-15GB VRAM)
- **`xlarge.yaml`**: Memory-optimized large (500M params, 8-10GB VRAM) - Uses gradient accumulation
