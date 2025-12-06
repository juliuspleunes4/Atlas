#!/usr/bin/env python3
"""
GGUF export script for Atlas models.

This script converts trained PyTorch models to GGUF format
for efficient inference with llama.cpp and compatible tools.
"""

import argparse
import sys
from pathlib import Path

import torch

from atlas.model import AtlasLM
from atlas.config import ModelConfig
from atlas.export import export_atlas_to_gguf, GGMLQuantizationType


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export Atlas model to GGUF format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output GGUF file path",
    )
    
    # Optional arguments
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["f32", "f16"],
        default="f32",
        help="Quantization type (f32=float32, f16=float16)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Tokenizer name for metadata",
    )
    
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str):
    """Load model checkpoint and extract config."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' not in checkpoint:
        raise ValueError("Invalid checkpoint: missing 'model_state_dict'")
    
    # Get config
    if 'config' in checkpoint:
        config = checkpoint['config']
        if hasattr(config, '__dict__'):
            # ModelConfig object
            model_config = config
        else:
            # Dict
            model_config = ModelConfig(**config)
    else:
        # Try to infer from state dict
        state_dict = checkpoint['model_state_dict']
        
        vocab_size = state_dict['embeddings.token_embedding.embedding.weight'].shape[0]
        max_seq_len = state_dict['embeddings.positional_embedding.embedding.weight'].shape[0]
        hidden_size = state_dict['embeddings.token_embedding.embedding.weight'].shape[1]
        num_layers = sum(1 for k in state_dict.keys() if k.startswith('blocks.') and k.endswith('.ln1.weight'))
        
        # Infer num_heads from attention weight shape
        # qkv_proj.weight shape is (3 * hidden_size, hidden_size)
        qkv_weight = state_dict['blocks.0.attention.qkv_proj.weight']
        num_heads = hidden_size // 64  # Assume head_dim = 64
        
        print("Warning: Config not found in checkpoint, inferring from state dict")
        print(f"  Inferred: vocab_size={vocab_size}, max_seq_len={max_seq_len}")
        print(f"  Inferred: hidden_size={hidden_size}, num_layers={num_layers}")
        print(f"  Assumed: num_heads={num_heads}")
        
        model_config = ModelConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
        )
    
    # Create model and load weights
    model = AtlasLM(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded:")
    print(f"  Vocab size: {model_config.vocab_size}")
    print(f"  Max sequence length: {model_config.max_seq_len}")
    print(f"  Hidden size: {model_config.hidden_size}")
    print(f"  Layers: {model_config.num_layers}")
    print(f"  Attention heads: {model_config.num_heads}")
    
    return model, model_config


def main():
    """Main export process."""
    args = parse_args()
    
    print("=" * 80)
    print("Atlas GGUF Export")
    print("=" * 80)
    
    # Load model
    try:
        model, model_config = load_checkpoint(args.checkpoint)
    except Exception as e:
        print(f"\n[!] Error loading checkpoint: {e}")
        sys.exit(1)
    
    # Determine quantization type
    if args.quantization == "f32":
        quantization = GGMLQuantizationType.F32
    elif args.quantization == "f16":
        quantization = GGMLQuantizationType.F16
    else:
        print(f"[!] Unknown quantization type: {args.quantization}")
        sys.exit(1)
    
    print(f"\nExport configuration:")
    print(f"  Output: {args.output}")
    print(f"  Quantization: {quantization.name}")
    print(f"  Tokenizer: {args.tokenizer}")
    
    # Prepare config dicts
    model_config_dict = {
        'vocab_size': model_config.vocab_size,
        'max_seq_len': model_config.max_seq_len,
        'hidden_size': model_config.hidden_size,
        'num_layers': model_config.num_layers,
        'num_heads': model_config.num_heads,
    }
    
    tokenizer_config = {
        'name': args.tokenizer,
    }
    
    # Export to GGUF
    print("\nExporting to GGUF...")
    try:
        export_atlas_to_gguf(
            model=model,
            output_path=args.output,
            model_config=model_config_dict,
            tokenizer_config=tokenizer_config,
            quantization=quantization,
        )
    except Exception as e:
        print(f"\n[!] Error during export: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Verify output file exists
    output_path = Path(args.output)
    if output_path.exists():
        file_size = output_path.stat().st_size
        print(f"\nExport successful!")
        print(f"  File: {args.output}")
        print(f"  Size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    else:
        print(f"\n[!] Export failed: output file not created")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("Export complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
