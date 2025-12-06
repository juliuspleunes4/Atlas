#!/usr/bin/env python3
"""
Inference script for Atlas LLM.

This script handles text generation from trained models:
- Loading checkpoints
- Interactive and batch generation modes
- Configurable sampling parameters
- Multiple prompts support
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

import torch

from atlas.model import AtlasLM
from atlas.tokenizer import Tokenizer
from atlas.inference import TextGenerator, GenerationConfig
from atlas.training import CheckpointManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate text with Atlas LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file",
    )
    
    # Input modes
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--prompt",
        type=str,
        help="Text prompt for generation",
    )
    input_group.add_argument(
        "--prompts-file",
        type=str,
        help="File containing prompts (one per line)",
    )
    input_group.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode (enter prompts interactively)",
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (higher = more random)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling (0 = disabled)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling (1.0 = disabled)",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling instead of greedy decoding",
    )
    
    # Model parameters
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda/cpu)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Tokenizer name (default: gpt2)",
    )
    
    # Output options
    parser.add_argument(
        "--output-file",
        type=str,
        help="Write generated text to file instead of stdout",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Include prompt in output",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default="\n" + "="*80 + "\n",
        help="Separator between multiple generations",
    )
    
    return parser.parse_args()


def load_model_and_tokenizer(checkpoint_path: str, tokenizer_name: str, device: str):
    """Load model and tokenizer from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model config from checkpoint
    if 'model_state_dict' not in checkpoint:
        raise ValueError(f"Invalid checkpoint: missing 'model_state_dict'")
    
    # Initialize tokenizer
    tokenizer = Tokenizer(tokenizer_name=tokenizer_name)
    print(f"Tokenizer: {tokenizer_name} (vocab_size={tokenizer.vocab_size})")
    
    # Get model config from checkpoint metadata
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Try to infer from state dict
        state_dict = checkpoint['model_state_dict']
        # Get vocab size from embedding weight
        vocab_size = state_dict['embedding.token_embedding.weight'].shape[0]
        max_seq_len = state_dict['embedding.positional_embedding.weight'].shape[0]
        hidden_size = state_dict['embedding.token_embedding.weight'].shape[1]
        
        # Count layers
        num_layers = sum(1 for k in state_dict.keys() if k.startswith('blocks.') and k.endswith('.ln1.weight'))
        
        # Get num_heads from attention weights
        num_heads = state_dict['blocks.0.attention.qkv_proj.weight'].shape[0] // (3 * hidden_size) * hidden_size
        num_heads = hidden_size // (state_dict['blocks.0.attention.qkv_proj.weight'].shape[0] // (3 * hidden_size))
        
        print(f"Inferred config from checkpoint:")
        print(f"  vocab_size={vocab_size}, max_seq_len={max_seq_len}")
        print(f"  hidden_size={hidden_size}, num_layers={num_layers}")
        
        # Create config dict
        from atlas.config import ModelConfig
        config = ModelConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
        )
    
    # Create model
    if hasattr(config, '__dict__'):
        model = AtlasLM(config)
    else:
        from atlas.config import ModelConfig
        model = AtlasLM(ModelConfig(**config))
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {num_params:,} parameters")
    print(f"Device: {device}")
    
    return model, tokenizer


def load_prompts_from_file(file_path: str) -> List[str]:
    """Load prompts from file (one per line)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def generate_interactive(generator: TextGenerator, config: GenerationConfig, show_prompt: bool):
    """Interactive generation mode."""
    print("\n" + "="*80)
    print("Interactive Mode")
    print("="*80)
    print("Enter prompts (Ctrl+C or 'quit' to exit)")
    print("="*80 + "\n")
    
    try:
        while True:
            try:
                prompt = input(">>> ")
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not prompt.strip():
                    continue
                
                print()
                
                # Generate
                generated = generator.generate_from_prompt(
                    prompt=prompt,
                    config=config,
                )
                
                # Display
                if show_prompt:
                    print(f"Prompt: {prompt}")
                    print(f"Generated: {generated}")
                else:
                    print(generated)
                
                print()
                
            except EOFError:
                break
    
    except KeyboardInterrupt:
        print("\n\nExiting...")


def generate_batch(
    generator: TextGenerator,
    prompts: List[str],
    config: GenerationConfig,
    show_prompt: bool,
    separator: str,
    output_file: Optional[str] = None,
):
    """Batch generation mode."""
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"Generating for prompt {i+1}/{len(prompts)}...", file=sys.stderr)
        
        # Generate
        generated = generator.generate_from_prompt(
            prompt=prompt,
            config=config,
        )
        
        # Format output
        if show_prompt:
            output = f"Prompt: {prompt}\nGenerated: {generated}"
        else:
            output = generated
        
        results.append(output)
    
    # Write results
    full_output = separator.join(results)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_output)
        print(f"\nOutput written to: {output_file}", file=sys.stderr)
    else:
        print(full_output)


def main():
    """Main inference loop."""
    args = parse_args()
    
    print("="*80)
    print("Atlas LLM Inference")
    print("="*80)
    
    # Load model and tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer(
            args.checkpoint,
            args.tokenizer,
            args.device,
        )
    except Exception as e:
        print(f"\n[!] Error loading model: {e}")
        sys.exit(1)
    
    # Create generation config
    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=args.top_p if args.top_p < 1.0 else None,
        do_sample=args.do_sample,
    )
    
    print("\nGeneration parameters:")
    print(f"  max_new_tokens: {config.max_new_tokens}")
    print(f"  temperature: {config.temperature}")
    print(f"  top_k: {config.top_k}")
    print(f"  top_p: {config.top_p}")
    print(f"  do_sample: {config.do_sample}")
    print("="*80)
    
    # Create generator
    generator = TextGenerator(model=model, tokenizer=tokenizer, device=args.device)
    
    # Determine mode and generate
    if args.interactive:
        generate_interactive(generator, config, args.show_prompt)
    
    elif args.prompts_file:
        prompts = load_prompts_from_file(args.prompts_file)
        print(f"\nLoaded {len(prompts)} prompts from {args.prompts_file}\n")
        generate_batch(
            generator,
            prompts,
            config,
            args.show_prompt,
            args.separator,
            args.output_file,
        )
    
    elif args.prompt:
        generate_batch(
            generator,
            [args.prompt],
            config,
            args.show_prompt,
            args.separator,
            args.output_file,
        )
    
    else:
        # Default to interactive if no input specified
        print("\nNo input specified. Starting interactive mode...\n")
        generate_interactive(generator, config, args.show_prompt)


if __name__ == "__main__":
    main()
