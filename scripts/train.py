#!/usr/bin/env python3
"""
Training script for Atlas LLM.

This script handles end-to-end training workflow:
- Loading configuration
- Initializing model, tokenizer, dataset, optimizer
- Running training loop with checkpointing
- Evaluating on validation set
- Handling interruptions gracefully
"""

import argparse
import signal
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml

from atlas.config import ModelConfig
from atlas.model import AtlasLM
from atlas.tokenizer import Tokenizer
from atlas.data import TextDataset, create_dataloader
from atlas.training import (
    Trainer,
    create_optimizer,
    create_scheduler,
    CheckpointManager,
    CheckpointMetadata,
)


# Global flag for graceful shutdown
interrupted = False


def signal_handler(sig, frame):
    """Handle Ctrl+C interrupt."""
    global interrupted
    print("\n\n[!] Interrupt received. Saving checkpoint and exiting gracefully...")
    interrupted = True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Atlas LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    
    # Data arguments
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training data file(s). Use comma-separated paths for multiple files.",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation data file(s). If not provided, no validation will be performed.",
    )
    
    # Training arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of training steps (overrides config)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1000,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log metrics every N steps",
    )
    
    # Model arguments
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (cuda/cpu)",
    )
    
    # Optimization overrides
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model_from_config(config: dict) -> AtlasLM:
    """Create model from configuration dictionary."""
    model_config = ModelConfig(
        vocab_size=config['model']['vocab_size'],
        max_seq_len=config['model']['max_seq_len'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        mlp_ratio=config['model'].get('mlp_ratio', 4.0),
        dropout=config['model'].get('dropout', 0.1),
    )
    
    return AtlasLM(model_config)


def create_datasets(train_paths: str, val_paths: Optional[str], tokenizer: Tokenizer, config: dict):
    """Create train and validation datasets."""
    # Parse paths (comma-separated)
    train_files = [p.strip() for p in train_paths.split(',')]
    
    train_dataset = TextDataset(
        file_paths=train_files,
        tokenizer=tokenizer,
        max_seq_len=config['data']['max_seq_len'],
    )
    
    val_dataset = None
    if val_paths:
        val_files = [p.strip() for p in val_paths.split(',')]
        val_dataset = TextDataset(
            file_paths=val_files,
            tokenizer=tokenizer,
            max_seq_len=config['data']['max_seq_len'],
        )
    
    return train_dataset, val_dataset


def main():
    """Main training loop."""
    global interrupted
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse arguments
    args = parse_args()
    
    print("=" * 80)
    print("Atlas LLM Training")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Train data: {args.train_data}")
    print(f"Val data: {args.val_data or 'None'}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Load configuration
    print("\n[1/6] Loading configuration...")
    config = load_config(args.config)
    
    # Apply command-line overrides
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
        print(f"  Override: learning_rate = {args.learning_rate}")
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
        print(f"  Override: batch_size = {args.batch_size}")
    if args.max_steps:
        config['training']['max_steps'] = args.max_steps
        print(f"  Override: max_steps = {args.max_steps}")
    
    # Initialize tokenizer
    print("\n[2/6] Initializing tokenizer...")
    tokenizer = Tokenizer(
        tokenizer_name=config['tokenizer']['name'],
        encoding_name=config['tokenizer'].get('encoding', 'cl100k_base'),
    )
    print(f"  Vocab size: {tokenizer.vocab_size}")
    
    # Create model
    print("\n[3/6] Creating model...")
    model = create_model_from_config(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {num_params:,}")
    print(f"  Model size: ~{num_params * 4 / 1024 / 1024:.2f} MB (fp32)")
    
    # Load checkpoint if resuming
    start_step = 0
    start_epoch = 0
    if args.resume:
        print(f"\n[*] Resuming from checkpoint: {args.resume}")
        checkpoint_manager = CheckpointManager(args.output_dir)
        
        # Create optimizer and scheduler (will be loaded from checkpoint)
        optimizer = create_optimizer(
            model,
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0.01),
        )
        scheduler = create_scheduler(
            optimizer,
            num_training_steps=config['training']['max_steps'],
            num_warmup_steps=config['training'].get('warmup_steps', 0),
            scheduler_type=config['training'].get('scheduler_type', 'cosine'),
        )
        
        metadata = checkpoint_manager.load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            device=args.device,
        )
        start_step = metadata.step
        start_epoch = metadata.epoch
        print(f"  Resumed from step {start_step}, epoch {start_epoch}")
    
    # Create datasets
    print("\n[4/6] Loading datasets...")
    train_dataset, val_dataset = create_datasets(
        args.train_data,
        args.val_data,
        tokenizer,
        config,
    )
    print(f"  Train samples: {len(train_dataset)}")
    if val_dataset:
        print(f"  Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 0),
    )
    
    val_loader = None
    if val_dataset:
        val_loader = create_dataloader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['data'].get('num_workers', 0),
        )
    
    # Create optimizer and scheduler (if not resuming)
    print("\n[5/6] Setting up training...")
    if not args.resume:
        optimizer = create_optimizer(
            model,
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0.01),
        )
        scheduler = create_scheduler(
            optimizer,
            num_training_steps=config['training']['max_steps'],
            num_warmup_steps=config['training'].get('warmup_steps', 0),
            scheduler_type=config['training'].get('scheduler_type', 'cosine'),
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        max_grad_norm=config['training'].get('max_grad_norm', 1.0),
        device=args.device,
    )
    
    # Set starting step if resuming
    if args.resume:
        trainer.global_step = start_step
    
    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=args.output_dir,
        model_name='atlas',
        keep_best=True,
        keep_last_n=config['training'].get('keep_checkpoints', 3),
    )
    
    print(f"  Optimizer: AdamW")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Scheduler: {config['training'].get('scheduler_type', 'cosine')}")
    print(f"  Gradient accumulation: {config['training'].get('gradient_accumulation_steps', 1)}")
    print(f"  Max grad norm: {config['training'].get('max_grad_norm', 1.0)}")
    
    # Training loop
    print("\n[6/6] Starting training...")
    print("=" * 80)
    
    max_steps = config['training']['max_steps']
    epoch = start_epoch
    best_val_loss = float('inf')
    
    try:
        while trainer.global_step < max_steps and not interrupted:
            epoch += 1
            print(f"\n>>> Epoch {epoch}")
            
            # Train for one epoch
            train_stats = trainer.train_epoch(
                train_loader,
                max_steps=max_steps,
                log_interval=args.log_interval,
            )
            
            print(f"\nEpoch {epoch} completed:")
            print(f"  Train loss: {train_stats['loss']:.4f}")
            print(f"  Train perplexity: {train_stats['perplexity']:.2f}")
            print(f"  Steps: {trainer.global_step}/{max_steps}")
            
            # Evaluate on validation set
            if val_loader and (trainer.global_step % args.eval_interval == 0 or interrupted):
                print(f"\n  Running validation...")
                val_stats = trainer.evaluate(val_loader, show_progress=False)
                print(f"  Val loss: {val_stats['loss']:.4f}")
                print(f"  Val perplexity: {val_stats['perplexity']:.2f}")
                
                # Save best model
                if val_stats['loss'] < best_val_loss:
                    best_val_loss = val_stats['loss']
                    print(f"  *** New best validation loss! ***")
                    is_best = True
                else:
                    is_best = False
            else:
                val_stats = None
                is_best = False
            
            # Save checkpoint
            if trainer.global_step % args.save_interval == 0 or interrupted:
                print(f"\n  Saving checkpoint...")
                metadata = CheckpointMetadata(
                    step=trainer.global_step,
                    epoch=epoch,
                    loss=train_stats['loss'],
                    perplexity=train_stats['perplexity'],
                    learning_rate=optimizer.param_groups[0]['lr'],
                    best_metric=best_val_loss if val_stats else None,
                )
                
                checkpoint_path = checkpoint_manager.save_checkpoint(
                    model,
                    optimizer,
                    metadata,
                    scheduler=scheduler,
                    is_best=is_best,
                )
                print(f"  Checkpoint saved: {checkpoint_path}")
            
            # Check if interrupted
            if interrupted:
                break
        
        # Training complete
        if not interrupted:
            print("\n" + "=" * 80)
            print("Training completed successfully!")
            print(f"Final step: {trainer.global_step}")
            print(f"Best validation loss: {best_val_loss:.4f}")
            print("=" * 80)
    
    except Exception as e:
        print(f"\n\n[!] Error during training: {e}")
        print("[!] Saving emergency checkpoint...")
        
        # Save emergency checkpoint
        metadata = CheckpointMetadata(
            step=trainer.global_step,
            epoch=epoch,
            loss=0.0,  # Unknown
            perplexity=0.0,  # Unknown
        )
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model,
            optimizer,
            metadata,
            scheduler=scheduler,
        )
        print(f"[!] Emergency checkpoint saved: {checkpoint_path}")
        raise
    
    finally:
        if interrupted:
            print("\nTraining interrupted by user.")
            print(f"Stopped at step {trainer.global_step}, epoch {epoch}")
            print("Resume training with --resume flag and the latest checkpoint.")


if __name__ == "__main__":
    main()
