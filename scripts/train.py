#!/usr/bin/env python3
"""
Training script for Atlas LLM.

This script handles end-to-end training workflow:
- Loading configuration
- Initializing model, tokenizer, dataset, optimizer
- Running training loop with checkpointing
- Evaluating on validation set
- Handling interruptions gracefully
- Comprehensive logging to console and file
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime
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

# Global logger
logger = None


def setup_logging(output_dir: str, resume_checkpoint: Optional[str] = None):
    """
    Setup logging to both console and file.
    
    Creates a training.log file in the output directory. If resuming from a checkpoint,
    adds a separator and timestamp to the existing log file.
    """
    log_file = Path(output_dir) / "training.log"
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if this is a resume (log file already exists)
    is_resume = log_file.exists() and resume_checkpoint is not None
    
    # If resuming, add separator to existing log
    if is_resume:
        with open(log_file, 'a') as f:
            f.write("\n\n")
            f.write("=" * 80 + "\n")
            f.write(f"RESUMED TRAINING SESSION: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Checkpoint: {resume_checkpoint}\n")
            f.write("=" * 80 + "\n\n")
    
    # Configure logger
    logger = logging.getLogger('atlas_training')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    
    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Log session start if not resuming
    if not is_resume:
        logger.info("=" * 80)
        logger.info(f"NEW TRAINING SESSION: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        logger.info("")
    
    return logger


def signal_handler(sig, frame):
    """Handle Ctrl+C interrupt."""
    global interrupted, logger
    msg = "\n\n[!] Interrupt received. Saving checkpoint and exiting gracefully..."
    print(msg)
    if logger:
        logger.info(msg)
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
    global interrupted, logger
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse arguments
    args = parse_args()
    
    # Setup logging first
    logger = setup_logging(args.output_dir, args.resume)
    
    logger.info("=" * 80)
    logger.info("Atlas LLM Training")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Train data: {args.train_data}")
    logger.info(f"Val data: {args.val_data or 'None'}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Device: {args.device}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info("=" * 80)
    
    # Load configuration
    logger.info("\n[1/6] Loading configuration...")
    config = load_config(args.config)
    logger.info(f"  Model: {config['model']['num_layers']} layers, {config['model']['hidden_size']} hidden, {config['model']['num_heads']} heads")
    logger.info(f"  Sequence length: {config['model']['max_seq_len']}")
    logger.info(f"  Batch size: {config['training']['batch_size']}")
    logger.info(f"  Max steps: {config['training']['max_steps']}")
    
    # Apply command-line overrides
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
        logger.info(f"  Override: learning_rate = {args.learning_rate}")
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
        logger.info(f"  Override: batch_size = {args.batch_size}")
    if args.max_steps:
        config['training']['max_steps'] = args.max_steps
        logger.info(f"  Override: max_steps = {args.max_steps}")
    
    # Initialize tokenizer
    logger.info("\n[2/6] Initializing tokenizer...")
    tokenizer = Tokenizer(
        tokenizer_name=config['tokenizer']['name'],
        encoding_name=config['tokenizer'].get('encoding', 'cl100k_base'),
    )
    logger.info(f"  Tokenizer: {config['tokenizer']['name']}")
    logger.info(f"  Vocab size: {tokenizer.vocab_size:,}")
    
    # Create model
    logger.info("\n[3/6] Creating model...")
    model = create_model_from_config(config)
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {num_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Model size: ~{num_params * 4 / 1024 / 1024:.2f} MB (fp32)")
    logger.info(f"  Model size: ~{num_params * 2 / 1024 / 1024:.2f} MB (fp16)")
    
    # Load checkpoint if resuming
    start_step = 0
    start_epoch = 0
    if args.resume:
        logger.info(f"\n[*] Resuming from checkpoint: {args.resume}")
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
        logger.info(f"  Resumed from step {start_step}, epoch {start_epoch}")
        logger.info(f"  Previous loss: {metadata.loss:.4f}")
        logger.info(f"  Previous perplexity: {metadata.perplexity:.2f}")
    
    # Create datasets
    logger.info("\n[4/6] Loading datasets...")
    load_start = time.time()
    train_dataset, val_dataset = create_datasets(
        args.train_data,
        args.val_data,
        tokenizer,
        config,
    )
    load_time = time.time() - load_start
    logger.info(f"  Train samples: {len(train_dataset):,}")
    logger.info(f"  Train tokens: {train_dataset.get_stats()['total_tokens']:,}")
    if val_dataset:
        logger.info(f"  Val samples: {len(val_dataset):,}")
        logger.info(f"  Val tokens: {val_dataset.get_stats()['total_tokens']:,}")
    logger.info(f"  Dataset loading time: {load_time:.2f}s")
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 0),
    )
    logger.info(f"  Train batches per epoch: {len(train_loader):,}")
    
    val_loader = None
    if val_dataset:
        val_loader = create_dataloader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['data'].get('num_workers', 0),
        )
        logger.info(f"  Val batches: {len(val_loader):,}")
    
    # Create optimizer and scheduler (if not resuming)
    logger.info("\n[5/6] Setting up training...")
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
    
    logger.info(f"  Optimizer: AdamW")
    logger.info(f"  Learning rate: {config['training']['learning_rate']}")
    logger.info(f"  Weight decay: {config['training'].get('weight_decay', 0.01)}")
    logger.info(f"  Scheduler: {config['training'].get('scheduler_type', 'cosine')}")
    logger.info(f"  Warmup steps: {config['training'].get('warmup_steps', 0):,}")
    logger.info(f"  Gradient accumulation: {config['training'].get('gradient_accumulation_steps', 1)}")
    logger.info(f"  Max grad norm: {config['training'].get('max_grad_norm', 1.0)}")
    logger.info(f"  Effective batch size: {config['training']['batch_size'] * config['training'].get('gradient_accumulation_steps', 1)}")
    
    # Estimate training time
    total_tokens = len(train_dataset) * config['model']['max_seq_len']
    tokens_per_step = config['training']['batch_size'] * config['model']['max_seq_len'] * config['training'].get('gradient_accumulation_steps', 1)
    steps_per_epoch = len(train_dataset) // config['training']['batch_size']
    estimated_epochs = (config['training']['max_steps'] - start_step) / steps_per_epoch
    logger.info(f"  Steps per epoch: ~{steps_per_epoch:,}")
    logger.info(f"  Estimated epochs to complete: ~{estimated_epochs:.1f}")
    logger.info(f"  Tokens per step: {tokens_per_step:,}")
    
    # Training loop
    logger.info("\n[6/6] Starting training...")
    logger.info("=" * 80)
    logger.info(f"Training from step {start_step} to {config['training']['max_steps']}")
    logger.info(f"Logging interval: every {args.log_interval} steps")
    logger.info(f"Eval interval: every {args.eval_interval} steps")
    logger.info(f"Save interval: every {args.save_interval} steps")
    logger.info("=" * 80)
    
    max_steps = config['training']['max_steps']
    epoch = start_epoch
    best_val_loss = float('inf')
    training_start_time = time.time()
    
    try:
        while trainer.global_step < max_steps and not interrupted:
            epoch += 1
            epoch_start_time = time.time()
            logger.info(f"\n{'='*80}")
            logger.info(f">>> EPOCH {epoch} | Step {trainer.global_step}/{max_steps}")
            logger.info(f"{'='*80}")
            
            # Train for one epoch
            train_stats = trainer.train_epoch(
                train_loader,
                max_steps=max_steps,
                log_interval=args.log_interval,
            )
            
            epoch_time = time.time() - epoch_start_time
            tokens_processed = len(train_dataset) * config['model']['max_seq_len']
            tokens_per_sec = tokens_processed / epoch_time
            
            logger.info(f"\n{'─'*80}")
            logger.info(f"Epoch {epoch} Summary:")
            logger.info(f"  Train loss: {train_stats['loss']:.4f}")
            logger.info(f"  Train perplexity: {train_stats['perplexity']:.2f}")
            logger.info(f"  Steps: {trainer.global_step}/{max_steps} ({trainer.global_step/max_steps*100:.1f}%)")
            logger.info(f"  Epoch time: {epoch_time:.2f}s")
            logger.info(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")
            logger.info(f"  Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"  GPU memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
            logger.info(f"{'─'*80}")
            
            # Evaluate on validation set
            if val_loader and (trainer.global_step % args.eval_interval == 0 or interrupted):
                logger.info(f"\n{'─'*80}")
                logger.info("Running validation...")
                val_start = time.time()
                val_stats = trainer.evaluate(val_loader, show_progress=False)
                val_time = time.time() - val_start
                logger.info(f"  Val loss: {val_stats['loss']:.4f}")
                logger.info(f"  Val perplexity: {val_stats['perplexity']:.2f}")
                logger.info(f"  Val time: {val_time:.2f}s")
                
                # Save best model
                if val_stats['loss'] < best_val_loss:
                    improvement = ((best_val_loss - val_stats['loss']) / best_val_loss) * 100
                    best_val_loss = val_stats['loss']
                    logger.info(f"  🌟 NEW BEST VALIDATION LOSS! (improved by {improvement:.2f}%)")
                    is_best = True
                else:
                    is_best = False
                logger.info(f"  Best val loss so far: {best_val_loss:.4f}")
                logger.info(f"{'─'*80}")
            else:
                val_stats = None
                is_best = False
            
            # Save checkpoint
            if trainer.global_step % args.save_interval == 0 or interrupted:
                logger.info(f"\n{'─'*80}")
                logger.info("Saving checkpoint...")
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
                logger.info(f"  ✓ Checkpoint saved: {checkpoint_path}")
                if is_best:
                    logger.info(f"  ✓ Best model saved")
                logger.info(f"{'─'*80}")
            
            # Check if interrupted
            if interrupted:
                break
        
        # Training complete
        if not interrupted:
            total_training_time = time.time() - training_start_time
            hours = total_training_time / 3600
            logger.info(f"\n{'='*80}")
            logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            logger.info(f"{'='*80}")
            logger.info(f"Final step: {trainer.global_step}")
            logger.info(f"Final epoch: {epoch}")
            logger.info(f"Final train loss: {train_stats['loss']:.4f}")
            logger.info(f"Final train perplexity: {train_stats['perplexity']:.2f}")
            if val_stats:
                logger.info(f"Best validation loss: {best_val_loss:.4f}")
            logger.info(f"Total training time: {hours:.2f} hours ({total_training_time:.0f}s)")
            logger.info(f"Average time per step: {total_training_time/(trainer.global_step-start_step):.2f}s")
            logger.info(f"{'='*80}")
    
    except Exception as e:
        logger.error(f"\n\n{'='*80}")
        logger.error(f"❌ ERROR DURING TRAINING: {str(e)}")
        logger.error(f"{'='*80}")
        logger.error("Saving emergency checkpoint...")
        
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
        logger.error(f"Emergency checkpoint saved: {checkpoint_path}")
        logger.error(f"{'='*80}")
        raise
    
    finally:
        if interrupted:
            logger.info(f"\n{'='*80}")
            logger.info("⚠️  TRAINING INTERRUPTED BY USER")
            logger.info(f"{'='*80}")
            logger.info(f"Stopped at step {trainer.global_step}, epoch {epoch}")
            logger.info(f"Progress: {trainer.global_step/max_steps*100:.1f}% complete")
            logger.info("Resume training with --resume flag and the latest checkpoint:")
            logger.info(f"  python scripts/train.py --config {args.config} \\")
            logger.info(f"    --train-data {args.train_data} \\")
            if args.val_data:
                logger.info(f"    --val-data {args.val_data} \\")
            logger.info(f"    --resume {args.output_dir}/checkpoint_step_{trainer.global_step}.pt")
            logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
