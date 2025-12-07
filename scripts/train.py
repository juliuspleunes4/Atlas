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

from atlas.config import load_config, AtlasConfig
from atlas.model import AtlasLM
from atlas.tokenizer import Tokenizer
from atlas.data import TextDataset, create_dataloader
from atlas.training import (
    Trainer,
    create_optimizer,
    create_scheduler,
    CheckpointManager,
    CheckpointMetadata,
    compute_perplexity,
)


# Global flag for graceful shutdown
interrupted = False

# Global logger
logger = None


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    # ANSI color codes
    COLORS = {
        'HEADER': '\033[95m',
        'BLUE': '\033[94m',
        'CYAN': '\033[96m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'RED': '\033[91m',
        'BOLD': '\033[1m',
        'UNDERLINE': '\033[4m',
        'RESET': '\033[0m',
    }
    
    def format(self, record):
        msg = record.getMessage()
        
        # Color section headers [1/6], [2/6], etc.
        if msg.startswith('[') and '/6]' in msg:
            return f"{self.COLORS['CYAN']}{self.COLORS['BOLD']}{msg}{self.COLORS['RESET']}"
        
        # Color EPOCH lines
        if '>>> EPOCH' in msg:
            return f"{self.COLORS['YELLOW']}{self.COLORS['BOLD']}{msg}{self.COLORS['RESET']}"
        
        # Color success/saved indicators
        if '[SAVED]' in msg or '[BEST]' in msg or '[SUCCESS]' in msg:
            return f"{self.COLORS['GREEN']}{msg}{self.COLORS['RESET']}"
        
        # Color error/warning
        if '[ERROR]' in msg or 'ERROR' in msg:
            return f"{self.COLORS['RED']}{self.COLORS['BOLD']}{msg}{self.COLORS['RESET']}"
        if '[WARNING]' in msg or 'WARNING' in msg:
            return f"{self.COLORS['YELLOW']}{msg}{self.COLORS['RESET']}"
        
        # Color separator lines
        if msg.strip().startswith('=') or msg.strip().startswith('-'):
            return f"{self.COLORS['BLUE']}{msg}{self.COLORS['RESET']}"
        
        # Color step/epoch summaries
        if 'Summary:' in msg or 'Configuration' in msg:
            return f"{self.COLORS['CYAN']}{msg}{self.COLORS['RESET']}"
        
        return msg


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
    console_formatter = ColoredFormatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (no colors in file)
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
    if not interrupted:  # Only handle first interrupt
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
        "--no-resume",
        action="store_true",
        help="Skip checkpoint detection and start fresh training (used by run_pipeline scripts)",
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


def create_model_from_config(config: AtlasConfig) -> AtlasLM:
    """Create model from configuration."""
    # Copy gradient checkpointing setting from training config to model config
    # This allows it to be specified in either place in YAML files
    if hasattr(config.training, 'gradient_checkpointing'):
        config.model.gradient_checkpointing = config.training.gradient_checkpointing
    return AtlasLM(config.model)


def create_datasets(train_paths: str, val_paths: Optional[str], tokenizer: Tokenizer, config: AtlasConfig):
    """Create train and validation datasets."""
    from pathlib import Path
    
    # Parse paths (comma-separated or directory)
    train_path_list = [p.strip() for p in train_paths.split(',')]
    train_files = []
    for path_str in train_path_list:
        path = Path(path_str)
        if path.is_dir():
            # If directory, glob for all .txt files
            train_files.extend([str(f) for f in path.glob('*.txt')])
        else:
            train_files.append(path_str)
    
    if not train_files:
        raise ValueError(f"No training files found in {train_paths}")
    
    # Use memory-mapped files for large datasets (>20M tokens) to reduce RAM usage
    use_mmap = config.training.batch_size == 1  # Enable for extreme memory optimization
    if use_mmap:
        logger.info("  Memory-mapped storage enabled (low RAM mode)")
    
    train_dataset = TextDataset(
        file_paths=train_files,
        tokenizer=tokenizer,
        max_seq_len=config.data.max_seq_len,
        use_mmap=use_mmap,
    )
    
    val_dataset = None
    if val_paths:
        val_path_list = [p.strip() for p in val_paths.split(',')]
        val_files = []
        for path_str in val_path_list:
            path = Path(path_str)
            if path.is_dir():
                val_files.extend([str(f) for f in path.glob('*.txt')])
            else:
                val_files.append(path_str)
        
        if val_files:
            val_dataset = TextDataset(
                file_paths=val_files,
                tokenizer=tokenizer,
                max_seq_len=config.data.max_seq_len,
                use_mmap=use_mmap,
            )
    
    return train_dataset, val_dataset


def main():
    """Main training loop."""
    global interrupted, logger
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse arguments
    args = parse_args()
    
    # Check for existing checkpoints and prompt user if not explicitly resuming
    checkpoint_manager_temp = CheckpointManager(args.output_dir)
    latest_checkpoint = checkpoint_manager_temp.find_latest_checkpoint()
    
    # Only prompt if checkpoint exists, user didn't specify --resume, and --no-resume wasn't set
    if latest_checkpoint and not args.resume and not args.no_resume:
        # Found checkpoint but user didn't specify --resume or --no-resume
        checkpoint_info = checkpoint_manager_temp.get_checkpoint_info(latest_checkpoint)
        
        print("\n" + "=" * 80)
        print("EXISTING CHECKPOINT DETECTED")
        print("=" * 80)
        print(f"Found checkpoint: {latest_checkpoint.name}")
        if checkpoint_info:
            print(f"  Step: {checkpoint_info.get('step', 'unknown')}")
            print(f"  Epoch: {checkpoint_info.get('epoch', 'unknown')}")
            print(f"  Loss: {checkpoint_info.get('loss', 'unknown'):.4f}" if checkpoint_info.get('loss') else "  Loss: unknown")
            print(f"  Perplexity: {checkpoint_info.get('perplexity', 'unknown'):.2f}" if checkpoint_info.get('perplexity') else "  Perplexity: unknown")
        print("=" * 80)
        
        while True:
            response = input("\nResume from checkpoint? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                args.resume = str(latest_checkpoint)
                print(f"[+] Will resume from {latest_checkpoint.name}\n")
                break
            elif response in ['n', 'no']:
                print("[-] Starting fresh training session\n")
                break
            else:
                print("Please enter 'y' or 'n'")
    
    # Setup logging
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
    logger.info(f"  Model: {config.model.num_layers} layers, {config.model.hidden_size} hidden, {config.model.num_heads} heads")
    logger.info(f"  Sequence length: {config.model.max_seq_len}")
    logger.info(f"  Batch size: {config.training.batch_size}")
    logger.info(f"  Max steps: {config.training.max_steps}")
    
    # Apply command-line overrides
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
        logger.info(f"  Override: learning_rate = {args.learning_rate}")
    if args.batch_size:
        config.training.batch_size = args.batch_size
        logger.info(f"  Override: batch_size = {args.batch_size}")
    if args.max_steps:
        config.training.max_steps = args.max_steps
        logger.info(f"  Override: max_steps = {args.max_steps}")
    
    # Initialize tokenizer
    logger.info("\n[2/6] Initializing tokenizer...")
    tokenizer = Tokenizer(encoding_name="gpt2")
    logger.info(f"  Tokenizer: gpt2")
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
    
    # Gradient checkpointing is enabled via config during model creation
    if config.training.gradient_checkpointing:
        logger.info(f"  Gradient checkpointing enabled (reduced memory usage)")
    
    # Load checkpoint if resuming
    start_step = 0
    start_epoch = 1  # Start from epoch 1 (will be decremented before loop)
    if args.resume:
        logger.info(f"\n[*] Resuming from checkpoint: {args.resume}")
        checkpoint_manager = CheckpointManager(args.output_dir)
        
        # Create optimizer and scheduler (will be loaded from checkpoint)
        optimizer = create_optimizer(
            model,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            optimizer_type=config.training.optimizer_type,
            momentum=config.training.momentum,
        )
        scheduler = create_scheduler(
            optimizer,
            num_training_steps=config.training.max_steps,
            num_warmup_steps=config.training.warmup_steps,
            scheduler_type=config.training.scheduler_type,
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
    stats = train_dataset.get_stats()
    logger.info(f"  Train samples: {len(train_dataset):,}")
    logger.info(f"  Train tokens: {stats['total_tokens']:,}")
    if stats.get('use_mmap'):
        logger.info(f"  Using memory-mapped storage (low RAM usage)")
    if val_dataset:
        logger.info(f"  Val samples: {len(val_dataset):,}")
        logger.info(f"  Val tokens: {val_dataset.get_stats()['total_tokens']:,}")
    logger.info(f"  Dataset loading time: {load_time:.2f}s")
    
    # Force garbage collection to free memory after dataset loading
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create dataloaders
    # Disable shuffle for batch_size=1 to reduce memory overhead
    use_shuffle = config.training.batch_size > 1
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=use_shuffle,
        num_workers=config.data.num_workers,
    )
    if not use_shuffle:
        logger.info("  Shuffle disabled for extreme memory optimization")
    logger.info(f"  Train batches per epoch: {len(train_loader):,}")
    
    val_loader = None
    if val_dataset:
        val_loader = create_dataloader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        logger.info(f"  Val batches: {len(val_loader):,}")
    
    # Create optimizer and scheduler (if not resuming)
    logger.info("\n[5/6] Setting up training...")
    if not args.resume:
        optimizer = create_optimizer(
            model,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            optimizer_type=config.training.optimizer_type,
            momentum=config.training.momentum,
        )
        scheduler = create_scheduler(
            optimizer,
            num_training_steps=config.training.max_steps,
            num_warmup_steps=config.training.warmup_steps,
            scheduler_type=config.training.scheduler_type,
        )
    
    # Initialize auto_save_interval (needed by trainer)
    # Save every 100 global steps = 1600 batches (~4-5 minutes at current speed)
    auto_save_interval = 100
    
    # Create checkpoint manager first (needed by trainer)
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=args.output_dir,
        model_name='atlas',
        keep_best=True,
        keep_last_n=config.training.keep_checkpoints,  # Step-based checkpoints
        keep_last_epochs=5,  # Keep last 5 epoch checkpoints
    )
    
    # Create trainer with built-in checkpoint manager
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        max_grad_norm=config.training.max_grad_norm,
        device=args.device,
        checkpoint_manager=checkpoint_manager,
        auto_save_interval=auto_save_interval,
    )
    
    # Set starting step if resuming
    if args.resume:
        trainer.global_step = start_step
    
    # Enable aggressive memory management for extreme optimization
    if config.training.batch_size == 1:
        import gc
        gc.set_threshold(700, 10, 10)  # More aggressive garbage collection
        logger.info("  Aggressive garbage collection enabled")
    
    logger.info(f"  Optimizer: AdamW")
    logger.info(f"  Learning rate: {config.training.learning_rate}")
    logger.info(f"  Weight decay: {config.training.weight_decay}")
    logger.info(f"  Scheduler: {config.training.scheduler_type}")
    logger.info(f"  Warmup steps: {config.training.warmup_steps:,}")
    logger.info(f"  Gradient accumulation: {config.training.gradient_accumulation_steps}")
    logger.info(f"  Max grad norm: {config.training.max_grad_norm}")
    logger.info(f"  Effective batch size: {config.training.batch_size * config.training.gradient_accumulation_steps}")
    
    # Estimate training time
    total_tokens = len(train_dataset) * config.model.max_seq_len
    tokens_per_step = config.training.batch_size * config.model.max_seq_len * config.training.gradient_accumulation_steps
    steps_per_epoch = len(train_dataset) // config.training.batch_size
    estimated_epochs = (config.training.max_steps - start_step) / steps_per_epoch
    logger.info(f"  Steps per epoch: ~{steps_per_epoch:,}")
    logger.info(f"  Estimated epochs to complete: ~{estimated_epochs:.1f}")
    logger.info(f"  Tokens per step: {tokens_per_step:,}")
    
    # Auto-adjust save_interval if not explicitly set to aim for ~10 minute intervals
    # We'll refine this after the first few steps based on actual throughput
    # (auto_save_interval already initialized earlier before trainer creation)
    save_interval_adjusted = False
    
    # Training loop
    logger.info("\n[6/6] Starting training...")
    logger.info("=" * 80)
    logger.info(f"Training from step {start_step} to {config.training.max_steps}")
    logger.info(f"Logging interval: every {args.log_interval} steps")
    logger.info(f"Eval interval: every {args.eval_interval} steps")
    logger.info(f"Save interval: every {auto_save_interval} steps (will auto-adjust for ~10 min)")
    logger.info(f"Epoch checkpoints: saved at end of each epoch (keep last 5)")
    
    # Warn about 8-bit optimizer initialization delay
    if config.training.optimizer_type == 'adamw8bit':
        logger.info("")
        logger.info("NOTE: First optimizer step will take 15-30 seconds (8-bit optimizer initialization)")
        logger.info("      This is normal - CUDA kernels are being compiled and loaded into memory")
        logger.info("      Your PC might look 'frozen' for about 20-60 seconds")
        logger.info("      Subsequent steps will be fast")
    
    logger.info("=" * 80)
    
    max_steps = config.training.max_steps
    epoch = start_epoch - 1  # Will be incremented at loop start
    best_val_loss = float('inf')
    best_train_loss = float('inf')  # Track best training loss
    
    # Restore best loss from checkpoint if resuming
    if args.resume and 'metadata' in locals():
        if metadata.best_metric is not None:
            best_train_loss = metadata.best_metric
            logger.info(f"  Restored best training loss from checkpoint: {best_train_loss:.4f}")
    
    training_start_time = time.time()
    
    # Define step callback for mid-epoch checkpointing
    def checkpoint_callback(trainer_obj, loss):
        """Called after each global step to save checkpoints."""
        # Save step-based checkpoint at intervals
        if trainer_obj.global_step % auto_save_interval == 0 and trainer_obj.global_step > 0:
            logger.info(f"\n{'-'*80}")
            logger.info(f"Saving checkpoint at step {trainer_obj.global_step}...")
            metadata = CheckpointMetadata(
                step=trainer_obj.global_step,
                epoch=epoch,
                loss=loss,
                perplexity=compute_perplexity(torch.tensor(loss)).item(),
                learning_rate=optimizer.param_groups[0]['lr'],
            )
            
            checkpoint_path = checkpoint_manager.save_checkpoint(
                model,
                optimizer,
                metadata,
                scheduler=scheduler,
                is_best=False,
                is_epoch_end=False,
            )
            logger.info(f"  [SAVED] Step checkpoint: {checkpoint_path}")
            logger.info(f"  [SAVED]   Step: {metadata.step}, Epoch: {metadata.epoch}")
            logger.info(f"  [SAVED]   Loss: {metadata.loss:.4f}, Perplexity: {metadata.perplexity:.2f}")
            logger.info(f"  [SAVED]   Learning Rate: {metadata.learning_rate:.2e}")
            logger.info(f"{'-'*80}")
    
    try:
        while trainer.global_step < max_steps and not interrupted:
            epoch += 1
            trainer.current_epoch = epoch  # Update epoch in trainer for checkpoint metadata
            epoch_start_time = time.time()
            logger.info(f"\n{'='*80}")
            logger.info(f">>> EPOCH {epoch} | Step {trainer.global_step}/{max_steps}")
            logger.info(f"{'='*80}")
            
            # Train for one epoch with checkpoint callback and interrupt checker
            train_stats = trainer.train_epoch(
                train_loader,
                max_steps=max_steps,
                log_interval=args.log_interval,
                step_callback=checkpoint_callback,
                check_interrupt=lambda: interrupted,
            )
            
            epoch_time = time.time() - epoch_start_time
            tokens_processed = len(train_dataset) * config.model.max_seq_len
            tokens_per_sec = tokens_processed / epoch_time
            
            # Auto-adjust save_interval after first epoch for ~10 minute intervals
            if not save_interval_adjusted and epoch == start_epoch + 1:
                steps_in_epoch = trainer.global_step - start_step
                avg_time_per_step = epoch_time / steps_in_epoch if steps_in_epoch > 0 else 10
                # Target 10 minutes = 600 seconds
                target_save_interval = max(50, int(600 / avg_time_per_step))
                if target_save_interval != auto_save_interval:
                    auto_save_interval = target_save_interval
                    logger.info(f"\n[INFO] Auto-adjusted save interval to {auto_save_interval} steps (~10 min based on {avg_time_per_step:.1f}s/step)")
                    save_interval_adjusted = True
            
            logger.info(f"\n{'-'*80}")
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
            logger.info(f"{'-'*80}")
            
            # Save epoch checkpoint (only if not interrupted)
            if not interrupted:
                logger.info(f"\n{'-'*80}")
                logger.info("Saving epoch checkpoint...")
                epoch_metadata = CheckpointMetadata(
                    step=trainer.global_step,
                    epoch=epoch,
                    loss=train_stats['loss'],
                    perplexity=train_stats['perplexity'],
                    learning_rate=optimizer.param_groups[0]['lr'],
                )
                epoch_checkpoint_path = checkpoint_manager.save_checkpoint(
                    model,
                    optimizer,
                    epoch_metadata,
                    scheduler=scheduler,
                    is_best=False,
                    is_epoch_end=True,
                )
                logger.info(f"  [SAVED] Epoch checkpoint: {epoch_checkpoint_path}")
                logger.info(f"{'-'*80}")
            
            # Check if current training loss is the best
            train_loss = train_stats['loss']
            is_best_train = train_loss < best_train_loss
            if is_best_train:
                best_train_loss = train_loss
            
            # Evaluate on validation set
            if val_loader and not interrupted and trainer.global_step % args.eval_interval == 0:
                logger.info(f"\n{'-'*80}")
                logger.info("Running validation...")
                val_start = time.time()
                val_stats = trainer.evaluate(val_loader, show_progress=False)
                val_time = time.time() - val_start
                logger.info(f"  Val loss: {val_stats['loss']:.4f}")
                logger.info(f"  Val perplexity: {val_stats['perplexity']:.2f}")
                logger.info(f"  Val time: {val_time:.2f}s")
                
                # Save best model based on validation loss
                if val_stats['loss'] < best_val_loss:
                    improvement = ((best_val_loss - val_stats['loss']) / best_val_loss) * 100
                    best_val_loss = val_stats['loss']
                    logger.info(f"  [BEST] NEW BEST VALIDATION LOSS! (improved by {improvement:.2f}%)")
                    is_best = True
                else:
                    is_best = False
                logger.info(f"  Best val loss so far: {best_val_loss:.4f}")
                logger.info(f"{'-'*80}")
            else:
                val_stats = None
                # Use training loss as "best" metric when no validation
                is_best = is_best_train
                if is_best:
                    logger.info(f"\n{'-'*80}")
                    improvement = ((float('inf') if best_train_loss == train_loss else best_train_loss - train_loss) / best_train_loss if best_train_loss != float('inf') else 0) * 100
                    logger.info(f"  [BEST] NEW BEST TRAINING LOSS: {train_loss:.4f}")
                    if improvement > 0:
                        logger.info(f"  [BEST] Improved by {improvement:.2f}%")
                    logger.info(f"{'-'*80}")
            
            # Check if interrupted before saving regular checkpoints
            if interrupted:
                # Save checkpoint on interrupt
                logger.info(f"\n{'-'*80}")
                logger.info(f"Saving checkpoint on interrupt at step {trainer.global_step}...")
                interrupt_metadata = CheckpointMetadata(
                    step=trainer.global_step,
                    epoch=epoch,
                    loss=train_stats['loss'],
                    perplexity=train_stats['perplexity'],
                    learning_rate=optimizer.param_groups[0]['lr'],
                )
                interrupt_checkpoint_path = checkpoint_manager.save_checkpoint(
                    model,
                    optimizer,
                    interrupt_metadata,
                    scheduler=scheduler,
                    is_best=False,
                    is_epoch_end=False,
                )
                logger.info(f"  [SAVED] Interrupt checkpoint: {interrupt_checkpoint_path}")
                logger.info(f"  [SAVED]   Step: {interrupt_metadata.step}, Epoch: {interrupt_metadata.epoch}")
                logger.info(f"  [SAVED]   Loss: {interrupt_metadata.loss:.4f}, Perplexity: {interrupt_metadata.perplexity:.2f}")
                logger.info(f"  [SAVED]   Learning Rate: {interrupt_metadata.learning_rate:.2e}")
                logger.info(f"{'-'*80}")
                break
            
            # Save step-based checkpoint (time-based interval)
            if trainer.global_step % auto_save_interval == 0:
                logger.info(f"\n{'-'*80}")
                logger.info("Saving step checkpoint...")
                metadata = CheckpointMetadata(
                    step=trainer.global_step,
                    epoch=epoch,
                    loss=train_stats['loss'],
                    perplexity=train_stats['perplexity'],
                    learning_rate=optimizer.param_groups[0]['lr'],
                    best_metric=best_val_loss if val_stats else best_train_loss,
                )
                
                checkpoint_path = checkpoint_manager.save_checkpoint(
                    model,
                    optimizer,
                    metadata,
                    scheduler=scheduler,
                    is_best=is_best,
                    is_epoch_end=False,
                )
                logger.info(f"  [SAVED] Step checkpoint: {checkpoint_path}")
                if is_best:
                    logger.info(f"  [BEST] Best model saved (always kept)")
                logger.info(f"{'-'*80}")
        
        # Training complete
        if not interrupted:
            total_training_time = time.time() - training_start_time
            hours = total_training_time / 3600
            logger.info(f"\n{'='*80}")
            logger.info("[SUCCESS] TRAINING COMPLETED SUCCESSFULLY!")
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
        logger.error(f"[ERROR] TRAINING ERROR: {str(e)}")
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
            logger.info("[WARNING] TRAINING INTERRUPTED BY USER")
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
