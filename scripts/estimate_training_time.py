#!/usr/bin/env python3
"""
Training Time Estimation Tool for Atlas.

This script performs a benchmark run to accurately estimate total training time
for a given configuration and dataset. It accounts for:
- First epoch warmup overhead
- Stable epoch timing
- Checkpoint saving overhead
- Validation overhead
- GPU thermal characteristics
- Dataset size and batch parameters
"""

""" 
!IMPORTANT: 
> This file contains some bugs and won't run yet. I am still working on fixing them.
> Please check back later! 

— bugs fear me (soon), Julius
"""

import argparse
import time
import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import json

import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from atlas.config import AtlasConfig, load_config
from atlas.model import AtlasLM
from atlas.tokenizer import Tokenizer
from atlas.data import TextDataset, create_dataloader
from atlas.training import Trainer, create_optimizer, create_scheduler


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )


def calculate_dataset_stats(
    train_files: list,
    tokenizer: Tokenizer,
    max_seq_len: int,
    stride: int
) -> Dict[str, int]:
    """
    Calculate dataset statistics without loading all data.
    
    Args:
        train_files: List of training file paths
        tokenizer: Tokenizer instance
        max_seq_len: Maximum sequence length
        stride: Stride for sliding window
        
    Returns:
        Dictionary with dataset statistics
    """
    logging.info("Analyzing dataset...")
    
    # Create dataset to get accurate sequence count
    dataset = TextDataset(
        file_paths=train_files,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        stride=stride
    )
    
    total_sequences = len(dataset)
    stats = dataset.get_stats()
    
    # Calculate average sequence length
    avg_seq_len = stats['total_tokens'] / total_sequences if total_sequences > 0 else 0
    
    return {
        'total_sequences': total_sequences,
        'total_tokens': stats['total_tokens'],
        'num_files': stats['num_files'],
        'avg_seq_len': avg_seq_len
    }


def benchmark_training_step(
    model: AtlasLM,
    trainer: Trainer,
    dataloader: torch.utils.data.DataLoader,
    num_steps: int,
    device: torch.device,
    warmup_steps: int = 50
) -> Dict[str, float]:
    """
    Benchmark training steps and measure throughput.
    
    Args:
        model: Atlas model
        trainer: Trainer instance
        dataloader: DataLoader instance
        num_steps: Number of steps to benchmark
        device: Device to train on
        warmup_steps: Number of warmup steps (excluded from timing)
        
    Returns:
        Dictionary with benchmark results
    """
    logging.info(f"Running benchmark: {warmup_steps} warmup + {num_steps} measured steps...")
    
    model.train()
    data_iter = iter(dataloader)
    
    # Warmup phase (JIT compilation, cache warming)
    logging.info("Warmup phase (JIT compilation, GPU cache warming)...")
    warmup_start = time.time()
    for i in range(warmup_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        batch = batch.to(device)
        _, _ = trainer.train_step(batch)
    
    # Ensure GPU operations complete
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    warmup_time = time.time() - warmup_start
    
    # Measured phase
    logging.info("Measurement phase...")
    step_times = []
    throughputs = []
    
    measure_start = time.time()
    
    for i in tqdm(range(num_steps), desc="Benchmarking"):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        batch = batch.to(device)
        
        step_start = time.time()
        loss, metrics = trainer.train_step(batch)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        # Calculate throughput (iterations per second)
        throughput = 1.0 / step_time if step_time > 0 else 0.0
        throughputs.append(throughput)
    
    measure_time = time.time() - measure_start
    
    # Statistics
    avg_step_time = sum(step_times) / len(step_times)
    min_step_time = min(step_times)
    max_step_time = max(step_times)
    avg_throughput = sum(throughputs) / len(throughputs)
    
    # Check for thermal throttling (increasing step times)
    first_half = step_times[:len(step_times)//2]
    second_half = step_times[len(step_times)//2:]
    first_avg = sum(first_half) / len(first_half)
    second_avg = sum(second_half) / len(second_half)
    slowdown_pct = ((second_avg - first_avg) / first_avg) * 100
    
    return {
        'warmup_time': warmup_time,
        'warmup_steps': warmup_steps,
        'measured_steps': num_steps,
        'total_measured_time': measure_time,
        'avg_step_time': avg_step_time,
        'min_step_time': min_step_time,
        'max_step_time': max_step_time,
        'avg_throughput': avg_throughput,
        'thermal_slowdown_pct': slowdown_pct,
        'is_thermal_throttling': slowdown_pct > 5.0  # >5% slowdown indicates throttling
    }


def benchmark_checkpoint_save(
    model: AtlasLM,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    checkpoint_dir: Path,
    num_trials: int = 3
) -> float:
    """
    Benchmark checkpoint saving time.
    
    Args:
        model: Atlas model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        checkpoint_dir: Directory to save checkpoints
        num_trials: Number of trials to average
        
    Returns:
        Average checkpoint save time in seconds
    """
    logging.info("Benchmarking checkpoint save time...")
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_times = []
    
    for i in range(num_trials):
        checkpoint_path = checkpoint_dir / f"benchmark_ckpt_{i}.pt"
        metadata_path = checkpoint_dir / f"benchmark_ckpt_{i}.json"
        
        start = time.time()
        
        # Save checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'step': i * 1000,
            'epoch': 1,
            'loss': 3.5,
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Save metadata
        metadata = {
            'step': i * 1000,
            'epoch': 1,
            'loss': 3.5,
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        save_time = time.time() - start
        save_times.append(save_time)
        
        # Clean up
        checkpoint_path.unlink()
        metadata_path.unlink()
    
    avg_save_time = sum(save_times) / len(save_times)
    logging.info(f"Average checkpoint save time: {avg_save_time:.2f}s")
    
    return avg_save_time


def benchmark_validation(
    model: AtlasLM,
    val_dataloader: Optional[torch.utils.data.DataLoader],
    device: torch.device,
    max_eval_batches: int = 100
) -> float:
    """
    Benchmark validation time.
    
    Args:
        model: Atlas model
        val_dataloader: Validation dataloader
        device: Device
        max_eval_batches: Maximum batches to evaluate
        
    Returns:
        Average validation time in seconds
    """
    if val_dataloader is None:
        return 0.0
    
    logging.info("Benchmarking validation time...")
    
    model.eval()
    
    start = time.time()
    batches_evaluated = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            if batches_evaluated >= max_eval_batches:
                break
            
            batch = batch.to(device)
            _ = model(batch)
            batches_evaluated += 1
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    val_time = time.time() - start
    
    model.train()
    
    logging.info(f"Validation time ({batches_evaluated} batches): {val_time:.2f}s")
    
    return val_time


def estimate_training_time(
    config: AtlasConfig,
    dataset_stats: Dict[str, int],
    benchmark_results: Dict[str, float],
    checkpoint_save_time: float,
    validation_time: float,
    num_epochs: int,
    save_interval: int,
    eval_interval: Optional[int]
) -> Dict[str, float]:
    """
    Estimate total training time based on benchmarks.
    
    Args:
        config: Atlas configuration
        dataset_stats: Dataset statistics
        benchmark_results: Benchmark results
        checkpoint_save_time: Average checkpoint save time
        validation_time: Validation time
        num_epochs: Number of epochs to train
        save_interval: Checkpoint save interval
        eval_interval: Evaluation interval
        
    Returns:
        Dictionary with time estimates
    """
    logging.info("Calculating time estimates...")
    
    # Calculate steps per epoch
    effective_batch_size = config.training.batch_size * config.training.gradient_accumulation_steps
    steps_per_epoch = dataset_stats['total_sequences'] // effective_batch_size
    total_steps = steps_per_epoch * num_epochs
    
    # Base training time
    avg_step_time = benchmark_results['avg_step_time']
    base_training_time = total_steps * avg_step_time
    
    # Account for thermal throttling
    if benchmark_results['is_thermal_throttling']:
        throttle_factor = 1 + (benchmark_results['thermal_slowdown_pct'] / 100)
        base_training_time *= throttle_factor
        logging.warning(f"Thermal throttling detected: {benchmark_results['thermal_slowdown_pct']:.1f}% slowdown")
    
    # Checkpoint overhead
    num_checkpoints = total_steps // save_interval
    checkpoint_overhead = num_checkpoints * checkpoint_save_time
    
    # Validation overhead
    validation_overhead = 0.0
    if eval_interval:
        num_validations = total_steps // eval_interval
        validation_overhead = num_validations * validation_time
    
    # Epoch-end checkpoint overhead (usually takes longer)
    epoch_checkpoint_overhead = num_epochs * checkpoint_save_time * 1.5
    
    # First epoch overhead (warmup, compilation, etc.)
    # Estimate 20% slower than subsequent epochs
    first_epoch_time = (steps_per_epoch * avg_step_time) * 1.2
    remaining_epochs_time = ((num_epochs - 1) * steps_per_epoch) * avg_step_time
    adjusted_training_time = first_epoch_time + remaining_epochs_time
    
    # Apply thermal throttling to adjusted time
    if benchmark_results['is_thermal_throttling']:
        adjusted_training_time *= throttle_factor
    
    # Total time
    total_time = (
        adjusted_training_time +
        checkpoint_overhead +
        validation_overhead +
        epoch_checkpoint_overhead
    )
    
    # Add 5% buffer for miscellaneous overhead
    total_time *= 1.05
    
    return {
        'steps_per_epoch': steps_per_epoch,
        'total_steps': total_steps,
        'base_training_time': base_training_time,
        'adjusted_training_time': adjusted_training_time,
        'checkpoint_overhead': checkpoint_overhead + epoch_checkpoint_overhead,
        'validation_overhead': validation_overhead,
        'total_time': total_time,
        'avg_step_time': avg_step_time,
        'avg_throughput': benchmark_results['avg_throughput'],
        'thermal_throttling': benchmark_results['is_thermal_throttling']
    }


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 24:
        days = hours // 24
        hours = hours % 24
        return f"{days}d {hours}h {minutes}m {secs}s"
    elif hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_results(
    config: AtlasConfig,
    dataset_stats: Dict[str, int],
    benchmark_results: Dict[str, float],
    estimates: Dict[str, float],
    num_epochs: int
):
    """Print formatted results."""
    print("\n" + "="*80)
    print("TRAINING TIME ESTIMATION RESULTS")
    print("="*80)
    
    print("\n[CONFIG] Configuration:")
    print(f"  Model: {config.model.num_layers} layers, {config.model.hidden_size} hidden, "
          f"{config.model.num_heads} heads")
    print(f"  Parameters: ~{sum(p.numel() for p in [torch.zeros(config.model.vocab_size, config.model.hidden_size)])/1e6:.1f}M "
          f"(rough estimate)")
    print(f"  Batch size: {config.training.batch_size} × {config.training.gradient_accumulation_steps} grad accum "
          f"= {config.training.batch_size * config.training.gradient_accumulation_steps} effective")
    print(f"  Sequence length: {config.model.max_seq_len}")
    print(f"  Epochs: {num_epochs}")
    
    print("\n[DATASET] Dataset:")
    print(f"  Total sequences: {dataset_stats['total_sequences']:,}")
    print(f"  Total tokens: {dataset_stats['total_tokens']:,}")
    print(f"  Files: {dataset_stats['num_files']}")
    print(f"  Average sequence length: {dataset_stats['avg_seq_len']:.1f} tokens")
    
    print("\n[BENCHMARK] Benchmark Results:")
    print(f"  Warmup: {benchmark_results['warmup_time']:.1f}s ({benchmark_results['warmup_steps']} steps)")
    print(f"  Measured steps: {benchmark_results['measured_steps']}")
    print(f"  Average step time: {benchmark_results['avg_step_time']:.3f}s")
    print(f"  Min/Max step time: {benchmark_results['min_step_time']:.3f}s / {benchmark_results['max_step_time']:.3f}s")
    print(f"  Average throughput: {benchmark_results['avg_throughput']:.2f} it/s")
    
    if benchmark_results['is_thermal_throttling']:
        print(f"  [WARNING] Thermal throttling: {benchmark_results['thermal_slowdown_pct']:.1f}% slowdown detected")
        print(f"            (GPU may be heating up - consider better cooling)")
    else:
        print(f"  [OK] No thermal throttling detected ({benchmark_results['thermal_slowdown_pct']:.1f}% variance)")
    
    print("\n[TIME] Time Estimates:")
    print(f"  Steps per epoch: {estimates['steps_per_epoch']:,}")
    print(f"  Total steps: {estimates['total_steps']:,}")
    print(f"  Base training time: {format_time(estimates['base_training_time'])}")
    print(f"  Adjusted training time: {format_time(estimates['adjusted_training_time'])}")
    print(f"  Checkpoint overhead: {format_time(estimates['checkpoint_overhead'])}")
    print(f"  Validation overhead: {format_time(estimates['validation_overhead'])}")
    
    print(f"\n[ESTIMATE] ESTIMATED TOTAL TIME: {format_time(estimates['total_time'])}")
    
    # Time per epoch
    time_per_epoch = estimates['total_time'] / num_epochs
    print(f"  Average per epoch: {format_time(time_per_epoch)}")
    
    # Breakdown percentages
    training_pct = (estimates['adjusted_training_time'] / estimates['total_time']) * 100
    checkpoint_pct = (estimates['checkpoint_overhead'] / estimates['total_time']) * 100
    validation_pct = (estimates['validation_overhead'] / estimates['total_time']) * 100
    
    print(f"\n[BREAKDOWN] Time Breakdown:")
    print(f"  Training: {training_pct:.1f}%")
    print(f"  Checkpointing: {checkpoint_pct:.1f}%")
    print(f"  Validation: {validation_pct:.1f}%")
    print(f"  Overhead: {100 - training_pct - checkpoint_pct - validation_pct:.1f}%")
    
    # Predictions
    print(f"\n[TIMELINE] Timeline Predictions:")
    now = time.time()
    end_time = now + estimates['total_time']
    
    from datetime import datetime, timedelta
    end_datetime = datetime.fromtimestamp(end_time)
    
    print(f"  If started now: {datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Estimated completion: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  ({(estimates['total_time'] / 3600 / 24):.1f} days from now)")
    
    print("\n" + "="*80)
    
    # Confidence note
    print("\n[NOTES] Notes:")
    print("  - These estimates assume consistent GPU performance")
    print("  - First epoch may be 15-20% slower (JIT compilation, caching)")
    print("  - Actual time may vary by ±10% depending on system load")
    if benchmark_results['is_thermal_throttling']:
        print("  - [WARNING] Thermal throttling detected - actual time may be longer")
        print("              Consider improving cooling or using ULTRA config")
    print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate training time for Atlas models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--train-data',
        type=str,
        required=True,
        help='Path to training data directory'
    )
    parser.add_argument(
        '--val-data',
        type=str,
        help='Path to validation data (optional)'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=10,
        help='Number of epochs to estimate (default: 10)'
    )
    parser.add_argument(
        '--benchmark-steps',
        type=int,
        default=200,
        help='Number of steps to benchmark (default: 200)'
    )
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=50,
        help='Number of warmup steps (default: 50)'
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1000,
        help='Checkpoint save interval (default: 1000)'
    )
    parser.add_argument(
        '--eval-interval',
        type=int,
        help='Evaluation interval (optional)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use (default: auto)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Save results to JSON file'
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Load configuration
    logging.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logging.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"GPU: {gpu_name}")
    
    # Setup tokenizer
    logging.info("Initializing tokenizer...")
    tokenizer = Tokenizer(encoding_name="gpt2")
    
    # Get training files
    train_dir = Path(args.train_data)
    train_files = sorted(train_dir.glob("*.txt"))
    
    if not train_files:
        logging.error(f"No training files found in {train_dir}")
        return 1
    
    logging.info(f"Found {len(train_files)} training files")
    
    # Calculate dataset statistics
    dataset_stats = calculate_dataset_stats(
        train_files=train_files,
        tokenizer=tokenizer,
        max_seq_len=config.model.max_seq_len,
        stride=config.model.max_seq_len  # Non-overlapping by default
    )
    
    # Create dataset and dataloader
    logging.info("Creating dataset and dataloader...")
    dataset = TextDataset(
        file_paths=train_files,
        tokenizer=tokenizer,
        max_seq_len=config.model.max_seq_len,
        stride=config.model.max_seq_len
    )
    
    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=device.type == 'cuda'
    )
    
    # Create validation dataloader if specified
    val_dataloader = None
    if args.val_data:
        val_dir = Path(args.val_data)
        val_files = sorted(val_dir.glob("*.txt"))
        if val_files:
            val_dataset = TextDataset(
                file_paths=val_files,
                tokenizer=tokenizer,
                max_seq_len=config.model.max_seq_len,
                stride=config.model.max_seq_len
            )
            val_dataloader = create_dataloader(
                dataset=val_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=config.data.num_workers,
                pin_memory=device.type == 'cuda'
            )
    
    # Create model
    logging.info("Creating model...")
    model = AtlasLM(config.model).to(device)
    
    num_params = model.count_parameters()
    logging.info(f"Model parameters: {num_params:,}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model=model,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(config.training.beta1, config.training.beta2)
    )
    scheduler = create_scheduler(
        optimizer=optimizer,
        num_training_steps=1000000,  # Dummy value for benchmark
        num_warmup_steps=config.training.warmup_steps,
        scheduler_type=config.training.scheduler_type,
        min_lr_ratio=config.training.min_lr_ratio
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config.training,
        scheduler=scheduler,
        device=device
    )
    
    # Run training benchmark
    benchmark_results = benchmark_training_step(
        model=model,
        trainer=trainer,
        dataloader=dataloader,
        num_steps=args.benchmark_steps,
        device=device,
        warmup_steps=args.warmup_steps
    )
    
    # Benchmark checkpoint saving
    checkpoint_dir = Path("checkpoints_benchmark_temp")
    checkpoint_save_time = benchmark_checkpoint_save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=checkpoint_dir,
        num_trials=3
    )
    
    # Clean up benchmark checkpoints
    if checkpoint_dir.exists():
        import shutil
        shutil.rmtree(checkpoint_dir)
    
    # Benchmark validation
    validation_time = benchmark_validation(
        model=model,
        val_dataloader=val_dataloader,
        device=device,
        max_eval_batches=100
    )
    
    # Estimate total training time
    estimates = estimate_training_time(
        config=config,
        dataset_stats=dataset_stats,
        benchmark_results=benchmark_results,
        checkpoint_save_time=checkpoint_save_time,
        validation_time=validation_time,
        num_epochs=args.num_epochs,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval
    )
    
    # Print results
    print_results(
        config=config,
        dataset_stats=dataset_stats,
        benchmark_results=benchmark_results,
        estimates=estimates,
        num_epochs=args.num_epochs
    )
    
    # Save to file if requested
    if args.output:
        output_data = {
            'config_file': args.config,
            'num_epochs': args.num_epochs,
            'dataset_stats': dataset_stats,
            'benchmark_results': benchmark_results,
            'estimates': estimates,
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else None
        }
        
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logging.info(f"Results saved to {output_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
