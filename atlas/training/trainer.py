"""
Training loop implementation for Atlas LLM.

Provides main training loop with:
- Forward/backward pass
- Loss computation
- Gradient accumulation
- Progress tracking
- Checkpointing integration
"""

import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .loss import compute_lm_loss_with_logits_shift, compute_perplexity
from .optimizer import clip_gradients


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    step: int
    loss: float
    perplexity: float
    learning_rate: float
    tokens_per_second: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'step': self.step,
            'loss': self.loss,
            'perplexity': self.perplexity,
            'learning_rate': self.learning_rate,
            'tokens_per_second': self.tokens_per_second,
        }


class Trainer:
    """
    Trainer for language model training.
    
    Handles:
    - Training loop with gradient accumulation
    - Progress tracking and logging
    - Metric computation
    - Integration with optimizer and scheduler
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer for parameter updates
            scheduler: Optional learning rate scheduler
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping (None to disable)
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Training state
        self.global_step = 0
        self.accumulated_loss = 0.0
        self.tokens_processed = 0
        self.start_time = time.time()
        
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        accumulation_step: int,
    ) -> float:
        """
        Execute single training step.
        
        Args:
            batch: Dictionary with 'input_ids' and optional 'labels'
            accumulation_step: Current step within accumulation cycle (0-indexed)
        
        Returns:
            Loss value for this step
        """
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch.get('labels', input_ids).to(self.device)
        
        # Forward pass
        logits = self.model(input_ids)
        
        # Compute loss
        loss = compute_lm_loss_with_logits_shift(
            model_output=logits,
            input_ids=labels,
            ignore_index=-100,
        )
        
        # Scale loss by accumulation steps
        scaled_loss = loss / self.gradient_accumulation_steps
        
        # Save loss value before any deletions
        loss_value = loss.item()
        
        # Backward pass
        scaled_loss.backward()
        
        # Free memory immediately after backward
        del scaled_loss
        
        # Update on last accumulation step
        is_last_accumulation = (accumulation_step + 1) == self.gradient_accumulation_steps
        if is_last_accumulation:
            # Clear memory BEFORE optimizer step to prevent spike
            if self.device == 'cuda' or (hasattr(self.device, 'type') and self.device.type == 'cuda'):
                torch.cuda.empty_cache()
            
            # Clip gradients if enabled
            if self.max_grad_norm is not None:
                clip_gradients(self.model, self.max_grad_norm)
            
            # Optimizer step (this causes the memory spike)
            self.optimizer.step()
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Zero gradients with set_to_none for memory efficiency
            self.optimizer.zero_grad(set_to_none=True)
            
            # Update global step
            self.global_step += 1
            
            # Aggressive memory cleanup after optimizer step
            if self.device == 'cuda' or (hasattr(self.device, 'type') and self.device.type == 'cuda'):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for GPU operations to complete
            
            # Force Python garbage collection after optimizer step
            import gc
            gc.collect()
        
        # Free batch tensors
        del input_ids, labels, logits, loss
        
        # Force garbage collection every 8 steps to prevent memory creep
        if accumulation_step % 8 == 0:
            import gc
            gc.collect()
        
        return loss_value
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        max_steps: Optional[int] = None,
        log_interval: int = 10,
        step_callback: Optional[callable] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data
            max_steps: Maximum number of steps (None for full epoch)
            log_interval: Log metrics every N global steps
            step_callback: Optional callback function called after each global step
                          Receives (trainer, loss) as arguments
        
        Returns:
            Dictionary with epoch statistics
        """
        self.model.train()
        
        epoch_loss = 0.0
        epoch_tokens = 0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(
            dataloader,
            desc='Training',
            total=max_steps if max_steps else len(dataloader),
        )
        
        accumulation_step = 0
        
        for batch_idx, batch in enumerate(pbar):
            # Check max steps
            if max_steps is not None and self.global_step >= max_steps:
                break
            
            # Training step
            loss = self.train_step(batch, accumulation_step)
            
            # Accumulate metrics
            epoch_loss += loss
            batch_tokens = batch['input_ids'].numel()
            epoch_tokens += batch_tokens
            self.tokens_processed += batch_tokens
            num_batches += 1
            
            # Update accumulation step
            accumulation_step = (accumulation_step + 1) % self.gradient_accumulation_steps
            
            # Call step callback if provided (for checkpointing, etc.)
            if step_callback is not None and accumulation_step == 0:
                step_callback(self, loss)
            
            # Log metrics
            if self.global_step > 0 and self.global_step % log_interval == 0:
                metrics = self.get_current_metrics(loss)
                pbar.set_postfix({
                    'loss': f'{metrics.loss:.4f}',
                    'ppl': f'{metrics.perplexity:.2f}',
                    'lr': f'{metrics.learning_rate:.2e}',
                    'tok/s': f'{metrics.tokens_per_second:.0f}',
                })
        
        pbar.close()
        
        # Compute epoch statistics
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_perplexity = compute_perplexity(torch.tensor(avg_loss)).item()
        
        return {
            'loss': avg_loss,
            'perplexity': avg_perplexity,
            'tokens': epoch_tokens,
            'steps': self.global_step,
        }
    
    def get_current_metrics(self, current_loss: float) -> TrainingMetrics:
        """
        Get current training metrics.
        
        Args:
            current_loss: Loss from current step
        
        Returns:
            TrainingMetrics object
        """
        # Compute tokens per second
        elapsed = time.time() - self.start_time
        tokens_per_second = self.tokens_processed / elapsed if elapsed > 0 else 0.0
        
        # Get current learning rate
        lr = self.optimizer.param_groups[0]['lr']
        
        # Compute perplexity (convert float to tensor if needed)
        loss_tensor = current_loss if isinstance(current_loss, torch.Tensor) else torch.tensor(current_loss)
        perplexity = compute_perplexity(loss_tensor).item()
        
        return TrainingMetrics(
            step=self.global_step,
            loss=current_loss,
            perplexity=perplexity,
            learning_rate=lr,
            tokens_per_second=tokens_per_second,
        )
    
    def reset_metrics(self):
        """Reset training metrics (e.g., at start of new epoch)."""
        self.tokens_processed = 0
        self.start_time = time.time()
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate model on validation data.
        
        Args:
            dataloader: DataLoader for validation data
            max_batches: Maximum number of batches (None for all)
            show_progress: Whether to show progress bar
        
        Returns:
            Dictionary with evaluation metrics
        """
        from .evaluator import Evaluator
        
        evaluator = Evaluator(self.model, device=self.device)
        metrics = evaluator.evaluate(dataloader, max_batches=max_batches, show_progress=show_progress)
        
        return metrics.to_dict()
