"""
Evaluation loop implementation for Atlas LLM.

Provides evaluation utilities for:
- Computing validation metrics
- Running evaluation without gradient computation
- Integration with training loop
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .loss import compute_lm_loss_with_logits_shift, compute_perplexity


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    loss: float
    perplexity: float
    num_tokens: int
    num_batches: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'loss': self.loss,
            'perplexity': self.perplexity,
            'num_tokens': self.num_tokens,
            'num_batches': self.num_batches,
        }


class Evaluator:
    """
    Evaluator for language model validation.
    
    Handles:
    - Evaluation loop without gradient computation
    - Metric computation (loss, perplexity)
    - Integration with training workflow
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            device: Device to evaluate on ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.device = device
    
    @torch.no_grad()
    def evaluate_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> tuple[float, int]:
        """
        Execute single evaluation step.
        
        Args:
            batch: Dictionary with 'input_ids' and optional 'labels'
        
        Returns:
            Tuple of (loss, num_tokens)
        """
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch.get('labels', input_ids).to(self.device)
        
        # Forward pass (no grad)
        logits = self.model(input_ids)
        
        # Compute loss
        loss = compute_lm_loss_with_logits_shift(
            model_output=logits,
            input_ids=labels,
            ignore_index=-100,
        )
        
        # Count tokens (excluding padding if applicable)
        num_tokens = (labels != -100).sum().item()
        
        return loss.item(), num_tokens
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
        show_progress: bool = True,
    ) -> EvaluationMetrics:
        """
        Evaluate model on validation data.
        
        Args:
            dataloader: DataLoader for validation data
            max_batches: Maximum number of batches to evaluate (None for all)
            show_progress: Whether to show progress bar
        
        Returns:
            EvaluationMetrics object with validation results
        """
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        # Progress bar
        iterator = tqdm(
            dataloader,
            desc='Evaluating',
            disable=not show_progress,
            total=max_batches if max_batches else len(dataloader),
        )
        
        for batch_idx, batch in enumerate(iterator):
            # Check max batches
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            # Evaluation step
            loss, num_tokens = self.evaluate_step(batch)
            
            # Accumulate metrics
            total_loss += loss * num_tokens  # Weight by number of tokens
            total_tokens += num_tokens
            num_batches += 1
            
            # Update progress bar
            if show_progress:
                current_avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
                current_perplexity = compute_perplexity(torch.tensor(current_avg_loss)).item()
                iterator.set_postfix({
                    'loss': f'{current_avg_loss:.4f}',
                    'ppl': f'{current_perplexity:.2f}',
                })
        
        iterator.close()
        
        # Compute final metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = compute_perplexity(torch.tensor(avg_loss)).item()
        
        return EvaluationMetrics(
            loss=avg_loss,
            perplexity=perplexity,
            num_tokens=total_tokens,
            num_batches=num_batches,
        )


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    max_batches: Optional[int] = None,
    show_progress: bool = True,
) -> EvaluationMetrics:
    """
    Convenience function to evaluate a model.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for validation data
        device: Device to evaluate on
        max_batches: Maximum number of batches (None for all)
        show_progress: Whether to show progress bar
    
    Returns:
        EvaluationMetrics object
    
    Example:
        >>> metrics = evaluate_model(model, val_dataloader)
        >>> print(f"Validation loss: {metrics.loss:.4f}, perplexity: {metrics.perplexity:.2f}")
    """
    evaluator = Evaluator(model, device=device)
    return evaluator.evaluate(dataloader, max_batches=max_batches, show_progress=show_progress)
