"""
Loss functions for language model training.

This module provides loss computation for autoregressive language modeling.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_lm_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Compute cross-entropy loss for language modeling.
    
    This function computes the standard cross-entropy loss for next-token
    prediction in autoregressive language models.
    
    Args:
        logits: Model output logits of shape (batch_size, seq_len, vocab_size)
        targets: Target token IDs of shape (batch_size, seq_len)
        ignore_index: Index to ignore in loss computation (e.g., padding tokens)
        reduction: Loss reduction method ('mean', 'sum', or 'none')
        
    Returns:
        Scalar loss tensor (if reduction='mean' or 'sum') or
        tensor of shape (batch_size, seq_len) (if reduction='none')
        
    Example:
        >>> logits = model(input_ids)  # (batch_size, seq_len, vocab_size)
        >>> # Shift targets: predict next token at each position
        >>> targets = input_ids[:, 1:]  # Remove first token
        >>> logits = logits[:, :-1]  # Remove last prediction
        >>> loss = compute_lm_loss(logits, targets)
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Reshape for cross_entropy: (batch_size * seq_len, vocab_size)
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=ignore_index,
        reduction=reduction,
    )
    
    # If reduction='none', reshape back to (batch_size, seq_len)
    if reduction == 'none':
        loss = loss.reshape(batch_size, seq_len)
    
    return loss


def compute_lm_loss_with_logits_shift(
    model_output: torch.Tensor,
    input_ids: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute language modeling loss with automatic shifting.
    
    This is a convenience function that automatically shifts the logits and
    targets for next-token prediction. At each position, the model predicts
    the next token.
    
    Args:
        model_output: Model output logits (batch_size, seq_len, vocab_size)
        input_ids: Input token IDs (batch_size, seq_len)
        ignore_index: Index to ignore in loss computation
        
    Returns:
        Scalar loss tensor
        
    Example:
        >>> logits = model(input_ids)
        >>> loss = compute_lm_loss_with_logits_shift(logits, input_ids)
    """
    # Shift logits and targets
    # Model predicts token at position i+1 from position i
    shift_logits = model_output[:, :-1, :].contiguous()
    shift_targets = input_ids[:, 1:].contiguous()
    
    return compute_lm_loss(shift_logits, shift_targets, ignore_index=ignore_index)


def compute_perplexity(loss: torch.Tensor) -> torch.Tensor:
    """
    Compute perplexity from cross-entropy loss.
    
    Perplexity is exp(loss) and is a common metric for language models.
    Lower perplexity indicates better model performance.
    
    Args:
        loss: Cross-entropy loss (scalar)
        
    Returns:
        Perplexity value
        
    Example:
        >>> loss = compute_lm_loss(logits, targets)
        >>> perplexity = compute_perplexity(loss)
    """
    return torch.exp(loss)
