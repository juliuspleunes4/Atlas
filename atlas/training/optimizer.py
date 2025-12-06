"""
Optimizer and learning rate scheduler setup.

This module provides utilities for creating optimizers and learning rate
schedulers for training.
"""

import torch
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional, Dict, Any
import math
import logging

logger = logging.getLogger(__name__)


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.95),
    eps: float = 1e-8,
    exclude_from_weight_decay: Optional[list] = None,
) -> AdamW:
    """
    Create AdamW optimizer with weight decay.
    
    By default, excludes biases and LayerNorm parameters from weight decay,
    following best practices for transformer training.
    
    Args:
        model: PyTorch model
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        betas: Adam betas (beta1, beta2)
        eps: Adam epsilon
        exclude_from_weight_decay: List of parameter name patterns to exclude
            from weight decay (e.g., ['bias', 'LayerNorm'])
        
    Returns:
        AdamW optimizer
        
    Example:
        >>> optimizer = create_optimizer(model, learning_rate=3e-4)
    """
    if exclude_from_weight_decay is None:
        exclude_from_weight_decay = ['bias', 'LayerNorm.weight', 'ln', 'norm']
    
    # Separate parameters into groups
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if parameter should be excluded from weight decay
        exclude = any(exclude_pattern in name for exclude_pattern in exclude_from_weight_decay)
        
        if exclude:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    # Create parameter groups
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    
    logger.info(
        f"Creating AdamW optimizer: lr={learning_rate}, weight_decay={weight_decay}, "
        f"decay_params={len(decay_params)}, no_decay_params={len(no_decay_params)}"
    )
    
    optimizer = AdamW(
        param_groups,
        lr=learning_rate,
        betas=betas,
        eps=eps,
    )
    
    return optimizer


def create_scheduler(
    optimizer: Optimizer,
    num_training_steps: int,
    num_warmup_steps: Optional[int] = None,
    warmup_ratio: float = 0.1,
    scheduler_type: str = 'cosine',
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Create learning rate scheduler with warmup and decay.
    
    Supports different decay schedules:
    - 'cosine': Cosine annealing (default)
    - 'linear': Linear decay
    - 'constant': Constant LR after warmup
    
    Args:
        optimizer: Optimizer instance
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps (overrides warmup_ratio if provided)
        warmup_ratio: Fraction of training for warmup (used if num_warmup_steps is None)
        scheduler_type: Type of decay schedule ('cosine', 'linear', 'constant')
        min_lr_ratio: Minimum learning rate as fraction of initial LR (for cosine/linear)
        
    Returns:
        LambdaLR scheduler
        
    Example:
        >>> scheduler = create_scheduler(
        ...     optimizer,
        ...     num_training_steps=10000,
        ...     warmup_ratio=0.1,
        ...     scheduler_type='cosine'
        ... )
    """
    # Validate scheduler_type early
    valid_types = ['cosine', 'linear', 'constant']
    if scheduler_type not in valid_types:
        raise ValueError(
            f"Unknown scheduler_type: {scheduler_type}. "
            f"Valid options are: {', '.join(valid_types)}"
        )
    
    if num_warmup_steps is None:
        num_warmup_steps = int(num_training_steps * warmup_ratio)
    
    logger.info(
        f"Creating {scheduler_type} scheduler: "
        f"warmup_steps={num_warmup_steps}, total_steps={num_training_steps}, "
        f"min_lr_ratio={min_lr_ratio}"
    )
    
    def lr_lambda(current_step: int) -> float:
        """Compute learning rate multiplier."""
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Decay phase
        if scheduler_type == 'constant':
            return 1.0
        
        # Progress after warmup
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        
        if scheduler_type == 'linear':
            # Linear decay from 1.0 to min_lr_ratio
            return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * progress)
        
        else:  # scheduler_type == 'cosine'
            # Cosine decay from 1.0 to min_lr_ratio
            return max(
                min_lr_ratio,
                min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
            )
    
    return LambdaLR(optimizer, lr_lambda)


def get_optimizer_state(optimizer: Optimizer) -> Dict[str, Any]:
    """
    Get optimizer state information.
    
    Returns current learning rates for all parameter groups and other
    optimizer statistics.
    
    Args:
        optimizer: Optimizer instance
        
    Returns:
        Dictionary with optimizer state information
    """
    state = {
        'learning_rates': [group['lr'] for group in optimizer.param_groups],
        'num_param_groups': len(optimizer.param_groups),
    }
    
    # Add parameter counts
    param_counts = [
        sum(p.numel() for p in group['params'])
        for group in optimizer.param_groups
    ]
    state['param_counts'] = param_counts
    
    return state


def clip_gradients(
    model: torch.nn.Module,
    max_grad_norm: float,
) -> float:
    """
    Clip gradients by global norm.
    
    Args:
        model: PyTorch model
        max_grad_norm: Maximum gradient norm
        
    Returns:
        Total gradient norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_grad_norm
    ).item()
