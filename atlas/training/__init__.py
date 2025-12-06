"""
Training and optimization components for Atlas.

This module contains the training loop, loss functions, optimizers,
learning rate schedulers, and evaluation logic.
"""

from atlas.training.optimizer import (
    create_optimizer,
    create_scheduler,
    get_optimizer_state,
    clip_gradients,
)
from atlas.training.loss import (
    compute_lm_loss,
    compute_lm_loss_with_logits_shift,
    compute_perplexity,
)

__all__ = [
    "create_optimizer",
    "create_scheduler",
    "get_optimizer_state",
    "clip_gradients",
    "compute_lm_loss",
    "compute_lm_loss_with_logits_shift",
    "compute_perplexity",
]
