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
from atlas.training.trainer import Trainer, TrainingMetrics
from atlas.training.evaluator import Evaluator, EvaluationMetrics, evaluate_model
from atlas.training.checkpoint import CheckpointManager, CheckpointMetadata

__all__ = [
    "create_optimizer",
    "create_scheduler",
    "get_optimizer_state",
    "clip_gradients",
    "compute_lm_loss",
    "compute_lm_loss_with_logits_shift",
    "compute_perplexity",
    "Trainer",
    "TrainingMetrics",
    "Evaluator",
    "EvaluationMetrics",
    "evaluate_model",
    "CheckpointManager",
    "CheckpointMetadata",
]
