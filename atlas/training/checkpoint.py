"""
Checkpointing utilities for Atlas LLM.

Provides functionality for:
- Saving model, optimizer, and scheduler state
- Loading checkpoints to resume training
- Tracking best model based on metrics
- Managing checkpoint directories
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata stored with each checkpoint."""
    step: int
    epoch: int
    loss: float
    perplexity: Optional[float] = None
    learning_rate: Optional[float] = None
    best_metric: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """Create from dictionary."""
        return cls(**data)


class CheckpointManager:
    """
    Manages model checkpointing during training.
    
    Features:
    - Save/load model, optimizer, scheduler state
    - Track best model based on validation metric
    - Automatic checkpoint directory management
    - Resume training from checkpoint
    - Separate tracking for step-based and epoch-based checkpoints
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        model_name: str = 'atlas',
        keep_best: bool = True,
        keep_last_n: Optional[int] = 3,
        keep_last_epochs: Optional[int] = 5,
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            model_name: Name prefix for checkpoint files
            keep_best: Whether to track and save best model
            keep_last_n: Number of recent step-based checkpoints to keep (None for all)
            keep_last_epochs: Number of recent epoch checkpoints to keep (None for all)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.keep_best = keep_best
        self.keep_last_n = keep_last_n
        self.keep_last_epochs = keep_last_epochs
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track best metric - load from existing best checkpoint if available
        self.best_metric = float('inf')  # Lower is better (loss)
        self.best_checkpoint_path = None
        
        # Check for existing best checkpoint and restore best metric
        if self.keep_best:
            best_checkpoint = self.checkpoint_dir / f'{self.model_name}_best.pt'
            if best_checkpoint.exists():
                try:
                    existing_best = torch.load(best_checkpoint, map_location='cpu', weights_only=False)
                    if 'metadata' in existing_best and 'loss' in existing_best['metadata']:
                        self.best_metric = existing_best['metadata']['loss']
                        self.best_checkpoint_path = str(best_checkpoint)
                        logger.info(f"Found existing best checkpoint with loss: {self.best_metric:.4f}")
                except Exception as e:
                    logger.warning(f"Could not load existing best checkpoint: {e}")
                    # Keep default float('inf') if loading fails
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        metadata: CheckpointMetadata,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        is_best: bool = False,
        is_epoch_end: bool = False,
    ) -> str:
        """
        Save checkpoint with model, optimizer, and scheduler state.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            metadata: Checkpoint metadata
            scheduler: Optional scheduler to save
            is_best: Whether this is the best model so far
            is_epoch_end: Whether this is an end-of-epoch checkpoint
        
        Returns:
            Path to saved checkpoint
        """
        # Prepare checkpoint data
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metadata': metadata.to_dict(),
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Checkpoint filename - distinguish between step and epoch checkpoints
        if is_epoch_end:
            checkpoint_filename = f'{self.model_name}_epoch_{metadata.epoch}_step_{metadata.step}.pt'
        else:
            checkpoint_filename = f'{self.model_name}_step_{metadata.step}.pt'
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        
        # Save checkpoint (overwrites if exists)
        torch.save(checkpoint, checkpoint_path)
        
        # Save metadata as JSON for easy inspection
        metadata_path = checkpoint_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Save as best if applicable (always keep separate from others)
        # Only save if this is truly better than any existing best checkpoint
        if is_best and self.keep_best and metadata.loss < self.best_metric:
            best_path = self.checkpoint_dir / f'{self.model_name}_best.pt'
            torch.save(checkpoint, best_path)
            best_metadata_path = best_path.with_suffix('.json')
            with open(best_metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            self.best_checkpoint_path = str(best_path)
            self.best_metric = metadata.loss
            logger.info(f"  [BEST] Saved new best checkpoint (loss: {metadata.loss:.4f})")
        
        # Clean up old checkpoints
        if is_epoch_end and self.keep_last_epochs is not None:
            self._cleanup_old_epoch_checkpoints()
        elif not is_epoch_end and self.keep_last_n is not None:
            self._cleanup_old_step_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        device: str = 'cpu',
    ) -> CheckpointMetadata:
        """
        Load checkpoint and restore state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            device: Device to load tensors to
        
        Returns:
            CheckpointMetadata from the checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load metadata
        metadata = CheckpointMetadata.from_dict(checkpoint['metadata'])
        
        return metadata
    
    def load_best_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        device: str = 'cpu',
    ) -> Optional[CheckpointMetadata]:
        """
        Load the best checkpoint if available.
        
        Args:
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            device: Device to load tensors to
        
        Returns:
            CheckpointMetadata if best checkpoint exists, None otherwise
        """
        best_path = self.checkpoint_dir / f'{self.model_name}_best.pt'
        
        if not best_path.exists():
            return None
        
        return self.load_checkpoint(str(best_path), model, optimizer, scheduler, device)
    
    def load_latest_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        device: str = 'cpu',
    ) -> Optional[CheckpointMetadata]:
        """
        Load the most recent checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            device: Device to load tensors to
        
        Returns:
            CheckpointMetadata if checkpoint exists, None otherwise
        """
        # Find all checkpoint files
        checkpoints = list(self.checkpoint_dir.glob(f'{self.model_name}_step_*.pt'))
        
        if not checkpoints:
            return None
        
        # Sort by modification time (most recent first)
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        return self.load_checkpoint(str(latest_checkpoint), model, optimizer, scheduler, device)
    
    def _cleanup_old_step_checkpoints(self):
        """Remove old step-based checkpoints, keeping only the most recent N."""
        # Only match step-based checkpoints (not epoch checkpoints)
        checkpoints = [
            p for p in self.checkpoint_dir.glob(f'{self.model_name}_step_*.pt')
            if '_epoch_' not in p.name  # Exclude epoch checkpoints
        ]
        
        if len(checkpoints) <= self.keep_last_n:
            return
        
        # Sort by modification time
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Remove old checkpoints (but never remove best checkpoint)
        for checkpoint in checkpoints[self.keep_last_n:]:
            # Skip if this is the best checkpoint
            if checkpoint.name == f'{self.model_name}_best.pt':
                continue
            logger.info(f"  [CLEANUP] Removing old checkpoint: {checkpoint.name}")
            checkpoint.unlink()
            # Also remove associated JSON metadata
            json_path = checkpoint.with_suffix('.json')
            if json_path.exists():
                json_path.unlink()
    
    def _cleanup_old_epoch_checkpoints(self):
        """Remove old epoch checkpoints, keeping only the most recent N."""
        # Only match epoch-based checkpoints
        checkpoints = list(self.checkpoint_dir.glob(f'{self.model_name}_epoch_*.pt'))
        
        if len(checkpoints) <= self.keep_last_epochs:
            return
        
        # Sort by modification time
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Remove old checkpoints
        for checkpoint in checkpoints[self.keep_last_epochs:]:
            logger.info(f"  [CLEANUP] Removing old epoch checkpoint: {checkpoint.name}")
            checkpoint.unlink()
            # Also remove associated JSON metadata
            json_path = checkpoint.with_suffix('.json')
            if json_path.exists():
                json_path.unlink()
    
    def find_latest_checkpoint(self) -> Optional[Path]:
        """
        Find the most recent checkpoint file.
        
        Returns:
            Path to latest checkpoint, or None if no checkpoints exist
        """
        # Find all checkpoint files (both step and epoch)
        checkpoints = list(self.checkpoint_dir.glob(f'{self.model_name}_*.pt'))
        
        # Exclude best checkpoint (handled separately)
        checkpoints = [p for p in checkpoints if 'best' not in p.name]
        
        if not checkpoints:
            return None
        
        # Sort by modification time (most recent first)
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        return latest_checkpoint
    
    def get_checkpoint_info(self, checkpoint_path: Path) -> Optional[Dict[str, Any]]:
        """
        Get metadata about a checkpoint without loading it.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary with checkpoint info, or None if metadata not found
        """
        json_path = checkpoint_path.with_suffix('.json')
        
        if json_path.exists():
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        
        return None
    
    def list_checkpoints(self) -> list[Dict[str, Any]]:
        """
        List all available checkpoints with their metadata.
        
        Returns:
            List of dictionaries with checkpoint information
        """
        checkpoints = []
        
        for checkpoint_path in self.checkpoint_dir.glob(f'{self.model_name}_step_*.pt'):
            json_path = checkpoint_path.with_suffix('.json')
            
            if json_path.exists():
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            checkpoints.append({
                'path': str(checkpoint_path),
                'size_mb': checkpoint_path.stat().st_size / (1024 * 1024),
                'metadata': metadata,
            })
        
        # Sort by step number
        checkpoints.sort(key=lambda x: x['metadata'].get('step', 0))
        
        return checkpoints
