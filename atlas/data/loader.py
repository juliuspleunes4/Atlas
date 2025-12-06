"""
Data loading utilities for training.

This module provides utilities for creating DataLoaders with proper
batching, collation, and efficient memory usage.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for training or evaluation.
    
    Args:
        dataset: PyTorch Dataset instance
        batch_size: Number of sequences per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory (faster GPU transfer)
        drop_last: Whether to drop the last incomplete batch
        
    Returns:
        DataLoader instance
        
    Example:
        >>> from atlas.data import TextDataset, create_dataloader
        >>> from atlas.tokenizer import Tokenizer
        >>> 
        >>> tokenizer = Tokenizer()
        >>> dataset = TextDataset("data/train.txt", tokenizer, max_seq_len=1024)
        >>> loader = create_dataloader(dataset, batch_size=8, shuffle=True)
        >>> 
        >>> for batch in loader:
        ...     # batch shape: (batch_size, seq_len)
        ...     print(batch.shape)
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    
    logger.info(
        f"Creating DataLoader: batch_size={batch_size}, shuffle={shuffle}, "
        f"num_workers={num_workers}, pin_memory={pin_memory}, drop_last={drop_last}"
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_batch,
    )


def collate_batch(batch):
    """
    Collate a batch of sequences.
    
    This is a simple collation function that stacks sequences into a batch.
    All sequences in TextDataset are already the same length, so we just
    need to stack them.
    
    Args:
        batch: List of tensors from the dataset
        
    Returns:
        Stacked tensor of shape (batch_size, seq_len)
    """
    # batch is a list of tensors, each of shape (seq_len,)
    # Stack them into (batch_size, seq_len)
    return torch.stack(batch, dim=0)


def get_dataloader_stats(loader: DataLoader) -> Dict[str, Any]:
    """
    Get statistics about a DataLoader.
    
    Args:
        loader: DataLoader instance
        
    Returns:
        Dictionary with DataLoader statistics
    """
    dataset = loader.dataset
    
    stats = {
        "dataset_size": len(dataset),
        "batch_size": loader.batch_size,
        "num_batches": len(loader),
        "shuffle": loader.sampler is not None,
        "num_workers": loader.num_workers,
        "pin_memory": loader.pin_memory,
        "drop_last": loader.drop_last,
    }
    
    # Add dataset-specific stats if available
    if hasattr(dataset, "get_stats"):
        stats["dataset_stats"] = dataset.get_stats()
    
    return stats


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: Optional[int] = 42,
) -> tuple:
    """
    Split a dataset into train/val/test subsets.
    
    Args:
        dataset: Dataset to split
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
        
    Example:
        >>> dataset = TextDataset(...)
        >>> train, val, test = split_dataset(dataset, 0.8, 0.1, 0.1)
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total_ratio <= 1.01):  # Allow small floating point errors
        raise ValueError(
            f"Ratios must sum to 1.0, got {total_ratio} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )
    
    if train_ratio <= 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("All ratios must be non-negative, and train_ratio must be positive")
    
    # Calculate sizes
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # Split using torch random_split
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )
    
    logger.info(
        f"Split dataset: train={train_size} ({train_ratio:.1%}), "
        f"val={val_size} ({val_ratio:.1%}), test={test_size} ({test_ratio:.1%})"
    )
    
    return train_dataset, val_dataset, test_dataset
