"""
Text dataset for language model training.

This module provides a PyTorch Dataset for loading and tokenizing text data
for autoregressive language modeling.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Union, Optional
import logging
import numpy as np
import tempfile
import os

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    Dataset for autoregressive language modeling.
    
    Loads text from files, tokenizes it, and splits into fixed-length sequences
    for training. Supports both single file and multiple file loading.
    
    Args:
        file_paths: Path or list of paths to text files
        tokenizer: Tokenizer instance with encode() method
        max_seq_len: Maximum sequence length (context window)
        stride: Stride for sliding window (if None, uses max_seq_len for non-overlapping)
        
    Example:
        >>> from atlas.tokenizer import Tokenizer
        >>> tokenizer = Tokenizer()
        >>> dataset = TextDataset(
        ...     file_paths="data/train.txt",
        ...     tokenizer=tokenizer,
        ...     max_seq_len=1024
        ... )
        >>> tokens = dataset[0]
        >>> tokens.shape
        torch.Size([1024])
    """
    
    def __init__(
        self,
        file_paths: Union[str, Path, List[Union[str, Path]]],
        tokenizer,
        max_seq_len: int = 1024,
        stride: Optional[int] = None,
        use_mmap: bool = False,
    ):
        super().__init__()
        
        # Convert to list of paths
        if isinstance(file_paths, (str, Path)):
            file_paths = [Path(file_paths)]
        else:
            file_paths = [Path(p) for p in file_paths]
        
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride if stride is not None else max_seq_len
        self.use_mmap = use_mmap
        self.mmap_file = None
        self.mmap_array = None
        
        # Validate parameters
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if self.stride <= 0 or self.stride > self.max_seq_len:
            raise ValueError(
                f"stride must be positive and <= max_seq_len, "
                f"got stride={stride}, max_seq_len={max_seq_len}"
            )
        
        # Load and tokenize all text
        logger.info(f"Loading {len(file_paths)} file(s)...")
        if use_mmap:
            self._setup_mmap()
        else:
            self.tokens = self._load_and_tokenize()
            logger.info(f"Loaded {len(self.tokens)} tokens")
        
        # Create sequence windows
        self.sequences = self._create_sequences()
        logger.info(f"Created {len(self.sequences)} sequences of length {max_seq_len}")
    
    def _setup_mmap(self):
        """Setup memory-mapped file for large datasets."""
        # Create temporary file for memory-mapped array
        self.mmap_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mmap')
        mmap_path = self.mmap_file.name
        self.mmap_file.close()
        
        # First pass: tokenize and save to mmap file
        all_tokens = []
        for file_path in self.file_paths:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            logger.info(f"Loading {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if not text.strip():
                logger.warning(f"File {file_path} is empty, skipping")
                continue
            
            # Tokenize in chunks to avoid memory spike
            tokens = self.tokenizer.encode(text)
            all_tokens.extend(tokens)
            logger.info(f"  Tokenized {len(tokens)} tokens from {file_path}")
            
            # Free memory
            del text
        
        if not all_tokens:
            raise ValueError("No tokens were loaded from any file")
        
        # Save to memory-mapped file
        logger.info(f"Creating memory-mapped file with {len(all_tokens)} tokens...")
        self.mmap_array = np.memmap(mmap_path, dtype='int32', mode='w+', shape=(len(all_tokens),))
        self.mmap_array[:] = all_tokens
        self.mmap_array.flush()
        logger.info(f"Loaded {len(all_tokens)} tokens (memory-mapped)")
        
        # Free the Python list
        del all_tokens
        
        # Reopen in read-only mode
        self.mmap_array = np.memmap(mmap_path, dtype='int32', mode='r', shape=self.mmap_array.shape)
    
    def _load_and_tokenize(self) -> List[int]:
        """Load all text files and tokenize them into a single token list."""
        all_tokens = []
        
        for file_path in self.file_paths:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            logger.info(f"Loading {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if not text.strip():
                logger.warning(f"File {file_path} is empty, skipping")
                continue
            
            # Tokenize
            tokens = self.tokenizer.encode(text)
            all_tokens.extend(tokens)
            logger.info(f"  Tokenized {len(tokens)} tokens from {file_path}")
        
        if not all_tokens:
            raise ValueError("No tokens were loaded from any file")
        
        return all_tokens
    
    def _create_sequences(self) -> List[int]:
        """
        Split tokens into sequences using a sliding window.
        
        Each sequence is max_seq_len tokens long. We use a stride to create
        overlapping or non-overlapping windows depending on the stride value.
        
        Returns:
            List of starting indices for each sequence
        """
        sequences = []
        if self.use_mmap:
            total_tokens = len(self.mmap_array)
        else:
            total_tokens = len(self.tokens)
        
        # Create sliding windows
        start_idx = 0
        while start_idx + self.max_seq_len <= total_tokens:
            sequences.append(start_idx)
            start_idx += self.stride
        
        # If we have leftover tokens and want to use them, pad the last sequence
        # (only if we haven't already captured them with overlapping windows)
        if start_idx < total_tokens and self.stride == self.max_seq_len:
            # We have leftover tokens with non-overlapping windows
            # We'll just skip them to avoid padding (cleaner for training)
            leftover = total_tokens - start_idx
            logger.info(f"Skipping {leftover} leftover tokens (less than max_seq_len)")
        
        return sequences
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a sequence at the given index.
        
        Args:
            idx: Index of the sequence (supports negative indexing)
            
        Returns:
            Tensor of token IDs of shape (max_seq_len,)
        """
        # Handle negative indexing
        if idx < 0:
            idx = len(self.sequences) + idx
        
        if idx < 0 or idx >= len(self.sequences):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        start_idx = self.sequences[idx]
        end_idx = start_idx + self.max_seq_len
        
        # Get the sequence
        if self.use_mmap:
            sequence = self.mmap_array[start_idx:end_idx]
            return torch.from_numpy(sequence.copy()).long()
        else:
            sequence = self.tokens[start_idx:end_idx]
            return torch.tensor(sequence, dtype=torch.long)
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size from the tokenizer."""
        return self.tokenizer.vocab_size
    
    def get_stats(self) -> dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.use_mmap:
            total_tokens = len(self.mmap_array)
        else:
            total_tokens = len(self.tokens)
        
        return {
            "num_files": len(self.file_paths),
            "total_tokens": total_tokens,
            "num_sequences": len(self.sequences),
            "max_seq_len": self.max_seq_len,
            "stride": self.stride,
            "vocab_size": self.get_vocab_size(),
            "use_mmap": self.use_mmap,
        }
    
    def __del__(self):
        """Cleanup memory-mapped file on deletion."""
        if self.use_mmap and self.mmap_file:
            try:
                mmap_path = self.mmap_file.name
                if os.path.exists(mmap_path):
                    os.unlink(mmap_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup mmap file: {e}")
