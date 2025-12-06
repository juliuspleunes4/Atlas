"""
Embedding layers for Atlas.

This module implements token embeddings and positional embeddings
for the transformer model.
"""

import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """
    Token embedding layer that converts token IDs to dense vectors.
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of the embedding vectors
        
    Example:
        >>> embedding = TokenEmbedding(vocab_size=50257, embedding_dim=768)
        >>> tokens = torch.tensor([[1, 2, 3, 4]])
        >>> embedded = embedding(tokens)
        >>> embedded.shape
        torch.Size([1, 4, 768])
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings.
        
        Args:
            tokens: Token IDs of shape (batch_size, seq_len)
            
        Returns:
            Embeddings of shape (batch_size, seq_len, embedding_dim)
        """
        return self.embedding(tokens)


class PositionalEmbedding(nn.Module):
    """
    Learned positional embedding layer.
    
    Adds position information to token embeddings. Uses learned embeddings
    rather than fixed sinusoidal encodings.
    
    Args:
        max_seq_len: Maximum sequence length
        embedding_dim: Dimension of the embedding vectors
        
    Example:
        >>> pos_emb = PositionalEmbedding(max_seq_len=1024, embedding_dim=768)
        >>> batch_size, seq_len = 2, 10
        >>> positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        >>> pos_embedded = pos_emb(positions)
        >>> pos_embedded.shape
        torch.Size([2, 10, 768])
    """
    
    def __init__(self, max_seq_len: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.max_seq_len = max_seq_len
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Get positional embeddings.
        
        Args:
            positions: Position indices of shape (batch_size, seq_len)
            
        Returns:
            Positional embeddings of shape (batch_size, seq_len, embedding_dim)
        """
        return self.embedding(positions)


class CombinedEmbedding(nn.Module):
    """
    Combined token and positional embeddings with dropout.
    
    This is the input layer for the transformer, combining token embeddings
    with positional information.
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of the embedding vectors
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        
    Example:
        >>> embedding = CombinedEmbedding(
        ...     vocab_size=50257,
        ...     embedding_dim=768,
        ...     max_seq_len=1024,
        ...     dropout=0.1
        ... )
        >>> tokens = torch.tensor([[1, 2, 3, 4]])
        >>> embedded = embedding(tokens)
        >>> embedded.shape
        torch.Size([1, 4, 768])
    """
    
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embedding_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.positional_embedding = PositionalEmbedding(max_seq_len, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Get combined token and positional embeddings.
        
        Args:
            tokens: Token IDs of shape (batch_size, seq_len)
            
        Returns:
            Combined embeddings of shape (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len = tokens.shape
        
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
            )
        
        # Get token embeddings
        token_emb = self.token_embedding(tokens)
        
        # Generate position indices
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        
        # Get positional embeddings
        pos_emb = self.positional_embedding(positions)
        
        # Combine and apply dropout
        embeddings = token_emb + pos_emb
        embeddings = self.dropout(embeddings)
        
        return embeddings
