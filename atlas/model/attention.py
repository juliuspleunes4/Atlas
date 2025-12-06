"""
Multi-head self-attention mechanism for Atlas.

This module implements the core attention mechanism used in transformers,
with causal masking for autoregressive language modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism with causal masking.
    
    This is the core component of the transformer that allows the model
    to attend to different positions in the sequence.
    
    Args:
        hidden_size: Dimension of the model
        num_heads: Number of attention heads
        dropout: Dropout probability for attention weights
        use_bias: Whether to use bias in linear projections
        
    Example:
        >>> attention = MultiHeadAttention(hidden_size=768, num_heads=12)
        >>> x = torch.randn(2, 10, 768)  # (batch, seq_len, hidden_size)
        >>> output = attention(x)
        >>> output.shape
        torch.Size([2, 10, 768])
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        use_bias: bool = True,
    ):
        super().__init__()
        
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        
        self.embed_dim = hidden_size  # Alias for testing
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Combined Q, K, V projection (more efficient)
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=use_bias)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        
        # Cache for causal mask
        self.register_buffer("causal_mask", None, persistent=False)
    
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Get or create causal mask for autoregressive attention.
        
        The mask prevents attending to future positions.
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
            
        Returns:
            Causal mask of shape (seq_len, seq_len)
        """
        if self.causal_mask is None or self.causal_mask.shape[0] < seq_len:
            # Create lower triangular matrix (1 = can attend, 0 = cannot attend)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
            self.causal_mask = mask
        
        return self.causal_mask[:seq_len, :seq_len]
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Apply multi-head self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            mask: Optional attention mask (for padding, etc.)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * hidden_size)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        # (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, seq_len)
        # -> (batch, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        causal_mask = self._get_causal_mask(seq_len, x.device)
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply additional mask if provided (e.g., for padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim)
        # -> (batch, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch, seq_len, num_heads, head_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, hidden_size)
        
        # Output projection and dropout
        output = self.out_proj(attn_output)
        output = self.out_dropout(output)
        
        return output
