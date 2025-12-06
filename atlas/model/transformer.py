"""
Transformer block for Atlas.

This module implements a complete transformer block with attention,
feed-forward network, layer normalization, and residual connections.
"""

import torch
import torch.nn as nn

from atlas.model.attention import MultiHeadAttention
from atlas.model.mlp import MLP


class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-norm architecture.
    
    Architecture:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    
    This is the pre-norm design used in modern transformers (GPT-2 style).
    
    Args:
        hidden_size: Dimension of the model
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden size to model hidden size
        dropout: Dropout probability
        attention_dropout: Dropout probability for attention weights
        activation: Activation function for MLP
        use_bias: Whether to use bias in linear layers
        norm_eps: Epsilon for layer normalization
        
    Example:
        >>> block = TransformerBlock(hidden_size=768, num_heads=12)
        >>> x = torch.randn(2, 10, 768)
        >>> output = block(x)
        >>> output.shape
        torch.Size([2, 10, 768])
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "gelu",
        use_bias: bool = True,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.ln2 = nn.LayerNorm(hidden_size, eps=norm_eps)
        
        # Attention
        self.attn = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=attention_dropout,
            use_bias=use_bias,
        )
        
        # Feed-forward network
        self.mlp = MLP(
            hidden_size=hidden_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            activation=activation,
            use_bias=use_bias,
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Apply transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Pre-norm + attention + residual
        x = x + self.attn(self.ln1(x), mask=mask)
        
        # Pre-norm + MLP + residual
        x = x + self.mlp(self.ln2(x))
        
        return x
