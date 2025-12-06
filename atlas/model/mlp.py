"""
Feed-forward network (MLP) for Atlas.

This module implements the position-wise feed-forward network
used in transformer blocks.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Position-wise feed-forward network.
    
    Two-layer MLP with activation and dropout, applied independently
    to each position in the sequence.
    
    Args:
        hidden_size: Dimension of the model
        mlp_ratio: Ratio of MLP hidden size to model hidden size
        dropout: Dropout probability
        activation: Activation function name ('gelu', 'silu', or 'relu')
        use_bias: Whether to use bias in linear layers
        
    Example:
        >>> mlp = MLP(hidden_size=768, mlp_ratio=4.0)
        >>> x = torch.randn(2, 10, 768)
        >>> output = mlp(x)
        >>> output.shape
        torch.Size([2, 10, 768])
    """
    
    def __init__(
        self,
        hidden_size: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_bias: bool = True,
    ):
        super().__init__()
        
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        
        self.fc1 = nn.Linear(hidden_size, mlp_hidden_size, bias=use_bias)
        self.fc2 = nn.Linear(mlp_hidden_size, hidden_size, bias=use_bias)
        self.dropout = nn.Dropout(dropout)
        
        # Select activation function
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
