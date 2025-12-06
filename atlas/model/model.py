"""
Complete Atlas language model.

This module implements the full decoder-only transformer model
for autoregressive language modeling.
"""

import torch
import torch.nn as nn
from typing import Optional

from atlas.config import ModelConfig
from atlas.model.embeddings import CombinedEmbedding
from atlas.model.transformer import TransformerBlock


class AtlasLM(nn.Module):
    """
    Atlas Language Model - Decoder-only transformer for autoregressive generation.
    
    Architecture:
        1. Token + Positional Embeddings
        2. N x Transformer Blocks
        3. Final Layer Norm
        4. Language Modeling Head (tied with input embeddings)
    
    Args:
        config: Model configuration
        
    Example:
        >>> from atlas.config import ModelConfig
        >>> config = ModelConfig(num_layers=6, hidden_size=768, num_heads=12)
        >>> model = AtlasLM(config)
        >>> tokens = torch.randint(0, config.vocab_size, (2, 10))
        >>> logits = model(tokens)
        >>> logits.shape
        torch.Size([2, 10, 50261])
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embeddings = CombinedEmbedding(
            vocab_size=config.vocab_size,
            embedding_dim=config.hidden_size,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                activation=config.activation,
                use_bias=config.use_bias,
                norm_eps=config.norm_eps,
            )
            for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights between input embeddings and output projection
        if config.tie_weights:
            self.lm_head.weight = self.embeddings.token_embedding.embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            tokens: Input token IDs of shape (batch_size, seq_len)
            mask: Optional attention mask
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        # Get embeddings
        x = self.embeddings(tokens)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        return logits
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Count the number of parameters in the model.
        
        Args:
            trainable_only: Whether to count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_num_params(self, non_embedding: bool = False) -> str:
        """
        Get formatted number of parameters, optionally excluding embeddings.
        
        Args:
            non_embedding: If True, exclude embedding parameters
            
        Returns:
            Formatted string with number of parameters (e.g., "1.5M")
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embeddings.token_embedding.embedding.weight.numel()
            n_params -= self.embeddings.positional_embedding.embedding.weight.numel()
        return f"{n_params / 1e6:.2f}M"
    
    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            tokens: Initial token IDs of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k most likely tokens
            eos_token_id: If set, stop generation when this token is generated
            
        Returns:
            Generated token IDs of shape (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Stop if we've reached max sequence length
            if tokens.size(1) >= self.config.max_seq_len:
                break
            
            # Crop sequence if it exceeds max_seq_len
            idx_cond = tokens if tokens.size(1) <= self.config.max_seq_len else tokens[:, -self.config.max_seq_len:]
            
            # Get logits
            logits = self(idx_cond)
            
            # Focus on last position
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample from distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Check for EOS token
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return tokens
    
    def __repr__(self) -> str:
        """String representation of the model."""
        n_params = self.count_parameters(trainable_only=True)
        return (
            f"AtlasLM(\n"
            f"  num_layers={self.config.num_layers},\n"
            f"  hidden_size={self.config.hidden_size},\n"
            f"  num_heads={self.config.num_heads},\n"
            f"  vocab_size={self.config.vocab_size},\n"
            f"  parameters={n_params:,}\n"
            f")"
        )
