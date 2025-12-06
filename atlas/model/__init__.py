"""
Model architecture components for Atlas.

This module contains the core transformer architecture including:
- Embeddings (token and positional)
- Multi-head self-attention
- Feed-forward networks (MLP)
- Transformer blocks
- Language model head
"""

from atlas.model.embeddings import TokenEmbedding, PositionalEmbedding, CombinedEmbedding
from atlas.model.attention import MultiHeadAttention
from atlas.model.mlp import MLP
from atlas.model.transformer import TransformerBlock
from atlas.model.model import AtlasLM

__all__ = [
    "TokenEmbedding",
    "PositionalEmbedding",
    "CombinedEmbedding",
    "MultiHeadAttention",
    "MLP",
    "TransformerBlock",
    "AtlasLM",
]
