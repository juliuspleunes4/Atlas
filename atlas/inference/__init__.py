"""
Inference and text generation for Atlas.

This module provides autoregressive generation, sampling strategies,
and inference utilities.
"""

from atlas.inference.generation import (
    GenerationConfig,
    TextGenerator,
    generate_text,
    sample_top_k,
    sample_top_p,
)

__all__ = [
    "GenerationConfig",
    "TextGenerator",
    "generate_text",
    "sample_top_k",
    "sample_top_p",
]
