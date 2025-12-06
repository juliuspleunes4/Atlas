"""
Text generation and inference for Atlas LLM.

Provides autoregressive generation with various sampling strategies:
- Greedy decoding (argmax)
- Temperature sampling
- Top-k sampling
- Top-p (nucleus) sampling
"""

from typing import Optional, List, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 50
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    do_sample: bool = True
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.top_p is not None and (self.top_p <= 0 or self.top_p > 1.0):
            raise ValueError("top_p must be in (0, 1]")


def sample_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Apply top-k filtering to logits.
    
    Args:
        logits: Logits tensor (batch_size, vocab_size)
        top_k: Number of top tokens to keep
    
    Returns:
        Filtered logits with non-top-k values set to -inf
    """
    # Get top-k values and indices
    top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
    
    # Set all non-top-k values to -inf
    logits_filtered = torch.full_like(logits, float('-inf'))
    logits_filtered.scatter_(-1, top_k_indices, top_k_values)
    
    return logits_filtered


def sample_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Apply top-p (nucleus) filtering to logits.
    
    Args:
        logits: Logits tensor (batch_size, vocab_size)
        top_p: Cumulative probability threshold
    
    Returns:
        Filtered logits with nucleus sampling applied
    """
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    
    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    
    # Shift right to keep at least one token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # Set removed indices to -inf
    logits_filtered = sorted_logits.clone()
    logits_filtered[sorted_indices_to_remove] = float('-inf')
    
    # Scatter back to original order
    logits_output = torch.full_like(logits, float('-inf'))
    logits_output.scatter_(-1, sorted_indices, logits_filtered)
    
    return logits_output


class TextGenerator:
    """
    Autoregressive text generator for Atlas LLM.
    
    Supports multiple sampling strategies:
    - Greedy decoding (temperature=1.0, do_sample=False)
    - Temperature sampling
    - Top-k sampling
    - Top-p (nucleus) sampling
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Initialize text generator.
        
        Args:
            model: Trained language model
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        config: Optional[GenerationConfig] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            config: Generation configuration
        
        Returns:
            Generated token IDs (batch_size, seq_len + max_new_tokens)
        """
        if config is None:
            config = GenerationConfig()
        
        # Move input to device
        input_ids = input_ids.to(self.device)
        batch_size = input_ids.size(0)
        
        # Initialize generated sequence with input
        generated = input_ids.clone()
        
        # Generate tokens one at a time
        for _ in range(config.max_new_tokens):
            # Get logits for next token
            logits = self.model(generated)
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            
            # Apply temperature
            if config.temperature != 1.0:
                next_token_logits = next_token_logits / config.temperature
            
            # Apply top-k filtering
            if config.top_k is not None:
                next_token_logits = sample_top_k(next_token_logits, config.top_k)
            
            # Apply top-p filtering
            if config.top_p is not None:
                next_token_logits = sample_top_p(next_token_logits, config.top_p)
            
            # Sample or take argmax
            if config.do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Check for EOS token
            if config.eos_token_id is not None:
                if (next_token == config.eos_token_id).all():
                    break
        
        return generated
    
    @torch.no_grad()
    def generate_from_prompt(
        self,
        prompt: Union[str, List[int]],
        tokenizer,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Generate text from a string prompt.
        
        Args:
            prompt: Text prompt or list of token IDs
            tokenizer: Tokenizer for encoding/decoding
            config: Generation configuration
        
        Returns:
            Generated text as string
        """
        # Encode prompt
        if isinstance(prompt, str):
            input_ids = tokenizer.encode(prompt)
        else:
            input_ids = prompt
        
        # Convert to tensor
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)
        
        # Generate
        output_ids = self.generate(input_ids_tensor, config)
        
        # Decode
        output_text = tokenizer.decode(output_ids[0].tolist())
        
        return output_text


def generate_text(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    do_sample: bool = True,
    eos_token_id: Optional[int] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> torch.Tensor:
    """
    Convenience function for text generation.
    
    Args:
        model: Trained language model
        input_ids: Input token IDs
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        do_sample: Whether to sample (vs. greedy)
        eos_token_id: End-of-sequence token ID
        device: Device to run on
    
    Returns:
        Generated token IDs
    
    Example:
        >>> output = generate_text(model, input_ids, max_new_tokens=100, temperature=0.8)
    """
    config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        eos_token_id=eos_token_id,
    )
    
    generator = TextGenerator(model, device=device)
    return generator.generate(input_ids, config)
