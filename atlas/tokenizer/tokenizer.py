"""
Tokenizer implementation for Atlas.

This module provides a wrapper around tiktoken (GPT-style BPE tokenizer)
with support for special tokens and batch processing.
"""

import tiktoken
from typing import List, Union, Optional
from pathlib import Path


class Tokenizer:
    """
    Tokenizer for encoding and decoding text.
    
    Uses tiktoken (GPT-style BPE) as the backend with support for special tokens.
    
    Special tokens:
        - BOS (Beginning of Sequence): <|bos|>
        - EOS (End of Sequence): <|eos|>
        - PAD (Padding): <|pad|>
        - UNK (Unknown): <|unk|>
    
    Example:
        >>> tokenizer = Tokenizer()
        >>> tokens = tokenizer.encode("Hello, world!")
        >>> text = tokenizer.decode(tokens)
        >>> print(text)
        Hello, world!
    """
    
    def __init__(
        self,
        encoding_name: str = "gpt2",
        add_special_tokens: bool = True,
    ):
        """
        Initialize the tokenizer.
        
        Args:
            encoding_name: Name of the tiktoken encoding to use (default: "gpt2")
            add_special_tokens: Whether to add special tokens to the vocabulary
        """
        self._base_tokenizer = tiktoken.get_encoding(encoding_name)
        self._add_special_tokens = add_special_tokens
        
        # Define special tokens
        self.bos_token = "<|bos|>"
        self.eos_token = "<|eos|>"
        self.pad_token = "<|pad|>"
        self.unk_token = "<|unk|>"
        
        # Get base vocab size
        self._base_vocab_size = self._base_tokenizer.n_vocab
        
        if add_special_tokens:
            # Assign IDs for special tokens (after base vocabulary)
            self.bos_token_id = self._base_vocab_size
            self.eos_token_id = self._base_vocab_size + 1
            self.pad_token_id = self._base_vocab_size + 2
            self.unk_token_id = self._base_vocab_size + 3
            
            # Create mapping for special tokens
            self._special_token_to_id = {
                self.bos_token: self.bos_token_id,
                self.eos_token: self.eos_token_id,
                self.pad_token: self.pad_token_id,
                self.unk_token: self.unk_token_id,
            }
            self._id_to_special_token = {
                v: k for k, v in self._special_token_to_id.items()
            }
        else:
            self.bos_token_id = None
            self.eos_token_id = None
            self.pad_token_id = None
            self.unk_token_id = None
            self._special_token_to_id = {}
            self._id_to_special_token = {}
    
    @property
    def vocab_size(self) -> int:
        """
        Get the vocabulary size including special tokens.
        
        Returns:
            Total vocabulary size
        """
        if self._add_special_tokens:
            return self._base_vocab_size + len(self._special_token_to_id)
        return self._base_vocab_size
    
    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        allowed_special: Union[str, set] = "none_raise",
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Text to encode
            add_bos: Whether to add BOS token at the beginning
            add_eos: Whether to add EOS token at the end
            allowed_special: Which special tokens are allowed in the text
                - "none_raise": Raise error if special tokens found
                - "all": Allow all special tokens
                - set of strings: Allow specific special tokens
        
        Returns:
            List of token IDs
            
        Example:
            >>> tokenizer = Tokenizer()
            >>> tokens = tokenizer.encode("Hello!", add_bos=True, add_eos=True)
            >>> tokens[0] == tokenizer.bos_token_id
            True
            >>> tokens[-1] == tokenizer.eos_token_id
            True
        """
        # Handle special tokens in text
        if allowed_special == "none_raise":
            allowed_special_set = set()
        elif allowed_special == "all":
            allowed_special_set = set(self._special_token_to_id.keys())
        else:
            allowed_special_set = allowed_special
        
        # Encode with base tokenizer
        tokens = self._base_tokenizer.encode(
            text,
            allowed_special=allowed_special_set,
        )
        
        # Add special tokens if requested
        if add_bos and self.bos_token_id is not None:
            tokens = [self.bos_token_id] + tokens
        if add_eos and self.eos_token_id is not None:
            tokens = tokens + [self.eos_token_id]
        
        return tokens
    
    def decode(
        self,
        tokens: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            tokens: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens in output
        
        Returns:
            Decoded text
            
        Example:
            >>> tokenizer = Tokenizer()
            >>> tokens = [15496, 11, 995, 0]
            >>> text = tokenizer.decode(tokens)
            >>> print(text)
            Hello, world!
        """
        if skip_special_tokens and self._add_special_tokens:
            # Filter out special tokens
            special_ids = set(self._id_to_special_token.keys())
            tokens = [t for t in tokens if t not in special_ids]
        
        # Separate base tokens from special tokens
        base_tokens = []
        special_parts = []
        
        for token in tokens:
            if token in self._id_to_special_token:
                # Decode accumulated base tokens if any
                if base_tokens:
                    special_parts.append(self._base_tokenizer.decode(base_tokens))
                    base_tokens = []
                # Add special token
                special_parts.append(self._id_to_special_token[token])
            else:
                base_tokens.append(token)
        
        # Decode remaining base tokens
        if base_tokens:
            special_parts.append(self._base_tokenizer.decode(base_tokens))
        
        return "".join(special_parts)
    
    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
        allowed_special: Union[str, set] = "none_raise",
    ) -> List[List[int]]:
        """
        Encode a batch of texts to token IDs.
        
        Args:
            texts: List of texts to encode
            add_bos: Whether to add BOS token at the beginning
            add_eos: Whether to add EOS token at the end
            allowed_special: Which special tokens are allowed
        
        Returns:
            List of token ID lists
            
        Example:
            >>> tokenizer = Tokenizer()
            >>> batch = ["Hello!", "World!"]
            >>> tokens = tokenizer.encode_batch(batch)
            >>> len(tokens) == 2
            True
        """
        return [
            self.encode(text, add_bos=add_bos, add_eos=add_eos, allowed_special=allowed_special)
            for text in texts
        ]
    
    def decode_batch(
        self,
        token_lists: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Decode a batch of token ID lists to texts.
        
        Args:
            token_lists: List of token ID lists to decode
            skip_special_tokens: Whether to skip special tokens in output
        
        Returns:
            List of decoded texts
            
        Example:
            >>> tokenizer = Tokenizer()
            >>> tokens = [[15496, 0], [10603, 0]]
            >>> texts = tokenizer.decode_batch(tokens)
            >>> len(texts) == 2
            True
        """
        return [
            self.decode(tokens, skip_special_tokens=skip_special_tokens)
            for tokens in token_lists
        ]
    
    def __repr__(self) -> str:
        """String representation of the tokenizer."""
        return (
            f"Tokenizer(vocab_size={self.vocab_size}, "
            f"special_tokens={self._add_special_tokens})"
        )
