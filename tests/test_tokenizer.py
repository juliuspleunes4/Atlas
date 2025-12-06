"""
Tests for tokenizer functionality.
"""

import pytest
from atlas.tokenizer import Tokenizer


class TestTokenizerBasic:
    """Tests for basic tokenizer functionality."""
    
    def test_initialization(self):
        """Test that tokenizer initializes correctly."""
        tokenizer = Tokenizer()
        assert tokenizer.vocab_size > 0
        assert tokenizer.bos_token_id is not None
        assert tokenizer.eos_token_id is not None
        assert tokenizer.pad_token_id is not None
        assert tokenizer.unk_token_id is not None
    
    def test_initialization_without_special_tokens(self):
        """Test tokenizer initialization without special tokens."""
        tokenizer = Tokenizer(add_special_tokens=False)
        assert tokenizer.bos_token_id is None
        assert tokenizer.eos_token_id is None
        assert tokenizer.pad_token_id is None
        assert tokenizer.unk_token_id is None
    
    def test_vocab_size(self):
        """Test that vocab size is correct."""
        tokenizer_with_special = Tokenizer(add_special_tokens=True)
        tokenizer_without_special = Tokenizer(add_special_tokens=False)
        
        # Vocab with special tokens should be larger
        assert tokenizer_with_special.vocab_size > tokenizer_without_special.vocab_size
        assert tokenizer_with_special.vocab_size == tokenizer_without_special.vocab_size + 4


class TestTokenizerEncoding:
    """Tests for encoding functionality."""
    
    def test_encode_simple_text(self):
        """Test encoding simple text."""
        tokenizer = Tokenizer()
        tokens = tokenizer.encode("Hello, world!")
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)
    
    def test_encode_empty_string(self):
        """Test encoding empty string."""
        tokenizer = Tokenizer()
        tokens = tokenizer.encode("")
        assert tokens == []
    
    def test_encode_with_bos(self):
        """Test encoding with BOS token."""
        tokenizer = Tokenizer()
        tokens_without = tokenizer.encode("Hello")
        tokens_with = tokenizer.encode("Hello", add_bos=True)
        
        assert len(tokens_with) == len(tokens_without) + 1
        assert tokens_with[0] == tokenizer.bos_token_id
        assert tokens_with[1:] == tokens_without
    
    def test_encode_with_eos(self):
        """Test encoding with EOS token."""
        tokenizer = Tokenizer()
        tokens_without = tokenizer.encode("Hello")
        tokens_with = tokenizer.encode("Hello", add_eos=True)
        
        assert len(tokens_with) == len(tokens_without) + 1
        assert tokens_with[-1] == tokenizer.eos_token_id
        assert tokens_with[:-1] == tokens_without
    
    def test_encode_with_bos_and_eos(self):
        """Test encoding with both BOS and EOS tokens."""
        tokenizer = Tokenizer()
        tokens = tokenizer.encode("Hello", add_bos=True, add_eos=True)
        
        assert tokens[0] == tokenizer.bos_token_id
        assert tokens[-1] == tokenizer.eos_token_id
    
    def test_encode_long_text(self):
        """Test encoding longer text."""
        tokenizer = Tokenizer()
        text = "This is a much longer piece of text that contains multiple words and punctuation marks!"
        tokens = tokenizer.encode(text)
        
        assert len(tokens) > 10
        assert all(isinstance(t, int) for t in tokens)


class TestTokenizerDecoding:
    """Tests for decoding functionality."""
    
    def test_decode_simple_tokens(self):
        """Test decoding token IDs to text."""
        tokenizer = Tokenizer()
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        assert decoded == text
    
    def test_encode_decode_round_trip(self):
        """Test that encode/decode is reversible."""
        tokenizer = Tokenizer()
        original_texts = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "Testing 123!",
            "Special characters: @#$%",
        ]
        
        for text in original_texts:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            assert decoded == text
    
    def test_decode_with_special_tokens(self):
        """Test decoding with special tokens."""
        tokenizer = Tokenizer()
        text = "Hello"
        tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
        
        # Decode without skipping special tokens
        decoded_with = tokenizer.decode(tokens, skip_special_tokens=False)
        assert tokenizer.bos_token in decoded_with
        assert tokenizer.eos_token in decoded_with
        
        # Decode skipping special tokens
        decoded_without = tokenizer.decode(tokens, skip_special_tokens=True)
        assert decoded_without == text
    
    def test_decode_empty_list(self):
        """Test decoding empty token list."""
        tokenizer = Tokenizer()
        decoded = tokenizer.decode([])
        assert decoded == ""


class TestTokenizerBatchOperations:
    """Tests for batch encoding/decoding."""
    
    def test_encode_batch(self):
        """Test batch encoding."""
        tokenizer = Tokenizer()
        texts = ["Hello", "World", "Test"]
        batch_tokens = tokenizer.encode_batch(texts)
        
        assert len(batch_tokens) == len(texts)
        assert all(isinstance(tokens, list) for tokens in batch_tokens)
        
        # Compare with individual encoding
        for text, tokens in zip(texts, batch_tokens):
            individual_tokens = tokenizer.encode(text)
            assert tokens == individual_tokens
    
    def test_encode_batch_with_bos_eos(self):
        """Test batch encoding with special tokens."""
        tokenizer = Tokenizer()
        texts = ["Hello", "World"]
        batch_tokens = tokenizer.encode_batch(texts, add_bos=True, add_eos=True)
        
        for tokens in batch_tokens:
            assert tokens[0] == tokenizer.bos_token_id
            assert tokens[-1] == tokenizer.eos_token_id
    
    def test_decode_batch(self):
        """Test batch decoding."""
        tokenizer = Tokenizer()
        texts = ["Hello", "World", "Test"]
        batch_tokens = tokenizer.encode_batch(texts)
        decoded_texts = tokenizer.decode_batch(batch_tokens)
        
        assert decoded_texts == texts
    
    def test_encode_decode_batch_round_trip(self):
        """Test batch encode/decode round trip."""
        tokenizer = Tokenizer()
        original_texts = [
            "First text",
            "Second text with more words",
            "Third!",
        ]
        
        batch_tokens = tokenizer.encode_batch(original_texts)
        decoded_texts = tokenizer.decode_batch(batch_tokens)
        
        assert decoded_texts == original_texts
    
    def test_encode_batch_empty_list(self):
        """Test encoding empty batch."""
        tokenizer = Tokenizer()
        batch_tokens = tokenizer.encode_batch([])
        assert batch_tokens == []
    
    def test_decode_batch_empty_list(self):
        """Test decoding empty batch."""
        tokenizer = Tokenizer()
        decoded = tokenizer.decode_batch([])
        assert decoded == []


class TestTokenizerEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_encode_very_long_string(self):
        """Test encoding very long string."""
        tokenizer = Tokenizer()
        long_text = "word " * 1000
        tokens = tokenizer.encode(long_text)
        
        assert len(tokens) > 100
        decoded = tokenizer.decode(tokens)
        assert decoded == long_text
    
    def test_encode_unicode_characters(self):
        """Test encoding Unicode characters."""
        tokenizer = Tokenizer()
        texts = [
            "Hello 世界",
            "Café ☕",
            "🚀 Rocket",
        ]
        
        for text in texts:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            assert decoded == text
    
    def test_encode_newlines_and_tabs(self):
        """Test encoding whitespace characters."""
        tokenizer = Tokenizer()
        text = "Line 1\nLine 2\tTabbed"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text
    
    def test_special_token_ids_are_unique(self):
        """Test that all special token IDs are unique."""
        tokenizer = Tokenizer()
        special_ids = [
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            tokenizer.unk_token_id,
        ]
        
        assert len(special_ids) == len(set(special_ids))
    
    def test_special_token_ids_outside_base_vocab(self):
        """Test that special token IDs don't overlap with base vocabulary."""
        tokenizer = Tokenizer()
        base_vocab_size = tokenizer._base_vocab_size
        
        assert tokenizer.bos_token_id >= base_vocab_size
        assert tokenizer.eos_token_id >= base_vocab_size
        assert tokenizer.pad_token_id >= base_vocab_size
        assert tokenizer.unk_token_id >= base_vocab_size
    
    def test_repr(self):
        """Test string representation of tokenizer."""
        tokenizer = Tokenizer()
        repr_str = repr(tokenizer)
        
        assert "Tokenizer" in repr_str
        assert "vocab_size" in repr_str
        assert str(tokenizer.vocab_size) in repr_str


class TestTokenizerDifferentEncodings:
    """Tests for different encoding backends."""
    
    def test_gpt2_encoding(self):
        """Test GPT-2 encoding."""
        tokenizer = Tokenizer(encoding_name="gpt2")
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text
    
    def test_different_encodings_have_different_vocab_sizes(self):
        """Test that different encodings have different vocab sizes."""
        tokenizer_gpt2 = Tokenizer(encoding_name="gpt2", add_special_tokens=False)
        
        # GPT-2 has a known vocab size
        assert tokenizer_gpt2.vocab_size == 50257
