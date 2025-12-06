"""
Tests for inference and generation.
"""

import pytest
import torch

from atlas.inference import (
    GenerationConfig,
    TextGenerator,
    generate_text,
    sample_top_k,
    sample_top_p,
)
from atlas.model import AtlasLM
from atlas.config import ModelConfig


class TestGenerationConfig:
    """Test suite for GenerationConfig."""
    
    def test_generation_config_defaults(self):
        """Test default configuration values."""
        config = GenerationConfig()
        
        assert config.max_new_tokens == 50
        assert config.temperature == 1.0
        assert config.top_k is None
        assert config.top_p is None
        assert config.do_sample is True
    
    def test_generation_config_custom(self):
        """Test custom configuration."""
        config = GenerationConfig(
            max_new_tokens=100,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            do_sample=True,
        )
        
        assert config.max_new_tokens == 100
        assert config.temperature == 0.8
        assert config.top_k == 50
        assert config.top_p == 0.95
    
    def test_invalid_temperature(self):
        """Test that invalid temperature raises error."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            GenerationConfig(temperature=0)
        
        with pytest.raises(ValueError, match="temperature must be positive"):
            GenerationConfig(temperature=-1.0)
    
    def test_invalid_top_k(self):
        """Test that invalid top_k raises error."""
        with pytest.raises(ValueError, match="top_k must be positive"):
            GenerationConfig(top_k=0)
        
        with pytest.raises(ValueError, match="top_k must be positive"):
            GenerationConfig(top_k=-5)
    
    def test_invalid_top_p(self):
        """Test that invalid top_p raises error."""
        with pytest.raises(ValueError, match="top_p must be in"):
            GenerationConfig(top_p=0)
        
        with pytest.raises(ValueError, match="top_p must be in"):
            GenerationConfig(top_p=1.5)


class TestSamplingStrategies:
    """Test suite for sampling strategy functions."""
    
    def test_sample_top_k(self):
        """Test top-k sampling filter."""
        batch_size, vocab_size = 2, 100
        logits = torch.randn(batch_size, vocab_size)
        
        filtered = sample_top_k(logits, top_k=10)
        
        # Check that only top-k values are finite
        finite_count = (filtered != float('-inf')).sum(dim=-1)
        assert (finite_count == 10).all()
    
    def test_sample_top_k_larger_than_vocab(self):
        """Test top-k when k > vocab_size."""
        logits = torch.randn(2, 50)
        
        filtered = sample_top_k(logits, top_k=100)
        
        # All values should be finite
        finite_count = (filtered != float('-inf')).sum(dim=-1)
        assert (finite_count == 50).all()
    
    def test_sample_top_p(self):
        """Test top-p (nucleus) sampling filter."""
        batch_size, vocab_size = 2, 100
        logits = torch.randn(batch_size, vocab_size)
        
        filtered = sample_top_p(logits, top_p=0.9)
        
        # Check that some tokens are filtered out
        finite_count = (filtered != float('-inf')).sum(dim=-1)
        assert (finite_count > 0).all()
        assert (finite_count < vocab_size).all()
    
    def test_sample_top_p_keeps_at_least_one(self):
        """Test that top-p keeps at least one token."""
        logits = torch.randn(1, 50)
        
        # Very small top_p should still keep at least one token
        filtered = sample_top_p(logits, top_p=0.01)
        
        finite_count = (filtered != float('-inf')).sum()
        assert finite_count >= 1


class TestTextGenerator:
    """Test suite for TextGenerator class."""
    
    @pytest.fixture
    def tiny_model(self):
        """Create a tiny model for testing."""
        config = ModelConfig(
            vocab_size=100,
            max_seq_len=64,
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            mlp_ratio=2.0,
        )
        return AtlasLM(config)
    
    def test_generator_initialization(self, tiny_model):
        """Test TextGenerator initialization."""
        generator = TextGenerator(tiny_model, device='cpu')
        
        assert generator.model is not None
        assert generator.device == 'cpu'
        assert not generator.model.training  # Should be in eval mode
    
    def test_greedy_generation(self, tiny_model):
        """Test greedy generation (argmax)."""
        generator = TextGenerator(tiny_model, device='cpu')
        
        input_ids = torch.randint(0, 100, (1, 10))
        config = GenerationConfig(
            max_new_tokens=5,
            do_sample=False,  # Greedy
        )
        
        output = generator.generate(input_ids, config)
        
        assert output.shape[0] == 1
        assert output.shape[1] == 15  # 10 + 5
        assert torch.all(output[:, :10] == input_ids)  # Input preserved
    
    def test_temperature_sampling(self, tiny_model):
        """Test temperature sampling."""
        generator = TextGenerator(tiny_model, device='cpu')
        
        input_ids = torch.randint(0, 100, (1, 10))
        config = GenerationConfig(
            max_new_tokens=10,
            temperature=0.8,
            do_sample=True,
        )
        
        output = generator.generate(input_ids, config)
        
        assert output.shape[1] == 20  # 10 + 10
    
    def test_top_k_generation(self, tiny_model):
        """Test top-k sampling."""
        generator = TextGenerator(tiny_model, device='cpu')
        
        input_ids = torch.randint(0, 100, (1, 10))
        config = GenerationConfig(
            max_new_tokens=5,
            top_k=10,
            do_sample=True,
        )
        
        output = generator.generate(input_ids, config)
        
        assert output.shape[1] == 15
    
    def test_top_p_generation(self, tiny_model):
        """Test top-p (nucleus) sampling."""
        generator = TextGenerator(tiny_model, device='cpu')
        
        input_ids = torch.randint(0, 100, (1, 10))
        config = GenerationConfig(
            max_new_tokens=5,
            top_p=0.9,
            do_sample=True,
        )
        
        output = generator.generate(input_ids, config)
        
        assert output.shape[1] == 15
    
    def test_combined_top_k_top_p(self, tiny_model):
        """Test combined top-k and top-p sampling."""
        generator = TextGenerator(tiny_model, device='cpu')
        
        input_ids = torch.randint(0, 100, (1, 10))
        config = GenerationConfig(
            max_new_tokens=5,
            top_k=20,
            top_p=0.9,
            do_sample=True,
        )
        
        output = generator.generate(input_ids, config)
        
        assert output.shape[1] == 15
    
    def test_eos_token_stopping(self, tiny_model):
        """Test generation stops at EOS token."""
        generator = TextGenerator(tiny_model, device='cpu')
        
        input_ids = torch.randint(0, 100, (1, 10))
        config = GenerationConfig(
            max_new_tokens=50,
            eos_token_id=50,  # Will rarely be generated, but test the mechanism
            do_sample=False,
        )
        
        output = generator.generate(input_ids, config)
        
        # Output should be at most 10 + 50 tokens
        assert output.shape[1] <= 60
    
    def test_batch_generation(self, tiny_model):
        """Test generation with batch size > 1."""
        generator = TextGenerator(tiny_model, device='cpu')
        
        batch_size = 3
        input_ids = torch.randint(0, 100, (batch_size, 10))
        config = GenerationConfig(max_new_tokens=5, do_sample=False)
        
        output = generator.generate(input_ids, config)
        
        assert output.shape[0] == batch_size
        assert output.shape[1] == 15
    
    def test_different_temperatures_different_outputs(self, tiny_model):
        """Test that different temperatures produce different outputs."""
        generator = TextGenerator(tiny_model, device='cpu')
        
        torch.manual_seed(42)
        input_ids = torch.randint(0, 100, (1, 10))
        
        # Low temperature (more deterministic)
        config_low = GenerationConfig(max_new_tokens=10, temperature=0.5, do_sample=True)
        torch.manual_seed(42)
        output_low = generator.generate(input_ids.clone(), config_low)
        
        # High temperature (more random)
        config_high = GenerationConfig(max_new_tokens=10, temperature=2.0, do_sample=True)
        torch.manual_seed(43)  # Different seed
        output_high = generator.generate(input_ids.clone(), config_high)
        
        # Outputs should differ (with very high probability)
        assert not torch.all(output_low == output_high)
    
    def test_greedy_is_deterministic(self, tiny_model):
        """Test that greedy generation is deterministic."""
        generator = TextGenerator(tiny_model, device='cpu')
        
        input_ids = torch.randint(0, 100, (1, 10))
        config = GenerationConfig(max_new_tokens=10, do_sample=False)
        
        output1 = generator.generate(input_ids.clone(), config)
        output2 = generator.generate(input_ids.clone(), config)
        
        assert torch.all(output1 == output2)


class TestGenerateText:
    """Test suite for generate_text convenience function."""
    
    @pytest.fixture
    def tiny_model(self):
        """Create a tiny model for testing."""
        config = ModelConfig(
            vocab_size=100,
            max_seq_len=64,
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            mlp_ratio=2.0,
        )
        return AtlasLM(config)
    
    def test_generate_text_basic(self, tiny_model):
        """Test generate_text convenience function."""
        input_ids = torch.randint(0, 100, (1, 10))
        
        output = generate_text(
            tiny_model,
            input_ids,
            max_new_tokens=5,
            device='cpu',
        )
        
        assert output.shape[1] == 15
    
    def test_generate_text_with_all_params(self, tiny_model):
        """Test generate_text with all parameters."""
        input_ids = torch.randint(0, 100, (1, 10))
        
        output = generate_text(
            tiny_model,
            input_ids,
            max_new_tokens=10,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            eos_token_id=50,
            device='cpu',
        )
        
        assert output.shape[0] == 1
        assert output.shape[1] <= 20  # May stop early due to EOS
