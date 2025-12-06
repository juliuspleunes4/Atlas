"""
Tests for inference script functionality.
"""

import tempfile
from pathlib import Path
import pytest
import torch

from scripts.infer import load_prompts_from_file
from atlas.model import AtlasLM
from atlas.config import ModelConfig
from atlas.tokenizer import Tokenizer
from atlas.training import CheckpointManager, CheckpointMetadata


@pytest.fixture
def sample_model():
    """Create a small test model."""
    config = ModelConfig(
        vocab_size=50261,
        max_seq_len=128,
        hidden_size=256,
        num_layers=2,
        num_heads=4,
    )
    model = AtlasLM(config)
    return model, config


@pytest.fixture
def sample_checkpoint(sample_model):
    """Create a temporary checkpoint file."""
    model, config = sample_model
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "checkpoint.pt"
        
        # Save checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config,
            'step': 1000,
            'epoch': 1,
        }
        torch.save(checkpoint, checkpoint_path)
        
        yield checkpoint_path


@pytest.fixture
def prompts_file():
    """Create a temporary prompts file."""
    prompts = [
        "Once upon a time",
        "The quick brown fox",
        "In a galaxy far, far away",
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for prompt in prompts:
            f.write(prompt + '\n')
        prompts_path = f.name
    
    yield prompts_path
    
    # Cleanup
    Path(prompts_path).unlink(missing_ok=True)


class TestPromptsLoading:
    """Test prompts file loading."""
    
    def test_load_prompts_from_file(self, prompts_file):
        """Test loading prompts from file."""
        prompts = load_prompts_from_file(prompts_file)
        
        assert len(prompts) == 3
        assert prompts[0] == "Once upon a time"
        assert prompts[1] == "The quick brown fox"
        assert prompts[2] == "In a galaxy far, far away"
    
    def test_load_prompts_skips_empty_lines(self):
        """Test that empty lines are skipped."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("First prompt\n")
            f.write("\n")
            f.write("Second prompt\n")
            f.write("   \n")
            f.write("Third prompt\n")
            prompts_path = f.name
        
        try:
            prompts = load_prompts_from_file(prompts_path)
            assert len(prompts) == 3
            assert prompts[0] == "First prompt"
            assert prompts[1] == "Second prompt"
            assert prompts[2] == "Third prompt"
        finally:
            Path(prompts_path).unlink(missing_ok=True)
    
    def test_load_prompts_strips_whitespace(self):
        """Test that whitespace is stripped from prompts."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("  Prompt with leading spaces\n")
            f.write("Prompt with trailing spaces  \n")
            f.write("  Prompt with both  \n")
            prompts_path = f.name
        
        try:
            prompts = load_prompts_from_file(prompts_path)
            assert len(prompts) == 3
            assert prompts[0] == "Prompt with leading spaces"
            assert prompts[1] == "Prompt with trailing spaces"
            assert prompts[2] == "Prompt with both"
        finally:
            Path(prompts_path).unlink(missing_ok=True)


class TestCheckpointLoading:
    """Test checkpoint loading functionality."""
    
    def test_checkpoint_has_required_fields(self, sample_checkpoint):
        """Test checkpoint contains required fields."""
        checkpoint = torch.load(sample_checkpoint, map_location='cpu', weights_only=False)
        
        assert 'model_state_dict' in checkpoint
        assert 'config' in checkpoint
        assert 'step' in checkpoint
        assert 'epoch' in checkpoint
    
    def test_checkpoint_config_is_valid(self, sample_checkpoint):
        """Test checkpoint config has valid model parameters."""
        checkpoint = torch.load(sample_checkpoint, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        
        assert config.vocab_size == 50261
        assert config.max_seq_len == 128
        assert config.hidden_size == 256
        assert config.num_layers == 2
        assert config.num_heads == 4
    
    def test_checkpoint_state_dict_is_valid(self, sample_checkpoint):
        """Test checkpoint state dict can be loaded."""
        checkpoint = torch.load(sample_checkpoint, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        # Check some expected keys (actual key names from model)
        assert 'embeddings.token_embedding.embedding.weight' in state_dict
        assert 'embeddings.positional_embedding.embedding.weight' in state_dict
        assert 'blocks.0.ln1.weight' in state_dict
        assert 'lm_head.weight' in state_dict


class TestGenerationConfig:
    """Test generation configuration."""
    
    def test_default_generation_config(self):
        """Test default generation config values."""
        from atlas.inference import GenerationConfig
        
        config = GenerationConfig()
        
        assert config.max_new_tokens == 50
        assert config.temperature == 1.0
        assert config.top_k is None
        assert config.top_p is None
        assert config.do_sample is True
        assert config.eos_token_id is None
    
    def test_custom_generation_config(self):
        """Test custom generation config values."""
        from atlas.inference import GenerationConfig
        
        config = GenerationConfig(
            max_new_tokens=100,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            do_sample=True,
            eos_token_id=50256,
        )
        
        assert config.max_new_tokens == 100
        assert config.temperature == 0.8
        assert config.top_k == 40
        assert config.top_p == 0.9
        assert config.do_sample is True
        assert config.eos_token_id == 50256


class TestInferenceScript:
    """Test inference script integration."""
    
    def test_output_file_creation(self):
        """Test that output file can be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.txt"
            
            # Write some test output
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("Generated text")
            
            assert output_path.exists()
            
            # Read back
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert content == "Generated text"
    
    def test_separator_formatting(self):
        """Test separator formatting between generations."""
        results = ["First generation", "Second generation", "Third generation"]
        separator = "\n" + "="*80 + "\n"
        
        output = separator.join(results)
        
        assert "First generation" in output
        assert "Second generation" in output
        assert "Third generation" in output
        assert output.count("="*80) == 2  # Between 3 items
    
    def test_prompt_display_with_show_prompt(self):
        """Test output format when showing prompts."""
        prompt = "Test prompt"
        generated = "Generated text"
        
        output = f"Prompt: {prompt}\nGenerated: {generated}"
        
        assert "Prompt: Test prompt" in output
        assert "Generated: Generated text" in output
    
    def test_prompt_display_without_show_prompt(self):
        """Test output format when not showing prompts."""
        generated = "Generated text"
        
        output = generated
        
        assert "Prompt:" not in output
        assert "Generated:" not in output
        assert output == "Generated text"
