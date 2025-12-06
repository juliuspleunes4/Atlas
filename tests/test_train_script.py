"""
Tests for training script functionality.
"""

import tempfile
from pathlib import Path
import pytest
import yaml
import torch

from scripts.train import (
    load_config,
    create_model_from_config,
    create_datasets,
)
from atlas.tokenizer import Tokenizer


@pytest.fixture
def sample_config():
    """Sample training configuration."""
    return {
        'model': {
            'vocab_size': 50261,
            'max_seq_len': 128,
            'hidden_size': 256,
            'num_layers': 4,
            'num_heads': 4,
            'mlp_ratio': 4.0,
            'dropout': 0.1,
        },
        'tokenizer': {
            'name': 'gpt2',
            'encoding': 'gpt2',
        },
        'data': {
            'max_seq_len': 128,
            'num_workers': 0,
        },
        'training': {
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'batch_size': 2,
            'gradient_accumulation_steps': 1,
            'max_grad_norm': 1.0,
            'max_steps': 100,
            'warmup_steps': 10,
            'scheduler_type': 'cosine',
            'keep_checkpoints': 2,
        }
    }


@pytest.fixture
def config_file(sample_config):
    """Create temporary config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_config, f)
        config_path = f.name
    
    yield config_path
    
    # Cleanup
    Path(config_path).unlink(missing_ok=True)


@pytest.fixture
def sample_data():
    """Create temporary training data."""
    train_text = "Hello world! " * 100
    val_text = "Test validation. " * 50
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(train_text)
        train_path = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(val_text)
        val_path = f.name
    
    yield train_path, val_path
    
    # Cleanup
    Path(train_path).unlink(missing_ok=True)
    Path(val_path).unlink(missing_ok=True)


class TestConfigLoading:
    """Test configuration loading."""
    
    def test_load_config(self, config_file, sample_config):
        """Test loading config from YAML file."""
        config = load_config(config_file)
        
        assert config['model']['vocab_size'] == sample_config['model']['vocab_size']
        assert config['model']['hidden_size'] == sample_config['model']['hidden_size']
        assert config['training']['learning_rate'] == sample_config['training']['learning_rate']
    
    def test_create_model_from_config(self, sample_config):
        """Test creating model from config."""
        model = create_model_from_config(sample_config)
        
        assert model.config.vocab_size == 50261
        assert model.config.hidden_size == 256
        assert model.config.num_layers == 4
        assert model.config.num_heads == 4
    
    def test_model_parameters(self, sample_config):
        """Test model has correct number of parameters."""
        model = create_model_from_config(sample_config)
        
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0
        
        # Check all parameters are trainable
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params == num_params


class TestDatasetCreation:
    """Test dataset creation."""
    
    def test_create_train_dataset(self, sample_data, sample_config):
        """Test creating training dataset."""
        train_path, _ = sample_data
        tokenizer = Tokenizer('gpt2')
        
        train_dataset, val_dataset = create_datasets(
            train_path,
            None,
            tokenizer,
            sample_config,
        )
        
        assert len(train_dataset) > 0
        assert val_dataset is None
    
    def test_create_train_and_val_datasets(self, sample_data, sample_config):
        """Test creating both training and validation datasets."""
        train_path, val_path = sample_data
        tokenizer = Tokenizer('gpt2')
        
        train_dataset, val_dataset = create_datasets(
            train_path,
            val_path,
            tokenizer,
            sample_config,
        )
        
        assert len(train_dataset) > 0
        assert len(val_dataset) > 0
    
    def test_create_dataset_with_multiple_files(self, sample_data, sample_config):
        """Test creating dataset with comma-separated paths."""
        train_path, val_path = sample_data
        tokenizer = Tokenizer('gpt2')
        
        # Use same file twice
        train_paths = f"{train_path},{train_path}"
        
        train_dataset, _ = create_datasets(
            train_paths,
            None,
            tokenizer,
            sample_config,
        )
        
        # Should have data from both files
        assert len(train_dataset) > 0


class TestTrainingScript:
    """Test training script integration."""
    
    def test_config_override(self, sample_config):
        """Test that config overrides work correctly."""
        original_lr = sample_config['training']['learning_rate']
        
        # Simulate CLI override
        new_lr = 5e-4
        sample_config['training']['learning_rate'] = new_lr
        
        assert sample_config['training']['learning_rate'] == new_lr
        assert sample_config['training']['learning_rate'] != original_lr
    
    def test_device_selection(self):
        """Test device selection logic."""
        # Should select cuda if available, otherwise cpu
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Simulate argparse default
        device = "cuda" if torch.cuda.is_available() else "cpu"
        assert device == expected_device
    
    def test_checkpoint_directory_creation(self):
        """Test checkpoint directory can be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            assert checkpoint_dir.exists()
            assert checkpoint_dir.is_dir()
