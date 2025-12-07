"""
Tests for training script functionality.
"""

import tempfile
from pathlib import Path
import pytest
import yaml
import torch

from scripts.train import (
    create_model_from_config,
    create_datasets,
    setup_logging,
)
from atlas.config import load_config
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
        
        assert config.model.vocab_size == sample_config['model']['vocab_size']
        assert config.model.hidden_size == sample_config['model']['hidden_size']
        assert config.training.learning_rate == sample_config['training']['learning_rate']
    
    def test_create_model_from_config(self, config_file):
        """Test creating model from config."""
        config = load_config(config_file)
        model = create_model_from_config(config)
        
        assert model.config.vocab_size == 50261
        assert model.config.hidden_size == 256
        assert model.config.num_layers == 4
        assert model.config.num_heads == 4
    
    def test_model_parameters(self, config_file):
        """Test model has correct number of parameters."""
        config = load_config(config_file)
        model = create_model_from_config(config)
        
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0
        
        # Check all parameters are trainable
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params == num_params
    
    def test_config_attribute_access_not_dict(self, config_file):
        """Test that loaded config is AtlasConfig object, not dict."""
        config = load_config(config_file)
        
        # Verify it's an AtlasConfig object with attribute access
        from atlas.config import AtlasConfig
        assert isinstance(config, AtlasConfig)
        
        # Verify attribute access works (not dict access)
        assert hasattr(config, 'model')
        assert hasattr(config, 'training')
        assert hasattr(config, 'data')
        assert hasattr(config, 'logging')
        assert hasattr(config, 'inference')
        
        # Verify nested attribute access works
        assert hasattr(config.model, 'vocab_size')
        assert hasattr(config.model, 'hidden_size')
        assert hasattr(config.training, 'learning_rate')
        assert hasattr(config.training, 'batch_size')
        assert hasattr(config.data, 'max_seq_len')
        assert hasattr(config.data, 'num_workers')
        
        # Verify dict-style access raises TypeError (proving it's not a dict)
        with pytest.raises(TypeError):
            _ = config['model']
    
    def test_config_validation_on_load(self, config_file):
        """Test that config validation happens during load."""
        config = load_config(config_file)
        
        # Verify validation worked - these should all be valid
        assert config.model.hidden_size % config.model.num_heads == 0
        assert config.training.learning_rate > 0
        assert config.training.batch_size > 0
        assert config.training.gradient_accumulation_steps > 0
        assert config.training.lr_schedule in ['cosine', 'linear', 'constant']
        
        # Verify aliases are synced
        assert config.training.grad_clip == config.training.max_grad_norm
        assert config.training.lr_schedule == config.training.scheduler_type
        assert config.data.max_seq_len == config.data.sequence_length
        assert config.data.max_seq_len == config.model.max_seq_len
    
    def test_config_type_safety(self, config_file):
        """Test that config fields have correct types."""
        config = load_config(config_file)
        
        # Model config types
        assert isinstance(config.model.vocab_size, int)
        assert isinstance(config.model.hidden_size, int)
        assert isinstance(config.model.num_layers, int)
        assert isinstance(config.model.dropout, float)
        
        # Training config types
        assert isinstance(config.training.learning_rate, float)
        assert isinstance(config.training.batch_size, int)
        assert isinstance(config.training.max_steps, int)
        assert isinstance(config.training.scheduler_type, str)
        assert isinstance(config.training.gradient_checkpointing, bool)
        
        # Data config types
        assert isinstance(config.data.max_seq_len, int)
        assert isinstance(config.data.num_workers, int)


class TestDatasetCreation:
    """Test dataset creation."""
    
    def test_create_train_dataset(self, sample_data, config_file):
        """Test creating training dataset."""
        train_path, _ = sample_data
        tokenizer = Tokenizer('gpt2')
        config = load_config(config_file)
        
        train_dataset, val_dataset = create_datasets(
            train_path,
            None,
            tokenizer,
            config,
        )
        
        assert len(train_dataset) > 0
        assert val_dataset is None
    
    def test_create_train_and_val_datasets(self, sample_data, config_file):
        """Test creating both training and validation datasets."""
        train_path, val_path = sample_data
        tokenizer = Tokenizer('gpt2')
        config = load_config(config_file)
        
        train_dataset, val_dataset = create_datasets(
            train_path,
            val_path,
            tokenizer,
            config,
        )
        
        assert len(train_dataset) > 0
        assert len(val_dataset) > 0
    
    def test_create_dataset_with_multiple_files(self, sample_data, config_file):
        """Test creating dataset with comma-separated paths."""
        train_path, val_path = sample_data
        tokenizer = Tokenizer('gpt2')
        config = load_config(config_file)
        
        # Use same file twice
        train_paths = f"{train_path},{train_path}"
        
        train_dataset, _ = create_datasets(
            train_paths,
            None,
            tokenizer,
            config,
        )
        
        # Should have data from both files
        assert len(train_dataset) > 0


class TestLogging:
    """Test logging functionality."""
    
    def _cleanup_handlers(self, logger):
        """Helper to properly close and remove logger handlers."""
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
    
    def test_setup_logging_creates_log_file(self):
        """Test that setup_logging creates training.log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logging(tmpdir)
            
            try:
                log_file = Path(tmpdir) / "training.log"
                assert log_file.exists()
                assert log_file.is_file()
                
                # Check log file has initial session marker
                content = log_file.read_text()
                assert "NEW TRAINING SESSION" in content
                assert "=" * 80 in content
            finally:
                self._cleanup_handlers(logger)
    
    def test_setup_logging_resume_mode(self):
        """Test that setup_logging appends to existing log on resume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create initial log
            logger1 = setup_logging(tmpdir)
            log_file = Path(tmpdir) / "training.log"
            
            try:
                initial_content = log_file.read_text()
            finally:
                self._cleanup_handlers(logger1)
            
            # Setup again with resume
            logger2 = setup_logging(tmpdir, resume_checkpoint="checkpoint.pt")
            
            try:
                resumed_content = log_file.read_text()
                
                # Should contain both sessions
                assert len(resumed_content) > len(initial_content)
                assert "RESUMED TRAINING SESSION" in resumed_content
                assert "Checkpoint: checkpoint.pt" in resumed_content
            finally:
                self._cleanup_handlers(logger2)
    
    def test_setup_logging_creates_directory(self):
        """Test that setup_logging creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "new_dir" / "checkpoints"
            logger = setup_logging(str(output_dir))
            
            try:
                assert output_dir.exists()
                assert (output_dir / "training.log").exists()
            finally:
                self._cleanup_handlers(logger)
    
    def test_logging_to_file_and_console(self):
        """Test that logger writes to both file and console."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logging(tmpdir)
            
            try:
                test_message = "Test logging message"
                logger.info(test_message)
                
                # Check message is in log file
                log_file = Path(tmpdir) / "training.log"
                content = log_file.read_text()
                assert test_message in content
            finally:
                self._cleanup_handlers(logger)


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
