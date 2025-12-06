"""
Tests for configuration management.
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from atlas.config import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    LoggingConfig,
    InferenceConfig,
    AtlasConfig,
    load_config,
    save_config,
    load_yaml,
)
from atlas.config.cli import (
    parse_override_string,
    build_override_dict,
    parse_args_to_overrides,
    create_config_parser,
)


class TestModelConfig:
    """Tests for ModelConfig validation."""
    
    def test_valid_config(self):
        """Test that valid configuration is accepted."""
        config = ModelConfig(
            num_layers=6,
            hidden_size=512,
            num_heads=8,
            vocab_size=50257,
            max_seq_len=1024,
        )
        assert config.num_layers == 6
        assert config.hidden_size == 512
    
    def test_hidden_size_not_divisible_by_heads(self):
        """Test that hidden_size must be divisible by num_heads."""
        with pytest.raises(ValueError, match="must be divisible"):
            ModelConfig(hidden_size=512, num_heads=7)
    
    def test_invalid_activation(self):
        """Test that invalid activation function is rejected."""
        with pytest.raises(ValueError, match="activation must be one of"):
            ModelConfig(activation="invalid")
    
    def test_dropout_out_of_range(self):
        """Test that dropout must be in [0, 1]."""
        with pytest.raises(ValueError, match="dropout must be in"):
            ModelConfig(dropout=1.5)
        
        with pytest.raises(ValueError, match="dropout must be in"):
            ModelConfig(dropout=-0.1)
    
    def test_attention_dropout_out_of_range(self):
        """Test that attention_dropout must be in [0, 1]."""
        with pytest.raises(ValueError, match="attention_dropout must be in"):
            ModelConfig(attention_dropout=1.2)
    
    def test_negative_mlp_ratio(self):
        """Test that mlp_ratio must be positive."""
        with pytest.raises(ValueError, match="mlp_ratio must be positive"):
            ModelConfig(mlp_ratio=-1.0)


class TestTrainingConfig:
    """Tests for TrainingConfig validation."""
    
    def test_valid_config(self):
        """Test that valid configuration is accepted."""
        config = TrainingConfig(
            learning_rate=3e-4,
            batch_size=32,
            max_steps=100000,
        )
        assert config.learning_rate == 3e-4
    
    def test_negative_learning_rate(self):
        """Test that learning_rate must be positive."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            TrainingConfig(learning_rate=-0.001)
    
    def test_invalid_batch_size(self):
        """Test that batch_size must be positive."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            TrainingConfig(batch_size=0)
    
    def test_invalid_gradient_accumulation(self):
        """Test that gradient_accumulation_steps must be positive."""
        with pytest.raises(ValueError, match="gradient_accumulation_steps must be positive"):
            TrainingConfig(gradient_accumulation_steps=-1)
    
    def test_invalid_lr_schedule(self):
        """Test that lr_schedule must be valid."""
        with pytest.raises(ValueError, match="lr_schedule must be one of"):
            TrainingConfig(lr_schedule="invalid")
    
    def test_effective_batch_size(self):
        """Test calculation of effective batch size."""
        config = TrainingConfig(
            batch_size=32,
            gradient_accumulation_steps=4,
        )
        assert config.effective_batch_size == 128


class TestInferenceConfig:
    """Tests for InferenceConfig validation."""
    
    def test_valid_config(self):
        """Test that valid configuration is accepted."""
        config = InferenceConfig(
            max_new_tokens=100,
            temperature=1.0,
            top_k=50,
        )
        assert config.max_new_tokens == 100
    
    def test_invalid_max_new_tokens(self):
        """Test that max_new_tokens must be positive."""
        with pytest.raises(ValueError, match="max_new_tokens must be positive"):
            InferenceConfig(max_new_tokens=0)
    
    def test_invalid_temperature(self):
        """Test that temperature must be positive."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            InferenceConfig(temperature=0)
    
    def test_invalid_top_p(self):
        """Test that top_p must be in (0, 1]."""
        with pytest.raises(ValueError, match="top_p must be in"):
            InferenceConfig(top_p=0)
        
        with pytest.raises(ValueError, match="top_p must be in"):
            InferenceConfig(top_p=1.5)


class TestAtlasConfig:
    """Tests for AtlasConfig validation."""
    
    def test_valid_config(self):
        """Test that valid configuration is accepted."""
        config = AtlasConfig()
        assert config.model.num_layers == 6
        assert config.training.batch_size == 32
    
    def test_sequence_length_mismatch(self):
        """Test that data.sequence_length must match model.max_seq_len."""
        with pytest.raises(ValueError, match="must match"):
            AtlasConfig(
                model=ModelConfig(max_seq_len=1024),
                data=DataConfig(sequence_length=512),
            )


class TestConfigLoader:
    """Tests for configuration loading."""
    
    def test_load_from_dict(self):
        """Test loading config from dictionary."""
        config = load_config(overrides={"model": {"num_layers": 12}})
        assert config.model.num_layers == 12
    
    def test_load_from_yaml_file(self):
        """Test loading config from YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"model": {"num_layers": 12, "hidden_size": 768}}, f)
            temp_path = f.name
        
        try:
            config = load_config(config_path=temp_path)
            assert config.model.num_layers == 12
            assert config.model.hidden_size == 768
        finally:
            Path(temp_path).unlink()
    
    def test_load_with_overrides(self):
        """Test loading config with overrides."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"model": {"num_layers": 6}}, f)
            temp_path = f.name
        
        try:
            config = load_config(
                config_path=temp_path,
                overrides={"model": {"num_layers": 12}},
            )
            assert config.model.num_layers == 12
        finally:
            Path(temp_path).unlink()
    
    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_yaml("nonexistent_file.yaml")
    
    def test_save_and_load_config(self):
        """Test saving and loading config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "test_config.yaml"
            
            # Create and save config
            original_config = AtlasConfig(
                model=ModelConfig(num_layers=12, hidden_size=768),
            )
            save_config(original_config, str(temp_path))
            
            # Load config
            loaded_config = load_config(config_path=str(temp_path))
            
            assert loaded_config.model.num_layers == 12
            assert loaded_config.model.hidden_size == 768


class TestCLIParsing:
    """Tests for CLI argument parsing."""
    
    def test_parse_override_string_int(self):
        """Test parsing integer override."""
        key, value = parse_override_string("model.num_layers=12")
        assert key == "model.num_layers"
        assert value == 12
        assert isinstance(value, int)
    
    def test_parse_override_string_float(self):
        """Test parsing float override."""
        key, value = parse_override_string("training.learning_rate=0.001")
        assert key == "training.learning_rate"
        assert value == 0.001
        assert isinstance(value, float)
    
    def test_parse_override_string_bool(self):
        """Test parsing boolean override."""
        key, value = parse_override_string("training.use_amp=true")
        assert key == "training.use_amp"
        assert value is True
        
        key, value = parse_override_string("training.use_amp=false")
        assert value is False
    
    def test_parse_override_string_none(self):
        """Test parsing None override."""
        key, value = parse_override_string("data.train_data_path=none")
        assert key == "data.train_data_path"
        assert value is None
    
    def test_parse_override_string_string(self):
        """Test parsing string override."""
        key, value = parse_override_string("model.activation=gelu")
        assert key == "model.activation"
        assert value == "gelu"
        assert isinstance(value, str)
    
    def test_parse_override_string_invalid(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="must be in format"):
            parse_override_string("invalid_format")
    
    def test_build_override_dict(self):
        """Test building nested override dictionary."""
        overrides = [
            ("model.num_layers", 12),
            ("model.hidden_size", 768),
            ("training.batch_size", 64),
        ]
        result = build_override_dict(overrides)
        
        assert result["model"]["num_layers"] == 12
        assert result["model"]["hidden_size"] == 768
        assert result["training"]["batch_size"] == 64
    
    def test_create_config_parser(self):
        """Test creating config parser."""
        parser = create_config_parser()
        args = parser.parse_args([
            "--config", "test.yaml",
            "--override", "model.num_layers=12",
            "--override", "training.batch_size=64",
        ])
        
        assert args.config == "test.yaml"
        assert len(args.overrides) == 2
        
        overrides = parse_args_to_overrides(args)
        assert overrides["model"]["num_layers"] == 12
        assert overrides["training"]["batch_size"] == 64
    
    def test_parse_args_no_overrides(self):
        """Test parsing args with no overrides."""
        parser = create_config_parser()
        args = parser.parse_args([])
        
        overrides = parse_args_to_overrides(args)
        assert overrides is None
