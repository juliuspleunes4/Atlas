"""
Configuration loading utilities for Atlas.

This module provides functions to load configurations from YAML files,
merge with defaults, and handle CLI overrides.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict

from atlas.config.config import (
    AtlasConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    LoggingConfig,
    InferenceConfig,
)


def load_yaml(path: str) -> Dict[str, Any]:
    """
    Load a YAML file and return its contents as a dictionary.
    
    Args:
        path: Path to the YAML file
        
    Returns:
        Dictionary containing the YAML contents
        
    Raises:
        FileNotFoundError: If the file does not exist
        yaml.YAMLError: If the file is not valid YAML
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    
    if config_dict is None:
        config_dict = {}
    
    return config_dict


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge override dictionary into base dictionary.
    
    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = merge_configs(result[key], value)
        else:
            # Override value
            result[key] = value
    
    return result


def dict_to_config(config_dict: Dict[str, Any]) -> AtlasConfig:
    """
    Convert a dictionary to an AtlasConfig object.
    
    Args:
        config_dict: Dictionary containing configuration values
        
    Returns:
        AtlasConfig object
    """
    # Extract nested configs
    model_dict = config_dict.get("model", {})
    training_dict = config_dict.get("training", {})
    data_dict = config_dict.get("data", {})
    logging_dict = config_dict.get("logging", {})
    inference_dict = config_dict.get("inference", {})
    
    # Create config objects
    model_config = ModelConfig(**model_dict)
    training_config = TrainingConfig(**training_dict)
    data_config = DataConfig(**data_dict)
    logging_config = LoggingConfig(**logging_dict)
    inference_config = InferenceConfig(**inference_dict)
    
    # Extract global settings
    seed = config_dict.get("seed", 42)
    device = config_dict.get("device", "cuda")
    
    return AtlasConfig(
        model=model_config,
        training=training_config,
        data=data_config,
        logging=logging_config,
        inference=inference_config,
        seed=seed,
        device=device,
    )


def config_to_dict(config: AtlasConfig) -> Dict[str, Any]:
    """
    Convert an AtlasConfig object to a dictionary.
    
    Args:
        config: AtlasConfig object
        
    Returns:
        Dictionary representation of the config
    """
    return asdict(config)


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> AtlasConfig:
    """
    Load configuration from YAML file with optional overrides.
    
    Args:
        config_path: Path to YAML config file (optional)
        overrides: Dictionary of values to override (optional)
        
    Returns:
        AtlasConfig object
        
    Example:
        >>> config = load_config("configs/small.yaml")
        >>> config = load_config("configs/small.yaml", {"training": {"batch_size": 64}})
        >>> config = load_config(overrides={"model": {"num_layers": 12}})
    """
    # Start with empty dict (will use dataclass defaults)
    config_dict: Dict[str, Any] = {}
    
    # Load from file if provided
    if config_path is not None:
        config_dict = load_yaml(config_path)
    
    # Apply overrides if provided
    if overrides is not None:
        config_dict = merge_configs(config_dict, overrides)
    
    # Convert to AtlasConfig
    return dict_to_config(config_dict)


def save_config(config: AtlasConfig, path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: AtlasConfig object to save
        path: Path where to save the YAML file
    """
    config_dict = config_to_dict(config)
    
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
