"""
Configuration management for Atlas.

This module provides configuration schemas, loading from files,
CLI argument parsing, and validation.
"""

from atlas.config.config import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    LoggingConfig,
    InferenceConfig,
    AtlasConfig,
)
from atlas.config.loader import (
    load_config,
    save_config,
    load_yaml,
    config_to_dict,
)
from atlas.config.cli import (
    create_config_parser,
    add_config_args,
    parse_args_to_overrides,
)

__all__ = [
    # Config classes
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "LoggingConfig",
    "InferenceConfig",
    "AtlasConfig",
    # Loader functions
    "load_config",
    "save_config",
    "load_yaml",
    "config_to_dict",
    # CLI functions
    "create_config_parser",
    "add_config_args",
    "parse_args_to_overrides",
]
