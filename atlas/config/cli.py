"""
CLI argument parsing utilities for Atlas.

This module provides functions to parse command-line arguments
and convert them into configuration overrides.
"""

import argparse
from typing import Dict, Any, Optional, List


def parse_override_string(override_str: str) -> tuple[str, Any]:
    """
    Parse a single override string in the format "key.subkey=value".
    
    Args:
        override_str: String like "model.num_layers=12" or "training.learning_rate=0.001"
        
    Returns:
        Tuple of (key_path, value) where key_path is like "model.num_layers"
        
    Raises:
        ValueError: If the string format is invalid
        
    Example:
        >>> parse_override_string("model.num_layers=12")
        ('model.num_layers', 12)
        >>> parse_override_string("training.use_amp=true")
        ('training.use_amp', True)
    """
    if "=" not in override_str:
        raise ValueError(f"Override must be in format 'key=value', got: {override_str}")
    
    key_path, value_str = override_str.split("=", 1)
    key_path = key_path.strip()
    value_str = value_str.strip()
    
    if not key_path:
        raise ValueError(f"Empty key in override: {override_str}")
    
    # Try to parse the value as the appropriate type
    value: Any
    
    # Boolean
    if value_str.lower() in ["true", "false"]:
        value = value_str.lower() == "true"
    # None
    elif value_str.lower() == "none":
        value = None
    # Try integer
    elif value_str.isdigit() or (value_str.startswith("-") and value_str[1:].isdigit()):
        value = int(value_str)
    # Try float
    else:
        try:
            value = float(value_str)
        except ValueError:
            # Keep as string
            value = value_str
    
    return key_path, value


def build_override_dict(overrides: List[tuple[str, Any]]) -> Dict[str, Any]:
    """
    Build a nested dictionary from a list of (key_path, value) tuples.
    
    Args:
        overrides: List of (key_path, value) tuples like [("model.num_layers", 12)]
        
    Returns:
        Nested dictionary like {"model": {"num_layers": 12}}
        
    Example:
        >>> build_override_dict([("model.num_layers", 12), ("training.batch_size", 64)])
        {'model': {'num_layers': 12}, 'training': {'batch_size': 64}}
    """
    result: Dict[str, Any] = {}
    
    for key_path, value in overrides:
        keys = key_path.split(".")
        current = result
        
        # Navigate/create nested structure
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
    
    return result


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add configuration-related arguments to an argument parser.
    
    Args:
        parser: ArgumentParser to add arguments to
        
    Returns:
        Modified ArgumentParser
    """
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    
    parser.add_argument(
        "--override",
        type=str,
        action="append",
        dest="overrides",
        help=(
            "Override config values (format: key.subkey=value). "
            "Can be specified multiple times. "
            "Example: --override model.num_layers=12 --override training.batch_size=64"
        ),
    )
    
    return parser


def parse_args_to_overrides(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """
    Convert parsed arguments to a config override dictionary.
    
    Args:
        args: Parsed arguments from argparse
        
    Returns:
        Override dictionary or None if no overrides
        
    Example:
        >>> parser = argparse.ArgumentParser()
        >>> parser = add_config_args(parser)
        >>> args = parser.parse_args(["--override", "model.num_layers=12"])
        >>> parse_args_to_overrides(args)
        {'model': {'num_layers': 12}}
    """
    if not hasattr(args, "overrides") or args.overrides is None:
        return None
    
    # Parse each override string
    parsed_overrides = [parse_override_string(s) for s in args.overrides]
    
    # Build nested dictionary
    return build_override_dict(parsed_overrides)


def create_config_parser(description: str = "Atlas configuration") -> argparse.ArgumentParser:
    """
    Create a standard argument parser for Atlas scripts.
    
    Args:
        description: Description for the parser
        
    Returns:
        ArgumentParser with config arguments added
    """
    parser = argparse.ArgumentParser(description=description)
    return add_config_args(parser)
