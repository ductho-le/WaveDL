"""
WaveDL - Configuration Management
==================================

YAML configuration file support for reproducible experiments.

Features:
    - Load experiment configs from YAML files
    - Merge configs with CLI arguments (CLI takes precedence)
    - Validate config values against known options
    - Save effective config for reproducibility

Usage:
    # Load config and merge with CLI args
    config = load_config("experiment.yaml")
    args = merge_config_with_args(config, args)
    
    # Save effective config
    save_config(args, "output/config.yaml")

Author: Ductho Le (ductho.le@outlook.com)
Version: 1.0.0
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import argparse


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Dictionary of configuration values
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    
    Example:
        >>> config = load_config("configs/experiment.yaml")
        >>> print(config['model'])
        'cnn'
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        config = {}
    
    # Handle nested configs (e.g., optimizer.lr -> optimizer_lr)
    config = _flatten_config(config)
    
    return config


def _flatten_config(config: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flatten nested dictionaries for argparse compatibility.
    
    Recursively flattens nested dicts, preserving the full key path.
    
    Example:
        {'optimizer': {'lr': 1e-3}} -> {'optimizer_lr': 1e-3}
        {'optimizer': {'params': {'beta1': 0.9}}} -> {'optimizer_params_beta1': 0.9}
        {'lr': 1e-3} -> {'lr': 1e-3}
    """
    items = []
    for key, value in config.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            # Recursively flatten, passing full accumulated key path
            items.extend(_flatten_config(value, new_key, sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def merge_config_with_args(
    config: Dict[str, Any],
    args: argparse.Namespace,
    parser: Optional[argparse.ArgumentParser] = None,
    ignore_unknown: bool = True
) -> argparse.Namespace:
    """
    Merge YAML config with CLI arguments. CLI args take precedence.
    
    Args:
        config: Dictionary from load_config()
        args: Parsed argparse Namespace
        parser: Optional ArgumentParser to detect defaults (if not provided, 
                uses heuristic comparison with common default values)
        ignore_unknown: If True, skip config keys not in args
    
    Returns:
        Updated argparse Namespace
    
    Note:
        CLI arguments (non-default values) always override config values.
        This allows: `--config base.yaml --lr 5e-4` to use config but override LR.
    """
    # Get parser defaults to detect which args were explicitly set by user
    if parser is not None:
        defaults = vars(parser.parse_args([]))
    else:
        # Fallback: reconstruct defaults from known patterns
        # This works because argparse stores actual values, and we compare
        defaults = {}
    
    # Track which args were explicitly set on CLI (differ from defaults)
    cli_overrides = set()
    for key, value in vars(args).items():
        if parser is not None:
            if key in defaults and value != defaults[key]:
                cli_overrides.add(key)
        # Without parser, we can't reliably detect CLI overrides
        # So we apply all config values (legacy behavior)
    
    # Apply config values only where CLI didn't override
    for key, value in config.items():
        if hasattr(args, key):
            # Skip if user explicitly set this via CLI
            if key in cli_overrides:
                logging.debug(f"Config key '{key}' skipped: CLI override detected")
                continue
            setattr(args, key, value)
        elif not ignore_unknown:
            logging.warning(f"Unknown config key: {key}")
    
    return args


def save_config(
    args: argparse.Namespace,
    output_path: str,
    exclude_keys: Optional[List[str]] = None
) -> str:
    """
    Save effective configuration to YAML for reproducibility.
    
    Args:
        args: Parsed argparse Namespace
        output_path: Path to save YAML file
        exclude_keys: Keys to exclude from saved config
    
    Returns:
        Path to saved config file
    
    Example:
        >>> save_config(args, "output/effective_config.yaml")
    """
    if exclude_keys is None:
        exclude_keys = ['list_models', 'fresh', 'resume']
    
    config = {}
    for key, value in vars(args).items():
        if key not in exclude_keys:
            # Convert Path objects to strings
            if isinstance(value, Path):
                value = str(value)
            config[key] = value
    
    # Add metadata
    config['_metadata'] = {
        'saved_at': datetime.now().isoformat(),
        'wavedl_version': '1.0.0',
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return str(output_path)


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration values against known options.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        List of warning messages (empty if valid)
    """
    warnings = []
    
    # Known valid options
    from utils import list_losses, list_optimizers, list_schedulers
    from models import list_models
    
    valid_options = {
        'model': list_models(),
        'loss': list_losses(),
        'optimizer': list_optimizers(),
        'scheduler': list_schedulers(),
    }
    
    for key, valid_values in valid_options.items():
        if key in config and config[key] not in valid_values:
            warnings.append(
                f"Invalid {key}='{config[key]}'. Valid options: {valid_values}"
            )
    
    # Validate numeric ranges
    numeric_checks = {
        'lr': (0, 1, "Learning rate should be between 0 and 1"),
        'epochs': (1, 100000, "Epochs should be positive"),
        'batch_size': (1, 10000, "Batch size should be positive"),
        'patience': (1, 1000, "Patience should be positive"),
        'cv': (0, 100, "CV folds should be 0-100"),
    }
    
    for key, (min_val, max_val, msg) in numeric_checks.items():
        if key in config:
            val = config[key]
            if not (min_val <= val <= max_val):
                warnings.append(f"{msg}: got {val}")
    
    return warnings


def create_default_config() -> Dict[str, Any]:
    """
    Create a default configuration dictionary.
    
    Returns:
        Dictionary with default training configuration
    """
    return {
        # Model
        'model': 'cnn',
        
        # Hyperparameters
        'batch_size': 128,
        'lr': 1e-3,
        'epochs': 1000,
        'patience': 20,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        
        # Training components
        'loss': 'mse',
        'optimizer': 'adamw',
        'scheduler': 'plateau',
        
        # Cross-validation
        'cv': 0,
        'cv_stratify': False,
        'cv_bins': 10,
        
        # Performance
        'precision': 'bf16',
        'compile': False,
        
        # Output
        'seed': 2025,
        'workers': 8,
    }
