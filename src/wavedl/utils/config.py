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

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from wavedl.core.configuration import (
    _flatten_config as _flatten_config,
    load_config as load_config,
    merge_config_with_args as merge_config_with_args,
)


def save_config(
    args: argparse.Namespace,
    output_path: str | Path,
    exclude_keys: list[str] | None = None,
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
        exclude_keys = ["list_models", "fresh", "resume"]

    config = {}
    for key, value in vars(args).items():
        if key not in exclude_keys:
            # Convert Path objects to strings
            if isinstance(value, Path):
                value = str(value)
            config[key] = value

    # Add metadata
    from wavedl import __version__

    config["_metadata"] = {
        "saved_at": datetime.now().isoformat(),
        "wavedl_version": __version__,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return str(output_path)


def validate_config(
    config: dict[str, Any],
    known_keys: list[str] | None = None,
    parser: "argparse.ArgumentParser | None" = None,
) -> list[str]:
    """
    Validate configuration values against known options.

    Args:
        config: Configuration dictionary
        known_keys: Optional list of valid keys (if None, derives from parser or uses defaults)
        parser: Optional ArgumentParser to derive valid keys from (most reliable)

    Returns:
        List of warning messages (empty if valid)
    """
    warnings = []

    # Known valid options
    from wavedl.models import list_models
    from wavedl.utils import list_losses, list_optimizers, list_schedulers

    valid_options = {
        "model": list_models(),
        "loss": [*list_losses(), "l1", "logcosh"],
        "optimizer": list_optimizers(),
        "scheduler": list_schedulers(),
    }

    for key, valid_values in valid_options.items():
        if key in config and config[key] not in valid_values:
            warnings.append(
                f"Invalid {key}='{config[key]}'. Valid options: {valid_values}"
            )

    # Validate argparse choice-constrained options (precision, mixed_precision,
    # cache_validate, constraint_reduction, ...). merge bypasses argparse, so
    # these `choices=` constraints are otherwise never enforced.
    if parser is not None:
        for action in parser._actions:
            if action.choices and action.dest in config:
                if config[action.dest] not in action.choices:
                    warnings.append(
                        f"Invalid {action.dest}='{config[action.dest]}'. "
                        f"Valid options: {list(action.choices)}"
                    )
            if action.dest not in config or config[action.dest] is None:
                continue
            value = config[action.dest]
            values = value if isinstance(value, (list, tuple)) else [value]
            if action.type in (int, float):
                for entry in values:
                    valid_type = (
                        type(entry) is int
                        if action.type is int
                        else type(entry) in (int, float)
                    )
                    if not valid_type or not math.isfinite(entry):
                        type_name = getattr(action.type, "__name__", str(action.type))
                        warnings.append(
                            f"Invalid type/value for '{action.dest}': expected a finite "
                            f"{type_name}, got {entry!r}"
                        )
            elif isinstance(
                action, (argparse._StoreTrueAction, argparse._StoreFalseAction)
            ):
                if type(value) is not bool:
                    warnings.append(
                        f"Invalid type for '{action.dest}': expected a YAML boolean, got {value!r}"
                    )
            if (
                not action.nargs
                and not isinstance(action, argparse._AppendAction)
                and isinstance(value, (list, tuple))
            ):
                warnings.append(
                    f"'{action.dest}' requires a single value, got {value!r}"
                )

    # Validate numeric ranges
    numeric_checks = {
        "lr": (1e-12, 10, "Learning rate must be > 0 and <= 10"),
        "epochs": (1, 100000, "Epochs should be positive"),
        "batch_size": (1, 10000, "Batch size should be positive"),
        # Unbounded: a very large patience is how runs disable early stopping.
        "patience": (1, math.inf, "Patience must be positive"),
        "cv": (0, 100, "CV folds should be 0-100"),
        "grad_accum_steps": (1, 256, "Gradient accumulation steps should be 1-256"),
        "weight_decay": (0, math.inf, "Weight decay must be non-negative"),
        "grad_clip": (0, math.inf, "Gradient clipping must be non-negative"),
        "huber_delta": (1e-300, math.inf, "Huber delta must be positive"),
        "min_lr": (0, math.inf, "Minimum learning rate must be non-negative"),
        "scheduler_patience": (0, math.inf, "Scheduler patience must be non-negative"),
        "scheduler_factor": (
            1e-300,
            1 - 1e-15,
            "Scheduler factor must be between 0 and 1",
        ),
        "warmup_epochs": (1, math.inf, "Warmup epochs must be positive"),
        "step_size": (1, math.inf, "Scheduler step size must be positive"),
        "save_every": (0, math.inf, "Save interval must be non-negative (0 disables)"),
        "seed": (0, 2**32 - 1, "Seed must fit an unsigned 32-bit integer"),
        "workers": (-1, math.inf, "Workers must be -1 (automatic) or non-negative"),
        "cv_bins": (2, math.inf, "CV bins must be at least 2"),
        "momentum": (0, 1, "Momentum must be between 0 and 1"),
    }

    for key, (min_val, max_val, msg) in numeric_checks.items():
        if key in config:
            val = config[key]
            # Type check: ensure value is numeric before comparison
            if type(val) not in (int, float):
                warnings.append(
                    f"Invalid type for '{key}': expected number, got {type(val).__name__} ({val!r})"
                )
                continue
            if not math.isfinite(val) or not (min_val <= val <= max_val):
                warnings.append(f"{msg}: got {val}")

    if config.get("cv") == 1:
        warnings.append("CV must be 0 (disabled) or at least 2 folds.")

    # Check for unknown/unrecognized keys (helps catch typos)
    # Priority: explicit known_keys > parser-derived > hardcoded fallback
    if known_keys is not None:
        check_keys = set(known_keys)
    elif parser is not None:
        # Derive from parser: always in sync, never lags behind new arguments
        check_keys = {
            action.dest for action in parser._actions if action.dest != "help"
        }
        # Also accept the raw flag names without leading dashes
        # (e.g., YAML may use 'pretrained' or 'mixed_precision')
        for action in parser._actions:
            for opt in action.option_strings:
                check_keys.add(opt.lstrip("-").replace("-", "_"))
        check_keys.add("_metadata")  # Internal metadata key
    else:
        # Hardcoded fallback (used when neither parser nor known_keys provided)
        check_keys = {
            # Model
            "model",
            "import_modules",
            "pretrained",
            # Hyperparameters
            "batch_size",
            "lr",
            "epochs",
            "patience",
            "weight_decay",
            "grad_clip",
            "grad_accum_steps",
            # Loss
            "loss",
            "huber_delta",
            "loss_weights",
            "selection_metric",
            # Optimizer
            "optimizer",
            "momentum",
            "nesterov",
            "betas",
            # Scheduler
            "scheduler",
            "scheduler_patience",
            "min_lr",
            "scheduler_factor",
            "warmup_epochs",
            "step_size",
            "milestones",
            # Data
            "data_path",
            "workers",
            "seed",
            "deterministic",
            "cache_validate",
            "single_channel",
            "allow_pickle",
            # Cross-validation
            "cv",
            "cv_stratify",
            "cv_bins",
            # Checkpointing
            "resume",
            "save_every",
            "output_dir",
            "fresh",
            # Performance
            "compile",
            "precision",
            "mixed_precision",
            # Logging
            "wandb",
            "wandb_watch",
            "project_name",
            "run_name",
            # Config
            "config",
            "list_models",
            # Physical Constraints
            "constraint",
            "constraint_file",
            "constraint_weight",
            "constraint_reduction",
            # Metadata (internal)
            "_metadata",
        }

    for key in config:
        if key not in check_keys:
            # Try stripping namespace prefix for nested YAML keys
            # (e.g., "optimizer_lr" from {optimizer: {lr: ...}} → check "lr")
            if "_" in key:
                stripped = key.split("_", 1)[1]
                if stripped in check_keys:
                    continue
            warnings.append(
                f"Unknown config key: '{key}' - check for typos or see wavedl-train --help"
            )

    return warnings


def create_default_config() -> dict[str, Any]:
    """
    Create a default configuration dictionary.

    Returns:
        Dictionary with default training configuration
    """
    return {
        # Model
        "model": "cnn",
        # Hyperparameters
        "batch_size": 128,
        "lr": 1e-3,
        "epochs": 1000,
        "patience": 20,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "grad_accum_steps": 1,
        # Training components
        "loss": "mse",
        "optimizer": "adamw",
        "scheduler": "plateau",
        # Cross-validation
        "cv": 0,
        "cv_stratify": False,
        "cv_bins": 10,
        # Performance
        "precision": "bf16",
        "compile": False,
        # Output
        "seed": 2025,
        "workers": 8,
    }
