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
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
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
        >>> print(config["model"])
        'cnn'
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError(
            "Configuration must be a YAML mapping of option names to values."
        )

    # Handle nested configs (e.g., optimizer.lr -> optimizer_lr)
    config = _flatten_config(config)

    return config


def _flatten_config(
    config: dict[str, Any], parent_key: str = "", sep: str = "_"
) -> dict[str, Any]:
    """
    Flatten nested dictionaries for argparse compatibility.

    Recursively flattens nested dicts, preserving the full key path.

    Example:
        {'optimizer': {'lr': 1e-3}} -> {'optimizer_lr': 1e-3}
        {'optimizer': {'params': {'beta1': 0.9}}} -> {'optimizer_params_beta1': 0.9}
        {'lr': 1e-3} -> {'lr': 1e-3}
    """
    items: list[tuple[str, Any]] = []
    for key, value in config.items():
        if not isinstance(key, str):
            raise ValueError("Configuration option names must be strings.")
        if key == "_metadata":
            continue
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            # Recursively flatten, passing full accumulated key path
            items.extend(_flatten_config(value, new_key, sep).items())
        else:
            items.append((new_key, value))
    flattened = {}
    for key, value in items:
        if key in flattened:
            raise ValueError(f"Conflicting configuration key after flattening: {key}")
        flattened[key] = value
    return flattened


def merge_config_with_args(
    config: dict[str, Any],
    args: argparse.Namespace,
    parser: argparse.ArgumentParser | None = None,
    ignore_unknown: bool = True,
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
    # Detect which args were explicitly passed on the command line.
    # We cannot simply compare values against parser defaults because a user
    # who explicitly passes the default (e.g., --workers -1) would be silently
    # overwritten by YAML.  Instead, inspect sys.argv for the actual flags.
    cli_overrides: set[str] = set()
    if parser is not None:
        import sys

        # Check both '--flag value' and '--flag=value' forms in sys.argv.
        # The latter is a single token (e.g., '--workers=-1') so a plain
        # set-membership check would miss it.
        argv_tokens = sys.argv
        for action in parser._option_string_actions.values():
            if action.dest == "help":
                continue
            for flag in action.option_strings:
                if any(
                    tok == flag or tok.startswith(flag + "=") for tok in argv_tokens
                ):
                    cli_overrides.add(action.dest)
                    break

    # Map argparse dest -> declared type, used to coerce native YAML sequences
    # into the comma-separated string form expected by str-typed list args
    # (betas/milestones/loss_weights). merge bypasses argparse type casting.
    arg_types: dict[str, Any] = {}
    sequence_args: set[str] = set()
    if parser is not None:
        for action in parser._actions:
            if action.type is not None:
                arg_types[action.dest] = action.type
            if action.nargs or isinstance(action, argparse._AppendAction):
                sequence_args.add(action.dest)

    # Apply config values only where CLI didn't override
    for key, value in config.items():
        target_key = key
        if not hasattr(args, key) and "_" in key:
            # Nested YAML key (e.g., "optimizer_lr" from {optimizer: {lr: ...}}).
            # Try removing the namespace prefix to match the argparse destination.
            stripped = key.split("_", 1)[1]
            if hasattr(args, stripped):
                target_key = stripped
                logging.warning(
                    f"Config: key '{key}' did not match an argument directly; "
                    f"mapped to '{stripped}' by stripping the prefix. "
                    "Verify this is intended (not a typo)."
                )

        if hasattr(args, target_key):
            # Skip if user explicitly set this via CLI
            if target_key in cli_overrides:
                logging.debug(
                    f"Config key '{target_key}' skipped: CLI override detected"
                )
                continue
            # Coerce native YAML sequences (e.g. betas: [0.9, 0.999]) to the
            # comma-separated string the str-typed arg + later .split(",") expects.
            if (
                target_key not in sequence_args
                and isinstance(value, (list, tuple))
                and (
                    arg_types.get(target_key) is str
                    or isinstance(getattr(args, target_key, None), str)
                )
            ):
                value = ",".join(str(v) for v in value)
            setattr(args, target_key, value)
        elif not ignore_unknown:
            logging.warning(f"Unknown config key: {key}")
        else:
            # Even in ignore_unknown mode, log for discoverability
            logging.debug(f"Config key '{key}' ignored: not a valid argument")

    return args


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
        "patience": (1, 1000, "Patience should be positive"),
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
