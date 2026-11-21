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
import json
import logging
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


class _WaveDLSafeLoader(yaml.SafeLoader):
    """YAML SafeLoader supporting YAML 1.2 scientific notation (e.g., 1e-4)."""


_WaveDLSafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:
         [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?\.[0-9][0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)

# Known nested YAML section prefixes allowed to map to flat CLI argument names.
_KNOWN_CONFIG_SECTIONS: tuple[str, ...] = (
    "cross_validation",
    "hyperparameters",
    "checkpointing",
    "constraints",
    "performance",
    "optimizer",
    "scheduler",
    "training",
    "logging",
    "output",
    "model",
    "wandb",
    "loss",
    "data",
    "cv",
)


def _strip_known_section_prefix(key: str) -> str | None:
    """Strip a recognized nested YAML section prefix (if any) from ``key``."""
    for section in _KNOWN_CONFIG_SECTIONS:
        prefix = f"{section}_"
        if key.startswith(prefix) and len(key) > len(prefix):
            return key[len(prefix) :]
    return None


def _normalize_yaml_bool_choices(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize YAML 1.1 boolean ``no`` (parsed as ``False``) for precision keys."""
    normalized = dict(config)
    for key, value in list(normalized.items()):
        stripped = _strip_known_section_prefix(key) or key
        if stripped in {"precision", "mixed_precision"} and value is False:
            normalized[key] = "no"
    return normalized


def parse_model_kwargs(value: str) -> dict[str, Any]:
    """Parse the CLI architecture overrides; YAML supplies a mapping directly."""
    try:
        overrides = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"model_kwargs must be a JSON object: {exc}"
        ) from exc
    if not isinstance(overrides, dict):
        raise argparse.ArgumentTypeError("model_kwargs must be a JSON object.")
    return overrides


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
        config = yaml.load(f, Loader=_WaveDLSafeLoader)

    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError(
            "Configuration must be a YAML mapping of option names to values."
        )

    # Handle nested configs (e.g., optimizer.lr -> optimizer_lr)
    config = _flatten_config(config)
    config = _normalize_yaml_bool_choices(config)

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
        if isinstance(value, dict) and "model_kwargs" not in (
            new_key,
            _strip_known_section_prefix(new_key),
        ):
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


_DEFAULT_KEY_ALIASES: dict[str, str] = {
    "import": "import_modules",
    "mixed_precision": "precision",
}


def _build_key_aliases(parser: argparse.ArgumentParser | None) -> dict[str, str]:
    """Build option-name -> argparse dest alias mapping."""
    aliases = dict(_DEFAULT_KEY_ALIASES)
    if parser is not None:
        for action in parser._actions:
            if action.dest == "help":
                continue
            for opt in action.option_strings:
                opt_name = opt.lstrip("-").replace("-", "_")
                if opt_name and opt_name != action.dest:
                    aliases[opt_name] = action.dest
    return aliases


def _is_sequence_action(action: argparse.Action) -> bool:
    """Return True only for multi-value argparse actions (not nargs='?')."""
    if isinstance(action, argparse._AppendAction):
        return True
    if action.nargs in ("+", "*"):
        return True
    return isinstance(action.nargs, int) and action.nargs > 1


def _resolve_effective_entries(
    config: dict[str, Any],
    valid_dests: set[str],
    aliases: dict[str, str],
) -> tuple[dict[str, tuple[str, Any]], list[str]]:
    """Resolve config keys with consistent flat-over-nested precedence.

    Returns:
        Tuple of (resolved_map, unknown_keys), where ``resolved_map`` maps
        canonical destination name -> ``(original_key, value)``.
    """
    resolved: dict[str, tuple[str, Any]] = {}
    nested_candidates: list[tuple[str, str, Any]] = []
    unknown: list[str] = []

    # Pass 1: Direct / flat keys (including CLI option aliases like mixed_precision)
    for key, value in config.items():
        canonical = aliases.get(key, key)
        if canonical in valid_dests:
            resolved[canonical] = (key, value)
            continue
        stripped = _strip_known_section_prefix(key)
        if stripped is not None:
            stripped_canonical = aliases.get(stripped, stripped)
            if stripped_canonical in valid_dests:
                nested_candidates.append((stripped_canonical, key, value))
                continue
        unknown.append(key)

    # Pass 2: Nested section-prefixed keys only apply when no flat key set the target
    for target, orig_key, value in nested_candidates:
        if target not in resolved:
            resolved[target] = (orig_key, value)

    return resolved, unknown


def merge_config_with_args(
    config: dict[str, Any],
    args: argparse.Namespace,
    parser: argparse.ArgumentParser | None = None,
    ignore_unknown: bool = True,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """
    Merge YAML config with CLI arguments. CLI args take precedence.

    Args:
        config: Dictionary from load_config()
        args: Parsed argparse Namespace
        parser: Optional ArgumentParser to detect defaults (if not provided,
                uses heuristic comparison with common default values)
        ignore_unknown: If True, skip config keys not in args
        argv: Optional explicit CLI token list (defaults to ``sys.argv[1:]``)

    Returns:
        Updated argparse Namespace

    Note:
        CLI arguments (non-default values) always override config values.
        This allows: `--config base.yaml --lr 5e-4` to use config but override LR.
    """
    # Detect which args were explicitly passed on the command line.
    # We cannot simply compare values against parser defaults because a user
    # who explicitly passes the default (e.g., --workers -1) would be silently
    # overwritten by YAML.  Instead, inspect argv for the actual flags.
    cli_overrides: set[str] = set()
    if parser is not None:
        import sys

        raw_argv = sys.argv[1:] if argv is None else list(argv)
        # Check both '--flag value' and '--flag=value' forms in argv,
        # stopping at '--' (end-of-options marker).
        argv_tokens: list[str] = []
        for tok in raw_argv:
            if tok == "--":
                break
            argv_tokens.append(tok)
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
    arg_choices: dict[str, Any] = {}
    sequence_args: set[str] = set()
    if parser is not None:
        for action in parser._actions:
            if action.type is not None:
                arg_types[action.dest] = action.type
            if action.choices is not None:
                arg_choices[action.dest] = action.choices
            if _is_sequence_action(action):
                sequence_args.add(action.dest)

    config = _normalize_yaml_bool_choices(config)
    valid_dests = set(vars(args).keys())
    aliases = _build_key_aliases(parser)
    resolved, unknown_keys = _resolve_effective_entries(config, valid_dests, aliases)

    # Apply config values only where CLI didn't override
    for target_key, (orig_key, value) in resolved.items():
        if orig_key != target_key:
            logging.debug(f"Config: mapped key '{orig_key}' -> '{target_key}'")
        # Skip if user explicitly set this via CLI
        if target_key in cli_overrides:
            logging.debug(f"Config key '{target_key}' skipped: CLI override detected")
            continue
        if (
            value is False
            and target_key in arg_choices
            and "no" in arg_choices[target_key]
        ):
            value = "no"
        # Wrap scalar YAML values for multi-value list arguments (e.g.
        # constraint: "y0 > 0" -> ["y0 > 0"]).
        if (
            target_key in sequence_args
            and value is not None
            and not isinstance(value, (list, tuple))
        ):
            value = [value]
        elif target_key in sequence_args and isinstance(value, tuple):
            value = list(value)
        # Coerce native YAML sequences (e.g. betas: [0.9, 0.999]) or numeric
        # scalars (e.g. milestones: 30) to the string form expected by str args.
        expects_str = arg_types.get(target_key) is str or isinstance(
            getattr(args, target_key, None), str
        )
        if target_key not in sequence_args and expects_str:
            if isinstance(value, (list, tuple)):
                value = ",".join(str(v) for v in value)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                value = str(value)
        setattr(args, target_key, value)

    for key in unknown_keys:
        if not ignore_unknown:
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

    def _sanitize_yaml_value(val: Any) -> Any:
        if isinstance(val, Path):
            return str(val)
        if isinstance(val, tuple):
            return [_sanitize_yaml_value(item) for item in val]
        if isinstance(val, list):
            return [_sanitize_yaml_value(item) for item in val]
        if isinstance(val, dict):
            return {str(k): _sanitize_yaml_value(v) for k, v in val.items()}
        if hasattr(val, "item") and callable(val.item):
            try:
                return val.item()
            except Exception:
                return str(val)
        return val

    config = {}
    for key, value in vars(args).items():
        if key not in exclude_keys and not key.startswith("_"):
            config[key] = _sanitize_yaml_value(value)

    # Add metadata
    from wavedl import __version__

    config["_metadata"] = {
        "saved_at": datetime.now().isoformat(),
        "wavedl_version": __version__,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    return str(output_path)


def _config_keys(
    known_keys: list[str] | None, parser: argparse.ArgumentParser | None
) -> set[str]:
    """Resolve accepted flat keys before interpreting nested section prefixes."""
    internal_keys = {"_metadata", "_explicit_scheduler_factor"}
    if known_keys is not None:
        return set(known_keys) | internal_keys
    if parser is not None:
        keys = {action.dest for action in parser._actions if action.dest != "help"}
        for action in parser._actions:
            keys.update(
                opt.lstrip("-").replace("-", "_") for opt in action.option_strings
            )
        return keys | internal_keys
    return {
        "model",
        "model_kwargs",
        "import",
        "import_modules",
        "pretrained",
        "batch_size",
        "lr",
        "epochs",
        "patience",
        "weight_decay",
        "grad_clip",
        "grad_accum_steps",
        "loss",
        "huber_delta",
        "loss_weights",
        "selection_metric",
        "optimizer",
        "momentum",
        "nesterov",
        "betas",
        "scheduler",
        "scheduler_patience",
        "min_lr",
        "scheduler_factor",
        "warmup_epochs",
        "step_size",
        "milestones",
        "data_path",
        "val_data_path",
        "cache_dir",
        "workers",
        "seed",
        "deterministic",
        "cache_validate",
        "single_channel",
        "input_channels",
        "input_transform",
        "input_range",
        "target_scaling",
        "target_range",
        "allow_pickle",
        "cv",
        "cv_stratify",
        "cv_bins",
        "resume",
        "save_every",
        "output_dir",
        "fresh",
        "compile",
        "precision",
        "mixed_precision",
        "no_tf32",
        "wandb",
        "wandb_watch",
        "project_name",
        "run_name",
        "config",
        "list_models",
        "constraint",
        "constraint_file",
        "constraint_weight",
        "constraint_reduction",
        *internal_keys,
    }


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
    config = _normalize_yaml_bool_choices(config)

    # Known valid options
    from wavedl.models import list_models
    from wavedl.utils import list_losses, list_optimizers, list_schedulers

    valid_options = {
        "model": list_models(),
        "loss": [*list_losses(), "l1", "logcosh"],
        "optimizer": list_optimizers(),
        "scheduler": list_schedulers(),
    }

    # Build effective view mapping aliases and recognized nested section keys
    # with the exact same flat-over-nested precedence as merge_config_with_args.
    check_keys = _config_keys(known_keys, parser)
    aliases = _build_key_aliases(parser)
    resolved_entries, unknown_keys = _resolve_effective_entries(
        config, check_keys, aliases
    )
    effective_config: dict[str, Any] = {
        dest: val for dest, (_, val) in resolved_entries.items()
    }

    for key, valid_values in valid_options.items():
        if key in effective_config and effective_config[key] not in valid_values:
            warnings.append(
                f"Invalid {key}='{effective_config[key]}'. Valid options: {valid_values}"
            )

    if "model_kwargs" in effective_config:
        overrides = effective_config["model_kwargs"]
        if not isinstance(overrides, dict) or any(
            not isinstance(key, str) for key in overrides
        ):
            warnings.append("model_kwargs must be a mapping with string keys.")

    # Validate argparse choice-constrained options (precision, mixed_precision,
    # cache_validate, constraint_reduction, ...). merge bypasses argparse, so
    # these `choices=` constraints are otherwise never enforced.
    comma_list_str_args = {"betas", "milestones", "loss_weights"}
    if parser is not None:
        for action in parser._actions:
            if (
                action.dest not in effective_config
                or effective_config[action.dest] is None
            ):
                continue
            value = effective_config[action.dest]
            if value is False and action.choices and "no" in action.choices:
                value = "no"
            if (
                action.choices
                and action.dest not in valid_options
                and value not in action.choices
            ):
                warnings.append(
                    f"Invalid {action.dest}='{value}'. "
                    f"Valid options: {list(action.choices)}"
                )
            values = value if isinstance(value, (list, tuple)) else [value]
            if isinstance(action.nargs, int) and action.nargs > 1:
                if not isinstance(value, (list, tuple)) or len(value) != action.nargs:
                    warnings.append(
                        f"'{action.dest}' requires exactly {action.nargs} values, "
                        f"got {value!r}"
                    )
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
                not _is_sequence_action(action)
                and action.dest not in comma_list_str_args
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
        "warmup_epochs": (0, math.inf, "Warmup epochs must be non-negative"),
        "step_size": (1, math.inf, "Scheduler step size must be positive"),
        "save_every": (0, math.inf, "Save interval must be non-negative (0 disables)"),
        "seed": (0, 2**32 - 1, "Seed must fit an unsigned 32-bit integer"),
        "workers": (-1, math.inf, "Workers must be -1 (automatic) or non-negative"),
        "cv_bins": (2, math.inf, "CV bins must be at least 2"),
        "momentum": (0, 1, "Momentum must be between 0 and 1"),
    }

    for key, (min_val, max_val, msg) in numeric_checks.items():
        if key in effective_config and effective_config[key] is not None:
            val = effective_config[key]
            # Type check: ensure value is numeric before comparison
            if type(val) not in (int, float):
                warnings.append(
                    f"Invalid type for '{key}': expected number, got {type(val).__name__} ({val!r})"
                )
                continue
            if not math.isfinite(val) or not (min_val <= val <= max_val):
                warnings.append(f"{msg}: got {val}")

    if effective_config.get("cv") == 1:
        warnings.append("CV must be 0 (disabled) or at least 2 folds.")

    # Check for unknown/unrecognized keys (helps catch typos).
    for key in unknown_keys:
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
        "workers": -1,
    }
