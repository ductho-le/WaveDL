"""Shared YAML loading and argument merging without ML imports.

PyYAML is loaded only when reading a configuration file. Runtime-only registry
validation and configuration saving remain in wavedl.utils.config.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Sequence


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

    import yaml

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
    *,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """
    Merge YAML config with CLI arguments. CLI args take precedence.

    Args:
        config: Dictionary from load_config()
        args: Parsed argparse Namespace
        parser: Optional ArgumentParser to detect defaults (if not provided,
                uses heuristic comparison with common default values)
        ignore_unknown: If True, skip config keys not in args
        argv: Explicit CLI tokens; defaults to sys.argv for runtime compatibility

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
        argv_tokens = sys.argv if argv is None else argv
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
            if (
                action.nargs in ("+", "*")
                or isinstance(action.nargs, int)
                or isinstance(action, argparse._AppendAction)
            ):
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
                isinstance(value, (list, tuple))
                and target_key not in sequence_args
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
