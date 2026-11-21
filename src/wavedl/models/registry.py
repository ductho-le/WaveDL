"""
Model Registry System
=====================

Provides the core model registration and factory functionality.
This module has no dependencies on other model modules to prevent circular imports.

Author: Ductho Le (ductho.le@outlook.com)
"""

from collections.abc import Mapping
from inspect import Parameter, signature
from numbers import Integral
from typing import Any

import torch.nn as nn


# ==============================================================================
# GLOBAL MODEL REGISTRY
# ==============================================================================
MODEL_REGISTRY: dict[str, type[nn.Module]] = {}


def register_model(name: str):
    """
    Decorator to register a model class in the global registry.

    Args:
        name: Unique identifier for the model (lowercase, used in CLI)

    Example:
        @register_model("resnet50")
        class ResNet50(BaseModel):
            ...

    Raises:
        ValueError: If model name is empty or already registered
        TypeError: If decorated object is not a ``torch.nn.Module`` subclass
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Model name must be a non-empty string.")

    def decorator(cls: type[nn.Module]) -> type[nn.Module]:
        if not isinstance(cls, type) or not issubclass(cls, nn.Module):
            raise TypeError(
                f"Registered model '{name}' must be a torch.nn.Module subclass, got {cls!r}."
            )
        name_lower = name.strip().lower()
        if name_lower in MODEL_REGISTRY and MODEL_REGISTRY[name_lower] is not cls:
            raise ValueError(
                f"Model '{name}' is already registered. "
                f"Choose a unique name or check for duplicate imports."
            )
        MODEL_REGISTRY[name_lower] = cls
        return cls

    return decorator


def get_model(name: str) -> type[nn.Module]:
    """
    Retrieve a model class from the registry by name.

    Args:
        name: Registered model name (case-insensitive)

    Returns:
        The model class (not instantiated)

    Raises:
        ValueError: If model name is not found in registry
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Model name must be a non-empty string.")
    name_lower = name.strip().lower()
    if name_lower not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys())) or "none"
        raise ValueError(f"Model '{name}' not found. Available models: [{available}]")
    return MODEL_REGISTRY[name_lower]


def list_models() -> list[str]:
    """
    List all registered model names.

    Returns:
        Sorted list of model names
    """
    return sorted(MODEL_REGISTRY.keys())


# These are controlled by dataset geometry or by the dedicated --pretrained flag.
_RESERVED_MODEL_KWARGS = {
    "in_shape",
    "out_size",
    "out_shape",
    "in_channels",
    "pretrained",
}


def validate_model_kwargs(
    name: str, overrides: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Validate architecture overrides separately from pipeline-owned arguments."""
    if overrides is None:
        return {}
    if not isinstance(overrides, Mapping) or any(
        not isinstance(key, str) for key in overrides
    ):
        raise ValueError("model_kwargs must be a mapping with string keys.")
    if not overrides:
        return {}

    model_class = get_model(name)
    get_defaults = getattr(model_class, "get_default_config", None)
    defaults = get_defaults() if callable(get_defaults) else {}
    allowed = (
        set(defaults)
        if defaults
        else {
            key
            for key, param in signature(model_class).parameters.items()
            if param.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
            and not key.startswith("_")
        }
    )
    allowed -= _RESERVED_MODEL_KWARGS
    unknown = sorted(overrides.keys() - allowed)
    if unknown:
        raise ValueError(
            f"Unknown model_kwargs for model '{name}': {', '.join(unknown)}. "
            f"Allowed keys: {', '.join(sorted(allowed)) or 'none'}. "
            "Input/output shapes and channels are set from data; use --pretrained for weight initialization."
        )
    return dict(overrides)


def build_model(
    name: str, in_shape: tuple[int, ...], out_size: int, **kwargs
) -> nn.Module:
    """
    Factory function to instantiate a model by name.

    Args:
        name: Registered model name
        in_shape: Input spatial dimensions, excluding batch and channel dims:
                  - 1D: (L,) for signal length
                  - 2D: (H, W) for image dimensions
                  - 3D: (D, H, W) for volume dimensions
        out_size: Number of vector targets or dense output channels. Dense
                  models receive their canonical ``out_shape`` through kwargs.
        **kwargs: Additional model-specific parameters, including optional
                  ``in_channels`` (default 1), validated against
                  ``supported_input_channels`` before construction.

    Returns:
        Instantiated model

    Example:
        model = build_model("cnn", in_shape=(500, 500), out_size=5)  # 2D
        model = build_model("cnn", in_shape=(1024,), out_size=3)     # 1D
        model = build_model("cnn", in_shape=(64, 128, 128), out_size=5) # 3D
    """
    ModelClass = get_model(name)
    if "in_channels" in kwargs:
        in_channels = kwargs["in_channels"]
        if (
            not isinstance(in_channels, Integral)
            or isinstance(in_channels, bool)
            or in_channels <= 0
        ):
            raise ValueError(
                f"in_channels must be a positive integer, got {in_channels!r}."
            )
        in_channels = int(in_channels)
        kwargs["in_channels"] = in_channels
        supported = getattr(ModelClass, "supported_input_channels", (1,))
        if supported is not None and in_channels not in supported:
            raise ValueError(
                f"Model '{name}' does not support in_channels={in_channels}; "
                f"supported_input_channels={supported}."
            )
    return ModelClass(in_shape=in_shape, out_size=out_size, **kwargs)


def import_model_modules(modules: list[str]) -> None:
    """Import explicitly trusted custom modules, idempotently and without name collisions.

    Python model definitions are executable code and must come from a trusted source.
    Checkpoints never trigger these imports automatically.
    """
    import hashlib
    import importlib
    import importlib.util
    import sys
    from pathlib import Path

    for name in modules:
        if not name.endswith(".py"):
            prior_registry = MODEL_REGISTRY.copy()
            already_loaded = name in sys.modules
            try:
                importlib.import_module(name)
            except BaseException:
                if not already_loaded:
                    sys.modules.pop(name, None)
                MODEL_REGISTRY.clear()
                MODEL_REGISTRY.update(prior_registry)
                raise
            continue
        path = Path(name).resolve(strict=True)
        module_name = (
            "wavedl_custom_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
        )
        if module_name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load custom model module from {path}")
        module = importlib.util.module_from_spec(spec)
        prior_registry = MODEL_REGISTRY.copy()
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            MODEL_REGISTRY.clear()
            MODEL_REGISTRY.update(prior_registry)
            raise
