"""
Model Registry System
=====================

Provides the core model registration and factory functionality.
This module has no dependencies on other model modules to prevent circular imports.

Author: Ductho Le (ductho.le@outlook.com)
"""

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
        class ResNet50(nn.Module):
            ...

    Raises:
        ValueError: If a model with the same name is already registered
    """

    def decorator(cls: type[nn.Module]) -> type[nn.Module]:
        name_lower = name.lower()
        if name_lower in MODEL_REGISTRY:
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
    name_lower = name.lower()
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
        **kwargs: Additional model-specific parameters

    Returns:
        Instantiated model

    Example:
        model = build_model("cnn", in_shape=(500, 500), out_size=5)  # 2D
        model = build_model("cnn1d", in_shape=(1024,), out_size=3)       # 1D
        model = build_model("cnn3d", in_shape=(64, 128, 128), out_size=5) # 3D
    """
    ModelClass = get_model(name)
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
            importlib.import_module(name)
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
