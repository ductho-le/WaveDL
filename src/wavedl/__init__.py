"""
WaveDL - Deep Learning Framework for Wave-Based Inverse Problems
=================================================================

A scalable deep learning framework for wave-based inverse problems,
from ultrasonic NDE and geophysics to biomedical tissue characterization.

Quick Start:
    from wavedl.models import build_model, list_models
    from wavedl.utils import prepare_data, load_test_data

For training:
    wavedl-train --model cnn --data_path train.npz
    wavedl-train --model unet_field --data_path field_train.npz
    # or: python -m wavedl.train --model cnn --data_path train.npz

For inference:
    wavedl-test --checkpoint best_checkpoint --data_path test.npz
    # or: python -m wavedl.test --checkpoint best_checkpoint --data_path test.npz
"""

from typing import TYPE_CHECKING


__version__ = "1.9.0"
__author__ = "Ductho Le"
__email__ = "ductho.le@outlook.com"

if TYPE_CHECKING:
    from wavedl.models import (
        build_model,
        get_model,
        import_model_modules,
        list_models,
        register_model,
    )
    from wavedl.utils import (
        load_test_data,
        load_training_data,
        prepare_data,
    )

_MODEL_EXPORTS = {
    "build_model",
    "get_model",
    "import_model_modules",
    "list_models",
    "register_model",
}
_UTIL_EXPORTS = {"load_test_data", "load_training_data", "prepare_data"}


def __getattr__(name: str):
    """Lazily import heavy ML submodules on first attribute access (PEP 562)."""
    if name in _MODEL_EXPORTS:
        from wavedl import models as _models

        value = getattr(_models, name)
        globals()[name] = value
        return value
    if name in _UTIL_EXPORTS:
        from wavedl import utils as _utils

        value = getattr(_utils, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "__author__",
    "__email__",
    "__version__",
    "build_model",
    "get_model",
    "import_model_modules",
    "list_models",
    "load_test_data",
    "load_training_data",
    "prepare_data",
    "register_model",
]
