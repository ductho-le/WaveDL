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

__version__ = "1.9.0"
__author__ = "Ductho Le"
__email__ = "ductho.le@outlook.com"

# Re-export key APIs for convenience
from wavedl.models import build_model, get_model, list_models, register_model
from wavedl.utils import (
    load_test_data,
    load_training_data,
    prepare_data,
)


__all__ = [
    "__version__",
    "build_model",
    "get_model",
    "list_models",
    "load_test_data",
    "load_training_data",
    "prepare_data",
    "register_model",
]
