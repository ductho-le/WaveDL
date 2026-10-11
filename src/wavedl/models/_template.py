"""
Model Template for Custom Architectures
========================================

Copy this file and modify to add custom model architectures to WaveDL.
The model will be automatically registered and available via --model flag.

Quick Start:
    1. Copy this file to your project: cp _template.py my_model.py
    2. Rename the class and update @register_model("my_model")
    3. Implement your architecture in __init__ and forward
    4. Train: wavedl-train --import my_model --model my_model --data_path data.npz

This file implements the vector-output contract. Requirements (your model MUST):
    1. Inherit from BaseModel
    2. Accept (in_shape, out_size, **kwargs) in __init__
    3. Return tensor of shape (batch, out_size) from forward()

For dense 2D regression, also accept out_shape=(C, H, W), set
supports_spatial_output = True (and requires_spatial_output = True for a
field-only model), and return exactly (batch, C, H, W). See UNetField in
unet.py for a complete implementation.

See README.md "Adding Custom Models" section for more details.

Author: Ductho Le (ductho.le@outlook.com)
"""

import torch
import torch.nn as nn

from wavedl.models.base import BaseModel
from wavedl.models.registry import register_model  # noqa: F401 - template decorator


# Uncomment the decorator to register this model
# @register_model("my_model")
class TemplateModel(BaseModel):
    """
    Template vector-regression architecture (2D input only).

    Replace this docstring with your model description.
    The first line will appear in --list_models output.

    NOTE: This template is hardcoded for 2D inputs using Conv2d/MaxPool2d.
    For 1D/3D support, use dimension-agnostic layer factories from
    _pretrained_utils.py (get_conv_layer, get_pool_layer, get_norm_layer).

    Args:
        in_shape: Input spatial dimensions as (H, W) for 2D images
        out_size: Number of regression targets (auto-detected from data)
        hidden_dim: Size of hidden layers (default: 256)
        dropout: Dropout rate (default: 0.1)

    Input Shape:
        (B, 1, H, W) - 2D grayscale images

    Output Shape:
        (B, out_size) - Regression predictions
    """

    def __init__(
        self,
        in_shape: tuple,
        out_size: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        **kwargs,  # Accept extra kwargs for flexibility
    ):
        # REQUIRED: Call parent __init__ with in_shape and out_size
        super().__init__(in_shape, out_size)

        # Store hyperparameters as attributes (optional but recommended)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        # =================================================================
        # BUILD YOUR ARCHITECTURE HERE
        # =================================================================

        # Example: Simple CNN encoder (assumes 2D input with 1 channel)
        self.encoder = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Layer 4
            nn.Conv2d(128, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
        )

        # Example: Regression head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        REQUIRED: Must accept (B, C, *spatial) and return (B, out_size)

        Args:
            x: Input tensor of shape (B, 1, *in_shape)

        Returns:
            Output tensor of shape (B, out_size)
        """
        # Encode
        features = self.encoder(x)

        # Predict
        output = self.head(features)

        return output


# =============================================================================
# USAGE EXAMPLE
# =============================================================================
if __name__ == "__main__":
    # Quick test of the model
    model = TemplateModel(in_shape=(64, 64), out_size=5)

    # Print model summary
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {model.count_parameters():,}")

    # Test forward pass
    dummy_input = torch.randn(2, 1, 64, 64)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
