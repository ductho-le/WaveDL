"""
CNN: A Dimension-Agnostic Convolutional Neural Network
======================================================

A flexible CNN architecture that automatically adapts to 1D, 2D, or 3D inputs.
Dynamically selects appropriate convolution, pooling, and dropout layers based
on input dimensionality.

**Dimensionality Support**:
    - 1D: Waveforms, signals, time-series (N, 1, L) → Conv1d
    - 2D: Images, spectrograms (N, 1, H, W) → Conv2d
    - 3D: Volumetric data, CT/MRI (N, 1, D, H, W) → Conv3d

Use this as:
    - A baseline for comparing more complex architectures
    - A lightweight option for any spatial data type
    - A starting point for custom modifications

Author: Ductho Le (ductho.le@outlook.com)
Version: 1.0.0
"""

from typing import Tuple, Dict, Any, Union, Type
import torch
import torch.nn as nn

from models.base import BaseModel
from models.registry import register_model


# Type alias for spatial shapes
SpatialShape = Union[Tuple[int], Tuple[int, int], Tuple[int, int, int]]


def _get_conv_layers(dim: int) -> Tuple[Type[nn.Module], Type[nn.Module], Type[nn.Module]]:
    """
    Get the appropriate Conv, MaxPool, and Dropout classes for a given dimensionality.
    
    Args:
        dim: Spatial dimensionality (1, 2, or 3)
        
    Returns:
        Tuple of (Conv, MaxPool, Dropout) layer classes
        
    Raises:
        ValueError: If dim is not 1, 2, or 3
    """
    if dim == 1:
        return nn.Conv1d, nn.MaxPool1d, nn.Dropout1d
    elif dim == 2:
        return nn.Conv2d, nn.MaxPool2d, nn.Dropout2d
    elif dim == 3:
        return nn.Conv3d, nn.MaxPool3d, nn.Dropout3d
    else:
        raise ValueError(f"Unsupported dimensionality: {dim}D. Supported: 1D, 2D, 3D.")


@register_model("cnn")
class CNN(BaseModel):
    """
    Universal CNN: A dimension-agnostic convolutional network for regression.
    
    Automatically detects input dimensionality from in_shape and builds the
    appropriate architecture using Conv1d/2d/3d layers.
    
    Architecture:
        - 5 Encoder blocks: Conv → GroupNorm → LeakyReLU → MaxPool [→ Dropout]
          Channel progression: 32 → 64 → 128 → 256 → 512 (with base_channels=32)
        - Global adaptive pooling to 1×1 (handles variable spatial dimensions)
        - 3-layer MLP regression head: 512 → 256 → 128 → 64 → out_size
    
    Args:
        in_shape: Spatial dimensions as tuple:
            - 1D: (L,) for signals/waveforms
            - 2D: (H, W) for images
            - 3D: (D, H, W) for volumes
        out_size: Number of regression output targets
        base_channels: Base channel count, multiplied through encoder (default: 32)
        dropout_rate: Dropout rate for regularization (default: 0.1)
    
    Input Shape:
        (B, 1, *spatial_dims) where spatial_dims matches in_shape
        
    Output Shape:
        (B, out_size)
    
    Example:
        >>> model = CNN(in_shape=(500, 500), out_size=3)  # 2D dispersion curve
        >>> x = torch.randn(4, 1, 500, 500)
        >>> out = model(x)  # Shape: (4, 3)
        
        >>> model = CNN(in_shape=(512,), out_size=5)  # 1D waveform input
        >>> x = torch.randn(4, 1, 512)
        >>> out = model(x)  # Shape: (4, 5)
    """
    
    def __init__(
        self, 
        in_shape: SpatialShape, 
        out_size: int,
        base_channels: int = 32,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(in_shape, out_size)
        
        self.base_channels = base_channels
        self.dropout_rate = dropout_rate
        self.dim = len(in_shape)
        
        # Get dimension-appropriate layer classes
        self._Conv, self._MaxPool, self._Dropout = _get_conv_layers(self.dim)
        
        # Adaptive pooling for consistent feature size regardless of input resolution
        self._AdaptivePool = (
            nn.AdaptiveAvgPool1d if self.dim == 1 else
            nn.AdaptiveAvgPool2d if self.dim == 2 else
            nn.AdaptiveAvgPool3d
        )
        
        # Channel progression: 32 → 64 → 128 → 256 → 512 (5 blocks for deeper features)
        c1, c2, c3, c4, c5 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16
        )
        
        # Encoder blocks with progressive dropout (5 blocks)
        self.block1 = self._make_conv_block(1, c1)
        self.block2 = self._make_conv_block(c1, c2)
        self.block3 = self._make_conv_block(c2, c3, dropout=0.05)
        self.block4 = self._make_conv_block(c3, c4, dropout=0.05)
        self.block5 = self._make_conv_block(c4, c5, dropout=dropout_rate)
        
        # Global average pooling (output size = 1) - ONNX compatible with any input size
        # Using output size 1 ensures compatibility regardless of encoder output dimensions
        self.adaptive_pool = self._AdaptivePool(1)
        
        # Compute flattened feature size (1 element per channel after global pooling)
        flat_size = c5  # 512 channels × 1 spatial element
        
        # Regression head: 512 → 256 → 128 → 64 → out_size
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(flat_size, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.01, inplace=True),
            
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.01, inplace=True),
            
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.01, inplace=True),
            
            nn.Dropout(dropout_rate),
            nn.Linear(64, out_size)
        )
    
    @staticmethod
    def _compute_num_groups(num_channels: int, target_groups: int = 4) -> int:
        """
        Compute valid num_groups for GroupNorm that divides num_channels.
        
        Finds the largest divisor of num_channels that is <= target_groups,
        or falls back to 1 if no suitable divisor exists.
        
        Args:
            num_channels: Number of channels (must be positive)
            target_groups: Desired number of groups (default: 4)
            
        Returns:
            Valid num_groups that satisfies num_channels % num_groups == 0
        """
        # Try target_groups down to 1, return first valid divisor
        for g in range(min(target_groups, num_channels), 0, -1):
            if num_channels % g == 0:
                return g
        return 1  # Fallback (always valid)
    
    def _make_conv_block(
        self, 
        in_channels: int, 
        out_channels: int, 
        dropout: float = 0.0
    ) -> nn.Sequential:
        """
        Create a convolutional block with dimension-appropriate layers.
        
        Args:
            in_channels: Input channel count
            out_channels: Output channel count
            dropout: Dropout rate (0 to disable)
            
        Returns:
            Sequential block: Conv → GroupNorm → LeakyReLU → MaxPool [→ Dropout]
        """
        num_groups = self._compute_num_groups(out_channels, target_groups=4)
        
        layers = [
            self._Conv(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.LeakyReLU(0.01, inplace=True),
            self._MaxPool(2)
        ]
        
        if dropout > 0:
            layers.append(self._Dropout(dropout))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder and regression head.
        
        Args:
            x: Input tensor of shape (B, 1, *spatial_dims)
            
        Returns:
            Output tensor of shape (B, out_size)
        """
        # Encoder (5 blocks)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        
        # Adaptive pooling ensures consistent feature size
        x = self.adaptive_pool(x)
        
        # Flatten and regress
        x = x.flatten(1)
        return self.head(x)
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Return default configuration for CNN."""
        return {
            "base_channels": 32,
            "dropout_rate": 0.1
        }
    
    def __repr__(self) -> str:
        return (
            f"CNN({self.dim}D, in_shape={self.in_shape}, out_size={self.out_size}, "
            f"channels={self.base_channels}, dropout={self.dropout_rate})"
        )
