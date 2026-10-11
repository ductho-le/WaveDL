"""
Temporal Convolutional Network (TCN): Dilated Causal Convolutions for 1D Signals
=================================================================================

A dedicated 1D architecture using dilated causal convolutions to capture
long-range temporal dependencies in waveforms and time-series data.
Provides exponentially growing receptive field with linear parameter growth.

**Key Features**:
    - Dilated convolutions: Exponentially growing receptive field
    - Causal convolution padding; GroupNorm uses the entire signal
    - Residual connections: Stable gradient flow
    - Group normalization across channels and sequence positions

**Variants**:
    - tcn: Standard TCN with configurable depth and channels
    - tcn_small: Lightweight variant for quick experiments
    - tcn_large: Higher capacity for complex patterns

**Receptive Field Calculation**:
    RF = 1 + 2 * (kernel_size - 1) * sum(dilation[i] for i in layers)
    With default settings (kernel=3, 8 layers, dilation=2^i):
    RF = 1 + 4 * (1+2+4+8+16+32+64+128) = 1021 samples

**Note**: GroupNorm makes the complete model non-causal even though its
convolutions use causal padding. It is intended for whole-signal regression,
not autoregressive generation or streaming predictions. The receptive field
below describes the convolution stack; normalization also couples all positions.

TCN is 1D-only. For 2D/3D data, use ResNet, EfficientNet, or Swin.

References:
    Bai, S., Kolter, J.Z., & Koltun, V. (2018). An Empirical Evaluation of
    Generic Convolutional and Recurrent Networks for Sequence Modeling.
    arXiv:1803.01271. https://arxiv.org/abs/1803.01271

    van den Oord, A., et al. (2016). WaveNet: A Generative Model for Raw Audio.
    arXiv:1609.03499. https://arxiv.org/abs/1609.03499

Author: Ductho Le (ductho.le@outlook.com)
"""

import copy
from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from wavedl.models.base import BaseModel, compute_num_groups
from wavedl.models.registry import register_model


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution with dilation.

    Ensures output at time t only depends on inputs at times <= t.
    Uses left-side padding to achieve causal behavior.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        # Causal padding: only pad on the left
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=0,  # We handle padding manually for causality
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad on the left only (causal)
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """
    Temporal block with two causal dilated convolutions and residual connection.

    Architecture:
        Input → CausalConv → GroupNorm → GELU → Dropout →
                CausalConv → GroupNorm → GELU → Dropout → (+Input) → Output
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # First causal convolution
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.GroupNorm(
            compute_num_groups(out_channels, preferred_groups=8), out_channels
        )
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        # Second causal convolution
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm2 = nn.GroupNorm(
            compute_num_groups(out_channels, preferred_groups=8), out_channels
        )
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection (1x1 conv if channels change)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)

        # First conv block
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dropout1(out)

        # Second conv block
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.dropout2(out)

        return out + residual


class TCNBase(BaseModel):
    """
    Base Temporal Convolutional Network for 1D regression.

    Architecture:
    1. Input projection (optional channel expansion)
    2. Stack of temporal blocks with exponentially increasing dilation
    3. Global average pooling
    4. Regression head

    The receptive field grows exponentially with depth:
    RF = 1 + 2 * (kernel_size - 1) * sum(2^i for i in range(num_layers))
    """

    def __init__(
        self,
        in_shape: tuple[int],
        out_size: int,
        num_channels: list[int],
        kernel_size: int = 3,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        """
        Initialize TCN for regression.

        Args:
            in_shape: (L,) input signal length
            out_size: Number of regression output targets
            num_channels: List of channel sizes for each temporal block
            kernel_size: Convolution kernel size (default: 3)
            dropout_rate: Dropout rate (default: 0.1)
        """
        super().__init__(in_shape, out_size)

        if len(in_shape) != 1:
            raise ValueError(
                f"TCN requires 1D input (L,), got {len(in_shape)}D. "
                "For 2D/3D data, use ResNet, EfficientNet, or Swin."
            )

        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        # Build temporal blocks with exponentially increasing dilation
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2**i
            in_ch = 1 if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout_rate)
            )

        self.network = nn.Sequential(*layers)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Regression head
        final_channels = num_channels[-1]
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(final_channels, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, out_size),
        )

        # Calculate and store receptive field
        self.receptive_field = self._compute_receptive_field()

        # Initialize weights
        self._init_weights()

    def _compute_receptive_field(self) -> int:
        """Compute the receptive field of the network."""
        rf = 1
        for i in range(len(self.num_channels)):
            dilation = 2**i
            # Each temporal block has 2 convolutions
            rf += 2 * (self.kernel_size - 1) * dilation
        return rf

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                # Kaiming init with 'relu' gain (~√2); a reasonable approximation
                # for GELU activation (PyTorch has no native GELU gain support).
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 1, L)

        Returns:
            Output tensor of shape (B, out_size)
        """
        # Temporal blocks
        x = self.network(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.flatten(1)

        # Regression head
        return self.head(x)

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """Return default configuration for TCN."""
        return {
            "num_channels": [64, 128, 256, 256, 512, 512, 512, 512],
            "kernel_size": 3,
            "dropout_rate": 0.1,
            **copy.deepcopy(getattr(cls, "_variant_defaults", {})),
        }


# =============================================================================
# REGISTERED MODEL VARIANTS
# =============================================================================


@register_model("tcn")
class TCN(TCNBase):
    """
    TCN: Standard Temporal Convolutional Network.

    ~6.9M backbone parameters. 8 temporal blocks with channels [64→128→256→256→512→512→512→512].
    Convolutional receptive field: 1021 samples with kernel_size=3.

    Recommended for:
        - Ultrasonic A-scan processing
        - Acoustic emission signals
        - Seismic waveform analysis
        - Any 1D time-series regression

    Args:
        in_shape: (L,) input signal length
        out_size: Number of regression targets
        kernel_size: Convolution kernel size (default: 3)
        dropout_rate: Dropout rate (default: 0.1)

    Example:
        >>> model = TCN(in_shape=(4096,), out_size=3)
        >>> x = torch.randn(4, 1, 4096)
        >>> out = model(x)  # (4, 3)
    """

    def __init__(self, in_shape: tuple[int], out_size: int, **kwargs):
        # Default: 8 layers, 64→512 channels
        num_channels = kwargs.pop(
            "num_channels", [64, 128, 256, 256, 512, 512, 512, 512]
        )
        super().__init__(
            in_shape=in_shape, out_size=out_size, num_channels=num_channels, **kwargs
        )

    def __repr__(self) -> str:
        return (
            f"TCN(in_shape={self.in_shape}, out={self.out_size}, "
            f"RF={self.receptive_field})"
        )


@register_model("tcn_small")
class TCNSmall(TCNBase):
    """
    TCN-Small: Lightweight variant for quick experiments.

    ~0.9M backbone parameters. 6 temporal blocks with channels [32→64→128→128→256→256].
    Convolutional receptive field: 253 samples with kernel_size=3.

    Recommended for:
        - Quick prototyping
        - Smaller datasets
        - Real-time inference on edge devices

    Args:
        in_shape: (L,) input signal length
        out_size: Number of regression targets
        kernel_size: Convolution kernel size (default: 3)
        dropout_rate: Dropout rate (default: 0.1)

    Example:
        >>> model = TCNSmall(in_shape=(1024,), out_size=3)
        >>> x = torch.randn(4, 1, 1024)
        >>> out = model(x)  # (4, 3)
    """

    _variant_defaults: ClassVar[dict[str, Any]] = {
        "num_channels": [32, 64, 128, 128, 256, 256]
    }

    def __init__(self, in_shape: tuple[int], out_size: int, **kwargs):
        num_channels = kwargs.pop(
            "num_channels", list(self._variant_defaults["num_channels"])
        )
        super().__init__(
            in_shape=in_shape, out_size=out_size, num_channels=num_channels, **kwargs
        )

    def __repr__(self) -> str:
        return (
            f"TCN_Small(in_shape={self.in_shape}, out={self.out_size}, "
            f"RF={self.receptive_field})"
        )


@register_model("tcn_large")
class TCNLarge(TCNBase):
    """
    TCN-Large: High-capacity variant for complex patterns.

    ~10.0M backbone parameters. 10 temporal blocks with channels [64→128→256→256→512→512→512→512→512→512].
    Convolutional receptive field: 4093 samples with kernel_size=3.

    Recommended for:
        - Long sequences (>4096 samples)
        - Complex temporal patterns
        - Large datasets with sufficient compute

    Args:
        in_shape: (L,) input signal length
        out_size: Number of regression targets
        kernel_size: Convolution kernel size (default: 3)
        dropout_rate: Dropout rate (default: 0.1)

    Example:
        >>> model = TCNLarge(in_shape=(8192,), out_size=3)
        >>> x = torch.randn(4, 1, 8192)
        >>> out = model(x)  # (4, 3)
    """

    _variant_defaults: ClassVar[dict[str, Any]] = {
        "num_channels": [64, 128, 256, 256, 512, 512, 512, 512, 512, 512]
    }

    def __init__(self, in_shape: tuple[int], out_size: int, **kwargs):
        num_channels = kwargs.pop(
            "num_channels", list(self._variant_defaults["num_channels"])
        )
        super().__init__(
            in_shape=in_shape, out_size=out_size, num_channels=num_channels, **kwargs
        )

    def __repr__(self) -> str:
        return (
            f"TCN_Large(in_shape={self.in_shape}, out={self.out_size}, "
            f"RF={self.receptive_field})"
        )
