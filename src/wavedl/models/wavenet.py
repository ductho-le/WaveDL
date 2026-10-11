"""
WaveNet: Gated Dilated Convolutional Network for 1D Waveform Regression
========================================================================

WaveNet-style architecture adapted for regression on wave-based signals.
Uses gated dilated convolutions with skip connections for large receptive
fields and expressive nonlinearities.

**Key Features**:
    - Gated activation: tanh(Wf * x) × sigmoid(Wg * x)
    - Skip connections: all blocks contribute to the final output
    - Exponentially growing dilation: receptive field grows with depth
    - Same padding (non-causal): appropriate for regression, not generation
    - No normalization layers: stability comes from the gated tanh×sigmoid
      activations and residual/skip connections

**Variants**:
    - wavenet_small: 32 channels, 6 dilation levels (~1.0M params)
    - wavenet: 64 channels, 8 dilation levels (~4.0M params)
    - wavenet_large: 128 channels, 10 dilation levels (~15M params)

**Receptive Field**:
    RF = 1 + (kernel_size - 1) * sum(2^i for i in range(n_layers))
    wavenet (kernel=3, 8 layers): RF = 1 + 2 * 255 = 511 samples

**Note**: WaveNet is 1D-only. For 2D data use EfficientNet, Swin, or ViT.

References:
    van den Oord, A., et al. (2016). WaveNet: A Generative Model for Raw Audio.
    arXiv:1609.03499. https://arxiv.org/abs/1609.03499

Author: Ductho Le (ductho.le@outlook.com)
"""

import copy
from numbers import Integral
from typing import Any, ClassVar

import torch
import torch.nn as nn

from wavedl.models.base import BaseModel
from wavedl.models.registry import register_model


# =============================================================================
# BUILDING BLOCKS
# =============================================================================


class GatedResidualBlock(nn.Module):
    """
    WaveNet-style gated residual block with dilated convolution.

    Architecture:
        x ──► DilatedConv ──► split ──► tanh ──► ×  ──► skip_out
                                  └──► sigmoid ──┘   │
               ┌─────────────────────────────────────┘
               └─► 1x1 Conv ──► + x ──► residual_out
    """

    def __init__(
        self,
        channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        total_padding = dilation * (kernel_size - 1)
        padding = total_padding // 2
        self._right_padding = total_padding % 2

        # Dilated conv outputs 2*channels for gated split
        self.dilated_conv = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.dropout = nn.Dropout(dropout)

        # 1x1 conv for residual
        self.residual_conv = nn.Conv1d(channels, channels, kernel_size=1)

        # 1x1 conv for skip
        self.skip_conv = nn.Conv1d(channels, skip_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, channels, L)

        Returns:
            residual: (B, channels, L) — fed into the next block
            skip: (B, skip_channels, L) — accumulated across all blocks
        """
        residual_in = x

        # Gated activation
        if self._right_padding:
            x = nn.functional.pad(x, (0, self._right_padding))
        h = self.dilated_conv(x)
        h_tanh, h_sigmoid = h.chunk(2, dim=1)
        h = torch.tanh(h_tanh) * torch.sigmoid(h_sigmoid)  # (B, channels, L)

        h = self.dropout(h)

        # Residual path
        residual = self.residual_conv(h) + residual_in

        # Skip path
        skip = self.skip_conv(h)

        return residual, skip


class WaveNetBase(BaseModel):
    """
    WaveNet-style model for 1D waveform regression.

    Architecture:
    1. Input conv (1 → channels)
    2. N gated residual blocks with dilation 1, 2, 4, ..., 2^(N-1)
    3. Sum all skip outputs
    4. ReLU → 1x1 Conv → ReLU → 1x1 Conv
    5. Adaptive avg pool → regression head

    The receptive field grows exponentially with depth.
    """

    n_layers: int
    kernel_size: int
    receptive_field: int
    input_conv: nn.Module
    blocks: nn.ModuleList
    post_skip: nn.Module
    global_pool: nn.Module
    head: nn.Module

    def __init__(
        self,
        in_shape: tuple[int],
        out_size: int,
        channels: int,
        skip_channels: int,
        n_layers: int,
        kernel_size: int = 3,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        """
        Args:
            in_shape: (L,) input signal length
            out_size: Number of regression output targets
            channels: Residual/dilation channel width
            skip_channels: Skip connection channel width
            n_layers: Number of gated residual blocks (dilation doubles each layer)
            kernel_size: Convolution kernel size (default: 3)
            dropout_rate: Dropout rate (default: 0.1)
        """
        super().__init__(in_shape, out_size)

        if len(in_shape) != 1:
            raise ValueError(
                f"WaveNet requires 1D input (L,), got {len(in_shape)}D. "
                "For 2D/3D data, use ResNet, EfficientNet, or Swin."
            )

        if (
            not isinstance(n_layers, Integral)
            or isinstance(n_layers, bool)
            or n_layers <= 0
        ):
            raise ValueError("n_layers must be a positive integer.")

        self.channels = channels
        self.skip_channels = skip_channels
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        # Input projection (handles single-channel input)
        self.input_conv = nn.Conv1d(1, channels, kernel_size=1)

        # Gated residual blocks with exponentially increasing dilation
        self.blocks = nn.ModuleList(
            [
                GatedResidualBlock(
                    channels=channels,
                    skip_channels=skip_channels,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout_rate,
                )
                for i in range(n_layers)
            ]
        )

        # The final residual output feeds no later block; only its skip output
        # contributes to regression. Retain its weights for old checkpoints,
        # but exclude these unused parameters from optimization/DDP reduction.
        self.blocks[-1].residual_conv.requires_grad_(False)
        # Keep their original optimizer positions for published checkpoints.
        # With grad=None optimizers leave these frozen parameters untouched.
        self._optimizer_checkpoint_parameters = (
            f"blocks.{n_layers - 1}.residual_conv.weight",
            f"blocks.{n_layers - 1}.residual_conv.bias",
        )

        # Post-skip processing (WaveNet output stack)
        self.post_skip = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(skip_channels, skip_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(skip_channels, skip_channels, kernel_size=1),
        )

        # Global aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Regression head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(skip_channels, skip_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(skip_channels // 2, out_size),
        )

        # Compute receptive field
        self.receptive_field = self._compute_rf()

        self._init_weights()

    def _compute_rf(self) -> int:
        """Receptive field = 1 + (kernel-1)*sum(dilations)."""
        total_dilation = sum(2**i for i in range(self.n_layers))
        return 1 + (self.kernel_size - 1) * total_dilation

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 1, L)

        Returns:
            Output tensor of shape (B, out_size)
        """
        # Input projection
        x = self.input_conv(x)  # (B, channels, L)

        # Gated residual blocks — accumulate skip outputs
        skip_sum = None
        for block in self.blocks:
            x, skip = block(x)
            skip_sum = skip if skip_sum is None else skip_sum + skip

        # Post-skip stack
        out = self.post_skip(skip_sum)  # (B, skip_channels, L)

        # Global pool + head
        out = self.global_pool(out)  # (B, skip_channels, 1)
        return self.head(out)

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        return {
            "channels": 64,
            "skip_channels": 128,
            "n_layers": 8,
            "kernel_size": 3,
            "dropout_rate": 0.1,
            **copy.deepcopy(getattr(cls, "_variant_defaults", {})),
        }


# =============================================================================
# REGISTERED MODEL VARIANTS
# =============================================================================


@register_model("wavenet_small")
class WaveNetSmall(WaveNetBase):
    """
    WaveNet-Small: Lightweight gated dilated conv network for 1D waveforms.

    ~1.0M parameters. 6 dilation layers, 32-channel residual, 64-channel skip.
    Receptive field: 127 samples (kernel=3).

    Recommended for:
        - Quick experiments and prototyping
        - Short signals (< 2048 samples)
        - Edge deployment

    Args:
        in_shape: (L,) input signal length
        out_size: Number of regression targets

    Example:
        >>> model = WaveNetSmall(in_shape=(2048,), out_size=3)
        >>> x = torch.randn(4, 1, 2048)
        >>> out = model(x)  # (4, 3)
    """

    _variant_defaults: ClassVar[dict[str, Any]] = {
        "channels": 32,
        "skip_channels": 64,
        "n_layers": 6,
    }

    def __init__(self, in_shape: tuple[int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            **{**copy.deepcopy(self._variant_defaults), **kwargs},
        )

    def __repr__(self) -> str:
        return (
            f"WaveNet_Small(in_shape={self.in_shape}, out={self.out_size}, "
            f"RF={self.receptive_field})"
        )


@register_model("wavenet")
class WaveNet(WaveNetBase):
    """
    WaveNet: Standard gated dilated conv network for 1D waveforms.

    ~4.0M parameters. 8 dilation layers, 64-channel residual, 128-channel skip.
    Receptive field: 511 samples (kernel=3).

    Recommended for:
        - Ultrasonic A-scan and pulse-echo signals
        - Acoustic emission waveforms
        - Seismic traces
        - 1D signals with long-range dependencies

    Args:
        in_shape: (L,) input signal length
        out_size: Number of regression targets

    Example:
        >>> model = WaveNet(in_shape=(4096,), out_size=3)
        >>> x = torch.randn(4, 1, 4096)
        >>> out = model(x)  # (4, 3)
    """

    _variant_defaults: ClassVar[dict[str, Any]] = {
        "channels": 64,
        "skip_channels": 128,
        "n_layers": 8,
    }

    def __init__(self, in_shape: tuple[int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            **{**copy.deepcopy(self._variant_defaults), **kwargs},
        )

    def __repr__(self) -> str:
        return (
            f"WaveNet(in_shape={self.in_shape}, out={self.out_size}, "
            f"RF={self.receptive_field})"
        )


@register_model("wavenet_large")
class WaveNetLarge(WaveNetBase):
    """
    WaveNet-Large: High-capacity gated dilated conv network for 1D waveforms.

    ~15M parameters. 10 dilation layers, 128-channel residual, 256-channel skip.
    Receptive field: 2047 samples (kernel=3).

    Recommended for:
        - Long sequences (> 8192 samples)
        - Complex waveform patterns requiring large receptive fields
        - Large datasets with sufficient compute

    Args:
        in_shape: (L,) input signal length
        out_size: Number of regression targets

    Example:
        >>> model = WaveNetLarge(in_shape=(8192,), out_size=3)
        >>> x = torch.randn(4, 1, 8192)
        >>> out = model(x)  # (4, 3)
    """

    _variant_defaults: ClassVar[dict[str, Any]] = {
        "channels": 128,
        "skip_channels": 256,
        "n_layers": 10,
    }

    def __init__(self, in_shape: tuple[int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            **{**copy.deepcopy(self._variant_defaults), **kwargs},
        )

    def __repr__(self) -> str:
        return (
            f"WaveNet_Large(in_shape={self.in_shape}, out={self.out_size}, "
            f"RF={self.receptive_field})"
        )
