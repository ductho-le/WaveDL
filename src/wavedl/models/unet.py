"""
U-Net Regression: Encoder-Decoder Architectures for Vectors and Fields
=======================================================================

A dimension-agnostic U-Net implementation for vector regression and dense
spatial field reconstruction.

**Dimensionality Support**:
    - 1D: Waveforms, signals (N, 1, L) → Conv1d
    - 2D: Images, spectrograms (N, 1, H, W) → Conv2d
    - 3D: Volumetric data (N, 1, D, H, W) → Conv3d

**Variants**:
    - unet_regression: U-Net with global pooling for vector regression
    - unet_field: U-Net with a dense convolutional head for spatial fields

The ``UNetField`` module is dimension-agnostic when used directly, but the
WaveDL data/training CLI currently defines a dense **2D** target contract only:
``(N,H,W)`` or ``(N,C,H,W)``.

Author: Ductho Le (ductho.le@outlook.com)
"""

from numbers import Integral
from typing import Any

import torch
import torch.nn as nn

from wavedl.models.base import BaseModel, SpatialShape, compute_num_groups
from wavedl.models.registry import register_model


def _get_layers(dim: int):
    """Get dimension-appropriate layer classes."""
    if dim == 1:
        return nn.Conv1d, nn.ConvTranspose1d, nn.MaxPool1d, nn.AdaptiveAvgPool1d
    elif dim == 2:
        return nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d
    elif dim == 3:
        return nn.Conv3d, nn.ConvTranspose3d, nn.MaxPool3d, nn.AdaptiveAvgPool3d
    else:
        raise ValueError(f"Unsupported dimensionality: {dim}D")


def _validate_config(
    in_shape: SpatialShape, out_size: int, base_channels: int, depth: int
) -> None:
    """Validate dimensions before constructing either U-Net variant."""
    if len(in_shape) not in (1, 2, 3):
        raise ValueError(f"Unsupported dimensionality: {len(in_shape)}D")
    if any(
        not isinstance(size, Integral) or isinstance(size, bool) or size <= 0
        for size in in_shape
    ):
        raise ValueError(
            f"in_shape dimensions must be positive integers, got {in_shape}."
        )
    for name, value, minimum in (
        ("out_size", out_size, 1),
        ("base_channels", base_channels, 1),
        ("depth", depth, 0),
    ):
        if (
            not isinstance(value, Integral)
            or isinstance(value, bool)
            or value < minimum
        ):
            qualifier = "positive" if minimum else "nonnegative"
            raise ValueError(f"{name} must be a {qualifier} integer, got {value!r}.")

    min_size = 2 ** int(depth)
    for axis, size in enumerate(in_shape):
        if size < min_size:
            raise ValueError(
                f"U-Net with depth={depth} requires each spatial axis "
                f">= {min_size}, but axis {axis} has size {size}."
            )


def _normalization_groups(channels: int, in_shape: SpatialShape, level: int) -> int:
    """Keep existing groups except where pooling leaves a singleton field."""
    groups = compute_num_groups(channels)
    if all(size // (2**level) == 1 for size in in_shape) and groups == channels:
        if channels == 1:
            raise ValueError(
                "U-Net GroupNorm requires at least two values per group. "
                "Increase base_channels or the input spatial size."
            )
        # GroupNorm needs multiple values per sample/group even in eval mode.
        # A one-channel group on a singleton grid otherwise fails for batch=1
        # and discards all feature variation for larger batches.
        groups = compute_num_groups(channels, preferred_groups=channels // 2)
    return groups


class DoubleConv(nn.Module):
    """Double convolution block: Conv-GN-ReLU-Conv-GN-ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int = 2,
        num_groups: int | None = None,
    ):
        super().__init__()
        Conv = _get_layers(dim)[0]

        if num_groups is None:
            num_groups = compute_num_groups(out_channels)

        self.double_conv = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
            Conv(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int = 2,
        num_groups: int | None = None,
    ):
        super().__init__()
        _, _, MaxPool, _ = _get_layers(dim)

        self.maxpool_conv = nn.Sequential(
            MaxPool(2), DoubleConv(in_channels, out_channels, dim, num_groups)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int = 2,
        num_groups: int | None = None,
    ):
        super().__init__()
        _, ConvTranspose, _, _ = _get_layers(dim)

        self.up = ConvTranspose(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dim, num_groups)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Handle size mismatch (pad x1 to match x2)
        if x1.shape[2:] != x2.shape[2:]:
            diff = [x2.size(i + 2) - x1.size(i + 2) for i in range(len(x1.shape) - 2)]
            pad = []
            for d in reversed(diff):
                pad.extend([d // 2, d - d // 2])
            x1 = nn.functional.pad(x1, pad)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# =============================================================================
# REGISTERED MODEL
# =============================================================================


@register_model("unet_regression")
class UNetRegression(BaseModel):
    """
    U-Net for vector regression output.

    Uses U-Net encoder-decoder architecture with skip connections,
    then applies global pooling for standard vector regression output.

    ~31.0M backbone parameters (2D). Good for leveraging multi-scale features
    and skip connections for regression tasks.

    Args:
        in_shape: (L,), (H, W), or (D, H, W)
        out_size: Number of regression targets
        base_channels: Base channel count (default: 64)
        depth: Number of encoder/decoder levels (default: 4)
        dropout_rate: Dropout rate (default: 0.1)

    Example:
        >>> model = UNetRegression(in_shape=(224, 224), out_size=3)
        >>> x = torch.randn(4, 1, 224, 224)
        >>> out = model(x)  # (4, 3)
    """

    def __init__(
        self,
        in_shape: SpatialShape,
        out_size: int,
        base_channels: int = 64,
        depth: int = 4,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(in_shape, out_size)
        _validate_config(in_shape, out_size, base_channels, depth)

        self.dim = len(in_shape)
        self.base_channels = base_channels
        self.depth = depth
        self.dropout_rate = dropout_rate

        _, _, _, AdaptivePool = _get_layers(self.dim)

        # Channel progression: 64 -> 128 -> 256 -> 512 (for depth=4)
        features = [base_channels * (2**i) for i in range(depth + 1)]
        groups = [
            _normalization_groups(channels, in_shape, level)
            for level, channels in enumerate(features)
        ]

        # Initial double conv (1 -> features[0])
        self.inc = DoubleConv(1, features[0], self.dim, groups[0])

        # Encoder (down path)
        self.downs = nn.ModuleList()
        for i in range(depth):
            self.downs.append(
                Down(features[i], features[i + 1], self.dim, groups[i + 1])
            )

        # Decoder (up path)
        self.ups = nn.ModuleList()
        for i in range(depth):
            self.ups.append(
                Up(
                    features[depth - i],
                    features[depth - 1 - i],
                    self.dim,
                    groups[depth - 1 - i],
                )
            )

        # Vector output: global pooling + regression head
        self.global_pool = AdaptivePool(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(features[0], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, out_size),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(
                m,
                (
                    nn.Conv1d,
                    nn.Conv2d,
                    nn.Conv3d,
                    nn.ConvTranspose1d,
                    nn.ConvTranspose2d,
                    nn.ConvTranspose3d,
                ),
            ):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Encoder path - collect skip connections
        x1 = self.inc(x)

        skips = [x1]
        x = x1
        for down in self.downs:
            x = down(x)
            skips.append(x)

        # Remove last (bottleneck output, not a skip)
        skips = skips[:-1]

        # Decoder path - use skips in reverse order
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        # Global pooling + regression head
        x = self.global_pool(x)
        x = x.flatten(1)
        return self.head(x)

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """Return default configuration."""
        return {"base_channels": 64, "depth": 4, "dropout_rate": 0.1}

    def __repr__(self) -> str:
        return f"UNet_Regression({self.dim}D, in_shape={self.in_shape}, out_size={self.out_size})"


@register_model("unet_field")
class UNetField(BaseModel):
    """U-Net for dense spatial regression output.

    Unlike :class:`UNetRegression`, this variant retains the decoder's spatial
    tensor and projects it to ``out_size`` channels with a 1x1 convolution. If
    the requested output grid differs from the input grid, the prediction is
    interpolated to that spatial size.

    Direct construction supports 1D/2D/3D spatial tensors. The WaveDL CLI
    currently routes only dense 2D targets to this model.

    Args:
        in_shape: Input spatial dimensions: ``(L,)``, ``(H, W)``, or
            ``(D, H, W)``.
        out_size: Number of output field channels.
        out_shape: Optional canonical output shape ``(C, *spatial)``. The
            channel dimension must equal ``out_size``. Defaults to
            ``(out_size, *in_shape)``.
        base_channels: Base channel count (default: 64).
        depth: Number of encoder/decoder levels (default: 4).
        dropout_rate: Dropout applied before the output projection
            (default: 0.1).

    Example:
        >>> model = UNetField(in_shape=(128, 256), out_size=1, out_shape=(1, 96, 192))
        >>> x = torch.randn(4, 1, 128, 256)
        >>> out = model(x)
        >>> out.shape
        torch.Size([4, 1, 96, 192])
    """

    supports_spatial_output = True
    requires_spatial_output = True

    def __init__(
        self,
        in_shape: SpatialShape,
        out_size: int,
        out_shape: tuple[int, ...] | None = None,
        base_channels: int = 64,
        depth: int = 4,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(in_shape, out_size)
        _validate_config(in_shape, out_size, base_channels, depth)

        self.dim = len(in_shape)
        self.base_channels = base_channels
        self.depth = depth
        self.dropout_rate = dropout_rate

        if out_shape is None:
            out_shape = (out_size, *in_shape)
        else:
            out_shape = tuple(out_shape)

        expected_rank = self.dim + 1
        if len(out_shape) != expected_rank:
            raise ValueError(
                "out_shape must use canonical (C, *spatial) form with "
                f"{expected_rank} dimensions for a {self.dim}D model, got "
                f"{out_shape}."
            )
        if any(
            not isinstance(size, Integral) or isinstance(size, bool) or size <= 0
            for size in out_shape
        ):
            raise ValueError(
                f"out_shape dimensions must be positive integers, got {out_shape}."
            )
        if out_shape[0] != out_size:
            raise ValueError(
                f"out_shape channel dimension ({out_shape[0]}) must match "
                f"out_size ({out_size})."
            )

        self.out_shape = tuple(int(size) for size in out_shape)
        self.output_spatial_shape = self.out_shape[1:]

        Conv, _, _, _ = _get_layers(self.dim)

        features = [base_channels * (2**i) for i in range(depth + 1)]
        groups = [
            _normalization_groups(channels, in_shape, level)
            for level, channels in enumerate(features)
        ]

        self.inc = DoubleConv(1, features[0], self.dim, groups[0])

        self.downs = nn.ModuleList()
        for i in range(depth):
            self.downs.append(
                Down(features[i], features[i + 1], self.dim, groups[i + 1])
            )

        self.ups = nn.ModuleList()
        for i in range(depth):
            self.ups.append(
                Up(
                    features[depth - i],
                    features[depth - 1 - i],
                    self.dim,
                    groups[depth - 1 - i],
                )
            )

        self.dropout = nn.Dropout(dropout_rate)
        self.head = Conv(features[0], out_size, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(
                module,
                (
                    nn.Conv1d,
                    nn.Conv2d,
                    nn.Conv3d,
                    nn.ConvTranspose1d,
                    nn.ConvTranspose2d,
                    nn.ConvTranspose3d,
                ),
            ):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.GroupNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a dense field with shape ``(B, C, *spatial)``."""
        x = self.inc(x)

        skips = [x]
        for down in self.downs:
            x = down(x)
            skips.append(x)

        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)

        x = self.head(self.dropout(x))

        if x.shape[2:] != self.output_spatial_shape:
            modes = {1: "linear", 2: "bilinear", 3: "trilinear"}
            x = nn.functional.interpolate(
                x,
                size=self.output_spatial_shape,
                mode=modes[self.dim],
                align_corners=False,
            )

        return x

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """Return default configuration."""
        return {"base_channels": 64, "depth": 4, "dropout_rate": 0.1}

    def __repr__(self) -> str:
        return (
            f"UNet_Field({self.dim}D, in_shape={self.in_shape}, "
            f"out_shape={self.out_shape})"
        )
