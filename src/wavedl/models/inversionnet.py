"""
InversionNet: Skip-Free 2D Encoder-Decoder Architectures for Field Regression
=============================================================================

General-purpose 2D-to-2D encoder-decoder architectures for dense spatial field
regression, supporting arbitrary input channels, input resolutions, and output
grid shapes.

Unlike skip-connected U-Nets (which assume spatial alignment between input and
output pixels), these architectures compress 2D inputs into a global latent
representation via an aspect-ratio-aware convolutional encoder before decoding
into the target 2D field. This makes them well suited for cross-domain inverse
problems where input coordinates (e.g., time-receiver gathers or spectrograms)
differ from output coordinates (e.g., depth-width physical property maps), as
well as general 2D-to-2D spatial regression.

**Variants**:
    - ``inversionnet``: Encoder-decoder with learned transposed-convolution
      upsampling blocks (``ConvTranspose2d``).
    - ``inversionnet_resize``: Encoder-decoder with resize-convolution
      upsampling blocks (``Upsample + Conv2d``) to mitigate transposed-
      convolution checkerboard artifacts.

Architectural inspiration:
    - Wu & Lin (2020), *InversionNet: An Efficient and Accurate Data-Driven
      Full Waveform Inversion*, IEEE TCI 6:419-433.
    - Odena, Dumoulin & Olah (2016), *Deconvolution and Checkerboard
      Artifacts*, Distill (resize-convolution upsampling).

Author: Ductho Le (ductho.le@outlook.com)
"""

from math import isfinite
from numbers import Integral
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from wavedl.models.base import BaseModel, SpatialShape, compute_num_groups
from wavedl.models.registry import register_model


KernelSize = int | tuple[int, int]
VALID_NORMS = ("gn", "bn", "none")
VALID_ACTIVATIONS = (None, "tanh")
VALID_UPSAMPLE_MODES = ("bilinear", "nearest")


def _make_norm2d(
    norm: str, channels: int, *, is_singleton: bool = False
) -> nn.Module | None:
    """Build a 2D normalization layer safe for singleton spatial grids."""
    if norm == "none":
        return None
    if norm == "bn":
        return nn.BatchNorm2d(channels)
    if norm == "gn":
        groups = compute_num_groups(channels)
        if is_singleton and groups == channels:
            if channels == 1:
                raise ValueError(
                    "GroupNorm on a 1x1 bottleneck requires at least 2 channels. "
                    "Increase base_channels or use norm='none'."
                )
            groups = compute_num_groups(channels, preferred_groups=channels // 2)
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unsupported norm {norm!r}; choose from {VALID_NORMS}.")


def _validate_positive_int(name: str, value: Any) -> int:
    """Validate that an architectural hyperparameter is a positive integer."""
    if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def _validate_out_shape(
    out_shape: tuple[int, ...] | None, out_size: int, in_shape: tuple[int, int]
) -> tuple[int, int, int]:
    """Validate canonical dense 2D output shape ``(C, H_out, W_out)``."""
    if out_shape is None:
        return (out_size, in_shape[0], in_shape[1])
    try:
        shape = tuple(out_shape)
    except TypeError as exc:
        raise ValueError(
            f"out_shape must be a 3-tuple (C, H, W), got {out_shape!r}."
        ) from exc
    if len(shape) != 3:
        raise ValueError(
            "out_shape must use canonical (C, H, W) form with 3 dimensions for "
            f"2D field regression, got {shape}."
        )
    if any(
        not isinstance(size, Integral) or isinstance(size, bool) or size <= 0
        for size in shape
    ):
        raise ValueError(
            f"out_shape dimensions must be positive integers, got {shape}."
        )
    if shape[0] != out_size:
        raise ValueError(
            f"out_shape channel dimension ({shape[0]}) must match out_size ({out_size})."
        )
    return (int(shape[0]), int(shape[1]), int(shape[2]))


class ConvBlock2d(nn.Module):
    """2D convolution, optional normalization, and LeakyReLU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: KernelSize = 3,
        stride: KernelSize = 1,
        padding: KernelSize = 1,
        *,
        norm: str = "gn",
        negative_slope: float = 0.2,
        is_singleton: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=(norm == "none"),
            )
        ]
        norm_layer = _make_norm2d(norm, out_channels, is_singleton=is_singleton)
        if norm_layer is not None:
            layers.append(norm_layer)
        layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeconvBlock2d(nn.Module):
    """2D transposed convolution, optional normalization, and LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: KernelSize = 4,
        stride: KernelSize = 2,
        padding: KernelSize = 1,
        *,
        norm: str = "gn",
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=(norm == "none"),
            )
        ]
        norm_layer = _make_norm2d(norm, out_channels)
        if norm_layer is not None:
            layers.append(norm_layer)
        layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResizeConvBlock2d(nn.Module):
    """2x spatial interpolation followed by Conv2d, normalization, and LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        upsample_mode: str = "bilinear",
        norm: str = "gn",
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if upsample_mode not in VALID_UPSAMPLE_MODES:
            raise ValueError(
                f"Unsupported upsample_mode {upsample_mode!r}; "
                f"choose from {VALID_UPSAMPLE_MODES}."
            )
        self.upsample_mode = upsample_mode
        self.conv = ConvBlock2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm=norm,
            negative_slope=negative_slope,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample_mode == "bilinear":
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


def _plan_encoder_stage(
    height: int, width: int, stage_idx: int, min_grid: int
) -> tuple[
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
    int,
    int,
]:
    """Choose asymmetric or isotropic stride/kernel for one encoder stage.

    High-aspect-ratio inputs (such as tall time-space gathers or wide
    spectrograms) downsample along the longer axis first before applying
    isotropic 2D downsampling. Once an axis reaches ``min_grid``, striding
    along that axis stops so coarse spatial structure is preserved into the
    bottleneck.
    """
    can_down_h = height > min_grid
    can_down_w = width > min_grid

    if stage_idx < 4 and can_down_h and height >= 1.5 * width:
        k_down = (7, 1) if stage_idx == 0 else (3, 1)
        s_down = (2, 1)
        p_down = (3, 0) if stage_idx == 0 else (1, 0)
        k_ref, p_ref = (3, 1), (1, 0)
    elif stage_idx < 4 and can_down_w and width >= 1.5 * height:
        k_down = (1, 7) if stage_idx == 0 else (1, 3)
        s_down = (1, 2)
        p_down = (0, 3) if stage_idx == 0 else (0, 1)
        k_ref, p_ref = (1, 3), (0, 1)
    else:
        k_down = (3, 3)
        s_down = (2 if can_down_h else 1, 2 if can_down_w else 1)
        p_down = (1, 1)
        k_ref, p_ref = (3, 3), (1, 1)

    next_h = (height + 2 * p_down[0] - k_down[0]) // s_down[0] + 1
    next_w = (width + 2 * p_down[1] - k_down[1]) // s_down[1] + 1
    return k_down, s_down, p_down, k_ref, p_ref, next_h, next_w


class _InversionNetBase(BaseModel):
    """Shared encoder, bottleneck, and output head for 2D InversionNet models."""

    supports_spatial_output = True
    requires_spatial_output = True
    supported_input_channels: tuple[int, ...] | None = None

    def __init__(
        self,
        in_shape: SpatialShape,
        out_size: int,
        out_shape: tuple[int, ...] | None = None,
        in_channels: int = 1,
        base_channels: int = 32,
        bottleneck_grid: int = 4,
        dropout_rate: float = 0.0,
        norm: str = "gn",
        output_activation: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(in_shape, out_size)
        if len(self.in_shape) != 2:
            raise ValueError(
                f"{self.__class__.__name__} requires 2D input (H, W), "
                f"got {len(self.in_shape)}D shape {self.in_shape}."
            )

        self.in_channels = _validate_positive_int("in_channels", in_channels)
        self.base_channels = _validate_positive_int("base_channels", base_channels)
        self.bottleneck_grid = _validate_positive_int(
            "bottleneck_grid", bottleneck_grid
        )

        if (
            isinstance(dropout_rate, bool)
            or not isinstance(dropout_rate, (int, float))
            or not isfinite(dropout_rate)
            or not (0.0 <= float(dropout_rate) < 1.0)
        ):
            raise ValueError(
                f"dropout_rate must be a finite number in [0, 1), got {dropout_rate!r}."
            )
        self.dropout_rate = float(dropout_rate)

        if norm not in VALID_NORMS:
            raise ValueError(f"Unsupported norm {norm!r}; choose from {VALID_NORMS}.")
        self.norm = norm

        if output_activation not in VALID_ACTIVATIONS:
            raise ValueError(
                f"Unsupported output_activation {output_activation!r}; "
                f"choose from {VALID_ACTIVATIONS}."
            )
        self.output_activation = output_activation

        self.out_shape = _validate_out_shape(out_shape, self.out_size, self.in_shape)
        self.output_spatial_shape = self.out_shape[1:]

        dim1 = self.base_channels
        dim2 = dim1 * 2
        dim3 = dim1 * 4
        dim4 = dim1 * 8
        dim5 = dim1 * 16
        self._dims = (dim1, dim2, dim3, dim4, dim5)

        # 7-stage aspect-ratio-aware convolutional encoder
        stage_channels = [
            (self.in_channels, dim1),
            (dim1, dim2),
            (dim2, dim2),
            (dim2, dim3),
            (dim3, dim3),
            (dim3, dim4),
            (dim4, dim4),
        ]
        encoder_layers: list[nn.Module] = []
        h, w = self.in_shape
        for stage_idx, (c_in, c_out) in enumerate(stage_channels):
            k_down, s_down, p_down, k_ref, p_ref, h, w = _plan_encoder_stage(
                h, w, stage_idx, self.bottleneck_grid
            )
            encoder_layers.append(
                ConvBlock2d(
                    c_in,
                    c_out,
                    kernel_size=k_down,
                    stride=s_down,
                    padding=p_down,
                    norm=norm,
                )
            )
            if stage_idx > 0:
                encoder_layers.append(
                    ConvBlock2d(
                        c_out,
                        c_out,
                        kernel_size=k_ref,
                        stride=1,
                        padding=p_ref,
                        norm=norm,
                    )
                )
        self.encoder = nn.Sequential(*encoder_layers)

        # Spatial-layout-preserving latent compression to (B, dim5, 1, 1)
        self.bottleneck = ConvBlock2d(
            dim4,
            dim5,
            kernel_size=self.bottleneck_grid,
            stride=1,
            padding=0,
            norm=norm,
            is_singleton=True,
        )

        # Seed spatial expansion (1x1 -> 5x5) followed by subclass-specific upsampling
        self.decoder_seed = nn.Sequential(
            DeconvBlock2d(dim5, dim5, kernel_size=5, stride=1, padding=0, norm=norm),
            ConvBlock2d(dim5, dim5, kernel_size=3, stride=1, padding=1, norm=norm),
        )
        self.decoder_stages = self._build_decoder_stages(dim1, dim2, dim3, dim4, dim5)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.head = nn.Conv2d(dim1, self.out_size, kernel_size=3, padding=1)

        self._init_weights()

    def _build_decoder_stages(
        self, dim1: int, dim2: int, dim3: int, dim4: int, dim5: int
    ) -> nn.Sequential:
        raise NotImplementedError

    def _init_weights(self) -> None:
        """Initialize hidden convs with Kaiming normal and linear head with Xavier."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    module.weight, a=0.2, mode="fan_out", nonlinearity="leaky_relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, in_channels, H_in, W_in)`` to ``(B, out_size, H_out, W_out)``."""
        self.validate_input_shape(x, in_channels=self.in_channels, strict_spatial=False)
        x = self.encoder(x)
        grid = (self.bottleneck_grid, self.bottleneck_grid)
        if x.shape[2:] != grid:
            x = F.interpolate(x, size=grid, mode="bilinear", align_corners=False)
        x = self.bottleneck(x)
        x = self.decoder_seed(x)
        x = self.decoder_stages(x)
        if x.shape[2:] != self.output_spatial_shape:
            x = F.interpolate(
                x,
                size=self.output_spatial_shape,
                mode="bilinear",
                align_corners=False,
            )
        x = self.head(self.dropout(x))
        if self.output_activation == "tanh":
            x = torch.tanh(x)
        return x


@register_model("inversionnet")
class InversionNet(_InversionNetBase):
    """Skip-free 2D encoder-decoder with transposed-convolution upsampling.

    Generalizes the InversionNet encoder-decoder architecture to arbitrary
    2D input channel counts, input resolutions (square or rectangular), and
    output field resolutions ``(C, H_out, W_out)``.

    ~17.3M parameters with default ``base_channels=32``.

    Args:
        in_shape: 2D input spatial dimensions ``(H_in, W_in)``.
        out_size: Number of output field channels.
        out_shape: Optional canonical output shape ``(C, H_out, W_out)``.
            Defaults to ``(out_size, *in_shape)``.
        in_channels: Number of input channels (default: 1).
        base_channels: Channel width of the first stage; stages scale by
            ``[1, 2, 4, 8, 16]`` (default: 32).
        bottleneck_grid: Spatial grid size fed into the bottleneck convolution
            (default: 4).
        dropout_rate: Dropout applied before the final output projection
            (default: 0.0).
        norm: Normalization layer: ``"gn"`` (GroupNorm, default), ``"bn"``
            (BatchNorm2d), or ``"none"``.
        output_activation: Optional output nonlinearity: ``None`` (linear,
            default) or ``"tanh"``.

    Example:
        >>> model = InversionNet(
        ...     in_shape=(128, 64), out_size=1, out_shape=(1, 70, 70), in_channels=5
        ... )
        >>> x = torch.randn(2, 5, 128, 64)
        >>> model(x).shape
        torch.Size([2, 1, 70, 70])
    """

    def _build_decoder_stages(
        self, dim1: int, dim2: int, dim3: int, dim4: int, dim5: int
    ) -> nn.Sequential:
        pairs = [(dim5, dim4), (dim4, dim3), (dim3, dim2), (dim2, dim1)]
        layers: list[nn.Module] = []
        for c_in, c_out in pairs:
            layers.append(
                DeconvBlock2d(
                    c_in, c_out, kernel_size=4, stride=2, padding=1, norm=self.norm
                )
            )
            layers.append(
                ConvBlock2d(
                    c_out, c_out, kernel_size=3, stride=1, padding=1, norm=self.norm
                )
            )
        return nn.Sequential(*layers)

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """Return default model hyperparameters."""
        return {
            "base_channels": 32,
            "bottleneck_grid": 4,
            "dropout_rate": 0.0,
            "norm": "gn",
            "output_activation": None,
        }

    def __repr__(self) -> str:
        return (
            f"InversionNet(in_channels={self.in_channels}, in_shape={self.in_shape}, "
            f"out_shape={self.out_shape})"
        )


@register_model("inversionnet_resize")
class InversionNetResize(_InversionNetBase):
    """Skip-free 2D encoder-decoder with resize-convolution upsampling.

    Replaces strided transposed convolutions in the decoder with explicit
    spatial interpolation followed by standard 3x3 convolutions to suppress
    checkerboard artifacts while supporting arbitrary 2D input and output
    resolutions and channel counts.

    ~16.1M parameters with default ``base_channels=32``.

    Args:
        in_shape: 2D input spatial dimensions ``(H_in, W_in)``.
        out_size: Number of output field channels.
        out_shape: Optional canonical output shape ``(C, H_out, W_out)``.
            Defaults to ``(out_size, *in_shape)``.
        in_channels: Number of input channels (default: 1).
        base_channels: Channel width of the first stage; stages scale by
            ``[1, 2, 4, 8, 16]`` (default: 32).
        bottleneck_grid: Spatial grid size fed into the bottleneck convolution
            (default: 4).
        dropout_rate: Dropout applied before the final output projection
            (default: 0.0).
        norm: Normalization layer: ``"gn"`` (GroupNorm, default), ``"bn"``
            (BatchNorm2d), or ``"none"``.
        output_activation: Optional output nonlinearity: ``None`` (linear,
            default) or ``"tanh"``.
        upsample_mode: Interpolation mode for decoder upsampling: ``"bilinear"``
            (default) or ``"nearest"``.

    Example:
        >>> model = InversionNetResize(
        ...     in_shape=(64, 64), out_size=2, out_shape=(2, 48, 48), in_channels=3
        ... )
        >>> x = torch.randn(2, 3, 64, 64)
        >>> model(x).shape
        torch.Size([2, 2, 48, 48])
    """

    def __init__(
        self,
        in_shape: SpatialShape,
        out_size: int,
        out_shape: tuple[int, ...] | None = None,
        in_channels: int = 1,
        base_channels: int = 32,
        bottleneck_grid: int = 4,
        dropout_rate: float = 0.0,
        norm: str = "gn",
        output_activation: str | None = None,
        upsample_mode: str = "bilinear",
        **kwargs: Any,
    ) -> None:
        if upsample_mode not in VALID_UPSAMPLE_MODES:
            raise ValueError(
                f"Unsupported upsample_mode {upsample_mode!r}; "
                f"choose from {VALID_UPSAMPLE_MODES}."
            )
        self.upsample_mode = upsample_mode
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            out_shape=out_shape,
            in_channels=in_channels,
            base_channels=base_channels,
            bottleneck_grid=bottleneck_grid,
            dropout_rate=dropout_rate,
            norm=norm,
            output_activation=output_activation,
            **kwargs,
        )

    def _build_decoder_stages(
        self, dim1: int, dim2: int, dim3: int, dim4: int, dim5: int
    ) -> nn.Sequential:
        pairs = [(dim5, dim4), (dim4, dim3), (dim3, dim2), (dim2, dim1)]
        layers: list[nn.Module] = []
        for c_in, c_out in pairs:
            layers.append(
                ResizeConvBlock2d(
                    c_in, c_out, upsample_mode=self.upsample_mode, norm=self.norm
                )
            )
            layers.append(
                ConvBlock2d(
                    c_out, c_out, kernel_size=3, stride=1, padding=1, norm=self.norm
                )
            )
        return nn.Sequential(*layers)

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """Return default model hyperparameters."""
        return {
            "base_channels": 32,
            "bottleneck_grid": 4,
            "dropout_rate": 0.0,
            "norm": "gn",
            "output_activation": None,
            "upsample_mode": "bilinear",
        }

    def __repr__(self) -> str:
        return (
            f"InversionNetResize(in_channels={self.in_channels}, "
            f"in_shape={self.in_shape}, out_shape={self.out_shape}, "
            f"upsample_mode={self.upsample_mode!r})"
        )
