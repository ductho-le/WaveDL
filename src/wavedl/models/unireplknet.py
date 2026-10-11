"""
UniRepLKNet: Universal Large-Kernel ConvNet for Regression
===========================================================

A dimension-agnostic implementation of UniRepLKNet featuring ultra-large kernels
(up to 31x31) for capturing long-range dependencies. Particularly effective for
wave-based problems where spatial correlations span large distances.

**Key Features**:
    - Large kernels (13x13 to 31x31) via efficient decomposition
    - Dilated small kernel reparam for efficient training
    - SE (Squeeze-and-Excitation) attention
    - GRN (Global Response Normalization) from ConvNeXt V2
    - Dimension-agnostic: supports 1D, 2D, 3D inputs

**Variants**:
    - unireplknet_tiny: 31M params, depths [3,3,18,3], dims [80,160,320,640]
    - unireplknet_small: 56M params, depths [3,3,27,3], dims [96,192,384,768]
    - unireplknet_base: 97M params, depths [3,3,27,3], dims [128,256,512,1024]

**Why Large Kernels for Wave Problems**:
    - Dispersion curves: Long-range frequency-wavenumber correlations
    - B-scans: Defect signatures span many pixels
    - Time-series: Capture multiple wave periods without deep stacking

Reference:
    Ding, X., et al. (2024). UniRepLKNet: A Universal Perception Large-Kernel
    ConvNet for Audio, Video, Point Cloud, Time-Series and Image Recognition.
    CVPR 2024. https://arxiv.org/abs/2311.15599

Author: Ductho Le (ductho.le@outlook.com)
"""

import copy
from typing import Any, ClassVar

import torch
import torch.nn as nn

from wavedl.models._pretrained_utils import (
    DropPath,
    LayerNormNd,
    get_conv_layer,
    get_grn_layer,
    get_pool_layer,
)
from wavedl.models.base import BaseModel, SpatialShape
from wavedl.models.registry import register_model


__all__ = [
    "UniRepLKNetBase",
    "UniRepLKNetBaseLarge",
    "UniRepLKNetSmall",
    "UniRepLKNetTiny",
]


# =============================================================================
# LARGE KERNEL CONVOLUTION BLOCK
# =============================================================================


class LargeKernelConv(nn.Module):
    """
    Large kernel depthwise convolution.

    Implements efficient large kernel convolutions following UniRepLKNet.
    Uses a single large depthwise conv for simplicity and reliability.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dim: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        Conv = get_conv_layer(dim)
        padding = kernel_size // 2

        # Large kernel depthwise conv
        self.conv = Conv(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.

    Adaptively recalibrates channel-wise feature responses by explicitly
    modeling interdependencies between channels.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, reduced, bias=False)
        self.fc2 = nn.Linear(reduced, channels, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global average pooling
        if x.ndim == 3:  # 1D: (B, C, L)
            gap = x.mean(dim=2)
        elif x.ndim == 4:  # 2D: (B, C, H, W)
            gap = x.mean(dim=(2, 3))
        else:  # 3D: (B, C, D, H, W)
            gap = x.mean(dim=(2, 3, 4))

        # FC layers
        scale = self.act(self.fc1(gap))
        scale = torch.sigmoid(self.fc2(scale))

        # Reshape for broadcasting
        if x.ndim == 3:
            scale = scale.unsqueeze(-1)
        elif x.ndim == 4:
            scale = scale.unsqueeze(-1).unsqueeze(-1)
        else:
            scale = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return x * scale


# =============================================================================
# UNIREPLKNET BLOCK
# =============================================================================


class UniRepLKNetBlock(nn.Module):
    """
    UniRepLKNet block with large kernel convolution, SE attention, and GRN.

    Architecture:
        Input → LargeKernelConv → LayerNorm → SE → Linear → GELU → GRN → Linear → Residual

    This combines the effective receptive field of large kernels with the
    feature recalibration of SE attention and the inter-channel competition
    of GRN from ConvNeXt V2.
    """

    def __init__(
        self,
        dim: int,
        spatial_dim: int,
        kernel_size: int = 13,
        drop_path: float = 0.0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.spatial_dim = spatial_dim

        GRN = get_grn_layer(spatial_dim)

        # Large kernel depthwise conv
        self.large_kernel = LargeKernelConv(
            dim, kernel_size=kernel_size, dim=spatial_dim
        )

        # Layer norm (applied in channels-last format)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # SE attention
        self.se = SEBlock(dim)

        # MLP with expansion
        hidden_dim = int(dim * mlp_ratio)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.grn = GRN(hidden_dim)
        self.pwconv2 = nn.Linear(hidden_dim, dim)

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def _to_channels_last(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from channels-first to channels-last."""
        if self.spatial_dim == 1:
            return x.permute(0, 2, 1)  # (B, C, L) -> (B, L, C)
        elif self.spatial_dim == 2:
            return x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        else:
            return x.permute(0, 2, 3, 4, 1)  # (B, C, D, H, W) -> (B, D, H, W, C)

    def _to_channels_first(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from channels-last to channels-first."""
        if self.spatial_dim == 1:
            return x.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)
        elif self.spatial_dim == 2:
            return x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        else:
            return x.permute(0, 4, 1, 2, 3)  # (B, D, H, W, C) -> (B, C, D, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Large kernel conv (channels-first)
        x = self.large_kernel(x)

        # SE attention (channels-first)
        x = self.se(x)

        # LayerNorm + MLP (channels-last)
        x = self._to_channels_last(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)

        # GRN (channels-first)
        x = self._to_channels_first(x)
        x = self.grn(x)

        # Final projection (channels-last)
        x = self._to_channels_last(x)
        x = self.pwconv2(x)
        x = self._to_channels_first(x)

        # Residual + drop path
        x = residual + self.drop_path(x)
        return x


# =============================================================================
# UNIREPLKNET BASE CLASS
# =============================================================================


class UniRepLKNetBase(BaseModel):
    """
    UniRepLKNet base class for regression.

    Dimension-agnostic implementation supporting 1D, 2D, and 3D inputs.
    Features large kernels for capturing long-range dependencies in wave data.

    Architecture:
        1. Stem: 4x downsampling conv
        2. 4 stages with UniRepLKNet blocks
        3. Downsampling between stages
        4. Global pooling + regression head
    """

    def __init__(
        self,
        in_shape: SpatialShape,
        out_size: int,
        depths: list[int],
        dims: list[int],
        kernel_sizes: list[int] | None = None,
        drop_path_rate: float = 0.1,
        dropout_rate: float = 0.3,
        **kwargs,
    ):
        super().__init__(in_shape, out_size)

        self.dim = len(in_shape)
        self.depths = depths
        self.dims = dims

        # Validate minimum spatial size (matches ConvNeXt/ConvNeXtV2 siblings):
        # stem stride-4 × (len(depths)-1) stride-2 downsamplers.
        min_size = 4 * (2 ** (len(depths) - 1))
        for i, s in enumerate(in_shape):
            if s < min_size:
                raise ValueError(
                    f"UniRepLKNet requires each spatial axis >= {min_size}, "
                    f"but axis {i} has size {s}. "
                    f"(stem stride 4 x {len(depths) - 1} stride-2 "
                    f"downsamplers = {min_size}x)"
                )

        # Default kernel sizes: larger in early stages, smaller in later stages
        # Early stages: large receptive field for low-level features
        # Later stages: smaller kernels sufficient for high-level features
        if kernel_sizes is None:
            kernel_sizes = [31, 29, 17, 13]

        Conv = get_conv_layer(self.dim)
        Pool = get_pool_layer(self.dim)

        # Stem: aggressive 4x downsampling (like ConvNeXt)
        self.stem = nn.Sequential(
            Conv(1, dims[0], kernel_size=4, stride=4),
            LayerNormNd(dims[0], self.dim),
        )

        # Stochastic depth decay
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        cur = 0

        for i in range(len(depths)):
            # Adjust kernel size for 1D (can use larger kernels)
            # Ensure kernel size is always odd for proper same-padding
            kernel_size = kernel_sizes[i]
            if self.dim == 1:
                kernel_size = min(kernel_size * 2 - 1, 63)  # Keep odd for 1D

            stage = nn.Sequential(
                *[
                    UniRepLKNetBlock(
                        dim=dims[i],
                        spatial_dim=self.dim,
                        kernel_size=kernel_size,
                        drop_path=dp_rates[cur + j],
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

            # Downsample between stages (except after last)
            if i < len(depths) - 1:
                downsample = nn.Sequential(
                    LayerNormNd(dims[i], self.dim),
                    Conv(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
                self.downsamples.append(downsample)

        # Global pooling and head
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.global_pool = Pool(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(dims[-1], dims[-1] // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(dims[-1] // 2, out_size),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with truncated normal."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 1, *in_shape)

        Returns:
            Output tensor (B, out_size)
        """
        x = self.stem(x)

        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        # Global pooling
        x = self.global_pool(x)
        x = x.flatten(1)

        # Final norm and head
        x = self.norm(x)
        x = self.head(x)

        return x

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        return {
            "depths": [3, 3, 18, 3],
            "dims": [80, 160, 320, 640],
            "kernel_sizes": [31, 29, 17, 13],
            "drop_path_rate": 0.1,
            "dropout_rate": 0.3,
            **copy.deepcopy(getattr(cls, "_variant_defaults", {})),
        }


# =============================================================================
# REGISTERED VARIANTS
# =============================================================================


@register_model("unireplknet_tiny")
class UniRepLKNetTiny(UniRepLKNetBase):
    """
    UniRepLKNet Tiny: ~30.8M backbone parameters.

    Large kernels [31, 29, 17, 13] for capturing long-range wave patterns.
    Depths [3,3,18,3], Dims [80,160,320,640].
    Supports 1D, 2D, 3D inputs.

    Example:
        >>> model = UniRepLKNetTiny(in_shape=(256, 256), out_size=3)
        >>> x = torch.randn(4, 1, 256, 256)
        >>> out = model(x)  # (4, 3)
    """

    _variant_defaults: ClassVar[dict[str, Any]] = {
        "depths": [3, 3, 18, 3],
        "dims": [80, 160, 320, 640],
        "kernel_sizes": [31, 29, 17, 13],
    }

    def __init__(self, in_shape: SpatialShape, out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            **{**copy.deepcopy(self._variant_defaults), **kwargs},
        )

    def __repr__(self) -> str:
        return (
            f"UniRepLKNet_Tiny({self.dim}D, in_shape={self.in_shape}, "
            f"out_size={self.out_size})"
        )


@register_model("unireplknet_small")
class UniRepLKNetSmall(UniRepLKNetBase):
    """
    UniRepLKNet Small: ~56.0M backbone parameters.

    Large kernels [31, 29, 17, 13] for capturing long-range wave patterns.
    Depths [3,3,27,3], Dims [96,192,384,768].
    Supports 1D, 2D, 3D inputs.
    """

    _variant_defaults: ClassVar[dict[str, Any]] = {
        "depths": [3, 3, 27, 3],
        "dims": [96, 192, 384, 768],
        "kernel_sizes": [31, 29, 17, 13],
    }

    def __init__(self, in_shape: SpatialShape, out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            **{**copy.deepcopy(self._variant_defaults), **kwargs},
        )

    def __repr__(self) -> str:
        return (
            f"UniRepLKNet_Small({self.dim}D, in_shape={self.in_shape}, "
            f"out_size={self.out_size})"
        )


@register_model("unireplknet_base")
class UniRepLKNetBaseLarge(UniRepLKNetBase):
    """
    UniRepLKNet Base: ~97.6M backbone parameters.

    Large kernels [31, 29, 17, 13] for capturing long-range wave patterns.
    Depths [3,3,27,3], Dims [128,256,512,1024].
    Supports 1D, 2D, 3D inputs.
    """

    _variant_defaults: ClassVar[dict[str, Any]] = {
        "depths": [3, 3, 27, 3],
        "dims": [128, 256, 512, 1024],
        "kernel_sizes": [31, 29, 17, 13],
    }

    def __init__(self, in_shape: SpatialShape, out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            **{**copy.deepcopy(self._variant_defaults), **kwargs},
        )

    def __repr__(self) -> str:
        return (
            f"UniRepLKNet_Base({self.dim}D, in_shape={self.in_shape}, "
            f"out_size={self.out_size})"
        )
