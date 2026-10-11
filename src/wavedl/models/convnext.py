"""
ConvNeXt: A Modern Convolutional Neural Network
=================================================

A dimension-agnostic ConvNeXt implementation inspired by Vision Transformers.
Features: inverted bottleneck, LayerNorm, GELU activation, depthwise convolutions.

**Dimensionality Support**:
    - 1D: Waveforms, signals, time-series (N, 1, L) → Conv1d
    - 2D: Images, spectrograms (N, 1, H, W) → Conv2d
    - 3D: Volumetric data, CT/MRI (N, 1, D, H, W) → Conv3d

**Variants**:
    - convnext_tiny: Smallest (~27.8M backbone params for 2D)
    - convnext_small: Medium (~49.5M backbone params for 2D)
    - convnext_base: Standard (~87.6M backbone params for 2D)

References:
    Liu, Z., et al. (2022). A ConvNet for the 2020s.
    CVPR 2022. https://arxiv.org/abs/2201.03545

Author: Ductho Le (ductho.le@outlook.com)
"""

from typing import Any

import torch
import torch.nn as nn

from wavedl.models._pretrained_utils import (
    DropPath,
    LayerNormNd,
    get_conv_layer,
    get_pool_layer,
)
from wavedl.models.base import BaseModel, SpatialShape
from wavedl.models.registry import register_model


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block matching the official Facebook implementation.

    Uses the second variant from the paper which is slightly faster in PyTorch:
    1. DwConv (channels-first)
    2. Permute to channels-last
    3. LayerNorm → Linear → GELU → Linear (all channels-last)
    4. LayerScale (gamma * x)
    5. Permute back to channels-first
    6. Residual connection

    The LayerScale mechanism is critical for stable training in deep networks.
    It scales the output by a learnable parameter initialized to 1e-6.

    References:
        Liu, Z., et al. (2022). A ConvNet for the 2020s. CVPR 2022.
        https://github.com/facebookresearch/ConvNeXt
    """

    def __init__(
        self,
        channels: int,
        dim: int = 2,
        expansion_ratio: float = 4.0,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        Conv = get_conv_layer(dim)
        hidden_dim = int(channels * expansion_ratio)

        # Depthwise conv (7x7) - operates in channels-first
        self.dwconv = Conv(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )

        # LayerNorm (channels-last format, using standard nn.LayerNorm)
        self.norm = nn.LayerNorm(channels, eps=1e-6)

        # Pointwise convs implemented with Linear layers (channels-last)
        # This matches the official implementation and is slightly faster
        self.pwconv1 = nn.Linear(channels, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, channels)

        # LayerScale: learnable per-channel scaling (critical for deep networks)
        # Initialized to small value (1e-6) to prevent gradient explosion
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones(channels), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )

        # Stochastic depth (drop path) for regularisation in deep networks
        # Uses the local DropPath from _pretrained_utils (always available)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Depthwise conv in channels-first format
        x = self.dwconv(x)

        # Permute to channels-last for LayerNorm and Linear layers
        if self.dim == 1:
            x = x.permute(0, 2, 1)  # (B, C, L) -> (B, L, C)
        elif self.dim == 2:
            x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        else:
            x = x.permute(0, 2, 3, 4, 1)  # (B, C, D, H, W) -> (B, D, H, W, C)

        # LayerNorm + MLP (all in channels-last)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # Apply LayerScale
        if self.gamma is not None:
            x = self.gamma * x

        # Permute back to channels-first
        if self.dim == 1:
            x = x.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)
        elif self.dim == 2:
            x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        else:
            x = x.permute(0, 4, 1, 2, 3)  # (B, D, H, W, C) -> (B, C, D, H, W)

        # Residual connection with drop path
        x = residual + self.drop_path(x)
        return x


class ConvNeXtBase(BaseModel):
    """
    Base ConvNeXt class for regression tasks.

    Architecture:
    1. Stem: Patchify with stride-4 conv
    2. 4 stages with downsampling between stages
    3. Global average pooling
    4. Regression head
    """

    def __init__(
        self,
        in_shape: SpatialShape,
        out_size: int,
        depths: list[int],
        dims: list[int],
        dropout_rate: float = 0.1,
        drop_path_rate: float | None = None,
        **kwargs,
    ):
        super().__init__(in_shape, out_size)

        self.dim = len(in_shape)
        self.depths = depths
        self.dims = dims
        self.dropout_rate = dropout_rate
        # Stochastic-depth rate, decoupled from head dropout (matches the
        # ConvNeXtV2/UniRepLKNet siblings). Falls back to the previous coupled
        # behavior (dropout_rate * 0.5) when not specified, to preserve defaults.
        self.drop_path_rate = drop_path_rate

        # Validate minimum spatial size:
        # stem stride-4 × (len(depths)-1) stride-2 downsamplers
        min_size = 4 * (2 ** (len(depths) - 1))
        for i, s in enumerate(in_shape):
            if s < min_size:
                raise ValueError(
                    f"ConvNeXt requires each spatial axis >= {min_size}, "
                    f"but axis {i} has size {s}. "
                    f"(stem stride 4 x {len(depths) - 1} stride-2 "
                    f"downsamplers = {min_size}x)"
                )

        Conv = get_conv_layer(self.dim)

        # Stem: Patchify with stride-4 conv (like ViT patch embedding)
        self.stem = nn.Sequential(
            Conv(1, dims[0], kernel_size=4, stride=4), LayerNormNd(dims[0], self.dim)
        )

        # Build stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        # Linearly increasing drop-path rates across all blocks (paper practice)
        total_blocks = sum(depths)
        dp_max = drop_path_rate if drop_path_rate is not None else dropout_rate * 0.5
        dp_rates = torch.linspace(0, dp_max, total_blocks).tolist()
        block_idx = 0

        for i in range(4):
            # Stage: multiple ConvNeXt blocks with per-block drop path rates
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock(dims[i], self.dim, drop_path=dp_rates[block_idx + j])
                    for j in range(depths[i])
                ]
            )
            block_idx += depths[i]
            self.stages.append(stage)

            # Downsample between stages (except after last stage)
            if i < 3:
                downsample = nn.Sequential(
                    LayerNormNd(dims[i], self.dim),
                    Conv(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
                self.downsamples.append(downsample)

        # Global average pooling
        self.global_pool = get_pool_layer(self.dim)(1)

        # Final norm and regression head (eps=1e-6 matches the in-block norms and
        # the ConvNeXtV2/UniRepLKNet siblings; the default 1e-5 was inconsistent)
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(dims[-1], 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, out_size),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Stem
        x = self.stem(x)

        # Stages with downsampling
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
        """Return default configuration."""
        return {"dropout_rate": 0.1}


# =============================================================================
# REGISTERED MODEL VARIANTS
# =============================================================================


@register_model("convnext_tiny")
class ConvNeXtTiny(ConvNeXtBase):
    """
    ConvNeXt-Tiny: Smallest variant.

    ~27.8M backbone parameters (2D). Good for: Limited compute, fast training.

    Args:
        in_shape: (L,), (H, W), or (D, H, W)
        out_size: Number of regression targets
        dropout_rate: Dropout rate (default: 0.1)
    """

    def __init__(self, in_shape: SpatialShape, out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            depths=[3, 3, 9, 3],
            dims=[96, 192, 384, 768],
            **kwargs,
        )

    def __repr__(self) -> str:
        return f"ConvNeXt_Tiny({self.dim}D, in_shape={self.in_shape}, out_size={self.out_size})"


@register_model("convnext_small")
class ConvNeXtSmall(ConvNeXtBase):
    """
    ConvNeXt-Small: Medium variant.

    ~49.5M backbone parameters (2D). Good for: Balanced performance.

    Args:
        in_shape: (L,), (H, W), or (D, H, W)
        out_size: Number of regression targets
        dropout_rate: Dropout rate (default: 0.1)
    """

    def __init__(self, in_shape: SpatialShape, out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            depths=[3, 3, 27, 3],
            dims=[96, 192, 384, 768],
            **kwargs,
        )

    def __repr__(self) -> str:
        return f"ConvNeXt_Small({self.dim}D, in_shape={self.in_shape}, out_size={self.out_size})"


@register_model("convnext_base")
class ConvNeXtBase_(ConvNeXtBase):
    """
    ConvNeXt-Base: Standard variant.

    ~87.6M backbone parameters (2D). Good for: High accuracy, larger datasets.

    Args:
        in_shape: (L,), (H, W), or (D, H, W)
        out_size: Number of regression targets
        dropout_rate: Dropout rate (default: 0.1)
    """

    def __init__(self, in_shape: SpatialShape, out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            depths=[3, 3, 27, 3],
            dims=[128, 256, 512, 1024],
            **kwargs,
        )

    def __repr__(self) -> str:
        return f"ConvNeXt_Base({self.dim}D, in_shape={self.in_shape}, out_size={self.out_size})"


# =============================================================================
# PRETRAINED VARIANT (2D only, ImageNet weights)
# =============================================================================

try:
    from torchvision.models import (
        ConvNeXt_Tiny_Weights,
        convnext_tiny as tv_convnext_tiny,
    )

    CONVNEXT_PRETRAINED_AVAILABLE = True
except ImportError:
    tv_convnext_tiny = None
    ConvNeXt_Tiny_Weights = None
    CONVNEXT_PRETRAINED_AVAILABLE = False


@register_model("convnext_tiny_pretrained")
class ConvNeXtTinyPretrained(BaseModel):
    """
    ConvNeXt-Tiny with ImageNet pretrained weights (2D only).

    ~27.8M backbone parameters. Good for: Transfer learning with modern CNN.

    Args:
        in_shape: (H, W) image dimensions
        out_size: Number of regression targets
        pretrained: Use ImageNet weights (default: True)
        dropout_rate: Dropout rate (default: 0.2)
        freeze_backbone: Freeze backbone for fine-tuning (default: False)
    """

    def __init__(
        self,
        in_shape: tuple[int, int],
        out_size: int,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        freeze_backbone: bool = False,
        **kwargs,
    ):
        super().__init__(in_shape, out_size)

        if not CONVNEXT_PRETRAINED_AVAILABLE:
            raise ImportError(
                "torchvision>=0.13 is required for pretrained ConvNeXt. "
                "Install with: pip install torchvision"
            )

        if len(in_shape) != 2:
            raise ValueError(
                f"Pretrained ConvNeXt requires 2D input (H, W), got {len(in_shape)}D. "
                "For 1D/3D data, use convnext_tiny instead."
            )

        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.freeze_backbone = freeze_backbone

        # Load pretrained model
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = tv_convnext_tiny(weights=weights)

        # Get classifier input features
        in_features = self.backbone.classifier[2].in_features

        # Replace classifier with regression head
        self.backbone.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, out_size),
        )

        # Modify first conv for single-channel input
        from wavedl.models._pretrained_utils import adapt_first_conv_for_single_channel

        adapt_first_conv_for_single_channel(
            self.backbone, "features.0.0", pretrained=pretrained
        )

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze feature extraction while retaining the regression head."""
        self._freeze_backbone_modules(("classifier",))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        return {"pretrained": True, "dropout_rate": 0.2, "freeze_backbone": False}

    def __repr__(self) -> str:
        pt = "pretrained" if self.pretrained else "scratch"
        return f"ConvNeXt_Tiny_Pretrained({pt}, in_shape={self.in_shape}, out_size={self.out_size})"
