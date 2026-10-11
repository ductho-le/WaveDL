"""
ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders
========================================================================

ConvNeXt V2 improves upon V1 by replacing LayerScale with Global Response
Normalization (GRN), which enhances inter-channel feature competition.

**Key Changes from V1**:
    - GRN layer replaces LayerScale
    - Better compatibility with masked autoencoder pretraining
    - Prevents feature collapse in deep networks

**Variants**:
    - convnext_v2_tiny: 28M params, depths [3,3,9,3], dims [96,192,384,768]
    - convnext_v2_small: 50M params, depths [3,3,27,3], dims [96,192,384,768]
    - convnext_v2_base: 89M params, depths [3,3,27,3], dims [128,256,512,1024]
    - convnext_v2_tiny_pretrained: 2D only, ImageNet weights (via timm)

**Supports**: 1D, 2D, 3D inputs

Reference:
    Woo, S., et al. (2023). ConvNeXt V2: Co-designing and Scaling ConvNets
    with Masked Autoencoders. CVPR 2023.
    https://arxiv.org/abs/2301.00808

Author: Ductho Le (ductho.le@outlook.com)
"""

import copy
from typing import Any, ClassVar

import torch
import torch.nn as nn

from wavedl.models._pretrained_utils import (
    DropPath,
    LayerNormNd,
    build_regression_head,
    get_conv_layer,
    get_grn_layer,
    get_pool_layer,
)
from wavedl.models.base import BaseModel, SpatialShape
from wavedl.models.registry import register_model


__all__ = [
    "ConvNeXtV2Base",
    "ConvNeXtV2BaseLarge",
    "ConvNeXtV2Small",
    "ConvNeXtV2Tiny",
    "ConvNeXtV2TinyPretrained",
]


# =============================================================================
# CONVNEXT V2 BLOCK
# =============================================================================


class ConvNeXtV2Block(nn.Module):
    """
    ConvNeXt V2 Block with GRN and residual scale.

    Architecture:
        Input → DwConv → LayerNorm → Linear → GELU → GRN → Linear
              → ResScale → Residual

    GRN is the key V2 innovation for feature normalization (replaces
    LayerScale's role as regularizer). The residual scale (init=1e-6)
    stabilizes gradient flow for from-scratch training by suppressing
    the residual branch near init — the same role LayerScale plays
    in V1, but decoupled from the normalization that GRN provides.

    Note: The official ConvNeXt V2 omits this scale because it assumes
    MAE pretraining. For from-scratch regression, it is essential.
    """

    def __init__(
        self,
        dim: int,
        spatial_dim: int,
        drop_path: float = 0.0,
        mlp_ratio: float = 4.0,
        res_scale_init: float = 1e-6,
    ):
        super().__init__()
        self.spatial_dim = spatial_dim

        Conv = get_conv_layer(spatial_dim)
        GRN = get_grn_layer(spatial_dim)

        # Depthwise convolution
        kernel_size = 7
        padding = 3
        self.dwconv = Conv(
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim
        )

        # LayerNorm (applied in forward with permutation)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # MLP with expansion
        hidden_dim = int(dim * mlp_ratio)
        self.pwconv1 = nn.Linear(dim, hidden_dim)  # Expansion
        self.act = nn.GELU()
        self.grn = GRN(hidden_dim)  # GRN after expansion (key V2 change)
        self.pwconv2 = nn.Linear(hidden_dim, dim)  # Projection

        # Residual scale: suppresses residual branch at init so blocks
        # start as identity, preventing gradient explosion in deep networks.
        # Serves the same init-stability role as V1's LayerScale.
        self.res_scale = nn.Parameter(
            res_scale_init * torch.ones(dim), requires_grad=True
        )

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: This forward has 4 permute round-trips (vs 2 in ConvNeXt V1)
        # because GRN operates in channels-first between channels-last linear
        # layers. A channels-last GRN implementation would eliminate 2 permutes.
        residual = x

        # Depthwise conv
        x = self.dwconv(x)

        # Move channels to last for LayerNorm and Linear layers
        if self.spatial_dim == 1:
            x = x.permute(0, 2, 1)  # (B, C, L) -> (B, L, C)
        elif self.spatial_dim == 2:
            x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        else:  # 3D
            x = x.permute(0, 2, 3, 4, 1)  # (B, C, D, H, W) -> (B, D, H, W, C)

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)

        # Move back to channels-first for GRN
        if self.spatial_dim == 1:
            x = x.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)
        elif self.spatial_dim == 2:
            x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        else:  # 3D
            x = x.permute(0, 4, 1, 2, 3)  # (B, D, H, W, C) -> (B, C, D, H, W)

        # Apply GRN (the key V2 innovation)
        x = self.grn(x)

        # Move to channels-last for final projection
        if self.spatial_dim == 1:
            x = x.permute(0, 2, 1)
        elif self.spatial_dim == 2:
            x = x.permute(0, 2, 3, 1)
        else:
            x = x.permute(0, 2, 3, 4, 1)

        x = self.pwconv2(x)

        # Residual scale (like LayerScale, suppresses branch at init)
        x = self.res_scale * x

        # Move back to channels-first
        if self.spatial_dim == 1:
            x = x.permute(0, 2, 1)
        elif self.spatial_dim == 2:
            x = x.permute(0, 3, 1, 2)
        else:
            x = x.permute(0, 4, 1, 2, 3)

        x = residual + self.drop_path(x)
        return x


# =============================================================================
# CONVNEXT V2 BASE CLASS
# =============================================================================


class ConvNeXtV2Base(BaseModel):
    """
    ConvNeXt V2 base class for regression.

    Dimension-agnostic implementation supporting 1D, 2D, and 3D inputs.
    Uses GRN (Global Response Normalization) instead of LayerScale.
    """

    def __init__(
        self,
        in_shape: SpatialShape,
        out_size: int,
        depths: list[int],
        dims: list[int],
        drop_path_rate: float = 0.1,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(in_shape, out_size)

        self.dim = len(in_shape)
        self.depths = depths
        self.dims = dims

        # Validate minimum spatial size:
        # stem stride-4 × (len(depths)-1) stride-2 downsamplers
        min_size = 4 * (2 ** (len(depths) - 1))
        for i, s in enumerate(in_shape):
            if s < min_size:
                raise ValueError(
                    f"ConvNeXtV2 requires each spatial axis >= {min_size}, "
                    f"but axis {i} has size {s}. "
                    f"(stem stride 4 x {len(depths) - 1} stride-2 "
                    f"downsamplers = {min_size}x)"
                )

        Conv = get_conv_layer(self.dim)
        Pool = get_pool_layer(self.dim)

        # Stem: aggressive downsampling (4x stride like ConvNeXt)
        self.stem = nn.Sequential(
            Conv(1, dims[0], kernel_size=4, stride=4),
            LayerNormNd(dims[0], self.dim),
        )

        # Stochastic depth decay rule
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        cur = 0

        for i in range(len(depths)):
            # Stage: sequence of ConvNeXt V2 blocks
            stage = nn.Sequential(
                *[
                    ConvNeXtV2Block(
                        dim=dims[i],
                        spatial_dim=self.dim,
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
        """Initialize weights with truncated normal.

        All conv and linear layers use trunc-normal(std=0.02). Residual branch
        suppression at init is handled by res_scale (init=1e-6), not by
        zero-initializing any specific layer.
        """
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
            "depths": [3, 3, 9, 3],
            "dims": [96, 192, 384, 768],
            "drop_path_rate": 0.1,
            "dropout_rate": 0.1,
            **copy.deepcopy(getattr(cls, "_variant_defaults", {})),
        }


# =============================================================================
# REGISTERED VARIANTS
# =============================================================================


@register_model("convnext_v2_tiny")
class ConvNeXtV2Tiny(ConvNeXtV2Base):
    """
    ConvNeXt V2 Tiny: ~27.9M backbone parameters.

    Depths [3,3,9,3], Dims [96,192,384,768].
    Supports 1D, 2D, 3D inputs.

    Example:
        >>> model = ConvNeXtV2Tiny(in_shape=(64, 64), out_size=3)
        >>> x = torch.randn(4, 1, 64, 64)
        >>> out = model(x)  # (4, 3)
    """

    _variant_defaults: ClassVar[dict[str, Any]] = {
        "depths": [3, 3, 9, 3],
        "dims": [96, 192, 384, 768],
    }

    def __init__(self, in_shape: SpatialShape, out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            **{**copy.deepcopy(self._variant_defaults), **kwargs},
        )

    def __repr__(self) -> str:
        return (
            f"ConvNeXtV2_Tiny({self.dim}D, in_shape={self.in_shape}, "
            f"out_size={self.out_size})"
        )


@register_model("convnext_v2_small")
class ConvNeXtV2Small(ConvNeXtV2Base):
    """
    ConvNeXt V2 Small: ~49.6M backbone parameters.

    Depths [3,3,27,3], Dims [96,192,384,768].
    Supports 1D, 2D, 3D inputs.
    """

    _variant_defaults: ClassVar[dict[str, Any]] = {
        "depths": [3, 3, 27, 3],
        "dims": [96, 192, 384, 768],
    }

    def __init__(self, in_shape: SpatialShape, out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            **{**copy.deepcopy(self._variant_defaults), **kwargs},
        )

    def __repr__(self) -> str:
        return (
            f"ConvNeXtV2_Small({self.dim}D, in_shape={self.in_shape}, "
            f"out_size={self.out_size})"
        )


@register_model("convnext_v2_base")
class ConvNeXtV2BaseLarge(ConvNeXtV2Base):
    """
    ConvNeXt V2 Base: ~87.7M backbone parameters.

    Depths [3,3,27,3], Dims [128,256,512,1024].
    Supports 1D, 2D, 3D inputs.
    """

    _variant_defaults: ClassVar[dict[str, Any]] = {
        "depths": [3, 3, 27, 3],
        "dims": [128, 256, 512, 1024],
    }

    def __init__(self, in_shape: SpatialShape, out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            **{**copy.deepcopy(self._variant_defaults), **kwargs},
        )

    def __repr__(self) -> str:
        return (
            f"ConvNeXtV2_Base({self.dim}D, in_shape={self.in_shape}, "
            f"out_size={self.out_size})"
        )


# =============================================================================
# PRETRAINED VARIANT (2D ONLY, via timm)
# =============================================================================


@register_model("convnext_v2_tiny_pretrained")
class ConvNeXtV2TinyPretrained(BaseModel):
    """
    ConvNeXt V2 Tiny with ImageNet pretrained weights (2D only).

    Uses timm's ConvNeXt V2 implementation (FCMAE-pretrained, ImageNet
    fine-tuned) with genuine V2 architecture including GRN layers.

    - Adapted input layer for single-channel input
    - Custom regression head replaces classifier

    Args:
        in_shape: (H, W) input shape (2D only)
        out_size: Number of regression targets
        pretrained: Whether to load pretrained weights
        freeze_backbone: Whether to freeze backbone for fine-tuning
        dropout_rate: Dropout rate for regression head
    """

    def __init__(
        self,
        in_shape: tuple[int, int],
        out_size: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.3,
        **kwargs,
    ):
        super().__init__(in_shape, out_size)

        if len(in_shape) != 2:
            raise ValueError(
                f"ConvNeXtV2TinyPretrained requires 2D input (H, W), "
                f"got {len(in_shape)}D"
            )

        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone

        # Load real ConvNeXt V2 from timm
        try:
            import timm

            self.backbone = timm.create_model(
                "convnextv2_tiny",
                pretrained=pretrained,
                num_classes=0,  # Remove classifier
            )

            # Get feature dimension.
            # IMPORTANT: The probe MUST happen before _adapt_input_channels()
            # because the backbone still expects 3-channel input here.
            with torch.no_grad():
                self.backbone.eval()
                dummy = torch.zeros(1, 3, *in_shape)
                features = self.backbone(dummy)
                in_features = features.shape[-1]
                self.backbone.train()

        except ImportError:
            raise ImportError(
                "timm >= 0.9.0 is required for pretrained ConvNeXt V2. "
                "Install with: pip install timm>=0.9.0"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ConvNeXt V2 model 'convnextv2_tiny': {e}"
            )

        # Adapt input channels (3 -> 1)
        self._adapt_input_channels()

        # Regression head
        self.head = build_regression_head(in_features, out_size, dropout_rate)

        if freeze_backbone:
            self._freeze_backbone()

    def _adapt_input_channels(self):
        """Adapt first conv layer for single-channel input."""
        from wavedl.models._pretrained_utils import find_and_adapt_input_convs

        adapted_count = find_and_adapt_input_convs(
            self.backbone, pretrained=self.pretrained, adapt_all=False
        )

        if adapted_count == 0:
            import warnings

            warnings.warn(
                "Could not adapt ConvNeXt V2 input channels. Model may fail.",
                stacklevel=2,
            )

    def _freeze_backbone(self):
        """Freeze feature extraction while retaining the regression head."""
        self._freeze_backbone_modules()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def __repr__(self) -> str:
        return (
            f"ConvNeXtV2_Tiny_Pretrained(in_shape={self.in_shape}, "
            f"out_size={self.out_size}, pretrained={self.pretrained})"
        )
