"""
CaFormer: MetaFormer with Convolution and Attention
====================================================

CaFormer implements the MetaFormer architecture using depthwise separable
convolutions in early stages and vanilla self-attention in later stages.

**Key Features**:
    - MetaFormer principle: architecture > token mixer
    - Hybrid: Conv (early) + Attention (late)
    - StarReLU activation for efficiency
    - State-of-the-art on ImageNet (85.5%)

**Variants**:
    - caformer_s18: 26M params
    - caformer_s36: 39M params
    - caformer_m36: 56M params

**Related Models**:
    - PoolFormer: Uses pooling instead of attention
    - ConvFormer: Uses only convolutions

**Requirements**:
    - timm >= 0.9.0 (for CaFormer models)

Reference:
    Yu, W., et al. (2023). MetaFormer Baselines for Vision.
    TPAMI 2023. https://arxiv.org/abs/2210.13452

Author: Ductho Le (ductho.le@outlook.com)
"""

import torch

from wavedl.models._pretrained_utils import build_regression_head
from wavedl.models.base import BaseModel
from wavedl.models.registry import register_model


__all__ = [
    "CaFormerBase",
    "CaFormerM36",
    "CaFormerS18",
    "CaFormerS36",
    "PoolFormerS12",
]


# =============================================================================
# CAFORMER BASE CLASS
# =============================================================================


class CaFormerBase(BaseModel):
    """
    CaFormer base class wrapping timm implementation.

    MetaFormer with conv (early) + attention (late) token mixing.
    2D only.
    """

    def __init__(
        self,
        in_shape: tuple[int, int],
        out_size: int,
        model_name: str = "caformer_s18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.3,
        **kwargs,
    ):
        super().__init__(in_shape, out_size)

        if len(in_shape) != 2:
            raise ValueError(f"CaFormer requires 2D input (H, W), got {len(in_shape)}D")

        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.model_name = model_name

        # Try to load from timm
        try:
            import timm

            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # Remove classifier
            )

            # Get feature dimension (eval mode to preserve pretrained BN stats)
            with torch.no_grad():
                self.backbone.eval()
                dummy = torch.zeros(1, 3, *in_shape)
                features = self.backbone(dummy)
                in_features = features.shape[-1]
                self.backbone.train()

        except ImportError:
            raise ImportError(
                "timm >= 0.9.0 is required for CaFormer. "
                "Install with: pip install timm>=0.9.0"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load CaFormer model '{model_name}': {e}"
            ) from e

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
                "Could not adapt CaFormer input channels. Model may fail.",
                stacklevel=2,
            )

    def _freeze_backbone(self):
        """Freeze feature extraction while retaining the regression head."""
        self._freeze_backbone_modules()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


# =============================================================================
# REGISTERED VARIANTS
# =============================================================================


@register_model("caformer_s18")
class CaFormerS18(CaFormerBase):
    """
    CaFormer-S18: ~23.2M backbone parameters.

    MetaFormer with conv + attention.
    2D only.

    Example:
        >>> model = CaFormerS18(in_shape=(224, 224), out_size=3)
        >>> x = torch.randn(4, 1, 224, 224)
        >>> out = model(x)  # (4, 3)
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="caformer_s18",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"CaFormer_S18(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )


@register_model("caformer_s36")
class CaFormerS36(CaFormerBase):
    """
    CaFormer-S36: ~36.2M backbone parameters.

    Deeper MetaFormer variant.
    2D only.
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="caformer_s36",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"CaFormer_S36(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )


@register_model("caformer_m36")
class CaFormerM36(CaFormerBase):
    """
    CaFormer-M36: ~52.6M backbone parameters.

    Medium-size MetaFormer variant.
    2D only.
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="caformer_m36",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"CaFormer_M36(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )


@register_model("poolformer_s12")
class PoolFormerS12(CaFormerBase):
    """
    PoolFormer-S12: ~11.4M backbone parameters.

    MetaFormer with simple pooling token mixer.
    Proves that architecture matters more than complex attention.
    2D only.
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="poolformer_s12",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"PoolFormer_S12(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )
