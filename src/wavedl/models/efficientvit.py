"""
EfficientViT: Memory-Efficient Vision Transformer with Cascaded Group Attention
================================================================================

EfficientViT (MIT) achieves state-of-the-art speed-accuracy trade-off by using
cascaded group attention (CGA) which reduces computational redundancy in
multi-head self-attention while maintaining model capability.

**Key Features**:
    - Cascaded Group Attention (CGA): Linear complexity attention
    - Memory-efficient design for edge deployment
    - Faster than Swin Transformer with similar accuracy
    - Excellent for real-time NDE applications

**Variants**:
    - efficientvit_m1: 2.6M params (mobile, ultra-lightweight)
    - efficientvit_b1: 7.5M params (small)
    - efficientvit_b2: 21.8M params (medium)
    - efficientvit_l2: 60.5M params (large)

**Requirements**:
    - timm >= 0.9.0 (for EfficientViT models)

Reference:
    Liu, X., et al. (2023). EfficientViT: Memory Efficient Vision Transformer
    with Cascaded Group Attention. CVPR 2023.
    https://arxiv.org/abs/2305.07027

Author: Ductho Le (ductho.le@outlook.com)
"""

import torch

from wavedl.models._pretrained_utils import build_regression_head
from wavedl.models.base import BaseModel
from wavedl.models.registry import register_model


__all__ = [
    "EfficientViTB1",
    "EfficientViTB2",
    "EfficientViTBase",
    "EfficientViTL2",
    "EfficientViTM1",
]


# =============================================================================
# EFFICIENTVIT BASE CLASS
# =============================================================================


class EfficientViTBase(BaseModel):
    """
    EfficientViT base class wrapping timm implementation.

    Uses Cascaded Group Attention for efficient multi-head attention with
    linear complexity. 2D only due to attention structure.

    Args:
        in_shape: (H, W) input shape (2D only)
        out_size: Number of regression targets
        model_name: timm model name
        pretrained: Whether to load pretrained weights
        freeze_backbone: Whether to freeze backbone for fine-tuning
        dropout_rate: Dropout rate for regression head
    """

    def __init__(
        self,
        in_shape: tuple[int, int],
        out_size: int,
        model_name: str = "efficientvit_b0",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.3,
        **kwargs,
    ):
        super().__init__(in_shape, out_size)

        if len(in_shape) != 2:
            raise ValueError(
                f"EfficientViT requires 2D input (H, W), got {len(in_shape)}D"
            )

        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.model_name = model_name
        self._target_size = None
        if "m1" in model_name or model_name.startswith("efficientvit_m"):
            if in_shape != (224, 224):
                self._target_size = (224, 224)
                import warnings

                warnings.warn(
                    f"EfficientViT input {in_shape} will be interpolated to "
                    f"{self._target_size} for attention window compatibility.",
                    UserWarning,
                    stacklevel=3,
                )

        # Load from timm
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
                dummy_shape = (
                    self._target_size if self._target_size is not None else in_shape
                )
                dummy = torch.zeros(1, 3, *dummy_shape)
                features = self.backbone(dummy)
                in_features = features.shape[-1]
                self.backbone.train()

        except ImportError:
            raise ImportError(
                "timm >= 0.9.0 is required for EfficientViT. "
                "Install with: pip install timm>=0.9.0"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load EfficientViT model '{model_name}': {e}"
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
                "Could not adapt EfficientViT input channels. Model may fail.",
                stacklevel=2,
            )

    def _freeze_backbone(self):
        """Freeze feature extraction while retaining the regression head."""
        self._freeze_backbone_modules()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_target_size", None) is not None:
            _, _, h, w = x.shape
            if (h, w) != self._target_size:
                x = torch.nn.functional.interpolate(
                    x,
                    size=self._target_size,
                    mode="bilinear",
                    align_corners=False,
                )
        features = self.backbone(x)
        return self.head(features)


# =============================================================================
# MOBILE VARIANTS (Ultra-lightweight)
# =============================================================================


@register_model("efficientvit_m1")
class EfficientViTM1(EfficientViTBase):
    """
    EfficientViT-M1: ~2.6M backbone parameters. Mobile-tier representative.

    Ultra-lightweight, edge-deployable. Cascaded group attention for real-time
    NDE applications. 2D only.

    Example:
        >>> model = EfficientViTM1(in_shape=(224, 224), out_size=3)
        >>> x = torch.randn(4, 1, 224, 224)
        >>> out = model(x)  # (4, 3)
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="efficientvit_m1",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"EfficientViT_M1(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )


# =============================================================================
# BALANCED VARIANTS (B-series)
# =============================================================================


@register_model("efficientvit_b1")
class EfficientViTB1(EfficientViTBase):
    """
    EfficientViT-B1: ~7.5M backbone parameters. Small-tier representative.

    Good accuracy-speed trade-off. 2D only.
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="efficientvit_b1",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"EfficientViT_B1(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )


@register_model("efficientvit_b2")
class EfficientViTB2(EfficientViTBase):
    """
    EfficientViT-B2: ~21.8M backbone parameters. Medium-tier representative.

    Strong capacity for complex 2D wave patterns. 2D only.
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="efficientvit_b2",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"EfficientViT_B2(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )


# =============================================================================
# LARGE VARIANTS (L-series)
# =============================================================================


@register_model("efficientvit_l2")
class EfficientViTL2(EfficientViTBase):
    """
    EfficientViT-L2: ~60.5M backbone parameters. Large-tier representative.

    Maximum accuracy for complex tasks with ample compute. 2D only.
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="efficientvit_l2",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"EfficientViT_L2(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )
