"""
FastViT: A Fast Hybrid Vision Transformer
==========================================

FastViT from Apple uses RepMixer for efficient token mixing with structural
reparameterization - train with skip connections, deploy without.

**Key Features**:
    - RepMixer: Reparameterizable token mixing
    - Train-time overparameterization
    - Faster than EfficientNet/ConvNeXt on mobile
    - CoreML compatible

**Variants**:
    - fastvit_t8: 4M params (fastest)
    - fastvit_t12: 7M params
    - fastvit_s12: 9M params
    - fastvit_sa12: 21M params (with attention)

**Requirements**:
    - timm >= 0.9.0 (for FastViT models)

Reference:
    Vasu, P.K.A., et al. (2023). FastViT: A Fast Hybrid Vision Transformer
    using Structural Reparameterization. ICCV 2023.
    https://arxiv.org/abs/2303.14189

Author: Ductho Le (ductho.le@outlook.com)
"""

import torch

from wavedl.models._pretrained_utils import build_regression_head
from wavedl.models.base import BaseModel
from wavedl.models.registry import register_model


__all__ = [
    "FastViTBase",
    "FastViTS12",
    "FastViTSA12",
    "FastViTT8",
    "FastViTT12",
]


# =============================================================================
# FASTVIT BASE CLASS
# =============================================================================


class FastViTBase(BaseModel):
    """
    FastViT base class wrapping timm implementation.

    Uses RepMixer for efficient token mixing with reparameterization.
    2D only.
    """

    def __init__(
        self,
        in_shape: tuple[int, int],
        out_size: int,
        model_name: str = "fastvit_t8",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.3,
        **kwargs,
    ):
        super().__init__(in_shape, out_size)

        if len(in_shape) != 2:
            raise ValueError(f"FastViT requires 2D input (H, W), got {len(in_shape)}D")

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
                "timm >= 0.9.0 is required for FastViT. "
                "Install with: pip install timm>=0.9.0"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load FastViT model '{model_name}': {e}"
            ) from e

        # Adapt input channels (3 -> 1)
        self._adapt_input_channels()

        # Regression head
        self.head = build_regression_head(in_features, out_size, dropout_rate)

        if freeze_backbone:
            self._freeze_backbone()

    def _adapt_input_channels(self):
        """Adapt all conv layers with 3 input channels for single-channel input."""
        # FastViT may have multiple modules with 3 input channels (e.g., conv_kxk, conv_scale)
        # We need to adapt all of them
        from wavedl.models._pretrained_utils import find_and_adapt_input_convs

        adapted_count = find_and_adapt_input_convs(
            self.backbone, pretrained=self.pretrained, adapt_all=True
        )

        if adapted_count == 0:
            import warnings

            warnings.warn(
                "Could not adapt FastViT input channels. Model may fail.", stacklevel=2
            )

    def _freeze_backbone(self):
        """Freeze feature extraction while retaining the regression head."""
        self._freeze_backbone_modules()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def reparameterize(self):
        """
        Reparameterize model for inference.

        Fuses RepMixer blocks for faster inference.
        Call this before deployment.
        """
        if hasattr(self.backbone, "reparameterize"):
            self.backbone.reparameterize()


# =============================================================================
# REGISTERED VARIANTS
# =============================================================================


@register_model("fastvit_t8")
class FastViTT8(FastViTBase):
    """
    FastViT-T8: ~3.3M backbone parameters (fastest variant).

    Optimized for mobile and edge deployment.
    2D only.

    Example:
        >>> model = FastViTT8(in_shape=(224, 224), out_size=3)
        >>> x = torch.randn(4, 1, 224, 224)
        >>> out = model(x)  # (4, 3)
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="fastvit_t8",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"FastViT_T8(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )


@register_model("fastvit_t12")
class FastViTT12(FastViTBase):
    """
    FastViT-T12: ~6.5M backbone parameters.

    Balanced speed and accuracy.
    2D only.
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="fastvit_t12",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"FastViT_T12(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )


@register_model("fastvit_s12")
class FastViTS12(FastViTBase):
    """
    FastViT-S12: ~8.5M backbone parameters.

    Slightly larger for better accuracy.
    2D only.
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="fastvit_s12",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"FastViT_S12(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )


@register_model("fastvit_sa12")
class FastViTSA12(FastViTBase):
    """
    FastViT-SA12: ~10.6M backbone parameters.

    With self-attention for better accuracy at the cost of speed.
    2D only.
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="fastvit_sa12",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"FastViT_SA12(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )
