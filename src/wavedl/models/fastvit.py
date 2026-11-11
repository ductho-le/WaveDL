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
    - fastvit_t8: ~3.3M params (fastest)
    - fastvit_t12: ~6.8M params
    - fastvit_s12: ~8.5M params
    - fastvit_sa12: ~10.6M params (with attention)

**Requirements**:
    - timm >= 0.9.0 (for FastViT models)

Reference:
    Vasu, P.K.A., et al. (2023). FastViT: A Fast Hybrid Vision Transformer
    using Structural Reparameterization. ICCV 2023.
    https://arxiv.org/abs/2303.14189

Author: Ductho Le (ductho.le@outlook.com)
"""

from typing import Any

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

        min_size = 16
        for i, s in enumerate(self.in_shape):
            if s < min_size:
                raise ValueError(
                    f"FastViT requires each spatial axis >= {min_size}, "
                    f"but axis {i} has size {s}."
                )

        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.freeze_backbone = freeze_backbone
        self.model_name = model_name
        self._is_reparameterized = False

        # Try to load from timm
        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm >= 0.9.0 is required for FastViT. "
                "Install with: pip install timm>=0.9.0"
            )

        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # Remove classifier
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load FastViT model '{model_name}': {e}"
            ) from e

        # Adapt input channels (3 -> 1) before probing actual output width
        self._adapt_input_channels()
        from wavedl.models._pretrained_utils import infer_backbone_features

        in_features = infer_backbone_features(
            self.backbone, self.in_shape, in_channels=1
        )

        # Regression head
        self.head = build_regression_head(in_features, out_size, dropout_rate)

        if freeze_backbone:
            self._freeze_backbone()

    def _adapt_input_channels(self):
        """Adapt stem conv layers with 3 input channels for single-channel input."""
        # FastViT's MobileOne stem has multiple 3-channel branches (conv_kxk, conv_scale)
        from wavedl.models._pretrained_utils import find_and_adapt_input_convs

        stem_target = getattr(self.backbone, "stem", self.backbone)
        adapted_count = find_and_adapt_input_convs(
            stem_target, pretrained=self.pretrained, adapt_all=True
        )

        if adapted_count == 0:
            raise RuntimeError(
                f"Could not adapt input channels for FastViT model '{self.model_name}'."
            )

    def _freeze_backbone(self):
        """Freeze feature extraction while retaining the regression head."""
        self._freeze_backbone_modules()

    def train(self, mode: bool = True):
        if mode and getattr(self, "_is_reparameterized", False):
            raise RuntimeError(
                "Cannot switch FastViT back to train() mode after reparameterize(); "
                "structural reparameterization is irreversible."
            )
        return super().train(mode)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        backbone_prefix = f"{prefix}backbone."
        has_fused_keys = any(
            ".reparam_conv." in k for k in state_dict if k.startswith(backbone_prefix)
        )
        has_unfused_keys = any(
            ".conv_kxk." in k or ".conv_scale." in k
            for k in state_dict
            if k.startswith(backbone_prefix)
        )
        if has_fused_keys and not self._is_reparameterized:
            self.reparameterize()
        elif has_unfused_keys and self._is_reparameterized:
            raise RuntimeError(
                "Cannot load an unfused training state_dict into an already "
                "reparameterized FastViT instance; instantiate a fresh model first."
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """Return default configuration for FastViT."""
        return {
            "pretrained": True,
            "dropout_rate": 0.3,
            "freeze_backbone": False,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def reparameterize(self) -> None:
        """
        Reparameterize model for inference.

        Fuses MobileOne/RepMixer blocks in-place for faster inference.
        Idempotent and preserves ``freeze_backbone`` parameter states.
        """
        if self._is_reparameterized:
            return
        self.eval()
        for module in self.backbone.modules():
            if module is not self.backbone and hasattr(module, "reparameterize"):
                module.reparameterize()
        self._is_reparameterized = True
        if self.freeze_backbone:
            self._freeze_backbone()


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
