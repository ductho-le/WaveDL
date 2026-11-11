"""
MaxViT: Multi-Axis Vision Transformer
======================================

MaxViT combines local and global attention with O(n) complexity using
multi-axis attention: block attention (local) + grid attention (global sparse).

**Key Features**:
    - Multi-axis attention for both local and global context
    - Hybrid design with MBConv + attention
    - Linear O(n) complexity
    - Hierarchical multi-scale features

**Variants**:
    - maxvit_tiny: 31M params
    - maxvit_small: 69M params
    - maxvit_base: 120M params

**Requirements**:
    - timm (for pretrained models and architecture)
    - torchvision (fallback, limited support)

Reference:
    Tu, Z., et al. (2022). MaxViT: Multi-Axis Vision Transformer.
    ECCV 2022. https://arxiv.org/abs/2204.01697

Author: Ductho Le (ductho.le@outlook.com)
"""

from typing import ClassVar

import torch
import torch.nn.functional as F

from wavedl.models._pretrained_utils import build_regression_head
from wavedl.models.base import BaseModel
from wavedl.models.registry import register_model


__all__ = [
    "MaxViTBase",
    "MaxViTBaseLarge",
    "MaxViTSmall",
    "MaxViTTiny",
]


# =============================================================================
# MAXVIT BASE CLASS
# =============================================================================


class MaxViTBase(BaseModel):
    """
    MaxViT base class wrapping timm implementation.

    Multi-axis attention with local block and global grid attention.
    2D only due to attention structure.

    Note:
        Inputs are rounded up to multiples of the model's native size (e.g. 224).
        This preserves the attention partitions and relative-position parameter
        shapes used by existing WaveDL checkpoints.
    """

    _NATIVE_SIZES: ClassVar[dict[str, int]] = {
        "224": 224,
        "384": 384,
        "512": 512,
    }

    def __init__(
        self,
        in_shape: tuple[int, int],
        out_size: int,
        model_name: str = "maxvit_tiny_tf_224",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.3,
        **kwargs,
    ):
        super().__init__(in_shape, out_size)

        if len(in_shape) != 2:
            raise ValueError(f"MaxViT requires 2D input (H, W), got {len(in_shape)}D")

        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.model_name = model_name
        self.dropout_rate = dropout_rate

        # Preserve the native-size rounding policy used by saved checkpoints.
        self._divisor = self._get_native_size(model_name)
        self._target_size = self._compute_compatible_size(self.in_shape)

        # Try to load from timm
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "timm is required for MaxViT. Install with: pip install timm"
            ) from e

        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # Remove classifier
                img_size=self._target_size,  # Configure for actual input size
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load MaxViT model '{model_name}': {e}"
            ) from e

        # Adapt input channels (3 -> 1) before probing actual output width
        self._adapt_input_channels()
        from wavedl.models._pretrained_utils import infer_backbone_features

        in_features = infer_backbone_features(
            self.backbone, self._target_size, in_channels=1
        )

        # Regression head
        self.head = build_regression_head(in_features, out_size, dropout_rate)

        if freeze_backbone:
            self._freeze_backbone()

    @classmethod
    def _get_native_size(cls, model_name: str) -> int:
        """Get the native input size from the model name suffix."""
        for suffix, size in cls._NATIVE_SIZES.items():
            if model_name.endswith(suffix):
                return size
        return 224

    def _adapt_input_channels(self):
        """Adapt first conv layer for single-channel input."""
        from wavedl.models._pretrained_utils import find_and_adapt_input_convs

        adapted_count = find_and_adapt_input_convs(
            self.backbone, pretrained=self.pretrained, adapt_all=False
        )

        if adapted_count == 0:
            raise RuntimeError(
                f"Could not adapt input channels for MaxViT model '{self.model_name}'."
            )

    def _freeze_backbone(self):
        """Freeze feature extraction while retaining the regression head."""
        self._freeze_backbone_modules()

    def _compute_compatible_size(self, in_shape: tuple[int, int]) -> tuple[int, int]:
        """
        Compute the nearest input size compatible with MaxViT attention partitions.

        Rounds each axis up to the nearest multiple of the model's native size.
        """
        import math

        h, w = in_shape
        target_h = max(self._divisor, math.ceil(h / self._divisor) * self._divisor)
        target_w = max(self._divisor, math.ceil(w / self._divisor) * self._divisor)

        # Warn if resize significantly changes pixel count
        original_pixels = h * w
        target_pixels = target_h * target_w
        if target_pixels > original_pixels * 1.5:
            import warnings

            warnings.warn(
                f"MaxViT input ({h}, {w}) rounded up to ({target_h}, {target_w}) "
                f"for attention window compatibility ({target_pixels / original_pixels:.1f}x "
                f"pixel count increase). Bilinear interpolation may degrade "
                f"data quality. Consider resizing input to a multiple of "
                f"{self._divisor}.",
                UserWarning,
                stacklevel=3,
            )

        return (target_h, target_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Resize input to compatible size if needed
        _, _, h, w = x.shape
        if (h, w) != self._target_size:
            x = F.interpolate(
                x,
                size=self._target_size,
                mode="bilinear",
                align_corners=False,
            )
        features = self.backbone(x)
        return self.head(features)

    @classmethod
    def get_default_config(cls) -> dict[str, bool | float]:
        return {"pretrained": True, "freeze_backbone": False, "dropout_rate": 0.3}


# =============================================================================
# REGISTERED VARIANTS
# =============================================================================


@register_model("maxvit_tiny")
class MaxViTTiny(MaxViTBase):
    """
    MaxViT Tiny: ~30.1M backbone parameters.

    Multi-axis attention with local+global context.
    2D only.

    Example:
        >>> model = MaxViTTiny(in_shape=(224, 224), out_size=3)
        >>> x = torch.randn(4, 1, 224, 224)
        >>> out = model(x)  # (4, 3)
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="maxvit_tiny_tf_224",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"MaxViT_Tiny(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )


@register_model("maxvit_small")
class MaxViTSmall(MaxViTBase):
    """
    MaxViT Small: ~67.6M backbone parameters.

    Multi-axis attention with local+global context.
    2D only.
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="maxvit_small_tf_224",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"MaxViT_Small(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )


@register_model("maxvit_base")
class MaxViTBaseLarge(MaxViTBase):
    """
    MaxViT Base: ~118.1M backbone parameters.

    Multi-axis attention with local+global context.
    2D only.
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="maxvit_base_tf_224",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"MaxViT_Base(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )
