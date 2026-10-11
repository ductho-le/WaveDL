"""
EfficientNet: Efficient and Scalable CNNs for Regression
=========================================================

Wrapper around torchvision's EfficientNet with a regression head.
Provides optional ImageNet pretrained weights for transfer learning.

**Variants**:
    - efficientnet_b0: Tiny (~5.3M backbone params)
    - efficientnet_b2: Small (~9.1M backbone params)
    - efficientnet_b4: Medium (~19M backbone params)
    - efficientnet_b7: Large (~66M backbone params)

**Note**: EfficientNet is 2D-only. For 1D data, use TCN. For 3D data, use ResNet3D.

References:
    Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling
    for Convolutional Neural Networks. ICML 2019.
    https://arxiv.org/abs/1905.11946

Author: Ductho Le (ductho.le@outlook.com)
"""

from typing import Any

import torch
import torch.nn as nn


try:
    from torchvision.models import (
        EfficientNet_B0_Weights,
        EfficientNet_B2_Weights,
        EfficientNet_B4_Weights,
        EfficientNet_B7_Weights,
        efficientnet_b0,
        efficientnet_b2,
        efficientnet_b4,
        efficientnet_b7,
    )

    TORCHVISION_AVAILABLE = True
except ImportError:
    efficientnet_b0 = None
    efficientnet_b2 = None
    efficientnet_b4 = None
    efficientnet_b7 = None
    EfficientNet_B0_Weights = None
    EfficientNet_B2_Weights = None
    EfficientNet_B4_Weights = None
    EfficientNet_B7_Weights = None
    TORCHVISION_AVAILABLE = False

from wavedl.models.base import BaseModel
from wavedl.models.registry import register_model


class EfficientNetBase(BaseModel):
    """
    Base EfficientNet class for regression tasks.

    Wraps torchvision EfficientNet with:
    - Optional pretrained weights
    - Automatic input channel adaptation (grayscale → 3ch)
    - Custom regression head

    Note: This is 2D-only. Input shape must be (H, W).
    """

    def __init__(
        self,
        in_shape: tuple[int, int],
        out_size: int,
        model_fn,
        weights_class,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        freeze_backbone: bool = False,
        **kwargs,
    ):
        super().__init__(in_shape, out_size)

        if not TORCHVISION_AVAILABLE:
            raise ImportError(
                "torchvision is required for EfficientNet. "
                "Install with: pip install torchvision"
            )

        if len(in_shape) != 2:
            raise ValueError(
                f"EfficientNet requires 2D input (H, W), got {len(in_shape)}D. "
                "For 1D/3D data, use ResNet or CNN instead."
            )

        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.freeze_backbone = freeze_backbone

        # Load pretrained model
        weights = weights_class.IMAGENET1K_V1 if pretrained else None
        self.backbone = model_fn(weights=weights)

        # Get the classifier input features
        in_features = self.backbone.classifier[1].in_features

        # Replace classifier with regression head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, out_size),
        )

        # Adapt first conv for single-channel input
        # EfficientNet expects 3 channels, we'll expand grayscale
        self._adapt_input_channels()

        # Optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone()

    def _adapt_input_channels(self):
        """Modify first conv to accept single-channel input.

        Instead of expanding 1→3 channels in forward (which triples memory),
        we replace the first conv layer with a 1-channel version and initialize
        weights as the mean of the pretrained RGB filters.
        """
        # EfficientNet stem conv is at: features[0][0]
        old_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(
            1,  # Single channel input
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            padding_mode=old_conv.padding_mode,
            bias=old_conv.bias is not None,
        )
        if self.pretrained:
            # Initialize with mean of pretrained RGB weights
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        self.backbone.features[0][0] = new_conv

    def _freeze_backbone(self):
        """Freeze feature extraction while retaining the regression head."""
        self._freeze_backbone_modules(("classifier",))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 1, H, W)

        Returns:
            Output tensor of shape (B, out_size)
        """
        return self.backbone(x)

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """Return default configuration."""
        return {"pretrained": True, "dropout_rate": 0.2, "freeze_backbone": False}


# =============================================================================
# REGISTERED MODEL VARIANTS
# =============================================================================


@register_model("efficientnet_b0")
class EfficientNetB0(EfficientNetBase):
    """
    EfficientNet-B0: Tiny variant, most efficient.

    ~5.3M backbone parameters. Good for: Quick training, limited compute, baseline.

    Args:
        in_shape: (H, W) image dimensions
        out_size: Number of regression targets
        pretrained: Use ImageNet pretrained weights (default: True)
        dropout_rate: Dropout rate (default: 0.2)
        freeze_backbone: Freeze backbone weights for fine-tuning (default: False)
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_fn=efficientnet_b0,
            weights_class=EfficientNet_B0_Weights,
            **kwargs,
        )

    def __repr__(self) -> str:
        pt = "pretrained" if self.pretrained else "scratch"
        return (
            f"EfficientNet_B0({pt}, in_shape={self.in_shape}, out_size={self.out_size})"
        )


@register_model("efficientnet_b2")
class EfficientNetB2(EfficientNetBase):
    """
    EfficientNet-B2: Small variant. Best accuracy among the compact tier.

    ~9.1M backbone parameters. Good for: High accuracy without excessive compute.

    Args:
        in_shape: (H, W) image dimensions
        out_size: Number of regression targets
        pretrained: Use ImageNet pretrained weights (default: True)
        dropout_rate: Dropout rate (default: 0.2)
        freeze_backbone: Freeze backbone weights for fine-tuning (default: False)
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_fn=efficientnet_b2,
            weights_class=EfficientNet_B2_Weights,
            **kwargs,
        )

    def __repr__(self) -> str:
        pt = "pretrained" if self.pretrained else "scratch"
        return (
            f"EfficientNet_B2({pt}, in_shape={self.in_shape}, out_size={self.out_size})"
        )


@register_model("efficientnet_b4")
class EfficientNetB4(EfficientNetBase):
    """
    EfficientNet-B4: Medium variant.

    ~19M backbone parameters. Good for: Strong accuracy with moderate compute budget.

    Args:
        in_shape: (H, W) image dimensions
        out_size: Number of regression targets
        pretrained: Use ImageNet pretrained weights (default: True)
        dropout_rate: Dropout rate (default: 0.2)
        freeze_backbone: Freeze backbone weights for fine-tuning (default: False)
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_fn=efficientnet_b4,
            weights_class=EfficientNet_B4_Weights,
            **kwargs,
        )

    def __repr__(self) -> str:
        pt = "pretrained" if self.pretrained else "scratch"
        return (
            f"EfficientNet_B4({pt}, in_shape={self.in_shape}, out_size={self.out_size})"
        )


@register_model("efficientnet_b7")
class EfficientNetB7(EfficientNetBase):
    """
    EfficientNet-B7: Large variant.

    ~66M backbone parameters. Good for: Maximum accuracy; large datasets with ample compute.

    Args:
        in_shape: (H, W) image dimensions
        out_size: Number of regression targets
        pretrained: Use ImageNet pretrained weights (default: True)
        dropout_rate: Dropout rate (default: 0.2)
        freeze_backbone: Freeze backbone weights for fine-tuning (default: False)
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_fn=efficientnet_b7,
            weights_class=EfficientNet_B7_Weights,
            **kwargs,
        )

    def __repr__(self) -> str:
        pt = "pretrained" if self.pretrained else "scratch"
        return (
            f"EfficientNet_B7({pt}, in_shape={self.in_shape}, out_size={self.out_size})"
        )
