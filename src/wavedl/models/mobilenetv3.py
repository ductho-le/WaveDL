"""
MobileNetV3: Efficient Networks for Edge Deployment
====================================================

Lightweight architecture optimized for mobile and embedded devices.
MobileNetV3 combines neural architecture search (NAS) with hardware-aware
optimization to achieve excellent accuracy with minimal computational cost.

**Key Features**:
    - Inverted residuals with depthwise separable convolutions
    - Squeeze-and-Excitation (SE) attention for channel weighting
    - h-swish activation: efficient approximation of swish
    - Designed for real-time inference on CPUs and edge devices

**Variants**:
    - mobilenet_v3_small: Ultra-lightweight (~0.9M backbone params) - Edge/embedded
    - mobilenet_v3_large: Balanced (~3.0M backbone params) - Mobile deployment

**Use Cases**:
    - Real-time structural health monitoring on embedded systems
    - Field inspection with portable devices
    - When model size and inference speed are critical

**Note**: MobileNetV3 is 2D-only. For 1D data, use TCN. For 3D data, use ResNet3D.

References:
    Howard, A., et al. (2019). Searching for MobileNetV3.
    ICCV 2019. https://arxiv.org/abs/1905.02244

Author: Ductho Le (ductho.le@outlook.com)
"""

from typing import Any

import torch
import torch.nn as nn


try:
    from torchvision.models import (
        MobileNet_V3_Large_Weights,
        MobileNet_V3_Small_Weights,
        mobilenet_v3_large,
        mobilenet_v3_small,
    )

    MOBILENETV3_AVAILABLE = True
except ImportError:
    mobilenet_v3_small = None
    mobilenet_v3_large = None
    MobileNet_V3_Small_Weights = None
    MobileNet_V3_Large_Weights = None
    MOBILENETV3_AVAILABLE = False

from wavedl.models.base import BaseModel
from wavedl.models.registry import register_model


class MobileNetV3Base(BaseModel):
    """
    Base MobileNetV3 class for regression tasks.

    Wraps torchvision MobileNetV3 with:
    - Optional pretrained weights (ImageNet-1K)
    - Automatic input channel adaptation (grayscale → 3ch)
    - Lightweight regression head (maintains efficiency)

    MobileNetV3 is ideal for:
    - Edge deployment (Raspberry Pi, Jetson, mobile)
    - Real-time inference requirements
    - Memory-constrained environments
    - Quick prototyping and experimentation

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
        regression_hidden: int = 256,
        **kwargs,
    ):
        """
        Initialize MobileNetV3 for regression.

        Args:
            in_shape: (H, W) input image dimensions
            out_size: Number of regression output targets
            model_fn: torchvision model constructor
            weights_class: Pretrained weights enum class
            pretrained: Use ImageNet pretrained weights (default: True)
            dropout_rate: Dropout rate in regression head (default: 0.2)
            freeze_backbone: Freeze backbone for fine-tuning (default: False)
            regression_hidden: Hidden units in regression head (default: 256)
        """
        super().__init__(in_shape, out_size)

        if not MOBILENETV3_AVAILABLE:
            raise ImportError(
                "torchvision is required for MobileNetV3. "
                "Install with: pip install torchvision"
            )

        if len(in_shape) != 2:
            raise ValueError(
                f"MobileNetV3 requires 2D input (H, W), got {len(in_shape)}D. "
                "For 1D data, use TCN. For 3D data, use ResNet3D."
            )

        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.freeze_backbone = freeze_backbone
        self.regression_hidden = regression_hidden

        # Load pretrained backbone
        weights = weights_class.IMAGENET1K_V1 if pretrained else None
        self.backbone = model_fn(weights=weights)

        # MobileNetV3 classifier structure:
        # classifier[0]: Linear (features → 1280 for Large, 1024 for Small)
        # classifier[1]: Hardswish
        # classifier[2]: Dropout
        # classifier[3]: Linear (1280/1024 → num_classes)

        # Get the input features to the final classifier
        in_features = self.backbone.classifier[0].in_features

        # Replace classifier with lightweight regression head
        # Keep it efficient to maintain MobileNet's speed advantage
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, regression_hidden),
            nn.Hardswish(inplace=True),  # Match MobileNetV3's activation
            nn.Dropout(dropout_rate),
            nn.Linear(regression_hidden, out_size),
        )

        # Adapt first conv for single-channel input (3× memory savings vs expand)
        self._adapt_input_channels()

        # Optionally freeze backbone for fine-tuning (after adaptation so new conv is frozen too)
        if freeze_backbone:
            self._freeze_backbone()

    def _adapt_input_channels(self):
        """Modify first conv to accept single-channel input.

        Instead of expanding 1→3 channels in forward (which triples memory),
        we replace the first conv layer with a 1-channel version and initialize
        weights as the mean of the pretrained RGB filters.
        """
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
        """Return default configuration for MobileNetV3."""
        return {
            "pretrained": True,
            "dropout_rate": 0.2,
            "freeze_backbone": False,
            "regression_hidden": 256,
        }


# =============================================================================
# REGISTERED MODEL VARIANTS
# =============================================================================


@register_model("mobilenet_v3_small")
class MobileNetV3Small(MobileNetV3Base):
    """
    MobileNetV3-Small: Ultra-lightweight for edge deployment.

    ~0.9M backbone parameters. Designed for the most constrained environments.
    Achieves ~67% ImageNet accuracy with minimal compute.

    Recommended for:
        - Embedded systems (Raspberry Pi, Arduino with accelerators)
        - Battery-powered devices
        - Ultra-low latency requirements (<10ms)
        - Quick training experiments

    Performance (approximate):
        - CPU inference: ~6ms (single core)
        - Parameters: ~0.9M backbone
        - MAdds: 56M

    Args:
        in_shape: (H, W) image dimensions
        out_size: Number of regression targets
        pretrained: Use ImageNet pretrained weights (default: True)
        dropout_rate: Dropout rate in head (default: 0.2)
        freeze_backbone: Freeze backbone for fine-tuning (default: False)
        regression_hidden: Hidden units in regression head (default: 256)

    Example:
        >>> model = MobileNetV3Small(in_shape=(224, 224), out_size=3)
        >>> x = torch.randn(1, 1, 224, 224)
        >>> out = model(x)  # (1, 3)
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_fn=mobilenet_v3_small,
            weights_class=MobileNet_V3_Small_Weights,
            **kwargs,
        )

    def __repr__(self) -> str:
        pt = "pretrained" if self.pretrained else "scratch"
        return f"MobileNetV3_Small({pt}, in={self.in_shape}, out={self.out_size})"


@register_model("mobilenet_v3_large")
class MobileNetV3Large(MobileNetV3Base):
    """
    MobileNetV3-Large: Balanced efficiency and accuracy.

    ~3.0M backbone parameters. Best trade-off for mobile/portable deployment.
    Achieves ~75% ImageNet accuracy with efficient inference.

    Recommended for:
        - Mobile deployment (smartphones, tablets)
        - Portable inspection devices
        - Real-time processing with moderate accuracy needs
        - Default choice for edge deployment

    Performance (approximate):
        - CPU inference: ~20ms (single core)
        - Parameters: ~3.0M backbone
        - MAdds: 219M

    Args:
        in_shape: (H, W) image dimensions
        out_size: Number of regression targets
        pretrained: Use ImageNet pretrained weights (default: True)
        dropout_rate: Dropout rate in head (default: 0.2)
        freeze_backbone: Freeze backbone for fine-tuning (default: False)
        regression_hidden: Hidden units in regression head (default: 256)

    Example:
        >>> model = MobileNetV3Large(in_shape=(224, 224), out_size=3)
        >>> x = torch.randn(1, 1, 224, 224)
        >>> out = model(x)  # (1, 3)
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_fn=mobilenet_v3_large,
            weights_class=MobileNet_V3_Large_Weights,
            **kwargs,
        )

    def __repr__(self) -> str:
        pt = "pretrained" if self.pretrained else "scratch"
        return f"MobileNetV3_Large({pt}, in={self.in_shape}, out={self.out_size})"
