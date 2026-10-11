"""
EfficientNetV2: Faster Training and Better Accuracy
====================================================

Next-generation EfficientNet with improved training efficiency and performance.
EfficientNetV2 replaces early depthwise convolutions with fused MBConv blocks,
enabling 2-4× faster training while achieving better accuracy.

**Key Improvements over EfficientNet**:
    - Fused-MBConv in early stages (faster on accelerators)
    - Progressive learning support (start small, grow)
    - Better NAS-optimized architecture

**Variants**:
    - efficientnet_v2_s: Small (21.5M params) - Recommended default
    - efficientnet_v2_m: Medium (54.1M params) - Higher accuracy
    - efficientnet_v2_l: Large (118.5M params) - Maximum accuracy

**Note**: EfficientNetV2 is 2D-only. For 1D data, use TCN. For 3D data, use ResNet3D.

References:
    Tan, M., & Le, Q. (2021). EfficientNetV2: Smaller Models and Faster Training.
    ICML 2021. https://arxiv.org/abs/2104.00298

Author: Ductho Le (ductho.le@outlook.com)
"""

from typing import Any

import torch
import torch.nn as nn


try:
    from torchvision.models import (
        EfficientNet_V2_L_Weights,
        EfficientNet_V2_M_Weights,
        EfficientNet_V2_S_Weights,
        efficientnet_v2_l,
        efficientnet_v2_m,
        efficientnet_v2_s,
    )

    EFFICIENTNETV2_AVAILABLE = True
except ImportError:
    efficientnet_v2_s = None
    efficientnet_v2_m = None
    efficientnet_v2_l = None
    EfficientNet_V2_S_Weights = None
    EfficientNet_V2_M_Weights = None
    EfficientNet_V2_L_Weights = None
    EFFICIENTNETV2_AVAILABLE = False

from wavedl.models.base import BaseModel
from wavedl.models.registry import register_model


class EfficientNetV2Base(BaseModel):
    """
    Base EfficientNetV2 class for regression tasks.

    Wraps torchvision EfficientNetV2 with:
    - Optional pretrained weights (ImageNet-1K)
    - Automatic input channel adaptation (grayscale → 3ch)
    - Custom multi-layer regression head

    Compared to EfficientNet (V1):
    - 2-4× faster training on GPU/TPU
    - Better accuracy at similar parameter counts
    - More efficient at higher resolutions

    Note: This is 2D-only. Input shape must be (H, W).
    """

    def __init__(
        self,
        in_shape: tuple[int, int],
        out_size: int,
        model_fn,
        weights_class,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
        regression_hidden: int = 512,
        **kwargs,
    ):
        """
        Initialize EfficientNetV2 for regression.

        Args:
            in_shape: (H, W) input image dimensions
            out_size: Number of regression output targets
            model_fn: torchvision model constructor
            weights_class: Pretrained weights enum class
            pretrained: Use ImageNet pretrained weights (default: True)
            dropout_rate: Dropout rate in regression head (default: 0.3)
            freeze_backbone: Freeze backbone for fine-tuning (default: False)
            regression_hidden: Hidden units in regression head (default: 512)
        """
        super().__init__(in_shape, out_size)

        if not EFFICIENTNETV2_AVAILABLE:
            raise ImportError(
                "torchvision >= 0.13 is required for EfficientNetV2. "
                "Install with: pip install torchvision>=0.13"
            )

        if len(in_shape) != 2:
            raise ValueError(
                f"EfficientNetV2 requires 2D input (H, W), got {len(in_shape)}D. "
                "For 1D data, use TCN. For 3D data, use ResNet3D."
            )

        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.freeze_backbone = freeze_backbone
        self.regression_hidden = regression_hidden

        # Load pretrained backbone
        weights = weights_class.IMAGENET1K_V1 if pretrained else None
        self.backbone = model_fn(weights=weights)

        # Get classifier input features (before the final classification layer)
        in_features = self.backbone.classifier[1].in_features

        # Replace classifier with regression head
        # EfficientNetV2 benefits from a deeper regression head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, regression_hidden),
            nn.SiLU(inplace=True),  # SiLU (Swish) matches EfficientNet's activation
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(regression_hidden, regression_hidden // 2),
            nn.SiLU(inplace=True),
            nn.Linear(regression_hidden // 2, out_size),
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
        """Return default configuration for EfficientNetV2."""
        return {
            "pretrained": True,
            "dropout_rate": 0.3,
            "freeze_backbone": False,
            "regression_hidden": 512,
        }


# =============================================================================
# REGISTERED MODEL VARIANTS
# =============================================================================


@register_model("efficientnet_v2_s")
class EfficientNetV2S(EfficientNetV2Base):
    """
    EfficientNetV2-S: Small variant, recommended default.

    ~20.2M backbone parameters. Best balance of speed and accuracy for most tasks.
    2× faster training than EfficientNet-B4 with better accuracy.

    Recommended for:
        - Default choice for 2D wave data
        - Moderate compute budgets
        - When training speed matters

    Args:
        in_shape: (H, W) image dimensions
        out_size: Number of regression targets
        pretrained: Use ImageNet pretrained weights (default: True)
        dropout_rate: Dropout rate in head (default: 0.3)
        freeze_backbone: Freeze backbone for fine-tuning (default: False)
        regression_hidden: Hidden units in regression head (default: 512)

    Example:
        >>> model = EfficientNetV2S(in_shape=(500, 500), out_size=3)
        >>> x = torch.randn(4, 1, 500, 500)
        >>> out = model(x)  # (4, 3)
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_fn=efficientnet_v2_s,
            weights_class=EfficientNet_V2_S_Weights,
            **kwargs,
        )

    def __repr__(self) -> str:
        pt = "pretrained" if self.pretrained else "scratch"
        return f"EfficientNetV2_S({pt}, in={self.in_shape}, out={self.out_size})"


@register_model("efficientnet_v2_m")
class EfficientNetV2M(EfficientNetV2Base):
    """
    EfficientNetV2-M: Medium variant for higher accuracy.

    ~52.9M backbone parameters. Use when accuracy is more important than speed.

    Recommended for:
        - Large datasets (>50k samples)
        - Complex wave patterns
        - When compute is not a bottleneck

    Args:
        in_shape: (H, W) image dimensions
        out_size: Number of regression targets
        pretrained: Use ImageNet pretrained weights (default: True)
        dropout_rate: Dropout rate in head (default: 0.3)
        freeze_backbone: Freeze backbone for fine-tuning (default: False)
        regression_hidden: Hidden units in regression head (default: 512)

    Example:
        >>> model = EfficientNetV2M(in_shape=(500, 500), out_size=3)
        >>> x = torch.randn(4, 1, 500, 500)
        >>> out = model(x)  # (4, 3)
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_fn=efficientnet_v2_m,
            weights_class=EfficientNet_V2_M_Weights,
            **kwargs,
        )

    def __repr__(self) -> str:
        pt = "pretrained" if self.pretrained else "scratch"
        return f"EfficientNetV2_M({pt}, in={self.in_shape}, out={self.out_size})"


@register_model("efficientnet_v2_l")
class EfficientNetV2L(EfficientNetV2Base):
    """
    EfficientNetV2-L: Large variant for maximum accuracy.

    ~117.2M backbone parameters. Use only with large datasets and sufficient compute.

    Recommended for:
        - Very large datasets (>100k samples)
        - When maximum accuracy is critical
        - HPC environments with ample GPU memory

    Args:
        in_shape: (H, W) image dimensions
        out_size: Number of regression targets
        pretrained: Use ImageNet pretrained weights (default: True)
        dropout_rate: Dropout rate in head (default: 0.3)
        freeze_backbone: Freeze backbone for fine-tuning (default: False)
        regression_hidden: Hidden units in regression head (default: 512)

    Example:
        >>> model = EfficientNetV2L(in_shape=(500, 500), out_size=3)
        >>> x = torch.randn(4, 1, 500, 500)
        >>> out = model(x)  # (4, 3)
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_fn=efficientnet_v2_l,
            weights_class=EfficientNet_V2_L_Weights,
            **kwargs,
        )

    def __repr__(self) -> str:
        pt = "pretrained" if self.pretrained else "scratch"
        return f"EfficientNetV2_L({pt}, in={self.in_shape}, out={self.out_size})"
