"""
RegNet: Designing Network Design Spaces
========================================

RegNet provides a family of models with predictable scaling behavior,
designed through systematic exploration of network design spaces.
Models scale smoothly from mobile to server deployments.

**Key Features**:
    - Predictable scaling: accuracy increases linearly with compute
    - Simple, uniform architecture (no complex compound scaling)
    - Group convolutions for efficiency
    - Optional Squeeze-and-Excitation (SE) attention

**Variants** (RegNetY includes SE attention):
    - regnet_y_400mf: Ultra-light (~3.9M backbone params, 0.4 GFLOPs)
    - regnet_y_800mf: Light (~5.7M backbone params, 0.8 GFLOPs)
    - regnet_y_1_6gf: Medium (~10.3M backbone params, 1.6 GFLOPs) - Recommended
    - regnet_y_3_2gf: Large (~17.9M backbone params, 3.2 GFLOPs)
    - regnet_y_8gf: Very large (~37.4M backbone params, 8.0 GFLOPs)

**When to Use RegNet**:
    - When you need predictable performance at a given compute budget
    - For systematic model selection experiments
    - When interpretability of design choices matters
    - As an efficient alternative to ResNet

**Note**: RegNet is 2D-only. For 1D data, use TCN. For 3D data, use ResNet3D.

References:
    Radosavovic, I., et al. (2020). Designing Network Design Spaces.
    CVPR 2020. https://arxiv.org/abs/2003.13678

Author: Ductho Le (ductho.le@outlook.com)
"""

from typing import Any

import torch
import torch.nn as nn


try:
    from torchvision.models import (
        RegNet_Y_1_6GF_Weights,
        RegNet_Y_3_2GF_Weights,
        RegNet_Y_8GF_Weights,
        RegNet_Y_400MF_Weights,
        RegNet_Y_800MF_Weights,
        regnet_y_1_6gf,
        regnet_y_3_2gf,
        regnet_y_8gf,
        regnet_y_400mf,
        regnet_y_800mf,
    )

    REGNET_AVAILABLE = True
except ImportError:
    regnet_y_400mf = None
    regnet_y_800mf = None
    regnet_y_1_6gf = None
    regnet_y_3_2gf = None
    regnet_y_8gf = None
    RegNet_Y_400MF_Weights = None
    RegNet_Y_800MF_Weights = None
    RegNet_Y_1_6GF_Weights = None
    RegNet_Y_3_2GF_Weights = None
    RegNet_Y_8GF_Weights = None
    REGNET_AVAILABLE = False

from wavedl.models.base import BaseModel
from wavedl.models.registry import register_model


class RegNetBase(BaseModel):
    """
    Base RegNet class for regression tasks.

    Wraps torchvision RegNetY (with SE attention) with:
    - Optional pretrained weights (ImageNet-1K)
    - Automatic input channel adaptation (grayscale → 3ch)
    - Custom regression head

    RegNet advantages:
    - Simple, uniform design (easy to understand and modify)
    - Predictable accuracy/compute trade-off
    - Efficient group convolutions
    - SE attention for channel weighting (RegNetY variants)

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
        Initialize RegNet for regression.

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

        if not REGNET_AVAILABLE:
            raise ImportError(
                "torchvision is required for RegNet. "
                "Install with: pip install torchvision"
            )

        if len(in_shape) != 2:
            raise ValueError(
                f"RegNet requires 2D input (H, W), got {len(in_shape)}D. "
                "For 1D data, use TCN. For 3D data, use ResNet3D."
            )

        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.freeze_backbone = freeze_backbone
        self.regression_hidden = regression_hidden

        # Load pretrained backbone
        weights = weights_class.IMAGENET1K_V1 if pretrained else None
        self.backbone = model_fn(weights=weights)

        # RegNet uses .fc as the classification head
        in_features = self.backbone.fc.in_features

        # Replace fc with regression head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, regression_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
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
        old_conv = self.backbone.stem[0]
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
        self.backbone.stem[0] = new_conv

    def _freeze_backbone(self):
        """Freeze feature extraction while retaining the regression head."""
        self._freeze_backbone_modules(("fc",))

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
        """Return default configuration for RegNet."""
        return {
            "pretrained": True,
            "dropout_rate": 0.2,
            "freeze_backbone": False,
            "regression_hidden": 256,
        }


# =============================================================================
# REGISTERED MODEL VARIANTS
# =============================================================================


@register_model("regnet_y_400mf")
class RegNetY400MF(RegNetBase):
    """
    RegNetY-400MF: Ultra-lightweight for constrained environments.

    ~3.9M backbone parameters, 0.4 GFLOPs. Smallest RegNet variant with SE attention.

    Recommended for:
        - Edge deployment with moderate accuracy needs
        - Quick training experiments
        - Baseline comparisons

    Args:
        in_shape: (H, W) image dimensions
        out_size: Number of regression targets
        pretrained: Use ImageNet pretrained weights (default: True)
        dropout_rate: Dropout rate in head (default: 0.2)
        freeze_backbone: Freeze backbone for fine-tuning (default: False)
        regression_hidden: Hidden units in regression head (default: 256)

    Example:
        >>> model = RegNetY400MF(in_shape=(224, 224), out_size=3)
        >>> x = torch.randn(4, 1, 224, 224)
        >>> out = model(x)  # (4, 3)
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_fn=regnet_y_400mf,
            weights_class=RegNet_Y_400MF_Weights,
            **kwargs,
        )

    def __repr__(self) -> str:
        pt = "pretrained" if self.pretrained else "scratch"
        return f"RegNetY_400MF({pt}, in={self.in_shape}, out={self.out_size})"


@register_model("regnet_y_800mf")
class RegNetY800MF(RegNetBase):
    """
    RegNetY-800MF: Light variant with good accuracy.

    ~5.7M backbone parameters, 0.8 GFLOPs. Good balance for mobile deployment.

    Recommended for:
        - Mobile/portable devices
        - When MobileNet isn't accurate enough
        - Moderate compute budgets

    Args:
        in_shape: (H, W) image dimensions
        out_size: Number of regression targets
        pretrained: Use ImageNet pretrained weights (default: True)
        dropout_rate: Dropout rate in head (default: 0.2)
        freeze_backbone: Freeze backbone for fine-tuning (default: False)
        regression_hidden: Hidden units in regression head (default: 256)

    Example:
        >>> model = RegNetY800MF(in_shape=(224, 224), out_size=3)
        >>> x = torch.randn(4, 1, 224, 224)
        >>> out = model(x)  # (4, 3)
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_fn=regnet_y_800mf,
            weights_class=RegNet_Y_800MF_Weights,
            **kwargs,
        )

    def __repr__(self) -> str:
        pt = "pretrained" if self.pretrained else "scratch"
        return f"RegNetY_800MF({pt}, in={self.in_shape}, out={self.out_size})"


@register_model("regnet_y_1_6gf")
class RegNetY1_6GF(RegNetBase):
    """
    RegNetY-1.6GF: Recommended default for balanced performance.

    ~10.3M backbone parameters, 1.6 GFLOPs. Best trade-off of accuracy and efficiency.
    Comparable to ResNet50 but more efficient.

    Recommended for:
        - Default choice for general wave-based tasks
        - When you want predictable scaling
        - Server deployment with efficiency needs

    Args:
        in_shape: (H, W) image dimensions
        out_size: Number of regression targets
        pretrained: Use ImageNet pretrained weights (default: True)
        dropout_rate: Dropout rate in head (default: 0.2)
        freeze_backbone: Freeze backbone for fine-tuning (default: False)
        regression_hidden: Hidden units in regression head (default: 256)

    Example:
        >>> model = RegNetY1_6GF(in_shape=(224, 224), out_size=3)
        >>> x = torch.randn(4, 1, 224, 224)
        >>> out = model(x)  # (4, 3)
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_fn=regnet_y_1_6gf,
            weights_class=RegNet_Y_1_6GF_Weights,
            **kwargs,
        )

    def __repr__(self) -> str:
        pt = "pretrained" if self.pretrained else "scratch"
        return f"RegNetY_1.6GF({pt}, in={self.in_shape}, out={self.out_size})"


@register_model("regnet_y_3_2gf")
class RegNetY3_2GF(RegNetBase):
    """
    RegNetY-3.2GF: Higher accuracy for demanding tasks.

    ~17.9M backbone parameters, 3.2 GFLOPs. Use when 1.6GF isn't sufficient.

    Recommended for:
        - Larger datasets requiring more capacity
        - When accuracy is more important than efficiency
        - Research experiments with multiple model sizes

    Args:
        in_shape: (H, W) image dimensions
        out_size: Number of regression targets
        pretrained: Use ImageNet pretrained weights (default: True)
        dropout_rate: Dropout rate in head (default: 0.2)
        freeze_backbone: Freeze backbone for fine-tuning (default: False)
        regression_hidden: Hidden units in regression head (default: 256)

    Example:
        >>> model = RegNetY3_2GF(in_shape=(224, 224), out_size=3)
        >>> x = torch.randn(4, 1, 224, 224)
        >>> out = model(x)  # (4, 3)
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_fn=regnet_y_3_2gf,
            weights_class=RegNet_Y_3_2GF_Weights,
            **kwargs,
        )

    def __repr__(self) -> str:
        pt = "pretrained" if self.pretrained else "scratch"
        return f"RegNetY_3.2GF({pt}, in={self.in_shape}, out={self.out_size})"


@register_model("regnet_y_8gf")
class RegNetY8GF(RegNetBase):
    """
    RegNetY-8GF: High capacity for large-scale tasks.

    ~37.4M backbone parameters, 8.0 GFLOPs. Use for maximum accuracy needs.

    Recommended for:
        - Very large datasets (>50k samples)
        - Complex wave patterns
        - HPC environments with ample GPU memory

    Args:
        in_shape: (H, W) image dimensions
        out_size: Number of regression targets
        pretrained: Use ImageNet pretrained weights (default: True)
        dropout_rate: Dropout rate in head (default: 0.2)
        freeze_backbone: Freeze backbone for fine-tuning (default: False)
        regression_hidden: Hidden units in regression head (default: 256)

    Example:
        >>> model = RegNetY8GF(in_shape=(224, 224), out_size=3)
        >>> x = torch.randn(4, 1, 224, 224)
        >>> out = model(x)  # (4, 3)
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_fn=regnet_y_8gf,
            weights_class=RegNet_Y_8GF_Weights,
            **kwargs,
        )

    def __repr__(self) -> str:
        pt = "pretrained" if self.pretrained else "scratch"
        return f"RegNetY_8GF({pt}, in={self.in_shape}, out={self.out_size})"
