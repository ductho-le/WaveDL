"""
ResNet3D: 3D Residual Networks for Volumetric Data
===================================================

3D extension of ResNet for processing volumetric data such as C-scans,
3D wavefield imaging, and spatiotemporal cubes. Wraps torchvision's
video models adapted for regression tasks.

**Key Features**:
    - Native 3D convolutions for volumetric processing
    - Pretrained weights from Kinetics-400 (video action recognition)
    - Adapted for single-channel input (grayscale volumes)
    - Custom regression head for parameter estimation

**Variants**:
    - resnet3d_18: Lightweight (33M params)
    - resnet3d_34: Medium depth
    - resnet3d_50: Higher capacity with bottleneck blocks

**Use Cases**:
    - C-scan volume analysis (ultrasonic NDT)
    - 3D wavefield imaging and inversion
    - Spatiotemporal data cubes (time × space × space)
    - Medical imaging (CT/MRI volumes)

**Note**: ResNet3D is 3D-only. For 1D/2D data, use TCN or standard ResNet.

References:
    Hara, K., Kataoka, H., & Satoh, Y. (2018). Can Spatiotemporal 3D CNNs
    Retrace the History of 2D CNNs and ImageNet? CVPR 2018.
    https://arxiv.org/abs/1711.09577

    He, K., et al. (2016). Deep Residual Learning for Image Recognition.
    CVPR 2016. https://arxiv.org/abs/1512.03385

Author: Ductho Le (ductho.le@outlook.com)
"""

from typing import Any

import torch
import torch.nn as nn


try:
    from torchvision.models.video import (
        MC3_18_Weights,
        R3D_18_Weights,
        mc3_18,
        r3d_18,
    )

    RESNET3D_AVAILABLE = True
except ImportError:
    r3d_18 = None
    mc3_18 = None
    R3D_18_Weights = None
    MC3_18_Weights = None
    RESNET3D_AVAILABLE = False

from wavedl.models.base import BaseModel
from wavedl.models.registry import register_model


class ResNet3DBase(BaseModel):
    """
    Base ResNet3D class for volumetric regression tasks.

    Wraps torchvision 3D ResNet with:
    - Optional pretrained weights (Kinetics-400)
    - Automatic input channel adaptation (grayscale → 3ch)
    - Custom regression head

    Note: This is 3D-only. Input shape must be (D, H, W).
    """

    def __init__(
        self,
        in_shape: tuple[int, int, int],
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
        Initialize ResNet3D for regression.

        Args:
            in_shape: (D, H, W) input volume dimensions
            out_size: Number of regression output targets
            model_fn: torchvision model constructor
            weights_class: Pretrained weights enum class
            pretrained: Use Kinetics-400 pretrained weights (default: True)
            dropout_rate: Dropout rate in regression head (default: 0.3)
            freeze_backbone: Freeze backbone for fine-tuning (default: False)
            regression_hidden: Hidden units in regression head (default: 512)
        """
        super().__init__(in_shape, out_size)

        if not RESNET3D_AVAILABLE:
            raise ImportError(
                "torchvision >= 0.12 is required for ResNet3D. "
                "Install with: pip install torchvision>=0.12"
            )

        if len(in_shape) != 3:
            raise ValueError(
                f"ResNet3D requires 3D input (D, H, W), got {len(in_shape)}D. "
                "For 1D data, use TCN. For 2D data, use standard ResNet."
            )

        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.freeze_backbone = freeze_backbone
        self.regression_hidden = regression_hidden

        # Load pretrained backbone
        weights = weights_class.DEFAULT if pretrained else None
        self.backbone = model_fn(weights=weights)

        # Get the fc input features
        in_features = self.backbone.fc.in_features

        # Replace fc with regression head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, regression_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(regression_hidden, regression_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(regression_hidden // 2, out_size),
        )

        # Adapt before freezing so the replacement stem is frozen too.
        self._adapt_stem_for_single_channel()

        if freeze_backbone:
            self._freeze_backbone()

    def _adapt_stem_for_single_channel(self):
        """Modify stem conv to accept 1 channel, averaging pretrained RGB weights."""
        old_conv = self.backbone.stem[0]
        new_conv = nn.Conv3d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        if self.pretrained:
            with torch.no_grad():
                # Average RGB weights for grayscale initialization
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
        self.backbone.stem[0] = new_conv

    def _freeze_backbone(self):
        """Freeze feature extraction while retaining the regression head."""
        self._freeze_backbone_modules(("fc",))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 1, D, H, W)

        Returns:
            Output tensor of shape (B, out_size)
        """
        return self.backbone(x)

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """Return default configuration for ResNet3D."""
        return {
            "pretrained": True,
            "dropout_rate": 0.3,
            "freeze_backbone": False,
            "regression_hidden": 512,
        }


# =============================================================================
# REGISTERED MODEL VARIANTS
# =============================================================================


@register_model("resnet3d_18")
class ResNet3D18(ResNet3DBase):
    """
    ResNet3D-18: Lightweight 3D ResNet for volumetric data.

    ~33.2M backbone parameters. Uses 3D convolutions throughout for true volumetric processing.
    Pretrained on Kinetics-400 (video action recognition).

    Recommended for:
        - C-scan ultrasonic inspection volumes
        - 3D wavefield data cubes
        - Medical imaging (CT/MRI)
        - Moderate compute budgets

    Args:
        in_shape: (D, H, W) volume dimensions
        out_size: Number of regression targets
        pretrained: Use Kinetics-400 pretrained weights (default: True)
        dropout_rate: Dropout rate in head (default: 0.3)
        freeze_backbone: Freeze backbone for fine-tuning (default: False)
        regression_hidden: Hidden units in regression head (default: 512)

    Example:
        >>> model = ResNet3D18(in_shape=(16, 112, 112), out_size=3)
        >>> x = torch.randn(2, 1, 16, 112, 112)
        >>> out = model(x)  # (2, 3)
    """

    def __init__(self, in_shape: tuple[int, int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_fn=r3d_18,
            weights_class=R3D_18_Weights,
            **kwargs,
        )

    def __repr__(self) -> str:
        pt = "pretrained" if self.pretrained else "scratch"
        return f"ResNet3D_18({pt}, in={self.in_shape}, out={self.out_size})"


@register_model("mc3_18")
class MC3_18(ResNet3DBase):
    """
    MC3-18: Mixed Convolution 3D ResNet (3D stem + 2D residual blocks).

    ~11.5M backbone parameters. More efficient than pure 3D ResNet while maintaining
    good spatiotemporal modeling. Uses 3D convolutions in early layers
    and 2D convolutions in later layers.

    Recommended for:
        - When pure 3D is too expensive
        - Volumes with limited temporal/depth extent
        - Faster training with reasonable accuracy

    Args:
        in_shape: (D, H, W) volume dimensions
        out_size: Number of regression targets
        pretrained: Use Kinetics-400 pretrained weights (default: True)
        dropout_rate: Dropout rate in head (default: 0.3)
        freeze_backbone: Freeze backbone for fine-tuning (default: False)
        regression_hidden: Hidden units in regression head (default: 512)

    Example:
        >>> model = MC3_18(in_shape=(16, 112, 112), out_size=3)
        >>> x = torch.randn(2, 1, 16, 112, 112)
        >>> out = model(x)  # (2, 3)
    """

    def __init__(self, in_shape: tuple[int, int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_fn=mc3_18,
            weights_class=MC3_18_Weights,
            **kwargs,
        )

    def __repr__(self) -> str:
        pt = "pretrained" if self.pretrained else "scratch"
        return f"MC3_18({pt}, in={self.in_shape}, out={self.out_size})"
