"""
DenseNet: Dense Convolutional Networks for Regression
=======================================================

A dimension-agnostic DenseNet implementation with dense connectivity.
Features: feature reuse, efficient gradient flow, compact model.

**Dimensionality Support**:
    - 1D: Waveforms, signals, time-series (N, 1, L) → Conv1d
    - 2D: Images, spectrograms (N, 1, H, W) → Conv2d
    - 3D: Volumetric data, CT/MRI (N, 1, D, H, W) → Conv3d

**Variants**:
    - densenet121: Standard (121 layers, ~7.0M backbone params for 2D)
    - densenet169: Deeper (169 layers, ~12.5M backbone params for 2D)

References:
    Huang, G., et al. (2017). Densely Connected Convolutional Networks.
    CVPR 2017 (Best Paper). https://arxiv.org/abs/1608.06993

Author: Ductho Le (ductho.le@outlook.com)
"""

from typing import Any

import torch
import torch.nn as nn

from wavedl.models.base import BaseModel, SpatialShape
from wavedl.models.registry import register_model


def _get_layers(dim: int):
    """Get dimension-appropriate layer classes."""
    if dim == 1:
        return nn.Conv1d, nn.BatchNorm1d, nn.AvgPool1d, nn.AdaptiveAvgPool1d
    elif dim == 2:
        return nn.Conv2d, nn.BatchNorm2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d
    elif dim == 3:
        return nn.Conv3d, nn.BatchNorm3d, nn.AvgPool3d, nn.AdaptiveAvgPool3d
    else:
        raise ValueError(f"Unsupported dimensionality: {dim}D")


class DenseLayer(nn.Module):
    """
    Single dense layer (BN-ReLU-Conv-BN-ReLU-Conv) with bottleneck.

    Produces `growth_rate` new feature maps that are concatenated
    with the input features.
    """

    def __init__(
        self, in_channels: int, growth_rate: int, bn_size: int = 4, dim: int = 2
    ):
        super().__init__()
        Conv, BN, _, _ = _get_layers(dim)

        # Bottleneck: 1x1 conv reduces to 4*growth_rate
        self.bn1 = BN(in_channels)
        self.conv1 = Conv(in_channels, bn_size * growth_rate, kernel_size=1, bias=False)

        # 3x3 conv produces growth_rate features
        self.bn2 = BN(bn_size * growth_rate)
        self.conv2 = Conv(
            bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bottleneck
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        # 3x3 conv
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        # Concatenate with input (dense connection)
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    """Dense block containing multiple dense layers."""

    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        growth_rate: int,
        bn_size: int = 4,
        dim: int = 2,
    ):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layer_in_channels = in_channels + i * growth_rate
            layers.append(DenseLayer(layer_in_channels, growth_rate, bn_size, dim))

        self.layers = nn.Sequential(*layers)
        self.out_channels = in_channels + num_layers * growth_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Transition(nn.Module):
    """Transition layer between dense blocks (compression + downsampling)."""

    def __init__(self, in_channels: int, out_channels: int, dim: int = 2):
        super().__init__()
        Conv, BN, AvgPool, _ = _get_layers(dim)

        self.bn = BN(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = Conv(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = AvgPool(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x


class DenseNetBase(BaseModel):
    """
    Base DenseNet class for regression tasks.

    Architecture:
    1. Stem: 7x7 conv + avg pool (AvgPool used instead of MaxPool for
       torch.compile compatibility — see stem comment)
    2. 4 dense blocks with transitions between them
    3. Global average pooling
    4. Regression head
    """

    def __init__(
        self,
        in_shape: SpatialShape,
        out_size: int,
        block_config: list[int],
        growth_rate: int = 32,
        num_init_features: int = 64,
        bn_size: int = 4,
        compression: float = 0.5,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(in_shape, out_size)

        self.dim = len(in_shape)
        self.growth_rate = growth_rate
        self.dropout_rate = dropout_rate

        Conv, BN, AvgPool, AdaptivePool = _get_layers(self.dim)

        # Stem (uses AvgPool instead of MaxPool for torch.compile compatibility;
        # MaxPool's backward triggers a Triton compiler bug with large tensors)
        self.stem = nn.Sequential(
            Conv(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            BN(num_init_features),
            nn.ReLU(inplace=True),
            # count_include_pad=False so border windows are not biased toward zero
            # by the zero padding (unlike MaxPool, which ignores padding).
            AvgPool(kernel_size=3, stride=2, padding=1, count_include_pad=False),
        )

        # Build dense blocks and transitions
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # Dense block
            block = DenseBlock(num_features, num_layers, growth_rate, bn_size, self.dim)
            self.blocks.append(block)
            num_features = block.out_channels

            # Transition (except after last block)
            if i < len(block_config) - 1:
                out_features = int(num_features * compression)
                trans = Transition(num_features, out_features, self.dim)
                self.transitions.append(trans)
                num_features = out_features

        # Final batch norm
        _, BN, _, _ = _get_layers(self.dim)
        self.final_bn = BN(num_features)
        self.final_relu = nn.ReLU(inplace=True)

        # Global pooling and regression head
        self.global_pool = AdaptivePool(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, out_size),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Stem
        x = self.stem(x)

        # Dense blocks with transitions
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)

        # Final norm and pooling
        x = self.final_bn(x)
        x = self.final_relu(x)
        x = self.global_pool(x)
        x = x.flatten(1)

        return self.head(x)

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """Return default configuration."""
        return {"growth_rate": 32, "dropout_rate": 0.1}


# =============================================================================
# REGISTERED MODEL VARIANTS
# =============================================================================


@register_model("densenet121")
class DenseNet121(DenseNetBase):
    """
    DenseNet-121: Standard variant with 121 layers.

    ~7.0M backbone parameters (2D). Good for: Balanced accuracy, efficient training.

    Args:
        in_shape: (L,), (H, W), or (D, H, W)
        out_size: Number of regression targets
        growth_rate: Growth rate per layer (default: 32)
        dropout_rate: Dropout rate (default: 0.1)
    """

    def __init__(self, in_shape: SpatialShape, out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape, out_size=out_size, block_config=[6, 12, 24, 16], **kwargs
        )

    def __repr__(self) -> str:
        return f"DenseNet121({self.dim}D, in_shape={self.in_shape}, out_size={self.out_size})"


@register_model("densenet169")
class DenseNet169(DenseNetBase):
    """
    DenseNet-169: Deeper variant with 169 layers.

    ~12.5M backbone parameters (2D). Good for: Higher capacity, more complex patterns.

    Args:
        in_shape: (L,), (H, W), or (D, H, W)
        out_size: Number of regression targets
        growth_rate: Growth rate per layer (default: 32)
        dropout_rate: Dropout rate (default: 0.1)
    """

    def __init__(self, in_shape: SpatialShape, out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape, out_size=out_size, block_config=[6, 12, 32, 32], **kwargs
        )

    def __repr__(self) -> str:
        return f"DenseNet169({self.dim}D, in_shape={self.in_shape}, out_size={self.out_size})"


# =============================================================================
# PRETRAINED VARIANT (2D only, ImageNet weights)
# =============================================================================

try:
    from torchvision.models import DenseNet121_Weights, densenet121 as tv_densenet121

    DENSENET_PRETRAINED_AVAILABLE = True
except ImportError:
    DENSENET_PRETRAINED_AVAILABLE = False


@register_model("densenet121_pretrained")
class DenseNet121Pretrained(BaseModel):
    """
    DenseNet-121 with ImageNet pretrained weights (2D only).

    ~7.0M backbone parameters. Good for: Transfer learning with efficient feature reuse.

    Args:
        in_shape: (H, W) image dimensions
        out_size: Number of regression targets
        pretrained: Use ImageNet weights (default: True)
        dropout_rate: Dropout rate (default: 0.2)
        freeze_backbone: Freeze backbone for fine-tuning (default: False)
    """

    def __init__(
        self,
        in_shape: tuple[int, int],
        out_size: int,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        freeze_backbone: bool = False,
        **kwargs,
    ):
        super().__init__(in_shape, out_size)

        if not DENSENET_PRETRAINED_AVAILABLE:
            raise ImportError(
                "torchvision is required for pretrained DenseNet. "
                "Install with: pip install torchvision"
            )

        if len(in_shape) != 2:
            raise ValueError(
                f"Pretrained DenseNet requires 2D input (H, W), got {len(in_shape)}D. "
                "For 1D/3D data, use densenet121 instead."
            )

        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.freeze_backbone = freeze_backbone

        # Load pretrained model
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = tv_densenet121(weights=weights)

        # Get classifier input features
        in_features = self.backbone.classifier.in_features

        # Replace classifier with regression head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, out_size),
        )

        # Modify first conv for single-channel input
        from wavedl.models._pretrained_utils import adapt_first_conv_for_single_channel

        adapt_first_conv_for_single_channel(
            self.backbone, "features.conv0", pretrained=pretrained
        )

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze feature extraction while retaining the regression head."""
        self._freeze_backbone_modules(("classifier",))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        return {"pretrained": True, "dropout_rate": 0.2, "freeze_backbone": False}

    def __repr__(self) -> str:
        pt = "pretrained" if self.pretrained else "scratch"
        return f"DenseNet121_Pretrained({pt}, in_shape={self.in_shape}, out_size={self.out_size})"
