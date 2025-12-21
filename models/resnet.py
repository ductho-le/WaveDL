"""
ResNet: Dimension-Agnostic Residual Networks
=============================================

A flexible ResNet implementation that automatically adapts to 1D, 2D, or 3D inputs.
Provides multiple depth variants (18, 34, 50) with optional pretrained weights for 2D.

**Dimensionality Support**:
    - 1D: Waveforms, signals, time-series (N, 1, L) → Conv1d
    - 2D: Images, spectrograms (N, 1, H, W) → Conv2d  
    - 3D: Volumetric data, CT/MRI (N, 1, D, H, W) → Conv3d

**Variants**:
    - resnet18: Lightweight, fast training
    - resnet34: Balanced capacity
    - resnet50: Higher capacity with bottleneck blocks

Author: Ductho Le (ductho.le@outlook.com)
Version: 1.0.0
"""

from typing import Tuple, Dict, Any, Union, Type, List, Optional
import torch
import torch.nn as nn

from models.base import BaseModel
from models.registry import register_model


# Type alias for spatial shapes
SpatialShape = Union[Tuple[int], Tuple[int, int], Tuple[int, int, int]]


def _get_conv_layers(dim: int) -> Tuple[Type[nn.Module], Type[nn.Module], Type[nn.Module]]:
    """Get dimension-appropriate Conv, MaxPool, and AdaptiveAvgPool classes."""
    if dim == 1:
        return nn.Conv1d, nn.MaxPool1d, nn.AdaptiveAvgPool1d
    elif dim == 2:
        return nn.Conv2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d
    elif dim == 3:
        return nn.Conv3d, nn.MaxPool3d, nn.AdaptiveAvgPool3d
    else:
        raise ValueError(f"Unsupported dimensionality: {dim}D. Supported: 1D, 2D, 3D.")


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-18/34.
    
    Architecture: Conv → GroupNorm → ReLU → Conv → GroupNorm → Add → ReLU
    """
    expansion = 1
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dim: int = 2
    ):
        super().__init__()
        Conv = _get_conv_layers(dim)[0]
        
        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, 
                          stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3,
                          stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.downsample = downsample
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.gn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    Bottleneck residual block for ResNet-50/101/152.
    
    Architecture: 1x1 Conv → 3x3 Conv → 1x1 Conv with expansion
    """
    expansion = 4
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dim: int = 2
    ):
        super().__init__()
        Conv = _get_conv_layers(dim)[0]
        
        # 1x1 reduce
        self.conv1 = Conv(in_channels, out_channels, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(min(32, out_channels), out_channels)
        
        # 3x3 conv
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3,
                          stride=stride, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(32, out_channels), out_channels)
        
        # 1x1 expand
        self.conv3 = Conv(out_channels, out_channels * self.expansion, 
                          kernel_size=1, bias=False)
        self.gn3 = nn.GroupNorm(min(32, out_channels * self.expansion), 
                                out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.gn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out


class ResNetBase(BaseModel):
    """
    Base ResNet class that handles dimension-agnostic construction.
    
    This is the core implementation; specific variants (resnet18, etc.) 
    are created via the factory functions below.
    """
    
    def __init__(
        self,
        in_shape: SpatialShape,
        out_size: int,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        base_width: int = 64,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(in_shape, out_size)
        
        self.dim = len(in_shape)
        self.base_width = base_width
        self.dropout_rate = dropout_rate
        self.in_channels = base_width
        
        Conv, MaxPool, AdaptivePool = _get_conv_layers(self.dim)
        
        # Stem: 7x7 conv (or equivalent for 1D/3D)
        self.conv1 = Conv(1, base_width, kernel_size=7, stride=2, padding=3, bias=False)
        self.gn1 = nn.GroupNorm(min(32, base_width), base_width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = MaxPool(kernel_size=3, stride=2, padding=1)
        
        # Residual stages
        self.layer1 = self._make_layer(block, base_width, layers[0], stride=1)
        self.layer2 = self._make_layer(block, base_width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_width * 8, layers[3], stride=2)
        
        # Global pooling and regression head
        self.avgpool = AdaptivePool(1)
        final_channels = base_width * 8 * block.expansion
        
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(final_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, out_size)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        out_channels: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """Create a residual stage with multiple blocks."""
        Conv = _get_conv_layers(self.dim)[0]
        
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                Conv(self.in_channels, out_channels * block.expansion,
                     kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(min(32, out_channels * block.expansion),
                            out_channels * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, self.dim))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, dim=self.dim))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet."""
        # Stem
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling and regression
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.head(x)
        
        return x
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "base_width": 64,
            "dropout_rate": 0.1
        }


# =============================================================================
# REGISTERED MODEL VARIANTS
# =============================================================================

@register_model("resnet18")
class ResNet18(ResNetBase):
    """
    ResNet-18: Lightweight residual network with 18 layers.
    
    Good for: Quick experiments, smaller datasets, real-time inference.
    
    Args:
        in_shape: Spatial dimensions (L,), (H, W), or (D, H, W)
        out_size: Number of regression targets
        base_width: Base channel width (default: 64)
        dropout_rate: Dropout rate (default: 0.1)
    """
    
    def __init__(self, in_shape: SpatialShape, out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            block=BasicBlock,
            layers=[2, 2, 2, 2],
            **kwargs
        )
    
    def __repr__(self) -> str:
        return f"ResNet18({self.dim}D, in_shape={self.in_shape}, out_size={self.out_size})"


@register_model("resnet34")
class ResNet34(ResNetBase):
    """
    ResNet-34: Medium-depth residual network with 34 layers.
    
    Good for: Balanced performance and speed.
    
    Args:
        in_shape: Spatial dimensions (L,), (H, W), or (D, H, W)
        out_size: Number of regression targets
        base_width: Base channel width (default: 64)
        dropout_rate: Dropout rate (default: 0.1)
    """
    
    def __init__(self, in_shape: SpatialShape, out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            block=BasicBlock,
            layers=[3, 4, 6, 3],
            **kwargs
        )
    
    def __repr__(self) -> str:
        return f"ResNet34({self.dim}D, in_shape={self.in_shape}, out_size={self.out_size})"


@register_model("resnet50")
class ResNet50(ResNetBase):
    """
    ResNet-50: Deep residual network with bottleneck blocks.
    
    Good for: High capacity, complex patterns, larger datasets.
    
    Args:
        in_shape: Spatial dimensions (L,), (H, W), or (D, H, W)
        out_size: Number of regression targets
        base_width: Base channel width (default: 64)
        dropout_rate: Dropout rate (default: 0.1)
    """
    
    def __init__(self, in_shape: SpatialShape, out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            **kwargs
        )
    
    def __repr__(self) -> str:
        return f"ResNet50({self.dim}D, in_shape={self.in_shape}, out_size={self.out_size})"


# =============================================================================
# PRETRAINED VARIANTS (2D only, ImageNet weights)
# =============================================================================

try:
    from torchvision.models import (
        resnet18 as tv_resnet18, resnet50 as tv_resnet50,
        ResNet18_Weights, ResNet50_Weights
    )
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


class PretrainedResNetBase(BaseModel):
    """
    Pretrained ResNet wrapper using torchvision backbone.
    
    **Note**: 2D only (requires ImageNet pretrained weights).
    For 1D/3D, use the regular ResNet variants.
    """
    
    def __init__(
        self,
        in_shape: Tuple[int, int],
        out_size: int,
        model_fn,
        weights_class,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        freeze_backbone: bool = False,
        **kwargs
    ):
        super().__init__(in_shape, out_size)
        
        if not TORCHVISION_AVAILABLE:
            raise ImportError(
                "torchvision is required for pretrained ResNet. "
                "Install with: pip install torchvision"
            )
        
        if len(in_shape) != 2:
            raise ValueError(
                f"Pretrained ResNet requires 2D input (H, W), got {len(in_shape)}D. "
                "For 1D/3D data, use resnet18/resnet50 instead."
            )
        
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.freeze_backbone = freeze_backbone
        
        # Load pretrained model
        weights = weights_class.IMAGENET1K_V1 if pretrained else None
        self.backbone = model_fn(weights=weights)
        
        # Get the fc input features
        in_features = self.backbone.fc.in_features
        
        # Replace fc with regression head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, out_size)
        )
        
        # Modify first conv for single-channel input
        # Original: Conv2d(3, 64, ...) → New: Conv2d(1, 64, ...)
        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        # Initialize new conv with mean of pretrained weights
        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight = nn.Parameter(
                    old_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters except the fc head."""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {"pretrained": True, "dropout_rate": 0.2, "freeze_backbone": False}


@register_model("resnet18_pretrained")
class ResNet18Pretrained(PretrainedResNetBase):
    """
    ResNet-18 with ImageNet pretrained weights (2D only).
    
    ~11M parameters. Good for: Transfer learning, faster convergence.
    
    Args:
        in_shape: (H, W) image dimensions
        out_size: Number of regression targets
        pretrained: Use ImageNet weights (default: True)
        dropout_rate: Dropout rate (default: 0.2)
        freeze_backbone: Freeze backbone for fine-tuning (default: False)
    """
    
    def __init__(self, in_shape: Tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_fn=tv_resnet18,
            weights_class=ResNet18_Weights,
            **kwargs
        )
    
    def __repr__(self) -> str:
        pt = "pretrained" if self.pretrained else "scratch"
        return f"ResNet18_Pretrained({pt}, in_shape={self.in_shape}, out_size={self.out_size})"


@register_model("resnet50_pretrained")
class ResNet50Pretrained(PretrainedResNetBase):
    """
    ResNet-50 with ImageNet pretrained weights (2D only).
    
    ~25M parameters. Good for: High accuracy with transfer learning.
    
    Args:
        in_shape: (H, W) image dimensions
        out_size: Number of regression targets
        pretrained: Use ImageNet weights (default: True)
        dropout_rate: Dropout rate (default: 0.2)
        freeze_backbone: Freeze backbone for fine-tuning (default: False)
    """
    
    def __init__(self, in_shape: Tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_fn=tv_resnet50,
            weights_class=ResNet50_Weights,
            **kwargs
        )
    
    def __repr__(self) -> str:
        pt = "pretrained" if self.pretrained else "scratch"
        return f"ResNet50_Pretrained({pt}, in_shape={self.in_shape}, out_size={self.out_size})"
