"""
ConvNeXt: A Modern Convolutional Neural Network
=================================================

A dimension-agnostic ConvNeXt implementation inspired by Vision Transformers.
Features: inverted bottleneck, LayerNorm, GELU activation, depthwise convolutions.

**Dimensionality Support**:
    - 1D: Waveforms, signals, time-series (N, 1, L) → Conv1d
    - 2D: Images, spectrograms (N, 1, H, W) → Conv2d  
    - 3D: Volumetric data, CT/MRI (N, 1, D, H, W) → Conv3d

**Variants**:
    - convnext_tiny: Smallest (~28M params for 2D)
    - convnext_small: Medium (~50M params for 2D)  
    - convnext_base: Standard (~89M params for 2D)

Author: Ductho Le (ductho.le@outlook.com)
Version: 1.0.0
"""

from typing import Tuple, Dict, Any, Union, Type, List
import torch
import torch.nn as nn

from models.base import BaseModel
from models.registry import register_model


# Type alias for spatial shapes
SpatialShape = Union[Tuple[int], Tuple[int, int], Tuple[int, int, int]]


def _get_conv_layer(dim: int) -> Type[nn.Module]:
    """Get dimension-appropriate Conv class."""
    if dim == 1:
        return nn.Conv1d
    elif dim == 2:
        return nn.Conv2d
    elif dim == 3:
        return nn.Conv3d
    else:
        raise ValueError(f"Unsupported dimensionality: {dim}D")


class LayerNormNd(nn.Module):
    """
    LayerNorm for N-dimensional tensors (channels-first format).
    
    Normalizes over the channel dimension, supporting Conv1d/2d/3d outputs.
    """
    
    def __init__(self, num_channels: int, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, ..spatial..)
        # Normalize over channel dimension
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / (var + self.eps).sqrt()
        
        # Apply learnable parameters
        shape = [1, -1] + [1] * self.dim  # (1, C, 1, ...) for broadcasting
        x = x * self.weight.view(*shape) + self.bias.view(*shape)
        return x


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block with inverted bottleneck design.
    
    Architecture:
    - 7x7 depthwise conv
    - LayerNorm
    - 1x1 conv (expand by 4x)
    - GELU
    - 1x1 conv (reduce back)
    - Residual connection
    """
    
    def __init__(
        self,
        channels: int,
        dim: int = 2,
        expansion_ratio: float = 4.0,
        drop_path: float = 0.0
    ):
        super().__init__()
        Conv = _get_conv_layer(dim)
        hidden_dim = int(channels * expansion_ratio)
        
        # Depthwise conv (7x7)
        self.dwconv = Conv(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = LayerNormNd(channels, dim)
        
        # Pointwise convs (1x1)
        self.pwconv1 = Conv(channels, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = Conv(hidden_dim, channels, kernel_size=1)
        
        # Stochastic depth (drop path) - simplified version
        self.drop_path = nn.Identity()  # Can be replaced with DropPath if needed
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.drop_path(x)
        
        return residual + x


class ConvNeXtBase(BaseModel):
    """
    Base ConvNeXt class for regression tasks.
    
    Architecture:
    1. Stem: Patchify with stride-4 conv
    2. 4 stages with downsampling between stages
    3. Global average pooling
    4. Regression head
    """
    
    def __init__(
        self,
        in_shape: SpatialShape,
        out_size: int,
        depths: List[int],
        dims: List[int],
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(in_shape, out_size)
        
        self.dim = len(in_shape)
        self.depths = depths
        self.dims = dims
        self.dropout_rate = dropout_rate
        
        Conv = _get_conv_layer(self.dim)
        
        # Stem: Patchify with stride-4 conv (like ViT patch embedding)
        self.stem = nn.Sequential(
            Conv(1, dims[0], kernel_size=4, stride=4),
            LayerNormNd(dims[0], self.dim)
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        for i in range(4):
            # Stage: multiple ConvNeXt blocks
            stage = nn.Sequential(*[
                ConvNeXtBlock(dims[i], self.dim)
                for _ in range(depths[i])
            ])
            self.stages.append(stage)
            
            # Downsample between stages (except after last stage)
            if i < 3:
                downsample = nn.Sequential(
                    LayerNormNd(dims[i], self.dim),
                    Conv(dims[i], dims[i + 1], kernel_size=2, stride=2)
                )
                self.downsamples.append(downsample)
        
        # Global average pooling
        if self.dim == 1:
            self.global_pool = nn.AdaptiveAvgPool1d(1)
        elif self.dim == 2:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Final norm and regression head
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(dims[-1], 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, out_size)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Stem
        x = self.stem(x)
        
        # Stages with downsampling
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.flatten(1)
        
        # Final norm and head
        x = self.norm(x)
        x = self.head(x)
        
        return x
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "dropout_rate": 0.1
        }


# =============================================================================
# REGISTERED MODEL VARIANTS
# =============================================================================

@register_model("convnext_tiny")
class ConvNeXtTiny(ConvNeXtBase):
    """
    ConvNeXt-Tiny: Smallest variant.
    
    ~28M parameters (2D). Good for: Limited compute, fast training.
    
    Args:
        in_shape: (L,), (H, W), or (D, H, W)
        out_size: Number of regression targets
        dropout_rate: Dropout rate (default: 0.1)
    """
    
    def __init__(self, in_shape: SpatialShape, out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            depths=[3, 3, 9, 3],
            dims=[96, 192, 384, 768],
            **kwargs
        )
    
    def __repr__(self) -> str:
        return f"ConvNeXt_Tiny({self.dim}D, in_shape={self.in_shape}, out_size={self.out_size})"


@register_model("convnext_small")
class ConvNeXtSmall(ConvNeXtBase):
    """
    ConvNeXt-Small: Medium variant.
    
    ~50M parameters (2D). Good for: Balanced performance.
    
    Args:
        in_shape: (L,), (H, W), or (D, H, W)
        out_size: Number of regression targets
        dropout_rate: Dropout rate (default: 0.1)
    """
    
    def __init__(self, in_shape: SpatialShape, out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            depths=[3, 3, 27, 3],
            dims=[96, 192, 384, 768],
            **kwargs
        )
    
    def __repr__(self) -> str:
        return f"ConvNeXt_Small({self.dim}D, in_shape={self.in_shape}, out_size={self.out_size})"


@register_model("convnext_base")
class ConvNeXtBase_(ConvNeXtBase):
    """
    ConvNeXt-Base: Standard variant.
    
    ~89M parameters (2D). Good for: High accuracy, larger datasets.
    
    Args:
        in_shape: (L,), (H, W), or (D, H, W)
        out_size: Number of regression targets
        dropout_rate: Dropout rate (default: 0.1)
    """
    
    def __init__(self, in_shape: SpatialShape, out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            depths=[3, 3, 27, 3],
            dims=[128, 256, 512, 1024],
            **kwargs
        )
    
    def __repr__(self) -> str:
        return f"ConvNeXt_Base({self.dim}D, in_shape={self.in_shape}, out_size={self.out_size})"


# =============================================================================
# PRETRAINED VARIANT (2D only, ImageNet weights)
# =============================================================================

try:
    from torchvision.models import convnext_tiny as tv_convnext_tiny, ConvNeXt_Tiny_Weights
    CONVNEXT_PRETRAINED_AVAILABLE = True
except ImportError:
    CONVNEXT_PRETRAINED_AVAILABLE = False


@register_model("convnext_tiny_pretrained")
class ConvNeXtTinyPretrained(BaseModel):
    """
    ConvNeXt-Tiny with ImageNet pretrained weights (2D only).
    
    ~28M parameters. Good for: Transfer learning with modern CNN.
    
    Args:
        in_shape: (H, W) image dimensions
        out_size: Number of regression targets
        pretrained: Use ImageNet weights (default: True)
        dropout_rate: Dropout rate (default: 0.2)
        freeze_backbone: Freeze backbone for fine-tuning (default: False)
    """
    
    def __init__(
        self,
        in_shape: Tuple[int, int],
        out_size: int,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        freeze_backbone: bool = False,
        **kwargs
    ):
        super().__init__(in_shape, out_size)
        
        if not CONVNEXT_PRETRAINED_AVAILABLE:
            raise ImportError(
                "torchvision>=0.13 is required for pretrained ConvNeXt. "
                "Install with: pip install torchvision"
            )
        
        if len(in_shape) != 2:
            raise ValueError(
                f"Pretrained ConvNeXt requires 2D input (H, W), got {len(in_shape)}D. "
                "For 1D/3D data, use convnext_tiny instead."
            )
        
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.freeze_backbone = freeze_backbone
        
        # Load pretrained model
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = tv_convnext_tiny(weights=weights)
        
        # Get classifier input features
        in_features = self.backbone.classifier[2].in_features
        
        # Replace classifier with regression head
        self.backbone.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, out_size)
        )
        
        # Modify first conv for single-channel input
        old_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        if pretrained:
            with torch.no_grad():
                self.backbone.features[0][0].weight = nn.Parameter(
                    old_conv.weight.mean(dim=1, keepdim=True)
                )
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {"pretrained": True, "dropout_rate": 0.2, "freeze_backbone": False}
    
    def __repr__(self) -> str:
        pt = "pretrained" if self.pretrained else "scratch"
        return f"ConvNeXt_Tiny_Pretrained({pt}, in_shape={self.in_shape}, out_size={self.out_size})"
