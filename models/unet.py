"""
U-Net: Encoder-Decoder Architecture for Regression
====================================================

A dimension-agnostic U-Net implementation for tasks requiring either:
- Spatial output (e.g., velocity field prediction)
- Vector output (global pooling → regression head)

**Dimensionality Support**:
    - 1D: Waveforms, signals (N, 1, L) → Conv1d
    - 2D: Images, spectrograms (N, 1, H, W) → Conv2d  
    - 3D: Volumetric data (N, 1, D, H, W) → Conv3d

**Variants**:
    - unet: Full encoder-decoder with spatial output capability
    - unet_regression: U-Net with global pooling for vector regression

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


def _get_layers(dim: int):
    """Get dimension-appropriate layer classes."""
    if dim == 1:
        return nn.Conv1d, nn.ConvTranspose1d, nn.MaxPool1d, nn.AdaptiveAvgPool1d
    elif dim == 2:
        return nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d
    elif dim == 3:
        return nn.Conv3d, nn.ConvTranspose3d, nn.MaxPool3d, nn.AdaptiveAvgPool3d
    else:
        raise ValueError(f"Unsupported dimensionality: {dim}D")


class DoubleConv(nn.Module):
    """Double convolution block: Conv-GN-ReLU-Conv-GN-ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, dim: int = 2):
        super().__init__()
        Conv = _get_layers(dim)[0]
        
        num_groups = min(32, out_channels)
        while out_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        
        self.double_conv = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
            Conv(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    
    def __init__(self, in_channels: int, out_channels: int, dim: int = 2):
        super().__init__()
        _, _, MaxPool, _ = _get_layers(dim)
        
        self.maxpool_conv = nn.Sequential(
            MaxPool(2),
            DoubleConv(in_channels, out_channels, dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""
    
    def __init__(self, in_channels: int, out_channels: int, dim: int = 2):
        super().__init__()
        _, ConvTranspose, _, _ = _get_layers(dim)
        
        # in_channels comes from previous layer
        # After upconv: in_channels // 2
        # After concat with skip (out_channels): in_channels // 2 + out_channels = in_channels
        # Then DoubleConv: in_channels -> out_channels
        self.up = ConvTranspose(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dim)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Handle size mismatch (pad x1 to match x2)
        if x1.shape[2:] != x2.shape[2:]:
            diff = [x2.size(i + 2) - x1.size(i + 2) for i in range(len(x1.shape) - 2)]
            # Pad x1 to match x2
            pad = []
            for d in reversed(diff):
                pad.extend([d // 2, d - d // 2])
            x1 = nn.functional.pad(x1, pad)
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetBase(BaseModel):
    """
    Base U-Net class for regression tasks.
    
    Standard U-Net architecture:
    - Encoder path with downsampling
    - Decoder path with upsampling and skip connections
    - Optional spatial or vector output
    """
    
    def __init__(
        self,
        in_shape: SpatialShape,
        out_size: int,
        base_channels: int = 64,
        depth: int = 4,
        dropout_rate: float = 0.1,
        spatial_output: bool = False,
        **kwargs
    ):
        super().__init__(in_shape, out_size)
        
        self.dim = len(in_shape)
        self.base_channels = base_channels
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.spatial_output = spatial_output
        
        Conv, _, _, AdaptivePool = _get_layers(self.dim)
        
        # Channel progression: 64 -> 128 -> 256 -> 512 (for depth=4)
        # features[i] = base_channels * 2^i
        features = [base_channels * (2 ** i) for i in range(depth + 1)]
        
        # Initial double conv (1 -> features[0])
        self.inc = DoubleConv(1, features[0], self.dim)
        
        # Encoder (down path)
        self.downs = nn.ModuleList()
        for i in range(depth):
            self.downs.append(Down(features[i], features[i + 1], self.dim))
        
        # Decoder (up path)
        self.ups = nn.ModuleList()
        for i in range(depth):
            # Input: features[depth - i], Skip: features[depth - 1 - i], Output: features[depth - 1 - i]
            self.ups.append(Up(features[depth - i], features[depth - 1 - i], self.dim))
        
        if spatial_output:
            # Spatial output: 1x1 conv to out_size channels
            self.outc = Conv(features[0], out_size, kernel_size=1)
        else:
            # Vector output: global pooling + regression head
            self.global_pool = AdaptivePool(1)
            self.head = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(features[0], 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(256, out_size)
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                             nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Encoder path - collect skip connections
        x1 = self.inc(x)
        
        skips = [x1]
        x = x1
        for down in self.downs:
            x = down(x)
            skips.append(x)
        
        # Remove last (bottleneck output, not a skip)
        skips = skips[:-1]
        
        # Decoder path - use skips in reverse order
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)
        
        if self.spatial_output:
            return self.outc(x)
        else:
            x = self.global_pool(x)
            x = x.flatten(1)
            return self.head(x)
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "base_channels": 64,
            "depth": 4,
            "dropout_rate": 0.1
        }


# =============================================================================
# REGISTERED MODEL VARIANTS
# =============================================================================

@register_model("unet")
class UNet(UNetBase):
    """
    U-Net with spatial output capability.
    
    Good for: Pixel/voxel-wise regression (velocity fields, spatial maps).
    
    Note: For spatial output, out_size is the number of output channels.
    Output shape: (B, out_size, *spatial_dims) for spatial_output=True.
    
    Args:
        in_shape: (L,), (H, W), or (D, H, W)
        out_size: Number of output channels for spatial output
        base_channels: Base channel count (default: 64)
        depth: Number of encoder/decoder levels (default: 4)
        spatial_output: If True, output spatial map; if False, output vector
        dropout_rate: Dropout rate (default: 0.1)
    """
    
    def __init__(self, in_shape: SpatialShape, out_size: int, spatial_output: bool = True, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            spatial_output=spatial_output,
            **kwargs
        )
    
    def __repr__(self) -> str:
        mode = "spatial" if self.spatial_output else "vector"
        return f"UNet({self.dim}D, {mode}, in_shape={self.in_shape}, out_size={self.out_size})"


@register_model("unet_regression")
class UNetRegression(UNetBase):
    """
    U-Net for vector regression output.
    
    Uses U-Net encoder-decoder but applies global pooling at the end
    for standard vector regression output.
    
    Good for: Leveraging U-Net features (multi-scale, skip connections)
    for standard regression tasks.
    
    Args:
        in_shape: (L,), (H, W), or (D, H, W)
        out_size: Number of regression targets
        base_channels: Base channel count (default: 64)
        depth: Number of encoder/decoder levels (default: 4)
        dropout_rate: Dropout rate (default: 0.1)
    """
    
    def __init__(self, in_shape: SpatialShape, out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            spatial_output=False,
            **kwargs
        )
    
    def __repr__(self) -> str:
        return f"UNet_Regression({self.dim}D, in_shape={self.in_shape}, out_size={self.out_size})"
