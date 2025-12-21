"""
EfficientNet: Efficient and Scalable CNNs for Regression
=========================================================

Wrapper around torchvision's EfficientNet with a regression head.
Provides optional ImageNet pretrained weights for transfer learning.

**Variants**:
    - efficientnet_b0: Smallest, fastest (5.3M params)
    - efficientnet_b1: Light (7.8M params)
    - efficientnet_b2: Balanced (9.1M params)

**Note**: EfficientNet is 2D-only. For 1D/3D data, use ResNet or CNN.

Author: Ductho Le (ductho.le@outlook.com)
Version: 1.0.0
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn

try:
    from torchvision.models import (
        efficientnet_b0, efficientnet_b1, efficientnet_b2,
        EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights
    )
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

from models.base import BaseModel
from models.registry import register_model


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
            nn.Linear(512, out_size)
        )
        
        # Adapt first conv for single-channel input
        # EfficientNet expects 3 channels, we'll expand grayscale
        self._adapt_input_channels()
        
        # Optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone()
    
    def _adapt_input_channels(self):
        """Modify first conv to handle single-channel input by expanding to 3ch."""
        # We'll handle this in forward by repeating channels
        pass
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters except the classifier."""
        for name, param in self.backbone.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 1, H, W)
            
        Returns:
            Output tensor of shape (B, out_size)
        """
        # Expand single channel to 3 channels for pretrained weights
        if x.size(1) == 1:
            x = x.expand(-1, 3, -1, -1)
        
        return self.backbone(x)
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "pretrained": True,
            "dropout_rate": 0.2,
            "freeze_backbone": False
        }


# =============================================================================
# REGISTERED MODEL VARIANTS
# =============================================================================

@register_model("efficientnet_b0")
class EfficientNetB0(EfficientNetBase):
    """
    EfficientNet-B0: Smallest, most efficient variant.
    
    ~5.3M parameters. Good for: Quick training, limited compute, baseline.
    
    Args:
        in_shape: (H, W) image dimensions
        out_size: Number of regression targets
        pretrained: Use ImageNet pretrained weights (default: True)
        dropout_rate: Dropout rate (default: 0.2)
        freeze_backbone: Freeze backbone weights for fine-tuning (default: False)
    """
    
    def __init__(self, in_shape: Tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_fn=efficientnet_b0,
            weights_class=EfficientNet_B0_Weights,
            **kwargs
        )
    
    def __repr__(self) -> str:
        pt = "pretrained" if self.pretrained else "scratch"
        return f"EfficientNet_B0({pt}, in_shape={self.in_shape}, out_size={self.out_size})"


@register_model("efficientnet_b1")
class EfficientNetB1(EfficientNetBase):
    """
    EfficientNet-B1: Slightly larger variant.
    
    ~7.8M parameters. Good for: Better accuracy with moderate compute.
    
    Args:
        in_shape: (H, W) image dimensions
        out_size: Number of regression targets
        pretrained: Use ImageNet pretrained weights (default: True)
        dropout_rate: Dropout rate (default: 0.2)
        freeze_backbone: Freeze backbone weights for fine-tuning (default: False)
    """
    
    def __init__(self, in_shape: Tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_fn=efficientnet_b1,
            weights_class=EfficientNet_B1_Weights,
            **kwargs
        )
    
    def __repr__(self) -> str:
        pt = "pretrained" if self.pretrained else "scratch"
        return f"EfficientNet_B1({pt}, in_shape={self.in_shape}, out_size={self.out_size})"


@register_model("efficientnet_b2")
class EfficientNetB2(EfficientNetBase):
    """
    EfficientNet-B2: Best balance of size and performance.
    
    ~9.1M parameters. Good for: High accuracy without excessive compute.
    
    Args:
        in_shape: (H, W) image dimensions
        out_size: Number of regression targets
        pretrained: Use ImageNet pretrained weights (default: True)
        dropout_rate: Dropout rate (default: 0.2)
        freeze_backbone: Freeze backbone weights for fine-tuning (default: False)
    """
    
    def __init__(self, in_shape: Tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_fn=efficientnet_b2,
            weights_class=EfficientNet_B2_Weights,
            **kwargs
        )
    
    def __repr__(self) -> str:
        pt = "pretrained" if self.pretrained else "scratch"
        return f"EfficientNet_B2({pt}, in_shape={self.in_shape}, out_size={self.out_size})"
