"""
Swin Transformer: Hierarchical Vision Transformer with Shifted Windows
=======================================================================

State-of-the-art vision transformer that computes self-attention within
local windows while enabling cross-window connections via shifting.
Achieves excellent accuracy with linear computational complexity.

**Key Innovations**:
    - Hierarchical feature maps (like CNNs) for multi-scale processing
    - Shifted window attention: O(n) complexity vs O(n²) for vanilla ViT
    - Local attention with global receptive field through layer stacking
    - Strong inductive bias for structured data

**Variants**:
    - swin_t: Tiny (28M params) - Efficient default
    - swin_s: Small (50M params) - Better accuracy
    - swin_b: Base (88M params) - High accuracy

**Why Swin over ViT?**:
    - Better for smaller datasets (stronger inductive bias)
    - Handles higher resolution inputs efficiently
    - Produces hierarchical features (useful for multi-scale patterns)
    - More efficient memory usage

**Note**: Swin Transformer is 2D-only. For 1D data, use TCN. For 3D data, use ResNet3D.

References:
    Liu, Z., et al. (2021). Swin Transformer: Hierarchical Vision Transformer
    using Shifted Windows. ICCV 2021 (Best Paper). https://arxiv.org/abs/2103.14030

Author: Ductho Le (ductho.le@outlook.com)
"""

from typing import Any

import torch
import torch.nn as nn


try:
    from torchvision.models import (
        Swin_B_Weights,
        Swin_S_Weights,
        Swin_T_Weights,
        swin_b,
        swin_s,
        swin_t,
    )

    SWIN_AVAILABLE = True
except ImportError:
    swin_t = None
    swin_s = None
    swin_b = None
    Swin_T_Weights = None
    Swin_S_Weights = None
    Swin_B_Weights = None
    SWIN_AVAILABLE = False

from wavedl.models.base import BaseModel
from wavedl.models.registry import register_model


class SwinTransformerBase(BaseModel):
    """
    Base Swin Transformer class for regression tasks.

    Wraps torchvision Swin Transformer with:
    - Optional pretrained weights (ImageNet-1K or ImageNet-22K)
    - Automatic input channel adaptation (grayscale → 3ch)
    - Custom regression head with layer normalization

    Swin Transformer excels at:
    - Multi-scale feature extraction (dispersion curves, spectrograms)
    - High-resolution inputs (efficient O(n) attention)
    - Tasks requiring both local and global context
    - Transfer learning from pretrained weights

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
        Initialize Swin Transformer for regression.

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

        if not SWIN_AVAILABLE:
            raise ImportError(
                "torchvision >= 0.12 is required for Swin Transformer. "
                "Install with: pip install torchvision>=0.12"
            )

        if len(in_shape) != 2:
            raise ValueError(
                f"Swin Transformer requires 2D input (H, W), got {len(in_shape)}D. "
                "For 1D data, use TCN. For 3D data, use ResNet3D."
            )

        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.freeze_backbone = freeze_backbone
        self.regression_hidden = regression_hidden

        # Load pretrained backbone
        weights = weights_class.IMAGENET1K_V1 if pretrained else None
        self.backbone = model_fn(weights=weights)

        # Swin Transformer head structure:
        # head: Linear (embed_dim → num_classes)
        # We need to get the embedding dimension from the head

        in_features = self.backbone.head.in_features

        # Replace head with regression head
        # Use LayerNorm for stability (matches Transformer architecture)
        self.backbone.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, regression_hidden),
            nn.GELU(),  # GELU matches Transformer's activation
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(regression_hidden, regression_hidden // 2),
            nn.GELU(),
            nn.Linear(regression_hidden // 2, out_size),
        )

        # Adapt patch embedding conv for single-channel input (3× memory savings vs expand)
        self._adapt_input_channels()

        # Optionally freeze backbone for fine-tuning (after adaptation so new conv is frozen too)
        if freeze_backbone:
            self._freeze_backbone()

    def _adapt_input_channels(self):
        """Modify patch embedding conv to accept single-channel input.

        Instead of expanding 1→3 channels in forward (which triples memory),
        we replace the patch embedding conv with a 1-channel version and
        initialize weights as the mean of the pretrained RGB filters.
        """
        # Swin's patch embedding is at features[0][0]
        try:
            old_conv = self.backbone.features[0][0]
        except (IndexError, AttributeError, TypeError) as e:
            raise RuntimeError(
                f"Swin patch embed structure changed in this torchvision version. "
                f"Cannot adapt input channels. Error: {e}"
            ) from e
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
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
        self.backbone.features[0][0] = new_conv

    def _freeze_backbone(self):
        """Freeze feature extraction while retaining the regression head."""
        self._freeze_backbone_modules(("head",))

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
        """Return default configuration for Swin Transformer."""
        return {
            "pretrained": True,
            "dropout_rate": 0.3,
            "freeze_backbone": False,
            "regression_hidden": 512,
        }

    def get_optimizer_groups(self, base_lr: float, weight_decay: float = 0.05) -> list:
        """
        Get parameter groups with layer-wise learning rate decay.

        Swin Transformer benefits from decaying learning rate for earlier layers.
        This is a common practice for fine-tuning vision transformers.

        Args:
            base_lr: Base learning rate (applied to head)
            weight_decay: Weight decay coefficient

        Returns:
            List of parameter group dictionaries
        """
        # Separate parameters into 4 groups for proper LR decay:
        # 1. Head params with decay (full LR)
        # 2. Backbone params with decay (0.1× LR)
        # 3. Head bias/norm without decay (full LR)
        # 4. Backbone bias/norm without decay (0.1× LR)
        head_params = []
        backbone_params = []
        head_no_decay = []
        backbone_no_decay = []

        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue

            is_head = "head" in name
            # Classify 1-D params (biases and norm scales) as no-decay by shape,
            # not just by name: the head LayerNorm weight is named 'head.0.weight'
            # (no 'norm' substring) and would otherwise have weight decay applied
            # to a normalization gain.
            is_no_decay = "bias" in name or "norm" in name or param.ndim <= 1

            if is_head:
                if is_no_decay:
                    head_no_decay.append(param)
                else:
                    head_params.append(param)
            else:
                if is_no_decay:
                    backbone_no_decay.append(param)
                else:
                    backbone_params.append(param)

        groups = []

        if head_params:
            groups.append(
                {
                    "params": head_params,
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                }
            )

        if backbone_params:
            # Apply 0.1x learning rate to backbone (common for fine-tuning)
            groups.append(
                {
                    "params": backbone_params,
                    "lr": base_lr * 0.1,
                    "weight_decay": weight_decay,
                }
            )

        if head_no_decay:
            groups.append(
                {
                    "params": head_no_decay,
                    "lr": base_lr,
                    "weight_decay": 0.0,
                }
            )

        if backbone_no_decay:
            # Backbone bias/norm also gets 0.1× LR to match intended decay
            groups.append(
                {
                    "params": backbone_no_decay,
                    "lr": base_lr * 0.1,
                    "weight_decay": 0.0,
                }
            )

        return groups if groups else [{"params": self.parameters(), "lr": base_lr}]


# =============================================================================
# REGISTERED MODEL VARIANTS
# =============================================================================


@register_model("swin_t")
class SwinTiny(SwinTransformerBase):
    """
    Swin-T (Tiny): Efficient default for most wave-based tasks.

    ~27.5M backbone parameters. Good balance of accuracy and computational cost.
    Outperforms ResNet50 while being more efficient.

    Recommended for:
        - Default choice for 2D wave data
        - Dispersion curves, spectrograms, B-scans
        - When hierarchical features matter
        - Transfer learning with limited data

    Architecture:
        - Patch size: 4×4
        - Window size: 7×7
        - Embed dim: 96
        - Depths: [2, 2, 6, 2]
        - Heads: [3, 6, 12, 24]

    Args:
        in_shape: (H, W) image dimensions
        out_size: Number of regression targets
        pretrained: Use ImageNet pretrained weights (default: True)
        dropout_rate: Dropout rate in head (default: 0.3)
        freeze_backbone: Freeze backbone for fine-tuning (default: False)
        regression_hidden: Hidden units in regression head (default: 512)

    Example:
        >>> model = SwinTiny(in_shape=(224, 224), out_size=3)
        >>> x = torch.randn(4, 1, 224, 224)
        >>> out = model(x)  # (4, 3)
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_fn=swin_t,
            weights_class=Swin_T_Weights,
            **kwargs,
        )

    def __repr__(self) -> str:
        pt = "pretrained" if self.pretrained else "scratch"
        return f"Swin_Tiny({pt}, in={self.in_shape}, out={self.out_size})"


@register_model("swin_s")
class SwinSmall(SwinTransformerBase):
    """
    Swin-S (Small): Higher accuracy with moderate compute.

    ~48.8M backbone parameters. Better accuracy than Swin-T for larger datasets.

    Recommended for:
        - Larger datasets (>20k samples)
        - When Swin-T doesn't provide enough capacity
        - Complex multi-scale patterns

    Architecture:
        - Patch size: 4×4
        - Window size: 7×7
        - Embed dim: 96
        - Depths: [2, 2, 18, 2]  (deeper stage 3)
        - Heads: [3, 6, 12, 24]

    Args:
        in_shape: (H, W) image dimensions
        out_size: Number of regression targets
        pretrained: Use ImageNet pretrained weights (default: True)
        dropout_rate: Dropout rate in head (default: 0.3)
        freeze_backbone: Freeze backbone for fine-tuning (default: False)
        regression_hidden: Hidden units in regression head (default: 512)

    Example:
        >>> model = SwinSmall(in_shape=(224, 224), out_size=3)
        >>> x = torch.randn(4, 1, 224, 224)
        >>> out = model(x)  # (4, 3)
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_fn=swin_s,
            weights_class=Swin_S_Weights,
            **kwargs,
        )

    def __repr__(self) -> str:
        pt = "pretrained" if self.pretrained else "scratch"
        return f"Swin_Small({pt}, in={self.in_shape}, out={self.out_size})"


@register_model("swin_b")
class SwinBase(SwinTransformerBase):
    """
    Swin-B (Base): Maximum accuracy for large-scale tasks.

    ~86.7M backbone parameters. Best accuracy but requires more compute and data.

    Recommended for:
        - Very large datasets (>50k samples)
        - When accuracy is more important than efficiency
        - HPC environments with ample GPU memory
        - Research experiments

    Architecture:
        - Patch size: 4×4
        - Window size: 7×7
        - Embed dim: 128
        - Depths: [2, 2, 18, 2]
        - Heads: [4, 8, 16, 32]

    Args:
        in_shape: (H, W) image dimensions
        out_size: Number of regression targets
        pretrained: Use ImageNet pretrained weights (default: True)
        dropout_rate: Dropout rate in head (default: 0.3)
        freeze_backbone: Freeze backbone for fine-tuning (default: False)
        regression_hidden: Hidden units in regression head (default: 512)

    Example:
        >>> model = SwinBase(in_shape=(224, 224), out_size=3)
        >>> x = torch.randn(4, 1, 224, 224)
        >>> out = model(x)  # (4, 3)
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_fn=swin_b,
            weights_class=Swin_B_Weights,
            **kwargs,
        )

    def __repr__(self) -> str:
        pt = "pretrained" if self.pretrained else "scratch"
        return f"Swin_Base({pt}, in={self.in_shape}, out={self.out_size})"
