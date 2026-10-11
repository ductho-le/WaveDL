"""
Vision Transformer (ViT): Transformer-Based Architecture for Regression
========================================================================

A flexible Vision Transformer implementation for regression tasks.
Supports both 1D (signals) and 2D (images) inputs via configurable patch embedding.

**Dimensionality Support**:
    - 1D: Waveforms/signals → patches are segments
    - 2D: Images/spectrograms → patches are grid squares

**Variants**:
    - vit_tiny: Smallest (~5.4M backbone params, embed_dim=192, depth=12, heads=3)
    - vit_small: Light (~21.4M backbone params, embed_dim=384, depth=12, heads=6)
    - vit_base: Standard (~85.3M backbone params, embed_dim=768, depth=12, heads=12)

References:
    Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words:
    Transformers for Image Recognition at Scale. ICLR 2021.
    https://arxiv.org/abs/2010.11929

Author: Ductho Le (ductho.le@outlook.com)
"""

from typing import Any

import torch
import torch.nn as nn

from wavedl.models.base import BaseModel, SpatialShape1D, SpatialShape2D
from wavedl.models.registry import register_model


# ViT supports 1D and 2D only
SpatialShape = SpatialShape1D | SpatialShape2D


class PatchEmbed(nn.Module):
    """
    Patch Embedding module that converts input into sequence of patch embeddings.

    Supports 1D and 2D inputs:
    - 1D: Input (B, 1, L) → (B, num_patches, embed_dim)
    - 2D: Input (B, 1, H, W) → (B, num_patches, embed_dim)

    Args:
        in_shape: Spatial shape (L,) for 1D or (H, W) for 2D
        patch_size: Size of each patch
        embed_dim: Embedding dimension
        pad_if_needed: If True, pad input to nearest patch-aligned size instead of
            dropping edge pixels. Important for NDE/QUS applications where edge
            effects matter. Default: False (original behavior with warning).
    """

    def __init__(
        self,
        in_shape: SpatialShape,
        patch_size: int,
        embed_dim: int,
        pad_if_needed: bool = False,
    ):
        super().__init__()

        self.dim = len(in_shape)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.pad_if_needed = pad_if_needed
        self._padding: tuple[int, ...] | None = None  # Will be set if padding is needed
        self.proj: nn.Conv1d | nn.Conv2d

        if self.dim == 1:
            # 1D: segment patches
            L = in_shape[0]
            remainder = L % patch_size
            if remainder != 0:
                if pad_if_needed:
                    # Pad to next multiple of patch_size
                    pad_amount = patch_size - remainder
                    self._padding = (0, pad_amount)  # (left, right)
                    L_padded = L + pad_amount
                    self.num_patches = L_padded // patch_size
                else:
                    import warnings

                    warnings.warn(
                        f"Input length {L} not divisible by patch_size {patch_size}. "
                        f"Last {remainder} elements will be dropped. "
                        f"Consider using pad_if_needed=True or padding input to "
                        f"{((L // patch_size) + 1) * patch_size}.",
                        UserWarning,
                        stacklevel=2,
                    )
                    self.num_patches = L // patch_size
            else:
                self.num_patches = L // patch_size
            self.proj = nn.Conv1d(
                1, embed_dim, kernel_size=patch_size, stride=patch_size
            )
        elif self.dim == 2:
            # 2D: grid patches
            H, W = in_shape[0], in_shape[1]
            h_rem, w_rem = H % patch_size, W % patch_size
            if h_rem != 0 or w_rem != 0:
                if pad_if_needed:
                    # Pad to next multiple of patch_size
                    h_pad = (patch_size - h_rem) % patch_size
                    w_pad = (patch_size - w_rem) % patch_size
                    # Padding format: (left, right, top, bottom)
                    self._padding = (0, w_pad, 0, h_pad)
                    H_padded, W_padded = H + h_pad, W + w_pad
                    self.num_patches = (H_padded // patch_size) * (
                        W_padded // patch_size
                    )
                else:
                    import warnings

                    warnings.warn(
                        f"Input shape ({H}, {W}) not divisible by patch_size {patch_size}. "
                        f"Border pixels will be dropped (H: {h_rem}, W: {w_rem}). "
                        f"Consider using pad_if_needed=True or padding to "
                        f"({((H // patch_size) + 1) * patch_size}, "
                        f"{((W // patch_size) + 1) * patch_size}).",
                        UserWarning,
                        stacklevel=2,
                    )
                    self.num_patches = (H // patch_size) * (W // patch_size)
            else:
                self.num_patches = (H // patch_size) * (W // patch_size)
            self.proj = nn.Conv2d(
                1, embed_dim, kernel_size=patch_size, stride=patch_size
            )
        else:
            raise ValueError(f"ViT supports 1D and 2D inputs, got {self.dim}D")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, 1, ..spatial..)

        Returns:
            Patch embeddings (B, num_patches, embed_dim)
        """
        # Apply padding if configured
        if self._padding is not None:
            x = nn.functional.pad(x, self._padding, mode="constant", value=0)

        x = self.proj(x)  # (B, embed_dim, ..reduced_spatial..)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Uses F.scaled_dot_product_attention (PyTorch 2.0+) for efficient,
    fused attention with automatic Flash Attention support when available.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Use fused SDPA (PyTorch 2.0+) for efficiency + Flash Attention
        x = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class MLP(nn.Module):
    """MLP block with GELU activation."""

    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block with pre-norm architecture."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTBase(BaseModel):
    """
    Vision Transformer base class for regression.

    Architecture:
    1. Patch embedding
    2. Add learnable position embeddings + CLS token
    3. Transformer encoder blocks
    4. Extract CLS token
    5. Regression head

    Args:
        in_shape: Spatial shape (L,) for 1D or (H, W) for 2D
        out_size: Number of regression targets
        patch_size: Size of each patch (default: 16)
        embed_dim: Embedding dimension (default: 768)
        depth: Number of transformer blocks (default: 12)
        num_heads: Number of attention heads (default: 12)
        mlp_ratio: MLP hidden dim multiplier (default: 4.0)
        dropout_rate: Dropout rate (default: 0.1)
        pad_if_needed: If True, pad input to nearest patch-aligned size instead
            of dropping edge pixels. Important for NDE/QUS applications.
    """

    def __init__(
        self,
        in_shape: SpatialShape,
        out_size: int,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.1,
        pad_if_needed: bool = False,
        **kwargs,
    ):
        super().__init__(in_shape, out_size)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.dim = len(in_shape)
        self.pad_if_needed = pad_if_needed

        # Patch embedding
        self.patch_embed = PatchEmbed(in_shape, patch_size, embed_dim, pad_if_needed)
        num_patches = self.patch_embed.num_patches

        # Learnable CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout_rate)

        # Transformer encoder
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout_rate)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Regression head
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, out_size),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 1, ..spatial..)

        Returns:
            Regression output (B, out_size)
        """
        B = x.size(0)

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Extract CLS token for regression
        cls_output = x[:, 0]

        return self.head(cls_output)

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """Return default configuration."""
        return {"patch_size": 16, "dropout_rate": 0.1}


# =============================================================================
# REGISTERED MODEL VARIANTS
# =============================================================================


@register_model("vit_tiny")
class ViTTiny(ViTBase):
    """
    ViT-Tiny: Smallest Vision Transformer variant.

    ~5.4M backbone parameters. Good for: Quick experiments, smaller datasets.

    Args:
        in_shape: (L,) for 1D or (H, W) for 2D
        out_size: Number of regression targets
        patch_size: Patch size (default: 16)
        dropout_rate: Dropout rate (default: 0.1)
    """

    def __init__(
        self, in_shape: SpatialShape, out_size: int, patch_size: int = 16, **kwargs
    ):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            patch_size=patch_size,
            embed_dim=192,
            depth=12,
            num_heads=3,
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"ViT_Tiny({self.dim}D, in_shape={self.in_shape}, out_size={self.out_size})"
        )


@register_model("vit_small")
class ViTSmall(ViTBase):
    """
    ViT-Small: Light Vision Transformer variant.

    ~21.4M backbone parameters. Good for: Balanced performance.

    Args:
        in_shape: (L,) for 1D or (H, W) for 2D
        out_size: Number of regression targets
        patch_size: Patch size (default: 16)
        dropout_rate: Dropout rate (default: 0.1)
    """

    def __init__(
        self, in_shape: SpatialShape, out_size: int, patch_size: int = 16, **kwargs
    ):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            patch_size=patch_size,
            embed_dim=384,
            depth=12,
            num_heads=6,
            **kwargs,
        )

    def __repr__(self) -> str:
        return f"ViT_Small({self.dim}D, in_shape={self.in_shape}, out_size={self.out_size})"


@register_model("vit_base")
class ViTBase_(ViTBase):
    """
    ViT-Base: Standard Vision Transformer variant.

    ~85.3M backbone parameters. Good for: High accuracy, larger datasets.

    Args:
        in_shape: (L,) for 1D or (H, W) for 2D
        out_size: Number of regression targets
        patch_size: Patch size (default: 16)
        dropout_rate: Dropout rate (default: 0.1)
    """

    def __init__(
        self, in_shape: SpatialShape, out_size: int, patch_size: int = 16, **kwargs
    ):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            patch_size=patch_size,
            embed_dim=768,
            depth=12,
            num_heads=12,
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"ViT_Base({self.dim}D, in_shape={self.in_shape}, out_size={self.out_size})"
        )
