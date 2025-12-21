"""
Vision Transformer (ViT): Transformer-Based Architecture for Regression
========================================================================

A flexible Vision Transformer implementation for regression tasks.
Supports both 1D (signals) and 2D (images) inputs via configurable patch embedding.

**Dimensionality Support**:
    - 1D: Waveforms/signals → patches are segments
    - 2D: Images/spectrograms → patches are grid squares

**Variants**:
    - vit_tiny: Smallest (embed_dim=192, depth=12, heads=3)  
    - vit_small: Light (embed_dim=384, depth=12, heads=6)
    - vit_base: Standard (embed_dim=768, depth=12, heads=12)

Author: Ductho Le (ductho.le@outlook.com)
Version: 1.0.0
"""

from typing import Tuple, Dict, Any, Union
import math
import torch
import torch.nn as nn

from models.base import BaseModel
from models.registry import register_model


# Type alias for spatial shapes
SpatialShape = Union[Tuple[int], Tuple[int, int]]


class PatchEmbed(nn.Module):
    """
    Patch Embedding module that converts input into sequence of patch embeddings.
    
    Supports 1D and 2D inputs:
    - 1D: Input (B, 1, L) → (B, num_patches, embed_dim)
    - 2D: Input (B, 1, H, W) → (B, num_patches, embed_dim)
    """
    
    def __init__(
        self,
        in_shape: SpatialShape,
        patch_size: int,
        embed_dim: int
    ):
        super().__init__()
        
        self.dim = len(in_shape)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        if self.dim == 1:
            # 1D: segment patches
            L = in_shape[0]
            self.num_patches = L // patch_size
            self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        elif self.dim == 2:
            # 2D: grid patches
            H, W = in_shape
            self.num_patches = (H // patch_size) * (W // patch_size)
            self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            raise ValueError(f"ViT supports 1D and 2D inputs, got {self.dim}D")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, 1, ..spatial..)
            
        Returns:
            Patch embeddings (B, num_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, ..reduced_spatial..)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
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
        dropout: float = 0.0
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
        **kwargs
    ):
        super().__init__(in_shape, out_size)
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.dim = len(in_shape)
        
        # Patch embedding
        self.patch_embed = PatchEmbed(in_shape, patch_size, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Learnable CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout_rate)
        
        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout_rate)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Regression head
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, out_size)
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
    def get_default_config(cls) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "patch_size": 16,
            "dropout_rate": 0.1
        }


# =============================================================================
# REGISTERED MODEL VARIANTS
# =============================================================================

@register_model("vit_tiny")
class ViTTiny(ViTBase):
    """
    ViT-Tiny: Smallest Vision Transformer variant.
    
    ~5.7M parameters. Good for: Quick experiments, smaller datasets.
    
    Args:
        in_shape: (L,) for 1D or (H, W) for 2D
        out_size: Number of regression targets
        patch_size: Patch size (default: 16)
        dropout_rate: Dropout rate (default: 0.1)
    """
    
    def __init__(self, in_shape: SpatialShape, out_size: int, patch_size: int = 16, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            patch_size=patch_size,
            embed_dim=192,
            depth=12,
            num_heads=3,
            **kwargs
        )
    
    def __repr__(self) -> str:
        return f"ViT_Tiny({self.dim}D, in_shape={self.in_shape}, out_size={self.out_size})"


@register_model("vit_small")
class ViTSmall(ViTBase):
    """
    ViT-Small: Light Vision Transformer variant.
    
    ~22M parameters. Good for: Balanced performance.
    
    Args:
        in_shape: (L,) for 1D or (H, W) for 2D
        out_size: Number of regression targets
        patch_size: Patch size (default: 16)
        dropout_rate: Dropout rate (default: 0.1)
    """
    
    def __init__(self, in_shape: SpatialShape, out_size: int, patch_size: int = 16, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            patch_size=patch_size,
            embed_dim=384,
            depth=12,
            num_heads=6,
            **kwargs
        )
    
    def __repr__(self) -> str:
        return f"ViT_Small({self.dim}D, in_shape={self.in_shape}, out_size={self.out_size})"


@register_model("vit_base")
class ViTBase_(ViTBase):
    """
    ViT-Base: Standard Vision Transformer variant.
    
    ~86M parameters. Good for: High accuracy, larger datasets.
    
    Args:
        in_shape: (L,) for 1D or (H, W) for 2D
        out_size: Number of regression targets
        patch_size: Patch size (default: 16)
        dropout_rate: Dropout rate (default: 0.1)
    """
    
    def __init__(self, in_shape: SpatialShape, out_size: int, patch_size: int = 16, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            patch_size=patch_size,
            embed_dim=768,
            depth=12,
            num_heads=12,
            **kwargs
        )
    
    def __repr__(self) -> str:
        return f"ViT_Base({self.dim}D, in_shape={self.in_shape}, out_size={self.out_size})"
