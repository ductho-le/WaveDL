"""
Vision Mamba: Efficient Visual Representation Learning with State Space Models
===============================================================================

Vision Mamba (Vim) adapts the Mamba selective state space model for vision tasks.
Provides O(n) linear complexity vs O(n²) for transformers, making it efficient
for long sequences and high-resolution images.

**Key Features**:
    - Bidirectional SSM for image understanding
    - O(n) linear complexity
    - Portable pure-PyTorch implementation; performance depends on hardware
    - Works for 1D (time-series) and 2D (images)

**Variants**:
    - mamba_1d: For 1D time-series (alternative to TCN)
    - vim_tiny: 7M params for 2D images
    - vim_small: 26M params for 2D images
    - vim_base: 98M params for 2D images

**Dependencies**:
    - PyTorch only; this module does not automatically dispatch to mamba-ssm

Reference:
    Zhu, L., et al. (2024). Vision Mamba: Efficient Visual Representation
    Learning with Bidirectional State Space Model. ICML 2024.
    https://arxiv.org/abs/2401.09417

Author: Ductho Le (ductho.le@outlook.com)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from wavedl.models.base import BaseModel, SpatialShape1D, SpatialShape2D
from wavedl.models.registry import register_model


# Type alias for Mamba models (1D and 2D only)
SpatialShape = SpatialShape1D | SpatialShape2D

__all__ = [
    "Mamba1D",
    "Mamba1DBase",
    "MambaBlock",
    "VimBase",
    "VimSmall",
    "VimTiny",
    "VisionMambaBase",
]


# =============================================================================
# SELECTIVE SSM CORE (Pure PyTorch Implementation)
# =============================================================================

# Maximum sequence length for stable parallel scan without chunking
# Beyond this, the chunked implementation is used automatically
MAX_SAFE_SEQUENCE_LENGTH = 512

# Recommended maximum for this pure-PyTorch implementation
# For longer sequences, consider using the optimized mamba-ssm package
MAX_RECOMMENDED_SEQUENCE_LENGTH = 2048


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S6) - Core of Mamba.

    The key innovation is making the SSM parameters (B, C, Δ) input-dependent,
    allowing the model to selectively focus on or ignore inputs.

    This is a pure-PyTorch implementation with a stable associative scan.
    For long sequences, a separate implementation using the optimized
    mamba-ssm CUDA kernels may be faster; installing that package alone does
    not change this implementation.

    Args:
        d_model: Model dimension
        d_state: SSM state dimension (default: 16)
        d_conv: Local convolution width (default: 4)
        expand: Expansion factor for inner dimension (default: 2)
        chunk_size: Chunk size for the stable associative scan (default: 256).
            Controls the size of intermediate tensors for long sequences.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        chunk_size: int = 256,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        if (
            not isinstance(chunk_size, int)
            or isinstance(chunk_size, bool)
            or chunk_size <= 0
        ):
            raise ValueError("chunk_size must be a positive integer.")
        self.chunk_size = chunk_size
        self.register_buffer("scan_version", torch.tensor(2, dtype=torch.int64))

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Conv for local context
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # SSM parameters (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        # Learnable SSM matrices
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = state_dict.get(prefix + "scan_version")
        if version is None or version.numel() != 1 or version.item() != 2:
            error_msgs.append(
                f"{prefix} uses an unversioned or unsupported Mamba selective scan. "
                "Published legacy weights use a different numerical recurrence. "
                "Use the original WaveDL version for legacy inference/resume, or "
                "retrain with --fresh in this version. Weights cannot be migrated "
                "by adding a version marker."
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) input sequence

        Returns:
            y: (B, L, D) output sequence
        """
        _B, L, _D = x.shape

        # Warn for very long sequences
        if L > MAX_RECOMMENDED_SEQUENCE_LENGTH and self.training:
            import warnings

            warnings.warn(
                f"Sequence length {L} > {MAX_RECOMMENDED_SEQUENCE_LENGTH}. "
                "This pure-PyTorch scan may be expensive; a separate implementation "
                "using mamba-ssm CUDA kernels may offer better performance.",
                UserWarning,
                stacklevel=2,
            )

        # Input projection and split
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each: (B, L, d_inner)

        # Conv for local context
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :L]  # Causal
        x = x.transpose(1, 2)  # (B, L, d_inner)
        x = F.silu(x)

        # SSM parameters from input
        x_proj = self.x_proj(x)  # (B, L, d_state*2 + 1)
        # NOTE: delta is derived from a single shared scalar channel (a reduced-rank
        # simplification of the reference Mamba, which uses a dt_rank>1 projection).
        # This pure-PyTorch fallback trades some selectivity for simplicity/speed.
        delta = F.softplus(self.dt_proj(x_proj[:, :, :1]))  # (B, L, d_inner)
        B_param = x_proj[:, :, 1 : self.d_state + 1]  # (B, L, d_state)
        C_param = x_proj[:, :, self.d_state + 1 :]  # (B, L, d_state)

        # Discretize A
        A = -torch.exp(self.A_log)  # (d_state,)

        # Use chunked scan for long sequences, direct scan for short
        if L > MAX_SAFE_SEQUENCE_LENGTH:
            y = self._chunked_selective_scan(x, delta, A, B_param, C_param, self.D)
        else:
            y = self._selective_scan_single(x, delta, A, B_param, C_param, self.D)

        # Gating
        y = y * F.silu(z)

        # Output projection
        return self.out_proj(y)

    @staticmethod
    def _scan_states(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute ``h[t] = a[t] * h[t-1] + b[t]`` with zero initial state.

        Pair adjacent affine recurrences, scan the pairs, then recover the even
        positions. This associative scan uses O(L) work and O(log L) depth,
        without dividing by cumulative products that underflow for ordinary
        Mamba decay rates. All operations remain differentiable.
        """
        length = b.shape[1]
        if length == 0:
            raise ValueError("Selective scan requires a nonempty sequence.")
        if length == 1:
            # Keep a zero gradient for transition parameters at length one,
            # including distributed training where unused parameters matter.
            return b + a * 0.0

        pairs = length // 2
        a_even, a_odd = a[:, : 2 * pairs : 2], a[:, 1 : 2 * pairs : 2]
        b_even, b_odd = b[:, : 2 * pairs : 2], b[:, 1 : 2 * pairs : 2]
        odd_states = SelectiveSSM._scan_states(a_odd * a_even, b_odd + a_odd * b_even)
        even_states = torch.cat(
            [b[:, :1], b[:, 2::2] + a[:, 2::2] * odd_states[:, : (length - 1) // 2]],
            dim=1,
        )
        states = torch.stack([even_states[:, :pairs], odd_states], dim=2).flatten(1, 2)
        if length % 2:
            states = torch.cat([states, even_states[:, -1:]], dim=1)
        return states

    def _selective_scan_single(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """Stable associative scan for short sequences."""
        A_bar = torch.exp(delta.unsqueeze(-1) * A)
        BX = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)
        h = self._scan_states(A_bar, BX)
        return (C.unsqueeze(2) * h).sum(-1) + D * x

    def _chunked_selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """Scan bounded chunks while preserving the exact recurrent state."""
        batch_size, seq_len, d_inner = x.shape
        if seq_len == 0:
            raise ValueError("Selective scan requires a nonempty sequence.")
        h_state = x.new_zeros(batch_size, d_inner, self.d_state)
        y_chunks = []
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            x_chunk = x[:, start:end]
            delta_chunk = delta[:, start:end]
            A_bar = torch.exp(delta_chunk.unsqueeze(-1) * A)
            BX = (
                delta_chunk.unsqueeze(-1)
                * B[:, start:end].unsqueeze(2)
                * x_chunk.unsqueeze(-1)
            )
            # Fold the previous chunk's final state into this chunk's first
            # input contribution, then use the same zero-state scan.
            BX = torch.cat(
                [BX[:, :1] + A_bar[:, :1] * h_state.unsqueeze(1), BX[:, 1:]],
                dim=1,
            )
            h = self._scan_states(A_bar, BX)
            y_chunks.append((C[:, start:end].unsqueeze(2) * h).sum(-1) + D * x_chunk)
            h_state = h[:, -1]
        return torch.cat(y_chunks, dim=1)


# =============================================================================
# MAMBA BLOCK
# =============================================================================


class MambaBlock(nn.Module):
    """
    Mamba Block with residual connection.

    Architecture:
        Input → Norm → SelectiveSSM → Residual
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))


# =============================================================================
# BIDIRECTIONAL MAMBA (For Vision)
# =============================================================================


class BidirectionalMambaBlock(nn.Module):
    """
    Bidirectional Mamba Block for vision tasks.

    Processes sequence in both forward and backward directions
    to capture global context in images.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm_forward = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.ssm_backward = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.merge = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)

        # Forward pass
        y_forward = self.ssm_forward(x_norm)

        # Backward pass (flip, process, flip back)
        y_backward = self.ssm_backward(x_norm.flip(dims=[1])).flip(dims=[1])

        # Merge
        y = self.merge(torch.cat([y_forward, y_backward], dim=-1))

        return x + y


# =============================================================================
# MAMBA 1D (For Time-Series)
# =============================================================================


class Mamba1DBase(BaseModel):
    """
    Mamba for 1D time-series data.

    Alternative to TCN with theoretically infinite receptive field
    and linear complexity.
    """

    def __init__(
        self,
        in_shape: tuple[int],
        out_size: int,
        d_model: int = 256,
        n_layers: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(in_shape, out_size)

        if len(in_shape) != 1:
            raise ValueError(f"Mamba1D requires 1D input (L,), got {len(in_shape)}D")

        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, in_shape[0], d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Mamba blocks
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model, d_state, d_conv, expand) for _ in range(n_layers)]
        )

        # Final norm
        self.norm = nn.LayerNorm(d_model)

        # Regression head
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(d_model // 2, out_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, L) input signal

        Returns:
            (B, out_size) regression output
        """
        _B, _C, L = x.shape

        # Validate positional encoding coverage
        pos_len = self.pos_embed.shape[1]
        if pos_len < L:
            import warnings

            warnings.warn(
                f"Input length {L} exceeds pos_embed size "
                f"{pos_len} (from in_shape). "
                f"Positions beyond {pos_len} will have "
                f"no positional encoding.",
                UserWarning,
                stacklevel=2,
            )

        # Reshape to sequence
        x = x.transpose(1, 2)  # (B, L, 1)
        x = self.input_proj(x)  # (B, L, d_model)

        # Add positional encoding to covered positions only;
        # positions beyond pos_embed length receive no encoding.
        embed_len = min(L, pos_len)
        x[:, :embed_len, :] = x[:, :embed_len, :] + self.pos_embed[:, :embed_len, :]

        # Mamba blocks
        for block in self.blocks:
            x = block(x)

        # Global pooling (mean over sequence)
        x = x.mean(dim=1)  # (B, d_model)

        # Final norm and head
        x = self.norm(x)
        return self.head(x)


# =============================================================================
# VISION MAMBA (For 2D Images)
# =============================================================================


class VisionMambaBase(BaseModel):
    """
    Vision Mamba (Vim) for 2D images.

    Uses bidirectional SSM to capture global context efficiently.
    O(n) complexity instead of O(n²) for transformers.
    """

    def __init__(
        self,
        in_shape: tuple[int, int],
        out_size: int,
        patch_size: int = 16,
        d_model: int = 192,
        n_layers: int = 12,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(in_shape, out_size)

        if len(in_shape) != 2:
            raise ValueError(
                f"VisionMamba requires 2D input (H, W), got {len(in_shape)}D"
            )

        self.patch_size = patch_size
        self.d_model = d_model

        H, W = in_shape
        h_rem, w_rem = H % patch_size, W % patch_size
        if h_rem != 0 or w_rem != 0:
            import warnings

            warnings.warn(
                f"Input shape ({H}, {W}) not divisible by patch_size "
                f"{patch_size}. Border pixels will be dropped "
                f"(H: {h_rem}, W: {w_rem}). Consider padding to "
                f"({((H // patch_size) + 1) * patch_size}, "
                f"{((W // patch_size) + 1) * patch_size}).",
                UserWarning,
                stacklevel=2,
            )
        self.num_patches = (H // patch_size) * (W // patch_size)
        self.grid_size = (H // patch_size, W // patch_size)

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            1, d_model, kernel_size=patch_size, stride=patch_size
        )

        # CLS token for classification/regression
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Bidirectional Mamba blocks
        self.blocks = nn.ModuleList(
            [
                BidirectionalMambaBlock(d_model, d_state, d_conv, expand)
                for _ in range(n_layers)
            ]
        )

        # Final norm
        self.norm = nn.LayerNorm(d_model)

        # Regression head
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(d_model // 2, out_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) input image

        Returns:
            (B, out_size) regression output
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, d_model, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1 + num_patches, d_model)

        # Add positional embedding
        x = x + self.pos_embed

        # Bidirectional Mamba blocks
        for block in self.blocks:
            x = block(x)

        # Extract CLS token
        cls_output = x[:, 0]  # (B, d_model)

        # Final norm and head
        cls_output = self.norm(cls_output)
        return self.head(cls_output)


# =============================================================================
# REGISTERED VARIANTS
# =============================================================================


@register_model("mamba_1d")
class Mamba1D(Mamba1DBase):
    """
    Mamba 1D: ~3.4M backbone parameters (for time-series regression).

    8 layers, 256 dim. Alternative to TCN for time-series.
    Pure PyTorch implementation.

    Example:
        >>> model = Mamba1D(in_shape=(4096,), out_size=3)
        >>> x = torch.randn(4, 1, 4096)
        >>> out = model(x)  # (4, 3)
    """

    def __init__(self, in_shape: tuple[int], out_size: int, **kwargs):
        kwargs.setdefault("d_model", 256)
        kwargs.setdefault("n_layers", 8)
        super().__init__(in_shape=in_shape, out_size=out_size, **kwargs)

    def __repr__(self) -> str:
        return f"Mamba1D(in_shape={self.in_shape}, out_size={self.out_size})"


@register_model("vim_tiny")
class VimTiny(VisionMambaBase):
    """
    Vision Mamba Tiny: ~6.6M backbone parameters.

    12 layers, 192 dim. For 2D images.
    Pure PyTorch implementation with O(n) complexity.

    Example:
        >>> model = VimTiny(in_shape=(224, 224), out_size=3)
        >>> x = torch.randn(4, 1, 224, 224)
        >>> out = model(x)  # (4, 3)
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        kwargs.setdefault("patch_size", 16)
        kwargs.setdefault("d_model", 192)
        kwargs.setdefault("n_layers", 12)
        super().__init__(in_shape=in_shape, out_size=out_size, **kwargs)

    def __repr__(self) -> str:
        return f"VisionMamba_Tiny(in_shape={self.in_shape}, out_size={self.out_size})"


@register_model("vim_small")
class VimSmall(VisionMambaBase):
    """
    Vision Mamba Small: ~51.1M backbone parameters.

    24 layers, 384 dim. For 2D images.
    Pure PyTorch implementation with O(n) complexity.
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        kwargs.setdefault("patch_size", 16)
        kwargs.setdefault("d_model", 384)
        kwargs.setdefault("n_layers", 24)
        super().__init__(in_shape=in_shape, out_size=out_size, **kwargs)

    def __repr__(self) -> str:
        return f"VisionMamba_Small(in_shape={self.in_shape}, out_size={self.out_size})"


@register_model("vim_base")
class VimBase(VisionMambaBase):
    """
    Vision Mamba Base: ~201.4M backbone parameters.

    24 layers, 768 dim. For 2D images.
    Pure PyTorch implementation with O(n) complexity.
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        kwargs.setdefault("patch_size", 16)
        kwargs.setdefault("d_model", 768)
        kwargs.setdefault("n_layers", 24)
        super().__init__(in_shape=in_shape, out_size=out_size, **kwargs)

    def __repr__(self) -> str:
        return f"VisionMamba_Base(in_shape={self.in_shape}, out_size={self.out_size})"
