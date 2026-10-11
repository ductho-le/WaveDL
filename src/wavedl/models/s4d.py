"""
S4D: Diagonal Structured State Space Model for 1D Waveform Regression
======================================================================

Implements the S4D (Diagonal S4) model for learning long-range dependencies
in physical wave signals. The S4D kernel computes a diagonal state space
model (SSM) as a 1D convolution via FFT, yielding O(L log L) complexity
and full compatibility with torch.compile.

**Key Features**:
    - S4D-Lin kernel: diagonal SSM with HiPPO-LegS initialization
    - O(L log L) convolution via torch.fft (no recurrence at train time)
    - Fully vectorized: no Python loops over sequence length
    - MPS-compatible: complex arithmetic via real/imag tensor pairs
    - torch.compile-safe: no dynamic control flow

**Variants**:
    - s4d_small:  d_model=64,  n_layers=4, d_state=32  (~0.8M params)
    - s4d:        d_model=128, n_layers=6, d_state=64  (~3.2M params)
    - s4d_large:  d_model=256, n_layers=8, d_state=64  (~11M params)

**Note**: S4D is 1D-only. For 2D data use CNN, ResNet, or ViT.

References:
    Gu, A., et al. (2022). On the Parameterization and Initialization of
    Diagonal State Space Models. NeurIPS 2022.
    https://arxiv.org/abs/2206.11893

    Gu, A., et al. (2021). Efficiently Modeling Long Sequences with
    Structured State Spaces. ICLR 2022.
    https://arxiv.org/abs/2111.00396

Author: Ductho Le (ductho.le@outlook.com)
"""

import copy
import math
from typing import Any, ClassVar

import torch
import torch.nn as nn

from wavedl.models.base import BaseModel
from wavedl.models.registry import register_model


# =============================================================================
# S4D KERNEL
# =============================================================================


class S4DKernel(nn.Module):
    """
    Diagonal S4D-Lin kernel computed via FFT convolution.

    Implements the closed-form convolution kernel for a diagonal SSM:
        h_n = C * (exp(A_n * L) - 1) / A_n   [S4D-Lin approximation]

    where A = diag(A_0, ..., A_{N-1}) with HiPPO-LegS initialization:
        A_n = -1/2 + i*pi*n

    The full kernel K(t) = sum_n C_n * exp(A_n * t) * B_n is computed
    as a 1D FFT convolution in O(L log L).

    To maintain MPS (Apple Silicon) compatibility, complex numbers are
    represented as pairs of real tensors (real, imag) throughout.

    Args:
        d_model: Number of channels (model dimension)
        d_state: Number of diagonal SSM states per channel
        dt_min: Minimum step size (log-uniformly sampled)
        dt_max: Maximum step size
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Log step size (learnable), shape: (d_model, d_state)
        log_dt = torch.rand(d_model, d_state) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # HiPPO-LegS initialization: A_n = -1/2 + i*pi*n
        # Shape: (d_state,)
        n = torch.arange(d_state, dtype=torch.float32)
        A_real = -0.5 * torch.ones(d_state)
        A_imag = math.pi * n
        # Store as learnable parameters (real and imag separately).
        # A_real is clamped to non-positive in forward() to prevent
        # exponential growth of the SSM kernel (divergence → NaN).
        self.A_real = nn.Parameter(A_real)  # (d_state,)
        self.A_imag = nn.Parameter(A_imag)  # (d_state,)

        # C and B parameters (complex, stored as real+imag)
        # Shape: (d_model, d_state)
        self.C_real = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.C_imag = nn.Parameter(torch.randn(d_model, d_state) * 0.02)

    # Maximum chunk size along L for kernel computation.
    # Limits peak intermediate memory to chunk_size * d_model * d_state * ~4 tensors.
    _KERNEL_CHUNK_SIZE: int = 1024

    def forward(self, L: int) -> torch.Tensor:
        """
        Compute the S4D-Lin convolution kernel for sequence length L.

        Uses chunked computation along the sequence dimension to avoid
        materializing full (L, d_model, d_state) intermediates when L is
        large (e.g. L=8192, d_model=256, d_state=64 → ~2 GB unchunked).

        Returns:
            kernel: (d_model, L) real-valued convolution kernel
        """
        dt = self.log_dt.exp()  # (d_model, d_state)

        # Discretize: A_bar = exp(dt * A)
        # A shape: (d_state,), dt shape: (d_model, d_state)
        # Clamp A_real to non-positive: guarantees exp(t*A_real) decays,
        # preventing exponential kernel growth if A_real drifts positive.
        dA_real = dt * self.A_real.clamp(max=0)  # (d_model, d_state)
        dA_imag = dt * self.A_imag  # (d_model, d_state)

        # C coefficients (broadcast-ready)
        C_r = self.C_real.unsqueeze(0)  # (1, d_model, d_state)
        C_i = self.C_imag.unsqueeze(0)  # (1, d_model, d_state)

        # Chunked computation: process L in blocks to limit peak memory.
        # Each chunk materializes (chunk_size, d_model, d_state) intermediates
        # and immediately sums over d_state, keeping only (d_model, chunk_size).
        chunk_size = self._KERNEL_CHUNK_SIZE
        kernel_chunks: list[torch.Tensor] = []

        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            t = torch.arange(start, end, device=dt.device, dtype=dt.dtype)
            t = t.view(-1, 1, 1)  # (chunk_len, 1, 1)

            mag_t = (t * dA_real).exp()  # (chunk_len, d_model, d_state)
            cos_t = (t * dA_imag).cos()
            sin_t = (t * dA_imag).sin()

            # kernel_t = Re(C * exp(A)^t * B) — B=1 for S4D-Lin
            k_real = mag_t * (C_r * cos_t - C_i * sin_t)
            kernel_chunks.append(k_real.sum(dim=-1))  # (chunk_len, d_model)

        kernel = torch.cat(kernel_chunks, dim=0)  # (L, d_model)
        return kernel.permute(1, 0)  # (d_model, L)


# =============================================================================
# S4D LAYER
# =============================================================================


class S4DLayer(nn.Module):
    """
    S4D layer: SSM mixing + position-wise FFN with residual.

    Architecture:
        x → LayerNorm → S4D conv → GELU → Dropout
          → FFN (expand → GELU → contract) → + residual
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        # SSM mixing (operates on sequence dim)
        self.kernel = S4DKernel(d_model, d_state)
        self.D = nn.Parameter(torch.ones(d_model))  # skip/feedthrough
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # FFN
        inner = d_model * expand
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model) — sequence-first format

        Returns:
            (B, L, d_model)
        """
        residual = x
        _B, L, _ = x.shape

        # SSM mixing
        z = self.norm(x)  # (B, L, d_model)
        z = z.permute(0, 2, 1)  # (B, d_model, L)

        # Compute kernel
        k = self.kernel(L)  # (d_model, L)

        # FFT convolution
        k_f = torch.fft.rfft(k, n=2 * L)  # (d_model, L+1) complex
        z_f = torch.fft.rfft(z, n=2 * L)  # (B, d_model, L+1) complex
        y_f = k_f.unsqueeze(0) * z_f  # broadcast over batch
        y = torch.fft.irfft(y_f, n=2 * L)[..., :L]  # (B, d_model, L)

        # Add skip connection D*x (feedthrough)
        y = y + self.D.view(1, -1, 1) * z

        y = y.permute(0, 2, 1)  # (B, L, d_model)
        y = self.act(y)
        y = self.dropout(y)
        y = self.out_proj(y)

        x = residual + y

        # FFN
        x = x + self.ffn(x)
        return x


# =============================================================================
# S4D BASE MODEL
# =============================================================================


class S4DBase(BaseModel):
    """
    S4D model for 1D regression.

    Architecture:
    1. Input projection (1 → d_model via Conv1d)
    2. N × S4D layers (SSM mixing + FFN)
    3. Global average pool
    4. Regression head
    """

    def __init__(
        self,
        in_shape: tuple[int],
        out_size: int,
        d_model: int,
        n_layers: int,
        d_state: int = 64,
        expand: int = 2,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        """
        Args:
            in_shape: (L,) input signal length
            out_size: Number of regression output targets
            d_model: Model (hidden) dimension
            n_layers: Number of S4D layers
            d_state: Number of SSM states per channel (default: 64)
            expand: FFN expansion factor (default: 2)
            dropout_rate: Dropout rate (default: 0.1)
        """
        super().__init__(in_shape, out_size)

        if len(in_shape) != 1:
            raise ValueError(
                f"S4D requires 1D input (L,), got {len(in_shape)}D. "
                "For 2D/3D data, use ResNet, EfficientNet, or Swin."
            )

        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state

        # Input projection: (B, 1, L) → (B, d_model, L) → (B, L, d_model)
        self.input_proj = nn.Conv1d(1, d_model, kernel_size=1)

        # S4D layers
        self.layers = nn.ModuleList(
            [
                S4DLayer(
                    d_model=d_model,
                    d_state=d_state,
                    expand=expand,
                    dropout=dropout_rate,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm_out = nn.LayerNorm(d_model)

        # Global aggregation + head
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(d_model // 2, out_size),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 1, L)

        Returns:
            Output tensor of shape (B, out_size)
        """
        # Input projection
        x = self.input_proj(x)  # (B, d_model, L)
        x = x.permute(0, 2, 1)  # (B, L, d_model)

        # S4D layers
        for layer in self.layers:
            x = layer(x)

        x = self.norm_out(x)  # (B, L, d_model)

        # Global average pool over sequence
        x = x.mean(dim=1)  # (B, d_model)

        return self.head(x)  # (B, out_size)

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        return {
            "d_model": 128,
            "n_layers": 6,
            "d_state": 64,
            "expand": 2,
            "dropout_rate": 0.1,
            **copy.deepcopy(getattr(cls, "_variant_defaults", {})),
        }


# =============================================================================
# REGISTERED MODEL VARIANTS
# =============================================================================


@register_model("s4d_small")
class S4DSmall(S4DBase):
    """
    S4D-Small: Lightweight diagonal SSM for 1D waveform regression.

    ~0.8M parameters. d_model=64, n_layers=4, d_state=32.

    Recommended for:
        - Quick experiments with SSM architectures
        - Short to medium signals (≤ 4096 samples)
        - Memory-constrained environments

    Args:
        in_shape: (L,) input signal length
        out_size: Number of regression targets

    Example:
        >>> model = S4DSmall(in_shape=(2048,), out_size=3)
        >>> x = torch.randn(4, 1, 2048)
        >>> out = model(x)  # (4, 3)
    """

    _variant_defaults: ClassVar[dict[str, Any]] = {
        "d_model": 64,
        "n_layers": 4,
        "d_state": 32,
    }

    def __init__(self, in_shape: tuple[int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            **{**copy.deepcopy(self._variant_defaults), **kwargs},
        )

    def __repr__(self) -> str:
        return (
            f"S4D_Small(d_model={self.d_model}, layers={self.n_layers}, "
            f"in_shape={self.in_shape}, out={self.out_size})"
        )


@register_model("s4d")
class S4D(S4DBase):
    """
    S4D: Standard diagonal SSM for 1D waveform regression.

    ~3.2M parameters. d_model=128, n_layers=6, d_state=64.

    Recommended for:
        - Ultrasonic waveform analysis (large receptive field)
        - Seismic and acoustic signals
        - Any 1D regression task benefiting from long-range dependencies
        - Complement to Mamba for comparing SSM families

    Args:
        in_shape: (L,) input signal length
        out_size: Number of regression targets

    Example:
        >>> model = S4D(in_shape=(4096,), out_size=3)
        >>> x = torch.randn(4, 1, 4096)
        >>> out = model(x)  # (4, 3)
    """

    _variant_defaults: ClassVar[dict[str, Any]] = {
        "d_model": 128,
        "n_layers": 6,
        "d_state": 64,
    }

    def __init__(self, in_shape: tuple[int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            **{**copy.deepcopy(self._variant_defaults), **kwargs},
        )

    def __repr__(self) -> str:
        return (
            f"S4D(d_model={self.d_model}, layers={self.n_layers}, "
            f"in_shape={self.in_shape}, out={self.out_size})"
        )


@register_model("s4d_large")
class S4DLarge(S4DBase):
    """
    S4D-Large: High-capacity diagonal SSM for 1D waveform regression.

    ~11M parameters. d_model=256, n_layers=8, d_state=64.

    Recommended for:
        - Long sequences with complex temporal dynamics (> 8192 samples)
        - Large datasets requiring high model capacity
        - Research comparisons against Mamba-Large

    Args:
        in_shape: (L,) input signal length
        out_size: Number of regression targets

    Example:
        >>> model = S4DLarge(in_shape=(8192,), out_size=3)
        >>> x = torch.randn(4, 1, 8192)
        >>> out = model(x)  # (4, 3)
    """

    _variant_defaults: ClassVar[dict[str, Any]] = {
        "d_model": 256,
        "n_layers": 8,
        "d_state": 64,
    }

    def __init__(self, in_shape: tuple[int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            **{**copy.deepcopy(self._variant_defaults), **kwargs},
        )

    def __repr__(self) -> str:
        return (
            f"S4D_Large(d_model={self.d_model}, layers={self.n_layers}, "
            f"in_shape={self.in_shape}, out={self.out_size})"
        )
