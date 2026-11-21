"""U-Net for OpenFWI Style-A velocity maps, built on WaveDL's UNetField."""

import math
from numbers import Integral

import torch
from torch import nn

from wavedl.models import register_model
from wavedl.models.base import SpatialShape
from wavedl.models.unet import DoubleConv, UNetField, _normalization_groups


def _time_stem(
    in_channels: int, base_channels: int, time_steps: int, grid: tuple[int, int]
) -> nn.Sequential:
    """Halve the time axis ``time_steps`` times with stride-(2, 1) conv blocks."""
    height, width = grid
    blocks = []
    channels = in_channels
    for level in range(time_steps):
        next_channels = max(1, base_channels // 2) if level == 0 else base_channels
        shape = (height * 2 ** (time_steps - level - 1), width)
        groups = _normalization_groups(next_channels, shape, 0)
        blocks.append(
            nn.Sequential(
                nn.Conv2d(
                    channels,
                    next_channels,
                    kernel_size=3,
                    stride=(2, 1),
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(groups, next_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    next_channels, next_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.GroupNorm(groups, next_channels),
                nn.ReLU(inplace=True),
            )
        )
        channels = next_channels
    # With zero or one stride steps, still supply base channels to the U-Net.
    if channels != base_channels:
        groups = _normalization_groups(base_channels, grid, 0)
        blocks.append(DoubleConv(channels, base_channels, 2, groups))
    return nn.Sequential(*blocks)


@register_model("unet_fwi")
class UNetFWI(UNetField):
    """WaveDL's UNetField on the output grid, fed by a time-compressing stem.

    Stride-(2, 1) convolutions compress the time axis of the shot gathers onto
    the depth axis of the velocity map, so the U-Net runs directly on the
    output grid and needs no final resize. It accepts the same options as
    UNetField (``base_channels``, ``depth``, ``dropout_rate``).
    """

    supported_input_channels = None  # any positive count

    def __init__(
        self,
        in_shape: SpatialShape,
        out_size: int,
        out_shape: tuple[int, ...] | None = None,
        base_channels: int = 64,
        depth: int = 4,
        dropout_rate: float = 0.1,
        in_channels: int = 1,
        **kwargs,
    ):
        if len(in_shape) != 2:
            raise ValueError("unet_fwi requires a 2D in_shape=(time, receivers).")
        if (
            not isinstance(in_channels, Integral)
            or isinstance(in_channels, bool)
            or in_channels < 1
        ):
            raise ValueError(
                f"in_channels must be a positive integer, got {in_channels}."
            )
        out_shape = (out_size, *in_shape) if out_shape is None else tuple(out_shape)
        if len(out_shape) != 3:
            raise ValueError(
                f"out_shape must use canonical (C, H, W) form, got {out_shape}."
            )
        grid = tuple(int(size) for size in out_shape[1:])
        # Build the U-Net on the output grid, then restore the gather shape.
        super().__init__(
            grid,
            out_size,
            out_shape=out_shape,
            base_channels=base_channels,
            depth=depth,
            dropout_rate=dropout_rate,
        )
        self.in_shape = tuple(int(size) for size in in_shape)
        self.in_channels = int(in_channels)

        time = self.in_shape[0]
        self.time_steps = max(0, math.ceil(math.log2(time / grid[0])))
        self.time_padding = grid[0] * 2**self.time_steps - time
        self.stem = _time_stem(self.in_channels, base_channels, self.time_steps, grid)
        # The stem supplies base_channels features, not UNetField's single channel.
        self.inc = DoubleConv(
            base_channels,
            base_channels,
            2,
            _normalization_groups(base_channels, grid, 0),
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compress time onto the output grid, then apply the U-Net."""
        x = nn.functional.pad(x, (0, 0, 0, self.time_padding))
        x = self.stem(x)
        if x.shape[2:] != self.output_spatial_shape:
            x = nn.functional.adaptive_avg_pool2d(x, self.output_spatial_shape)
        return super().forward(x)

    def __repr__(self) -> str:
        return f"UNet_FWI(in_shape={self.in_shape}, out_shape={self.out_shape})"
