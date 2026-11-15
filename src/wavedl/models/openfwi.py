# © 2022. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos

# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.

# Department of Energy/National Nuclear Security Administration. All rights in the program are

# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear

# Security Administration. The Government is granted for itself and others acting on its behalf a

# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare

# derivative works, distribute copies to the public, perform publicly and display publicly, and to permit

# others to do so.

# Adapted from lanl/OpenFWI@48754806 (BSD-3-Clause); modifications: WaveDL model
# registration and shape contracts, type hints, documentation, and safe checkpoint loading.
"""OpenFWI InversionNet and VelocityGAN baselines, with upstream-compatible weights.

Credit: OmniWave Lab (Youzuo Lin's group, UNC–Chapel Hill) and Los Alamos
National Laboratory (LANL). Adapted from lanl/OpenFWI at commit
48754806b7b4c5877259c6b958a87f4513fcdc0b.
See THIRD_PARTY_NOTICES.md for the upstream BSD-3-Clause license.

References:
    Wu & Lin (2020), InversionNet, IEEE TCI 6:419-433,
    doi:10.1109/TCI.2019.2956866.
    Zhang & Lin (2020), VelocityGAN, IEEE TGRS 58(10):6900-6913,
    doi:10.1109/TGRS.2020.2977635.
    Deng et al. (2022), OpenFWI, NeurIPS Datasets & Benchmarks.

The original layer names, initialization, normalization, activations, and
80-to-70 crop are retained. Inputs are five-shot gathers of shape
``(B, 5, 1000, 70)``; outputs are velocity fields ``(B, 1, 70, 70)`` in [-1, 1].
"""

from argparse import Namespace
from collections import OrderedDict
from collections.abc import Mapping
from math import ceil
from numbers import Integral
from os import PathLike
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from wavedl.models.base import BaseModel
from wavedl.models.registry import register_model


NORM_LAYERS = {"bn": nn.BatchNorm2d, "in": nn.InstanceNorm2d, "ln": nn.LayerNorm}
KernelSize = int | tuple[int, int]


def replace_legacy(old_dict: Mapping[str, torch.Tensor]) -> OrderedDict:
    """Apply OpenFWI's legacy building-block checkpoint key conversion."""
    converted = []
    for key, value in old_dict.items():
        key = (
            key.replace("Conv2DwithBN", "layers")
            .replace("Conv2DwithBN_Tanh", "layers")
            .replace("Deconv2DwithBN", "layers")
            .replace("ResizeConv2DwithBN", "layers")
        )
        converted.append((key, value))
    return OrderedDict(converted)


class ConvBlock(nn.Module):
    """Upstream convolution, optional normalization, and LeakyReLU block."""

    def __init__(
        self,
        in_fea: int,
        out_fea: int,
        kernel_size: KernelSize = 3,
        stride: KernelSize = 1,
        padding: KernelSize = 1,
        norm: str = "bn",
        relu_slop: float = 0.2,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels=in_fea,
                out_channels=out_fea,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        ]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            # Upstream uses a fixed probability whenever dropout is enabled.
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the upstream block without changing its operation order."""
        return self.layers(x)


class ConvBlock_Tanh(nn.Module):
    """Upstream convolution and optional normalization with a final Tanh."""

    def __init__(
        self,
        in_fea: int,
        out_fea: int,
        kernel_size: KernelSize = 3,
        stride: KernelSize = 1,
        padding: KernelSize = 1,
        norm: str = "bn",
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels=in_fea,
                out_channels=out_fea,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        ]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce the bounded output of the upstream block."""
        return self.layers(x)


class DeconvBlock(nn.Module):
    """Upstream transposed convolution, normalization, and LeakyReLU."""

    def __init__(
        self,
        in_fea: int,
        out_fea: int,
        kernel_size: KernelSize = 2,
        stride: KernelSize = 2,
        padding: KernelSize = 0,
        output_padding: KernelSize = 0,
        norm: str = "bn",
    ) -> None:
        super().__init__()
        layers = [
            nn.ConvTranspose2d(
                in_channels=in_fea,
                out_channels=out_fea,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            )
        ]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample using the original OpenFWI block."""
        return self.layers(x)


@register_model("inversionnet")
class InversionNet(BaseModel):
    """OpenFWI InversionNet for fixed-geometry five-channel seismic gathers.

    ``in_shape=(1000, 70)``, ``in_channels=5``, ``out_size=1``, and
    ``out_shape=(1, 70, 70)`` are required. Omitting ``out_shape`` selects this
    fixed output shape. Channel widths mirror upstream defaults; only
    ``sample_spatial=1.0`` is supported. Extra WaveDL keywords (including
    ``pretrained``) are ignored; load official weights explicitly with
    :func:`load_openfwi_checkpoint`.
    """

    supports_spatial_output = True
    requires_spatial_output = True
    supported_input_channels = (5,)
    fixed_in_shape = (1000, 70)
    fixed_out_shape = (1, 70, 70)

    def __init__(
        self,
        in_shape: tuple[int, ...],
        out_size: int,
        out_shape: tuple[int, ...] | None = None,
        in_channels: int = 5,
        dim1: int = 32,
        dim2: int = 64,
        dim3: int = 128,
        dim4: int = 256,
        dim5: int = 512,
        sample_spatial: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(in_shape, out_size)
        if self.in_shape != self.fixed_in_shape:
            raise ValueError(
                f"InversionNet requires in_shape={self.fixed_in_shape}, "
                f"got {self.in_shape}."
            )
        if (
            not isinstance(in_channels, Integral)
            or isinstance(in_channels, bool)
            or in_channels != 5
        ):
            raise ValueError(f"InversionNet requires in_channels=5, got {in_channels}.")
        if out_size != 1:
            raise ValueError(f"InversionNet requires out_size=1, got {out_size}.")
        if out_shape is None:
            out_shape = self.fixed_out_shape
        try:
            out_shape = tuple(out_shape)
        except TypeError as exc:
            raise ValueError(
                f"InversionNet requires out_shape={self.fixed_out_shape}."
            ) from exc
        if out_shape != self.fixed_out_shape or any(
            not isinstance(size, Integral) or isinstance(size, bool)
            for size in out_shape
        ):
            raise ValueError(
                f"InversionNet requires out_shape={self.fixed_out_shape}, "
                f"got {out_shape}."
            )
        if isinstance(sample_spatial, bool) or sample_spatial != 1.0:
            raise ValueError(
                "InversionNet supports only sample_spatial=1.0 and in_shape=(1000, 70)."
            )
        self.in_channels = int(in_channels)
        self.out_shape = self.fixed_out_shape
        self.output_spatial_shape = self.out_shape[1:]

        # Keep upstream initialization and attribute names for official weights.
        self.convblock1 = ConvBlock(
            5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0)
        )
        self.convblock2_1 = ConvBlock(
            dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)
        )
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(
            dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)
        )
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(
            dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)
        )
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.convblock8 = ConvBlock(
            dim4, dim5, kernel_size=(8, ceil(70 * sample_spatial / 8)), padding=0
        )

        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, 5, 1000, 70)`` gathers to ``(B, 1, 70, 70)`` fields."""
        x = self.convblock1(x)
        x = self.convblock2_1(x)
        x = self.convblock2_2(x)
        x = self.convblock3_1(x)
        x = self.convblock3_2(x)
        x = self.convblock4_1(x)
        x = self.convblock4_2(x)
        x = self.convblock5_1(x)
        x = self.convblock5_2(x)
        x = self.convblock6_1(x)
        x = self.convblock6_2(x)
        x = self.convblock7_1(x)
        x = self.convblock7_2(x)
        x = self.convblock8(x)
        x = self.deconv1_1(x)
        x = self.deconv1_2(x)
        x = self.deconv2_1(x)
        x = self.deconv2_2(x)
        x = self.deconv3_1(x)
        x = self.deconv3_2(x)
        x = self.deconv4_1(x)
        x = self.deconv4_2(x)
        x = self.deconv5_1(x)
        x = self.deconv5_2(x)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0)
        x = self.deconv6(x)
        return x

    def get_optimizer_groups(
        self, base_lr: float, weight_decay: float = 1e-4
    ) -> list[dict[str, Any]]:
        """Apply weight decay to every parameter, as upstream ``AdamW(model.parameters())``.

        WaveDL's default grouping exempts biases and normalization parameters;
        OpenFWI does not, and reproduction is sensitive to such details.
        """
        return [
            {
                "params": [p for p in self.parameters() if p.requires_grad],
                "lr": base_lr,
                "weight_decay": weight_decay,
            }
        ]

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """Return the original OpenFWI generator's channel widths and sampling."""
        return {
            "dim1": 32,
            "dim2": 64,
            "dim3": 128,
            "dim4": 256,
            "dim5": 512,
            "sample_spatial": 1.0,
        }


class Discriminator(nn.Module):
    """Original OpenFWI VelocityGAN critic, including its final BN/LeakyReLU."""

    def __init__(
        self,
        dim1: int = 32,
        dim2: int = 64,
        dim3: int = 128,
        dim4: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.convblock1_1 = ConvBlock(1, dim1, stride=2)
        self.convblock1_2 = ConvBlock(dim1, dim1)
        self.convblock2_1 = ConvBlock(dim1, dim2, stride=2)
        self.convblock2_2 = ConvBlock(dim2, dim2)
        self.convblock3_1 = ConvBlock(dim2, dim3, stride=2)
        self.convblock3_2 = ConvBlock(dim3, dim3)
        self.convblock4_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock4_2 = ConvBlock(dim4, dim4)
        self.convblock5 = ConvBlock(dim4, 1, kernel_size=5, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score ``(B, 1, 70, 70)`` velocity fields, returning ``(B, 1)``."""
        x = self.convblock1_1(x)
        x = self.convblock1_2(x)
        x = self.convblock2_1(x)
        x = self.convblock2_2(x)
        x = self.convblock3_1(x)
        x = self.convblock3_2(x)
        x = self.convblock4_1(x)
        x = self.convblock4_2(x)
        x = self.convblock5(x)
        x = x.view(x.shape[0], -1)
        return x


@register_model("velocitygan")
class VelocityGAN(InversionNet):
    """InversionNet generator with a factory for a separate VelocityGAN critic."""

    def build_discriminator(self) -> Discriminator:
        """Return a fresh upstream critic, never attached to the generator."""
        return Discriminator()


def load_openfwi_checkpoint(
    model: nn.Module, path: str | PathLike[str]
) -> dict[str, Any]:
    """Strictly load official OpenFWI weights and return epoch/args metadata.

    Accepts either a bare state dict or an upstream checkpoint with a ``model``
    entry. Only ``argparse.Namespace`` is additionally allow-listed, scoped to
    this CPU, weights-only load; arbitrary pickle globals are not enabled.
    Legacy building-block names are converted exactly as in upstream OpenFWI.
    Metadata contains ``epoch`` and ``args`` when present, with ``args``
    converted to a plain dict. A bare state dict returns an empty dict.
    """
    with torch.serialization.safe_globals([Namespace]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("OpenFWI checkpoint must be a state dict or a mapping.")
    metadata = {}
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
        metadata = {
            key: checkpoint[key] for key in ("epoch", "args") if key in checkpoint
        }
        if "args" in metadata:
            args = metadata["args"]
            if isinstance(args, Namespace):
                metadata["args"] = vars(args).copy()
            elif isinstance(args, Mapping):
                metadata["args"] = dict(args)
            else:
                raise ValueError("OpenFWI checkpoint args must be a Namespace or dict.")
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, Mapping):
        raise ValueError("OpenFWI checkpoint model entry must be a state dict.")
    model.load_state_dict(replace_legacy(state_dict), strict=True)
    return metadata
