# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Host capability information, retaining unknown probe states."""

from __future__ import annotations

from wavedl_daemon.http.schemas.common import WireModel


class CudaDevice(WireModel):
    name: str
    memory_bytes: int


class Accelerators(WireModel):
    torch: str | None
    cuda: list[CudaDevice]
    mps: bool


class HostProfile(WireModel):
    hostname: str
    os: str
    os_release: str
    arch: str
    cpu_count: int | None
    memory_bytes: int | None
    python: str
    python_version: str
    accelerators: Accelerators | None
    backends: list[str]
