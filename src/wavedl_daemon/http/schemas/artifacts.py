# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""A run's ``output/`` listing; download bytes have no response model."""

from __future__ import annotations

from wavedl_daemon.http.schemas.common import WireModel


class ArtifactFile(WireModel):
    path: str
    size: int
    modified: str


class ArtifactListing(WireModel):
    files: list[ArtifactFile]
    checkpoints: list[str]
