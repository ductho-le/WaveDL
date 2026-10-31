# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Published inventory metadata; download bytes have no response model."""

from __future__ import annotations

from wavedl_daemon.http.schemas.common import WireModel


class ArtifactFile(WireModel):
    size: int
    sha256: str


class ArtifactGeneration(WireModel):
    generation: str
    attempt: int
    step: int
    files: dict[str, ArtifactFile]


class ArtifactListing(WireModel):
    generations: list[ArtifactGeneration]
