"""
A run's ``output/`` listing. Downloads are raw bytes with no response model.
Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""

from __future__ import annotations

from wavedl_daemon.http.schemas.common import WireModel


class ArtifactFile(WireModel):
    path: str
    size: int
    modified: str


class ArtifactListing(WireModel):
    files: list[ArtifactFile]
    checkpoints: list[str]
