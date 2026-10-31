# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Published inventory authority and owned, no-follow artifact streams."""

from __future__ import annotations

import hashlib
import re
from contextlib import ExitStack
from typing import TYPE_CHECKING
from uuid import UUID

from wavedl.core.generations import (
    FileRecord,
    _manifest_digest,
    _read_open_generation,
    published_generations,
)
from wavedl.core.safe_files import open_directory


if TYPE_CHECKING:
    from collections.abc import Callable

    from wavedl.core.generations import PublishedGeneration
    from wavedl.core.layout import RunLayout


class UnsafeArtifactPath(ValueError):
    """The public path is not a portable relative artifact identifier."""


class ArtifactNotFound(FileNotFoundError):
    """A requested member has no verified publication authority."""


def artifact_path(value: str) -> tuple[str, ...]:
    """Accept decoded portable relative names, excluding percent and controls.

    Ordinary spaces and Unicode are permitted; literal percent names are
    excluded so an encoded alias can never trigger a second decoding pass.
    """
    if (
        not isinstance(value, str)
        or any(char in value for char in "\\:%")
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise UnsafeArtifactPath("Invalid artifact path")
    parts = tuple(value.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise UnsafeArtifactPath("Invalid artifact path")
    return parts


class OpenedArtifact:
    """A single validated file handle plus its retained directory anchors."""

    def __init__(self, stream, anchors: ExitStack, record: FileRecord, filename: str):
        self.stream, self.anchors = stream, anchors
        self.size, self.sha256, self.filename = record.size, record.sha256, filename
        self._remaining = record.size

    def read(self, size: int = 65536) -> bytes:
        if type(size) is not int or not 1 <= size <= 1024 * 1024:
            raise ValueError("Invalid chunk size")
        if not self._remaining:
            return b""
        chunk = self.stream.read(min(size, self._remaining))
        if not chunk:
            raise ValueError("Artifact was truncated during download")
        self._remaining -= len(chunk)
        return chunk

    def close(self) -> None:
        self.anchors.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ArtifactStore:
    """Resolve registered layouts; journal artifact paths are never authority."""

    def __init__(self, run_lookup: Callable[[str], RunLayout]):
        self.run_lookup = run_lookup

    def _layout(self, run_id: str) -> RunLayout:
        try:
            canonical = str(UUID(run_id))
        except (ValueError, TypeError, AttributeError):
            raise UnsafeArtifactPath("Invalid run identity") from None
        if canonical != run_id:
            raise UnsafeArtifactPath("Invalid run identity")
        layout = self.run_lookup(run_id)
        if layout.run_id != run_id:
            raise ValueError("Run layout identity mismatch")
        return layout

    def list(self, run_id: str) -> list[PublishedGeneration]:
        return self.list_layout(self._layout(run_id))

    def list_layout(self, layout: RunLayout) -> list[PublishedGeneration]:
        """Filesystem-only work against an immutable owner-resolved layout."""
        UUID(layout.run_id)
        return published_generations(layout)

    def open(self, run_id: str, generation: str, relative_path: str) -> OpenedArtifact:
        return self.open_layout(self._layout(run_id), generation, relative_path)

    def open_layout(
        self, layout: RunLayout, generation: str, relative_path: str
    ) -> OpenedArtifact:
        """Revalidate COMPLETE and open through anchors, never reopen by pathname."""
        UUID(layout.run_id)
        parts = artifact_path(relative_path)
        if not isinstance(generation, str) or not re.fullmatch(
            r"gen-\d{6,}", generation
        ):
            raise UnsafeArtifactPath("Invalid generation")
        anchors = ExitStack()
        try:
            root = anchors.enter_context(open_directory(layout.run_dir))
            digest = _manifest_digest(layout, root)
            checkpoints = anchors.enter_context(
                root.open_directory(layout.checkpoints.name)
            )
            try:
                directory = anchors.enter_context(
                    checkpoints.open_directory(generation)
                )
            except FileNotFoundError:
                raise ArtifactNotFound("Generation is not published") from None
            try:
                publication = _read_open_generation(
                    layout, layout.checkpoints / generation, digest, directory
                )
            except FileNotFoundError:
                raise ArtifactNotFound("Generation is not published") from None
            if publication is None or relative_path not in publication.files:
                raise ArtifactNotFound("Artifact is not published")
            parent = directory
            for part in parts[:-1]:
                parent = anchors.enter_context(parent.open_directory(part))
            stream = anchors.enter_context(parent.open_file(parts[-1]))
            # Inventory verification above is insufficient: a member may have
            # changed since that probe. Hash the actual download handle.
            expected = publication.files[relative_path]
            size, content_digest = 0, hashlib.sha256()
            while chunk := stream.read(65536):
                size += len(chunk)
                content_digest.update(chunk)
                if size > expected.size:
                    raise ValueError("Artifact changed after publication")
            if FileRecord(size, content_digest.hexdigest()) != expected:
                raise ValueError("Artifact changed after publication")
            stream.seek(0)
            return OpenedArtifact(stream, anchors.pop_all(), expected, parts[-1])
        finally:
            anchors.close()
