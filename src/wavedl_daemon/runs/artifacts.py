"""
What a run's ``output/`` holds, and opening one file for download.

``output/`` is upstream's ``--output_dir``. A live trainer rewrites files there
in place, so a listing is a snapshot. A download streams the size seen at open
and aborts if the file changes.

Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, BinaryIO

from wavedl_daemon import paths
from wavedl_daemon.execution.budget import BudgetExhausted


if TYPE_CHECKING:
    from pathlib import Path


# The directories upstream saves an Accelerate checkpoint into.
CHECKPOINT_NAME = re.compile(r"(?:best|interrupted|epoch_[0-9]+)_checkpoint")
MAX_ENTRIES = 16384
CHUNK_BYTES = 65536


class UnsafeArtifactPath(ValueError):
    """The name is not a relative path of plain segments inside ``output/``."""


class ArtifactNotFound(FileNotFoundError):
    """No regular file of that name is in ``output/``."""


def _segment(value: str) -> bool:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False  # undecodable filesystem bytes, which JSON cannot carry
    # The browser also refuses "%", so every listed name downloads.
    return paths.is_segment(value) and "%" not in value


def artifact_path(value: str) -> tuple[str, ...]:
    if (
        not isinstance(value, str)
        or len(value) > 4096
        or not all(map(_segment, value.split("/")))
    ):
        raise UnsafeArtifactPath("Invalid artifact path")
    return tuple(value.split("/"))


@dataclass(frozen=True)
class OutputFile:
    path: str  # POSIX, relative to output/
    size: int
    modified: str  # ISO 8601, UTC


@dataclass(frozen=True)
class Inventory:
    files: tuple[OutputFile, ...]
    checkpoints: tuple[str, ...]


def list_output(output: Path) -> Inventory:
    """Regular files under ``output/``, never following a symlink."""
    files: list[OutputFile] = []
    checkpoints: list[str] = []
    pending = [("", str(output))]
    seen = 0
    while pending:
        prefix, directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    seen += 1
                    if seen > MAX_ENTRIES:
                        raise BudgetExhausted("Output listing exceeds its bound")
                    if not _segment(entry.name):
                        continue
                    name = prefix + entry.name
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if not prefix and CHECKPOINT_NAME.fullmatch(entry.name):
                                checkpoints.append(entry.name)
                            pending.append((name + "/", entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            info = entry.stat(follow_symlinks=False)
                            modified = datetime.fromtimestamp(info.st_mtime, UTC)
                            files.append(
                                OutputFile(name, info.st_size, modified.isoformat())
                            )
                    except FileNotFoundError:
                        continue  # removed by the trainer mid-walk
        except FileNotFoundError:
            continue  # output/ before the first attempt, or a pruned checkpoint
    files.sort(key=lambda item: item.path)
    return Inventory(tuple(files), tuple(sorted(checkpoints)))


class OpenedArtifact:
    """One open file and the byte count the download promised."""

    def __init__(self, stream: BinaryIO, filename: str, source: Path):
        self.stream, self.filename = stream, filename
        self._source = source
        self._identity = paths.file_identity(os.fstat(stream.fileno()))
        self.size = self._remaining = self._identity["st_size"]

    def read(self) -> bytes:
        if not self._remaining:
            return b""
        chunk = self.stream.read(min(CHUNK_BYTES, self._remaining))
        # A held Windows handle can retain an old last-write timestamp.
        # Also stat the path afresh, without blocking a trainer's writes.
        try:
            current = paths.file_identity(self._source.stat())
        except OSError:
            raise ValueError("Artifact changed during download") from None
        if (
            not chunk
            or self._identity != current
            or self._identity != paths.file_identity(os.fstat(self.stream.fileno()))
        ):
            raise ValueError("Artifact changed during download")
        self._remaining -= len(chunk)
        return chunk

    def close(self) -> None:
        self.stream.close()


def open_output(output: Path, name: str) -> OpenedArtifact:
    parts = artifact_path(name)
    try:
        source = paths.inside(output, parts)
        stream = paths.open_regular(source)
    except paths.UnsafePath:
        raise UnsafeArtifactPath("Invalid artifact path") from None
    except (FileNotFoundError, NotADirectoryError, ValueError):
        raise ArtifactNotFound("No such artifact") from None
    return OpenedArtifact(stream, parts[-1], source)
