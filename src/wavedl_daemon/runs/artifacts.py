# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""What a run's ``output/`` holds, and one file of it opened for download.

``output/`` is upstream's ``--output_dir``. While a run is live the trainer
rewrites files there in place, so a listing is a snapshot and a download
streams the size it saw at open, aborting if the file shrinks meanwhile.
"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, BinaryIO

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
    # The browser refuses the same characters, so every listed name downloads.
    return value not in {"", ".", ".."} and not any(
        char in "/\\:%" or ord(char) < 32 or ord(char) == 127 for char in value
    )


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

    def __init__(self, stream: BinaryIO, size: int, filename: str):
        self.stream, self.size, self.filename = stream, size, filename
        self._remaining = size

    def read(self) -> bytes:
        if not self._remaining:
            return b""
        chunk = self.stream.read(min(CHUNK_BYTES, self._remaining))
        if not chunk:
            raise ValueError("Artifact was truncated during download")
        self._remaining -= len(chunk)
        return chunk

    def close(self) -> None:
        self.stream.close()


def open_output(output: Path, name: str) -> OpenedArtifact:
    parts = artifact_path(name)
    root = output.resolve()
    target = root.joinpath(*parts).resolve()
    if not target.is_relative_to(root) or target == root:
        raise UnsafeArtifactPath("Invalid artifact path")
    try:
        # Nonblocking, so a FIFO cannot stall the worker before the type check.
        descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    except (FileNotFoundError, NotADirectoryError):
        raise ArtifactNotFound("No such artifact") from None
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise ArtifactNotFound("No such artifact")
    return OpenedArtifact(os.fdopen(descriptor, "rb"), info.st_size, parts[-1])
