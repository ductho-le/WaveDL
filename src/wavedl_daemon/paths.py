# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Plain checks for names a request sends and for files only the user may read."""

from __future__ import annotations

import os
import stat
from typing import TYPE_CHECKING, BinaryIO


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class UnsafePath(ValueError):
    """A name is not a plain segment, or it resolves outside its root."""


def is_segment(name: str) -> bool:
    return name not in {"", ".", ".."} and not any(
        char in "/\\:" or ord(char) < 32 or ord(char) == 127 for char in name
    )


def inside(root: Path, parts: Sequence[str]) -> Path:
    """``root`` joined with ``parts`` and resolved, refusing any escape from it."""
    if not all(map(is_segment, parts)):
        raise UnsafePath("Invalid path segment")
    base = root.resolve()
    target = base.joinpath(*parts).resolve()
    if not target.is_relative_to(base):
        raise UnsafePath("Path escapes its root")
    return target


def open_regular(path: Path) -> BinaryIO:
    # Nonblocking, so a FIFO cannot stall the open before the type check.
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError("Not a regular file")
    return os.fdopen(descriptor, "rb")


def read_regular(path: Path, limit: int) -> bytes:
    with open_regular(path) as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("File exceeds its byte bound")
    return data


def is_private(info: os.stat_result) -> bool:
    """No permission bits for group or others."""
    # Windows mode bits are synthesized and say nothing about its ACLs.
    return os.name != "posix" or not info.st_mode & 0o077


def file_identity(info: os.stat_result) -> dict[str, int]:
    """What changes when a file is replaced or rewritten."""
    return {
        name: int(getattr(info, name))
        for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    }
