# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Safe, capability-scoped dataset preview selection.

Only administrator-configured roots may be selected.  Paths are lexical slash
relative names and are opened through :mod:`wavedl.core.safe_files`, retaining
all anchors until the caller's context exits.
"""

from __future__ import annotations

import os
import re
import stat
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from wavedl.core.safe_files import SafeDirectory, SafeFile, open_directory


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


class PreviewForbidden(ValueError):
    """Selection is outside configured preview authority."""


class PreviewMissing(ValueError):
    """Configured root or selected member no longer exists."""


class PreviewChanged(ValueError):
    """A pinned root, ancestry, or file fingerprint changed."""


_EXTENSIONS = {".npz", ".mat", ".h5", ".hdf5"}


def _identity_from_stat(info: os.stat_result) -> dict[str, int]:
    return {
        "st_dev": int(info.st_dev),
        "st_ino": int(info.st_ino),
        "st_size": int(info.st_size),
        "st_mtime_ns": int(info.st_mtime_ns),
    }


def _validate_relative(path: str, extensions: frozenset[str]) -> list[str]:
    if type(path) is not str or not path:
        raise PreviewForbidden("A non-empty relative path is required")
    try:
        encoded = path.encode("utf-8")
    except UnicodeEncodeError:
        raise PreviewForbidden("Preview paths must be valid UTF-8") from None
    if len(encoded) > 4096:
        raise PreviewForbidden("A non-empty relative path is required")
    if "\0" in path or "\\" in path or ":" in path:
        raise PreviewForbidden("Preview paths must use slash-separated members")
    parts = path.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        raise PreviewForbidden("Preview path escapes its configured root")
    if len(parts) > 128:
        raise PreviewForbidden("Preview path is too deep")
    if Path(parts[-1]).suffix.lower() not in extensions:
        raise PreviewForbidden("Unsupported dataset extension")
    return parts


@dataclass(frozen=True)
class PreviewSelection:
    path: Path
    identity: dict[str, int]
    root_id: str
    relative_path: str
    root_path: Path
    root_identity: dict[str, int]
    ancestry: tuple[dict[str, int], ...]

    @property
    def expected(self) -> dict:
        return {
            "path": str(self.path),
            "identity": dict(self.identity),
            "root_id": self.root_id,
            "relative_path": self.relative_path,
            "root_path": str(self.root_path),
            "root_identity": dict(self.root_identity),
            "ancestry": tuple(dict(item) for item in self.ancestry),
        }


def _stat_file(handle: SafeFile) -> dict[str, int]:
    info = os.fstat(handle.handle) if handle.api is None else None
    if info is None:
        native = handle.api.info(handle.handle)
        return {
            "st_dev": int(native.volume),
            "st_ino": int(native.index_high << 32 | native.index_low),
            "st_size": int(native.size_high << 32 | native.size_low),
            "st_mtime_ns": (
                (int(native.write.dwHighDateTime) << 32)
                | int(native.write.dwLowDateTime)
            )
            * 100
            - 11644473600000000000,
        }
    if not stat.S_ISREG(info.st_mode):
        raise PreviewForbidden("Selected dataset is not a regular file")
    return _identity_from_stat(info)


def _stat_directory(directory: SafeDirectory) -> dict[str, int]:
    if directory.api is None:
        info = os.fstat(directory.handle)
        return _identity_from_stat(info)
    native = directory.api.info(directory.handle)
    return {
        "st_dev": int(native.volume),
        "st_ino": int(native.index_high << 32 | native.index_low),
        "st_size": 0,
        "st_mtime_ns": 0,
    }


class PreviewAuthority:
    """Resolve configured dataset roots and hold native capabilities."""

    def __init__(self, roots: Mapping[str, Path], *, extensions=None):
        if type(roots) is not dict and not hasattr(roots, "items"):
            raise ValueError("Preview roots must be a mapping")
        if len(roots) > 32:
            raise ValueError("Too many preview roots")
        self.roots: dict[str, Path] = {}
        self.extensions = frozenset(extensions or _EXTENSIONS)
        if not self.extensions or any(
            type(extension) is not str or not extension.startswith(".")
            for extension in self.extensions
        ):
            raise ValueError("Invalid preview extensions")
        for root_id, root in roots.items():
            if (
                type(root_id) is not str
                or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", root_id) is None
            ):
                raise ValueError("Invalid preview root identifier")
            path = Path(root)
            if not path.is_absolute():
                raise ValueError("Preview roots must be absolute")
            self.roots[root_id] = path

    @contextmanager
    def open_preview_selection(
        self, root_id: str, relative_path: str
    ) -> Iterator[PreviewSelection]:
        if type(root_id) is not str or root_id not in self.roots:
            raise PreviewForbidden("Unknown preview root")
        parts = _validate_relative(relative_path, self.extensions)
        in_body = False
        try:
            with ExitStack() as stack:
                root = stack.enter_context(open_directory(self.roots[root_id]))
                root.guard()
                root_identity = _stat_directory(root)
                current = root
                ancestry = [{"path": str(self.roots[root_id]), **root_identity}]
                for part in parts[:-1]:
                    current = stack.enter_context(current.open_directory(part))
                    current.guard()
                    ancestry.append(
                        {"path": str(current.path), **_stat_directory(current)}
                    )
                selected = stack.enter_context(current.open_file(parts[-1]))
                selected.guard()
                identity = _stat_file(selected)
                in_body = True
                yield PreviewSelection(
                    path=self.roots[root_id].joinpath(*parts),
                    identity=identity,
                    root_id=root_id,
                    relative_path=relative_path,
                    root_path=self.roots[root_id],
                    root_identity=root_identity,
                    ancestry=tuple(ancestry),
                )
                in_body = False
                for anchor in (root, current):
                    anchor.guard()
                selected.guard()
                if _stat_file(selected) != identity:
                    raise PreviewChanged("Selected dataset changed during preview")
        except (FileNotFoundError, NotADirectoryError):
            if in_body:
                raise
            raise PreviewMissing("Preview location does not exist") from None
        except PermissionError:
            if in_body:
                raise
            raise PreviewForbidden("Preview location is unreadable") from None
        except (OSError, ValueError) as error:
            if in_body:
                raise
            if isinstance(error, (PreviewForbidden, PreviewMissing, PreviewChanged)):
                raise
            raise PreviewChanged("Preview ancestry or file changed") from error

    def resolve_selection(self, root_id: str, relative_path: str) -> PreviewSelection:
        with self.open_preview_selection(root_id, relative_path) as selection:
            return selection


def open_preview_selection(
    roots: Mapping[str, Path], root_id: str, relative_path: str
) -> Iterator[PreviewSelection]:
    """Convenience context manager retaining all authority anchors."""
    return PreviewAuthority(roots).open_preview_selection(root_id, relative_path)


def verify_selection(selection: PreviewSelection) -> None:
    """Reopen and verify a worker's expected fingerprint before payload I/O."""
    with PreviewAuthority(
        {selection.root_id: selection.root_path},
        extensions=_EXTENSIONS | {Path(selection.relative_path).suffix.lower()},
    ).open_preview_selection(selection.root_id, selection.relative_path) as current:
        if current.identity != selection.identity or [
            (item["st_dev"], item["st_ino"]) for item in current.ancestry
        ] != [(item["st_dev"], item["st_ino"]) for item in selection.ancestry]:
            raise PreviewChanged("Preview fingerprint changed")


FrozenPreviewSelection = PreviewSelection
