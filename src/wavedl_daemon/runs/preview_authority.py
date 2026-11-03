# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Safe, capability-scoped dataset preview selection.

Only administrator-configured roots may be selected. Paths are slash-separated
names that must stay inside their root. The selected file stays open until the
caller's context exits, and its fingerprint and ancestry are checked again then.
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from wavedl_daemon import paths


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


class PreviewForbidden(ValueError):
    """Selection is outside configured preview authority."""


class PreviewMissing(ValueError):
    """Configured root or selected member no longer exists."""


class PreviewChanged(ValueError):
    """The root, an ancestor directory, or the file fingerprint changed."""


_EXTENSIONS = {".npz", ".mat", ".h5", ".hdf5"}


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
    ancestry: tuple[dict[str, object], ...]  # path, st_dev, st_ino of each directory

    @property
    def expected(self) -> dict:
        return {
            "path": str(self.path),
            "identity": dict(self.identity),
            "root_id": self.root_id,
            "relative_path": self.relative_path,
            "root_path": str(self.root_path),
            "ancestry": tuple(dict(item) for item in self.ancestry),
        }


def _ancestry(root: Path, parts: list[str]) -> list[dict[str, object]]:
    result = []
    for depth in range(len(parts)):
        directory = root.joinpath(*parts[:depth])
        info = directory.stat()
        result.append(
            {"path": str(directory), "st_dev": info.st_dev, "st_ino": info.st_ino}
        )
    return result


class PreviewAuthority:
    """Select datasets inside administrator-configured roots."""

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
        root = self.roots[root_id]
        in_body = False
        try:
            try:
                target = paths.inside(root, parts)
            except paths.UnsafePath:
                raise PreviewForbidden("Preview path escapes its root") from None
            ancestry = _ancestry(root, parts)
            with paths.open_regular(target) as stream:
                identity = paths.file_identity(os.fstat(stream.fileno()))
                in_body = True
                yield PreviewSelection(
                    path=root.joinpath(*parts),
                    identity=identity,
                    root_id=root_id,
                    relative_path=relative_path,
                    root_path=root,
                    ancestry=tuple(ancestry),
                )
                in_body = False
                if (
                    _ancestry(root, parts) != ancestry
                    or paths.file_identity(os.fstat(stream.fileno())) != identity
                    or paths.file_identity(target.stat()) != identity
                ):
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
    """Convenience context manager over a one-off authority."""
    return PreviewAuthority(roots).open_preview_selection(root_id, relative_path)


def verify_selection(selection: PreviewSelection) -> None:
    """Reopen and verify a worker's expected fingerprint before payload I/O."""
    with PreviewAuthority(
        {selection.root_id: selection.root_path},
        extensions=_EXTENSIONS | {Path(selection.relative_path).suffix.lower()},
    ).open_preview_selection(selection.root_id, selection.relative_path) as current:
        if (current.identity, current.ancestry) != (
            selection.identity,
            selection.ancestry,
        ):
            raise PreviewChanged("Preview fingerprint changed")


FrozenPreviewSelection = PreviewSelection
