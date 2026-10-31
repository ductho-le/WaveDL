"""Immutable checkpoint publication, independent of any daemon installation."""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from wavedl.core.layout import OperationKind
from wavedl.core.manifest import (
    Manifest,
    RuntimeSelection,
    StorageMapping,
    describes_layout,
)
from wavedl.core.persistence import sync_directory, write_once
from wavedl.core.safe_files import SafeDirectory, open_directory


if TYPE_CHECKING:
    from pathlib import Path

    from wavedl.core.layout import RunLayout


@dataclass(frozen=True)
class FileRecord:
    size: int
    sha256: str


@dataclass(frozen=True)
class PublishedGeneration:
    path: Path
    generation: str
    attempt: int
    step: int
    manifest_sha256: str
    files: dict[str, FileRecord]


def _copy_file(source: Path, target: Path) -> FileRecord:
    import os

    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        while chunk := incoming.read(1024 * 1024):
            outgoing.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    return FileRecord(size, digest.hexdigest())


def _allocate_generation(layout: RunLayout) -> Path:
    numbers = [
        int(path.name[4:])
        for path in layout.checkpoints.iterdir()
        if re.fullmatch(r"gen-\d{6,}", path.name)
    ]
    number = max(numbers, default=0) + 1
    while True:
        target = layout.checkpoints / f"gen-{number:06d}"
        try:
            target.mkdir()
        except FileExistsError:
            number += 1
            continue
        sync_directory(layout.checkpoints)
        return target


def _bad_constant(value: str):
    raise ValueError(f"Invalid JSON constant: {value}")


def _manifest_digest(layout: RunLayout, root: SafeDirectory | None = None) -> str:
    with ExitStack() as anchors:
        if root is None:
            root = anchors.enter_context(open_directory(layout.run_dir))
        raw = root.read_file(layout.manifest.name, limit=1024 * 1024)
    data = json.loads(raw, parse_constant=_bad_constant)
    manifest = Manifest(
        **(
            data
            | {
                "operation": OperationKind(data["operation"]),
                "runtime": RuntimeSelection(**data["runtime"]),
                "storage_map": tuple(
                    StorageMapping(**item) for item in data["storage_map"]
                ),
                "path_fields": tuple(data["path_fields"]),
            }
        )
    )
    if not describes_layout(manifest, layout):
        raise ValueError("Manifest/layout identity mismatch")
    return hashlib.sha256(raw).hexdigest()


def publish_generation(
    layout: RunLayout, source: Path, *, attempt: int, step: int
) -> PublishedGeneration:
    """Copy a quiesced staging tree, sync contents, then publish COMPLETE.json."""
    if (
        type(attempt) is not int
        or attempt < 1
        or not layout.attempt_dir(attempt).is_dir()
    ):
        raise ValueError("A reserved attempt is required")
    if type(step) is not int or step < 0:
        raise ValueError("step must be nonnegative")
    manifest_sha256 = _manifest_digest(layout)
    source = source.resolve()
    if not source.is_dir() or layout.checkpoints.is_relative_to(source):
        raise ValueError("Invalid staging directory")
    members = sorted(source.rglob("*"))
    if not members or not any(item.is_file() for item in members):
        raise ValueError("Empty generation")
    for item in members:
        if item.is_symlink() or not (item.is_file() or item.is_dir()):
            raise ValueError("Generation contents must be ordinary files/directories")
        if item.relative_to(source).as_posix() == "COMPLETE.json":
            raise ValueError("COMPLETE.json is reserved for publication")
    target = _allocate_generation(layout)
    directories = {target}
    files = {}
    for item in members:
        relative = item.relative_to(source)
        destination = target / relative
        if item.is_dir():
            destination.mkdir()
            directories.add(destination)
        else:
            files[relative.as_posix()] = _copy_file(item, destination)
    # Persist child entries before their parents and before the publication marker.
    for directory in sorted(
        directories, key=lambda item: len(item.parts), reverse=True
    ):
        sync_directory(directory)
    result = PublishedGeneration(
        target, target.name, attempt, step, manifest_sha256, files
    )
    completion = {
        "schema_version": 1,
        "run_id": layout.run_id,
        "generation": target.name,
        "attempt": attempt,
        "step": step,
        "manifest_sha256": manifest_sha256,
        "files": {name: asdict(record) for name, record in files.items()},
    }
    content = (
        json.dumps(completion, allow_nan=False, sort_keys=True).encode("utf-8") + b"\n"
    )
    if len(content) > 1024 * 1024:
        raise ValueError("Completion manifest exceeds discovery byte bound")
    write_once(target / "COMPLETE.json", content)
    return result


def _read_generation(
    layout: RunLayout,
    path: Path,
    manifest_sha256: str,
    checkpoints: SafeDirectory,
) -> PublishedGeneration | None:
    try:
        with checkpoints.open_directory(path.name) as generation:
            return _read_open_generation(layout, path, manifest_sha256, generation)
    except (
        KeyError,
        TypeError,
        ValueError,
        UnicodeError,
        FileNotFoundError,
        RecursionError,
    ):
        return None


def _read_open_generation(
    layout: RunLayout,
    path: Path,
    manifest_sha256: str,
    generation: SafeDirectory,
    *,
    check=None,
) -> PublishedGeneration | None:
    # The marker and every inventoried member are opened under owned anchors.
    # There is no pathname check/open gap and no allocation proportional to weights.
    check = check or (lambda: None)
    check()
    raw = generation.read_file("COMPLETE.json", limit=1024 * 1024)
    data = json.loads(raw, parse_constant=_bad_constant)
    if (
        type(data["schema_version"]) is not int
        or data["schema_version"] != 1
        or data["run_id"] != layout.run_id
        or data["generation"] != path.name
        or data["manifest_sha256"] != manifest_sha256
        or type(data["attempt"]) is not int
        or data["attempt"] < 1
        or type(data["step"]) is not int
        or data["step"] < 0
        or not isinstance(data["files"], dict)
        or not data["files"]
    ):
        return None
    files = {}
    for name, record in data["files"].items():
        check()
        # Strict portable spelling also excludes Windows drives/streams on POSIX.
        if not isinstance(name, str) or any(c in name for c in "\\:\0"):
            return None
        parts = name.split("/")
        if (
            any(part in ("", ".", "..") for part in parts)
            or name == "COMPLETE.json"
            or type(record["size"]) is not int
            or record["size"] < 0
            or not isinstance(record["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
        ):
            return None
        expected = FileRecord(record["size"], record["sha256"])
        with ExitStack() as anchors:
            parent = generation
            for part in parts[:-1]:
                check()
                parent = anchors.enter_context(parent.open_directory(part))
            if (
                FileRecord(
                    *parent.digest_file(parts[-1], check=check, limit=expected.size)
                )
                != expected
            ):
                return None
        files[name] = expected
    return PublishedGeneration(
        path, path.name, data["attempt"], data["step"], manifest_sha256, files
    )


def published_generations(layout: RunLayout) -> list[PublishedGeneration]:
    with (
        open_directory(layout.run_dir) as root,
        root.open_directory(layout.checkpoints.name) as checkpoints,
    ):
        manifest_sha256 = _manifest_digest(layout, root)
        paths = sorted(
            (
                layout.checkpoints / entry.name
                for entry in checkpoints.entries()
                if entry.is_directory and re.fullmatch(r"gen-\d{6,}", entry.name)
            ),
            key=lambda path: int(path.name[4:]),
        )
        return [
            generation
            for path in paths
            if (
                generation := _read_generation(
                    layout, path, manifest_sha256, checkpoints
                )
            )
            is not None
        ]
