"""Exclusive run directories and per-execution attempt reservations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING
from uuid import UUID

from wavedl.core.persistence import sync_directory


if TYPE_CHECKING:
    from pathlib import Path

    from wavedl.core.safe_files import SafeDirectory


class OperationKind(StrEnum):
    TRAIN = "train"
    TEST = "test"
    PLOT = "plot"


@dataclass(frozen=True)
class RunLayout:
    run_dir: Path
    run_id: str

    def __post_init__(self):
        UUID(self.run_id)
        object.__setattr__(self, "run_dir", self.run_dir.resolve())

    @classmethod
    def create(cls, root: Path, run_id: str) -> RunLayout:
        UUID(run_id)
        layout = cls(root / run_id, run_id)
        root.mkdir(parents=True, exist_ok=True)
        layout.run_dir.mkdir(mode=0o700)
        layout.attempts.mkdir()
        layout.checkpoints.mkdir()
        (layout.run_dir / "plots").mkdir()
        return layout

    @property
    def manifest(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def attempts(self) -> Path:
        return self.run_dir / "attempts"

    @property
    def checkpoints(self) -> Path:
        return self.run_dir / "checkpoints"

    def attempt_dir(self, attempt: int) -> Path:
        if type(attempt) is not int or attempt < 1:
            raise ValueError("attempt must be a positive integer")
        return self.attempts / str(attempt)

    def journal(self, attempt: int) -> Path:
        return self.attempt_dir(attempt) / "events.jsonl"


def allocate_attempt(layout: RunLayout, *, attempts_anchor=None) -> int:
    """Reserve by mkdir; gaps are reservations and are never recycled."""
    if attempts_anchor is not None:
        return _allocate_anchored_attempt(layout, attempts_anchor)
    layout.attempts.mkdir(exist_ok=True)
    sync_directory(layout.run_dir)
    numbers = [
        int(path.name) for path in layout.attempts.iterdir() if path.name.isdecimal()
    ]
    attempt = max(numbers, default=0) + 1
    while True:
        try:
            layout.attempt_dir(attempt).mkdir(mode=0o700)
        except FileExistsError:
            attempt += 1
            continue
        (layout.attempt_dir(attempt) / "rank").mkdir()
        sync_directory(layout.attempt_dir(attempt))
        sync_directory(layout.attempts)
        return attempt


def _allocate_anchored_attempt(layout, anchor):
    """POSIX startup allocation relative to a verified private directory FD."""
    if os.name != "posix" or anchor.api is not None:
        raise OSError("Anchored scheduler allocation requires POSIX")
    if anchor.path != layout.attempts:
        raise ValueError("Attempt anchor belongs to a different run")
    anchor.guard()
    anchor.verify_private_owner()
    maximum = 0
    for count, entry in enumerate(anchor.entries(), start=1):
        if count > 4096:
            raise ValueError("Attempt reservation inventory exceeds bound")
        if entry.name.isdecimal():
            if len(entry.name) > 19 or int(entry.name) >= 2**63 - 1:
                raise ValueError("Attempt reservation integer exceeds bound")
            maximum = max(maximum, int(entry.name))
    attempt = maximum + 1
    collisions = 0
    while True:
        if attempt > 2**63 - 1:
            raise ValueError("Attempt reservation exhausted integer range")
        anchor.guard()
        try:
            os.mkdir(str(attempt), mode=0o700, dir_fd=anchor.handle)
        except FileExistsError:
            collisions += 1
            if collisions >= 4096:
                raise ValueError("Attempt reservation contention exceeds bound")
            attempt += 1
            continue
        anchor.guard()
        with anchor.open_directory(str(attempt)) as selected:
            selected.verify_private_owner()
            selected.guard()
            os.mkdir("rank", mode=0o700, dir_fd=selected.handle)
            os.fsync(selected.handle)
            selected.guard()
        os.fsync(anchor.handle)
        anchor.guard()
        return attempt


def allocate_attempt_in_directory(root: SafeDirectory) -> int:
    """Reserve a runtime attempt under one retained run-directory capability.

    Numeric names are reservations regardless of their type; gaps and torn
    allocations are never recycled. POSIX writes are directory-FD relative.
    Windows requires a writable leaf capability with held no-delete ancestry.
    Native filesystem acceptance is separate from this allocation contract.
    """
    if not getattr(root, "writable", False):
        raise PermissionError(
            "Runtime allocation requires writable directory capability"
        )
    root.guard()
    root.verify_private_owner()
    try:
        _mkdir_in_directory(root, "attempts")
    except FileExistsError:
        pass
    with root.open_directory("attempts", writable=True) as attempts:
        attempts.verify_private_owner()
        maximum = 0
        for count, entry in enumerate(attempts.entries(), start=1):
            if count > 4096:
                raise ValueError("Attempt reservation inventory exceeds bound")
            if entry.name.isdecimal():
                if len(entry.name) > 19 or int(entry.name) >= 2**63 - 1:
                    raise ValueError("Attempt reservation integer exceeds bound")
                maximum = max(maximum, int(entry.name))
        attempt = maximum + 1
        for _ in range(4096):
            if attempt > 2**63 - 1:
                raise ValueError("Attempt reservation exhausted integer range")
            try:
                _mkdir_in_directory(attempts, str(attempt))
            except FileExistsError:
                attempt += 1
                continue
            with attempts.open_directory(str(attempt), writable=True) as selected:
                selected.verify_private_owner(confidential=True)
                _mkdir_in_directory(selected, "rank")
            attempts.guard()
            root.guard()
            return attempt
    raise ValueError("Attempt reservation contention exceeds bound")


def _mkdir_in_directory(directory, name):
    directory.guard()
    if os.name == "posix":
        os.mkdir(name, mode=0o700, dir_fd=directory.handle)
        os.fsync(directory.handle)
    else:
        # The native safe-files layer must hold ancestry and grant directory
        # durability; no path-only or read-capability fallback is permitted.
        if not getattr(directory, "writable", False):
            raise OSError("Runtime allocation requires writable directory capability")
        (directory.path / name).mkdir(mode=0o700)
        directory.sync()
    directory.guard()
