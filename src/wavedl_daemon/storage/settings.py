# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Daemon roots and the storage guarantees each one has to provide.

The daemon owns two roots with opposite requirements. The *state root* holds the
SQLite registry and the dispatch lock, so it must be local: SQLite's locking is
unreliable over NFS, and so is ``flock``. The *run root* holds run directories
that compute nodes read and write, so it is normally on exactly the kind of
network filesystem the state root forbids.

Nothing here inherits the GUI's ``Path("AppData").absolute()`` behaviour, which
resolves relative to the working directory and therefore puts application data
wherever the process happened to start.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from wavedl_daemon.storage.filesystems import (
    is_ephemeral_filesystem,
    is_network_filesystem,
)


APP_NAME = "WaveDL"


class UnsupportedStorage(RuntimeError):
    """A root cannot provide the guarantee its purpose requires."""


def default_state_dir() -> Path:
    """Per-user, absolute, and independent of the working directory."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if base and not Path(base).is_absolute():
            base = None
        return Path(base or Path.home() / "AppData" / "Local") / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    base = os.environ.get("XDG_STATE_HOME")
    if base and not Path(base).is_absolute():
        base = None
    return Path(base or Path.home() / ".local" / "state") / APP_NAME.lower()


def default_run_dir() -> Path:
    """Runs default beside the state root but are expected to be relocated."""
    return default_state_dir() / "runs"


def _require_writable(path: Path, purpose: str) -> None:
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise UnsupportedStorage(f"{purpose} is not creatable: {path} ({exc})") from exc
    try:
        with tempfile.TemporaryFile(dir=path) as probe:
            probe.write(b"wavedl")
            probe.flush()
    except OSError as exc:
        raise UnsupportedStorage(f"{purpose} is not writable: {path} ({exc})") from exc


@dataclass(frozen=True)
class Settings:
    """Validated daemon roots.

    ``shared_execution`` records whether any configured backend runs work on a
    different machine. It changes what the run root must satisfy, and it is an
    explicit input because the daemon cannot discover it from the filesystem.
    """

    state_root: Path
    run_root: Path
    shared_execution: bool = False

    def __post_init__(self):
        object.__setattr__(self, "state_root", Path(self.state_root).expanduser())
        object.__setattr__(self, "run_root", Path(self.run_root).expanduser())
        for name in ("state_root", "run_root"):
            if not getattr(self, name).is_absolute():
                raise UnsupportedStorage(f"{name} must be absolute")
            object.__setattr__(self, name, getattr(self, name).resolve())

    @classmethod
    def resolved(
        cls,
        *,
        state_root: Path | None = None,
        run_root: Path | None = None,
        shared_execution: bool = False,
    ) -> Settings:
        state = state_root or default_state_dir()
        settings = cls(
            state_root=state,
            run_root=run_root or state / "runs",
            shared_execution=shared_execution,
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.state_root == self.run_root:
            raise UnsupportedStorage("state root and run root must be distinct")

        # The registry and the dispatch lock both depend on POSIX advisory
        # locking, which is unreliable over the network.
        if is_network_filesystem(self.state_root):
            raise UnsupportedStorage(
                f"state root is on a network filesystem, where SQLite and lock "
                f"ownership are unreliable: {self.state_root}"
            )

        if self.shared_execution:
            # A run root nested in the state root cannot be relocated to shared
            # storage without dragging the registry onto the network with it.
            if self.run_root.is_relative_to(self.state_root):
                raise UnsupportedStorage(
                    f"run root is inside the state root, so it cannot be shared "
                    f"with compute nodes: {self.run_root}"
                )
            if is_ephemeral_filesystem(self.run_root):
                raise UnsupportedStorage(
                    f"run root is on an ephemeral filesystem and cannot be "
                    f"visible to another machine: {self.run_root}"
                )
        _require_writable(self.state_root, "state root")
        _require_writable(self.run_root, "run root")
