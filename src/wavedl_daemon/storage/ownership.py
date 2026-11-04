"""
Single-writer dispatch ownership, released by the kernel when the process dies.

Only one daemon may mutate run state at a time. A recorded pid does not prove
liveness (pids are reused, and SIGKILL runs no cleanup), so ownership is an
advisory file lock, which the kernel drops however the holder dies. The pid is
recorded for operators and never trusted.

Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""

from __future__ import annotations

import errno
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator


if sys.platform == "win32":  # pragma: no cover - exercised on Windows CI
    import msvcrt

    def _try_lock(handle) -> bool:
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                return False
            raise
        return True

    def _unlock(handle) -> None:
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:
    import fcntl

    def _try_lock(handle) -> bool:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                return False
            raise
        return True

    def _unlock(handle) -> None:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass


class DispatchHeld(RuntimeError):
    """Another process already owns dispatch."""


class DispatchRequired(RuntimeError):
    """A state mutation was attempted without owning dispatch."""


class DispatchLock:
    """Ownership of the right to mutate run state.

    Not reentrant across open file descriptions: a second ``DispatchLock`` on
    the same path fails even inside the same process, because that is what a
    second daemon would look like.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._handle = None
        self._pid = None

    @property
    def held(self) -> bool:
        return self._handle is not None and self._pid == os.getpid()

    def acquire(self) -> DispatchLock:
        if self._handle is not None:
            raise DispatchHeld(f"dispatch already acquired by this object: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+")
        try:
            acquired = _try_lock(handle)
        except BaseException:
            handle.close()
            raise
        if not acquired:
            handle.close()
            raise DispatchHeld(f"another process owns dispatch: {self.path}")
        self._handle = handle
        self._pid = os.getpid()
        # Recorded for operators only. Never read back as a liveness signal.
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(f"{os.getpid()}\n")
            handle.flush()
        except OSError:
            pass
        return self

    def release(self) -> None:
        if self._handle is None:
            return
        owned_here = self.held
        handle, self._handle = self._handle, None
        self._pid = None
        if owned_here:
            _unlock(handle)
        handle.close()

    def __enter__(self) -> DispatchLock:
        return self.acquire()

    def __exit__(self, *exc_info) -> None:
        self.release()


@contextmanager
def acquire_dispatch_lock(path: Path) -> Iterator[DispatchLock]:
    """Own dispatch for the duration of the block."""
    lock = DispatchLock(path)
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()


def require_dispatch(lock: DispatchLock | None) -> None:
    """Gate every externally visible state mutation."""
    if lock is None or not lock.held:
        raise DispatchRequired("this operation requires dispatch ownership")
