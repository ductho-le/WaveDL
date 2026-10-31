"""One exclusive writer per attempt; failed writers must roll over."""

from __future__ import annotations

import errno
import os
import stat
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from wavedl.core.layout import RunLayout, allocate_attempt
from wavedl.core.persistence import sync_directory
from wavedl.core.protocol import Envelope, encode


if TYPE_CHECKING:
    from pathlib import Path


class RecordTooLarge(ValueError):
    """A record was rejected before writing any bytes."""


class AbandonedTail(FileExistsError):
    """An existing attempt ends before its newline commit boundary."""

    def __init__(self, path: Path):
        super().__init__(f"Abandoned tail at {path}; start a new attempt")
        self.path = path


def _has_abandoned_tail(path: Path) -> bool:
    """Inspect one tail byte without blocking on replaced special files."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("Existing journal must be a regular non-symlink file")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise ValueError("Existing journal changed during tail inspection")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            if not stream.seek(0, os.SEEK_END):
                return False
            stream.seek(-1, os.SEEK_END)
            return stream.read(1) != b"\n"
    finally:
        os.close(descriptor)


class JournalWriter:
    def __init__(
        self,
        layout: RunLayout,
        *,
        attempt: int | None = None,
        max_record_bytes: int = 64 * 1024,
        fsync_every: int = 1,
    ):
        if max_record_bytes < 1 or fsync_every < 1:
            raise ValueError("Positive record bound and fsync cadence required")
        self.layout = layout
        self.attempt = allocate_attempt(layout) if attempt is None else attempt
        self.path = layout.journal(self.attempt)
        self.max_record_bytes = max_record_bytes
        self.fsync_every = fsync_every
        self.abandoned_tail: AbandonedTail | None = None
        self.seq = 0
        self._unsynced = 0
        self._failed = False
        self._entry_synced = False
        try:
            self._stream = self.path.open("xb", buffering=0)
        except FileExistsError:
            if _has_abandoned_tail(self.path):
                raise AbandonedTail(self.path) from None
            raise

    @classmethod
    def restart(
        cls,
        layout: RunLayout,
        previous_attempt: int,
        *,
        max_record_bytes: int = 64 * 1024,
        fsync_every: int = 1,
    ) -> JournalWriter:
        """Inspect the old tail for diagnostics, always allocate a new execution."""
        old = layout.journal(previous_attempt)
        abandoned = None
        if _has_abandoned_tail(old):
            abandoned = AbandonedTail(old)
        writer = cls(layout, max_record_bytes=max_record_bytes, fsync_every=fsync_every)
        writer.abandoned_tail = abandoned
        return writer

    def append(self, event_type: str, payload: dict[str, object]) -> Envelope:
        if self._failed or self._stream.closed:
            raise RuntimeError("Writer unavailable; start a new attempt")
        event = Envelope(
            run_id=self.layout.run_id,
            attempt=self.attempt,
            type=event_type,
            seq=self.seq,
            ts=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            payload=payload,
        )
        data = encode(event)
        if len(data) > self.max_record_bytes:
            raise RecordTooLarge(f"{len(data)} exceeds {self.max_record_bytes} bytes")
        try:
            remaining = memoryview(data)
            while remaining:
                written = self._stream.write(remaining)
                if not written:
                    raise OSError(errno.EIO, "Journal write made no progress")
                remaining = remaining[written:]
            self.seq += 1
            self._unsynced += 1
            if self._unsynced >= self.fsync_every or event_type == "exit":
                self.flush()
        except OSError:
            self._failed = True
            raise
        # Return the same sanitized representation that was committed to disk.
        from wavedl.core.protocol import decode

        return decode(data)

    def flush(self) -> None:
        try:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            if not self._entry_synced:
                sync_directory(self.path.parent)
                self._entry_synced = True
            self._unsynced = 0
        except OSError:
            self._failed = True
            raise

    def close(self) -> None:
        if self._stream.closed:
            return
        try:
            if self._unsynced and not self._failed:
                self.flush()
        finally:
            self._stream.close()

    def __enter__(self) -> JournalWriter:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
