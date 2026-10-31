"""Bounded binary replay; EOF never implies runtime completion."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from wavedl.core.protocol import Envelope, MalformedJSON, ProtocolError, decode
from wavedl.core.safe_files import open_directory


if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class JournalBookmark:
    """Compact framing boundary; no pending or ready journal payload retained."""

    run_id: str
    attempt: int
    offset: int
    last_seq: int
    file_identity: tuple | None
    observed_size: int
    malformed_count: int
    private_owner: bool
    historical_hello: bool


class JournalReader:
    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        attempt: int,
        read_bytes: int = 256 * 1024,
        max_record_bytes: int = 64 * 1024,
        max_diagnostics: int = 16,
        diagnostic_bytes: int = 160,
        verify_private_owner: bool = False,
    ):
        if min(read_bytes, max_record_bytes, max_diagnostics, diagnostic_bytes) < 1:
            raise ValueError("Reader bounds must be positive")
        self.path = path
        self.run_id = run_id
        self.attempt = attempt
        self.read_bytes = read_bytes
        self.max_record_bytes = max_record_bytes
        self.diagnostic_bytes = diagnostic_bytes
        self.verify_private_owner = verify_private_owner
        self.diagnostics: deque[tuple[int, bytes]] = deque(maxlen=max_diagnostics)
        self.malformed_count = 0
        self._file_identity = None
        self._observed_size = 0
        self.historical_hello = False
        self._reset(0)

    def snapshot(self) -> JournalBookmark:
        """Snapshot only after the caller commits every returned event.

        Buffered complete lines and partial tails are reread from their first
        byte after restoration; fetched-but-undelivered bytes are never skipped.
        """
        if self._error is not None:
            raise self._error
        offset = self._ready[0][0] if self._ready else self.offset - len(self._pending)
        return JournalBookmark(
            self.run_id,
            self.attempt,
            offset,
            self._last_seq,
            self._file_identity,
            self._observed_size,
            self.malformed_count,
            self.verify_private_owner,
            self.historical_hello,
        )

    def restore(self, bookmark: JournalBookmark) -> None:
        """Restore a committed byte boundary, refusing mismatched authority."""
        if (
            not isinstance(bookmark, JournalBookmark)
            or (bookmark.run_id, bookmark.attempt, bookmark.private_owner)
            != (self.run_id, self.attempt, self.verify_private_owner)
            or type(bookmark.offset) is not int
            or bookmark.offset < 0
            or type(bookmark.last_seq) is not int
            or bookmark.last_seq < -1
            or bookmark.observed_size < bookmark.offset
        ):
            raise ProtocolError("Invalid journal resume bookmark")
        self._reset(bookmark.offset, bookmark.last_seq)
        self._file_identity = bookmark.file_identity
        self._observed_size = bookmark.observed_size
        self.malformed_count = bookmark.malformed_count
        self.historical_hello = bookmark.historical_hello

    def _reset(self, offset: int, last_seq: int = -1) -> None:
        """Reposition the cursor. ``last_seq`` carries the replay floor across.

        Seeking must not forget which sequence numbers the client already holds:
        dropping the floor back to -1 lets records at or below the cursor be
        delivered again as if they were new.
        """
        self.offset = offset
        self._pending = b""
        self._ready: deque[tuple[int, bytes]] = deque()
        self._error: ProtocolError | None = None
        self.last_record_start = offset
        self._last_seq = last_seq

    def _fill(self) -> None:
        # Reopening on every poll also serves close-to-open shared filesystems.
        with (
            open_directory(self.path.parent) as directory,
            directory.open_file(self.path.name, share_write=True) as stream,
        ):
            if self.verify_private_owner:
                directory.verify_private_owner()
                stream.verify_private_owner()
            identity = stream.identity
            if self._file_identity is not None and identity != self._file_identity:
                raise ProtocolError("Journal replaced; select a new attempt")
            self._file_identity = identity
            size = stream.stat().st_size
            if size < max(self.offset, self._observed_size):
                raise ProtocolError("Journal truncated; select a new attempt")
            self._observed_size = max(self._observed_size, size)
            stream.seek(self.offset)
            chunk = stream.read(min(self.read_bytes, 1024 * 1024))
        start = self.offset - len(self._pending)
        self.offset += len(chunk)
        data = self._pending + chunk
        complete, newline, pending = data.rpartition(b"\n")
        if newline:
            self._pending = pending
            for line in complete.split(b"\n"):
                self._ready.append((start, line))
                start += len(line) + 1
        else:
            self._pending = data
        if len(self._pending) >= self.max_record_bytes:
            raise ProtocolError("Unterminated record exceeds byte bound")

    def read(self, *, max_records: int = 256) -> list[Envelope]:
        if max_records < 1:
            raise ValueError("max_records must be positive")
        if self._error is not None:
            raise self._error
        events = []
        filled = False
        try:
            while len(events) < max_records:
                if not self._ready:
                    if filled:
                        break
                    self._fill()
                    filled = True
                    if not self._ready:
                        break
                start, line = self._ready[0]
                if len(line) + 1 > self.max_record_bytes:
                    raise ProtocolError("Record exceeds byte bound")
                try:
                    event = decode(line)
                except MalformedJSON:
                    self.diagnostics.append((start, line[: self.diagnostic_bytes]))
                    self.malformed_count += 1
                    self._ready.popleft()
                    continue
                if (event.run_id, event.attempt) != (self.run_id, self.attempt):
                    raise ProtocolError("Journal identity mismatch")
                if event.seq <= self._last_seq:
                    raise ProtocolError("Sequence must increase within an attempt")
                self._ready.popleft()
                self._last_seq = event.seq
                self.last_record_start = start
                events.append(event)
        except ProtocolError as exc:
            self._error = exc
            raise
        return events

    def seek_to_seq(
        self, cursor: int, *, max_scan_bytes: int = 64 * 1024 * 1024
    ) -> None:
        """Position BEFORE the first seq > cursor, with bounded binary scanning."""
        if type(cursor) is not int or cursor < -1 or max_scan_bytes < 1:
            raise ValueError("Invalid cursor or scan budget")
        self._reset(0)
        if cursor == -1:
            return
        original_read_bytes = self.read_bytes
        try:
            while True:
                remaining = max_scan_bytes - self.offset
                if remaining <= 0 and not self._ready:
                    raise ProtocolError("Seek scan budget exhausted")
                self.read_bytes = min(original_read_bytes, max(1, remaining))
                before = self.offset
                events = self.read(max_records=1)
                if events and events[0].seq > cursor:
                    self._reset(self.last_record_start, last_seq=cursor)
                    return
                if not events and self.offset == before:
                    # Preserve the incomplete record for the next poll. The
                    # floor is retained here too: a cursor past the visible end
                    # of the journal is the live-tailing case, and records that
                    # arrive afterwards must still be measured against it.
                    self._reset(self.offset - len(self._pending), last_seq=cursor)
                    return
        finally:
            self.read_bytes = original_read_bytes


def seek_to_seq(
    reader: JournalReader, cursor: int, *, max_scan_bytes: int = 64 * 1024 * 1024
) -> None:
    reader.seek_to_seq(cursor, max_scan_bytes=max_scan_bytes)
