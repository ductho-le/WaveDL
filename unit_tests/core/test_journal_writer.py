import errno
import os
import stat
import threading
import time

import pytest

from wavedl.core import journal_writer, persistence, safe_files
from wavedl.core.journal_reader import JournalReader
from wavedl.core.journal_writer import AbandonedTail, JournalWriter, RecordTooLarge
from wavedl.core.layout import RunLayout, allocate_attempt
from wavedl.core.protocol import decode


RUN_ID = "00000000-0000-4000-8000-000000000001"


def test_oversize_leaves_bytes_and_sequence_unchanged(tmp_path):
    layout = RunLayout.create(tmp_path, RUN_ID)
    with JournalWriter(layout, max_record_bytes=512) as writer:
        writer.append("hello", {})
        before = writer.path.read_bytes()
        with pytest.raises(RecordTooLarge):
            writer.append("log", {"message": "x" * 1000})
        assert writer.path.read_bytes() == before
        assert writer.seq == 1
        assert writer.append("state", {}).seq == 1


def test_abandoned_tail_rolls_to_new_attempt_without_appending(tmp_path):
    """Restart preserves the crash fragment and emits seq zero in attempt 2."""
    layout = RunLayout.create(tmp_path, RUN_ID)
    first = allocate_attempt(layout)
    fragment = b'{"seq":0,"payload":'
    layout.journal(first).write_bytes(fragment)
    with pytest.raises(AbandonedTail):
        JournalWriter(layout, attempt=first)
    with JournalWriter.restart(layout, first) as writer:
        assert isinstance(writer.abandoned_tail, AbandonedTail)
        assert writer.abandoned_tail.path == layout.journal(first)
        event = writer.append("hello", {})
        assert (event.attempt, event.seq) == (2, 0)
        assert decode(writer.path.read_bytes()) == event
    assert layout.journal(first).read_bytes() == fragment


def test_intact_journal_is_not_counted_and_resumed(tmp_path):
    layout = RunLayout.create(tmp_path, RUN_ID)
    with JournalWriter(layout) as writer:
        writer.append("hello", {})
    before = writer.path.read_bytes()
    with pytest.raises(FileExistsError):
        JournalWriter(layout, attempt=1)
    with JournalWriter.restart(layout, 1) as replacement:
        assert replacement.abandoned_tail is None
        assert replacement.append("hello", {}).seq == 0
        assert replacement.attempt == 2
    assert writer.path.read_bytes() == before


def test_fsync_cadence_exit_and_close(tmp_path, monkeypatch):
    layout = RunLayout.create(tmp_path, RUN_ID)
    calls = []
    real_fsync = journal_writer.os.fsync

    def sync(fd):
        if stat.S_ISREG(os.fstat(fd).st_mode):
            calls.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(journal_writer.os, "fsync", sync)
    writer = JournalWriter(layout, fsync_every=3)
    writer.append("hello", {})
    writer.append("state", {})
    assert len(calls) == 0
    writer.append("metric", {})
    assert len(calls) == 1
    writer.append("exit", {})
    assert len(calls) == 2
    writer.append("log", {})
    writer.close()
    assert len(calls) == 3
    writer.close()
    assert len(calls) == 3


def test_disk_full_after_short_write_poisoned_and_recovered(tmp_path):
    """A partial write followed by ENOSPC cannot be joined to a later record."""
    layout = RunLayout.create(tmp_path, RUN_ID)
    writer = JournalWriter(layout)
    original = writer._stream

    class ShortThenFull:
        closed = False
        calls = 0

        def write(self, data):
            self.calls += 1
            if self.calls == 1:
                return original.write(data[:17])
            raise OSError(errno.ENOSPC, "disk full")

        def close(self):
            self.closed = True
            original.close()

    writer._stream = ShortThenFull()
    with pytest.raises(OSError) as caught:
        writer.append("hello", {})
    assert caught.value.errno == errno.ENOSPC
    assert writer.seq == 0
    assert len(writer.path.read_bytes()) == 17
    with pytest.raises(RuntimeError, match="new attempt"):
        writer.append("state", {})
    writer.close()
    with JournalWriter.restart(layout, 1) as replacement:
        assert replacement.append("hello", {}).attempt == 2
    assert len(writer.path.read_bytes()) == 17


def test_fsync_failure_is_visible_and_writer_cannot_continue(tmp_path, monkeypatch):
    layout = RunLayout.create(tmp_path, RUN_ID)
    writer = JournalWriter(layout)

    def fail(fd):
        raise OSError(errno.EIO, "sync failed")

    monkeypatch.setattr(journal_writer.os, "fsync", fail)
    with pytest.raises(OSError, match="sync failed"):
        writer.append("hello", {})
    assert writer.path.read_bytes().endswith(b"\n")
    with pytest.raises(RuntimeError):
        writer.append("state", {})
    writer.close()


def test_default_fsync_and_sanitized_return(tmp_path, monkeypatch):
    layout = RunLayout.create(tmp_path, RUN_ID)
    calls = []

    def sync(fd):
        if stat.S_ISREG(os.fstat(fd).st_mode):
            calls.append(fd)

    monkeypatch.setattr(journal_writer.os, "fsync", sync)
    with JournalWriter(layout) as writer:
        event = writer.append("metric", {"best_val_loss": float("inf")})
        assert event.payload == {"best_val_loss": None}
        assert decode(writer.path.read_bytes()) == event
        assert len(calls) == 1


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX special file")
def test_existing_fifo_never_blocks_tail_inspection(tmp_path):
    layout = RunLayout.create(tmp_path, RUN_ID)
    attempt = allocate_attempt(layout)
    os.mkfifo(layout.journal(attempt))
    began = time.monotonic()
    with pytest.raises(ValueError, match="regular"):
        JournalWriter(layout, attempt=attempt)
    with pytest.raises(ValueError, match="regular"):
        JournalWriter.restart(layout, attempt)
    assert time.monotonic() - began < 1


def test_existing_symlink_never_reads_target_tail(tmp_path):
    layout = RunLayout.create(tmp_path, RUN_ID)
    attempt = allocate_attempt(layout)
    target = tmp_path / "unrelated"
    target.write_bytes(b"unrelated unfinished tail")
    try:
        layout.journal(attempt).symlink_to(target)
    except OSError:
        pytest.skip("Native platform does not permit test symlink creation")
    with pytest.raises(ValueError, match="regular"):
        JournalWriter(layout, attempt=attempt)
    assert target.read_bytes() == b"unrelated unfinished tail"


def test_first_flush_syncs_file_then_journal_directory(tmp_path, monkeypatch):
    layout = RunLayout.create(tmp_path, RUN_ID)
    attempt = allocate_attempt(layout)
    calls = []
    real_fsync = os.fsync
    real_directory_sync = journal_writer.sync_directory

    def sync(fd):
        if stat.S_ISREG(os.fstat(fd).st_mode):
            calls.append("file")
        real_fsync(fd)

    def sync_parent(path):
        assert path == layout.journal(attempt).parent
        calls.append("directory")
        real_directory_sync(path)

    monkeypatch.setattr(journal_writer.os, "fsync", sync)
    monkeypatch.setattr(journal_writer, "sync_directory", sync_parent)
    with JournalWriter(layout, attempt=attempt) as writer:
        writer.append("hello", {})
        assert calls == ["file", "directory"]
        writer.append("state", {})
        assert calls == ["file", "directory", "file"]


@pytest.mark.skipif(os.name != "nt", reason="Native Windows sharing contract")
def test_first_flush_waits_for_native_journal_reader_release(tmp_path, monkeypatch):
    """A pinned journal poll delays durability without losing or duplicating bytes."""
    layout = RunLayout.create(tmp_path, RUN_ID)
    reader_pinned = threading.Event()
    release_reader = threading.Event()
    writer_waiting = threading.Event()
    writer_done = threading.Event()
    failures = []
    real_read = safe_files.SafeFile.read
    real_sleep = persistence.time.sleep
    real_sync = journal_writer.sync_directory
    syncs = []

    def held_read(stream, count):
        reader_pinned.set()
        if not release_reader.wait(5):
            raise TimeoutError("Test reader was not released")
        return real_read(stream, count)

    def observed_wait(seconds):
        writer_waiting.set()
        real_sleep(seconds)

    def sync_parent(path):
        real_sync(path)
        syncs.append(path)

    def poll(reader):
        try:
            reader.read()
        except BaseException as error:
            failures.append(error)

    def append(writer):
        try:
            writer.append("hello", {})
        except BaseException as error:
            failures.append(error)
        finally:
            writer_done.set()

    monkeypatch.setattr(safe_files.SafeFile, "read", held_read)
    monkeypatch.setattr(persistence.time, "sleep", observed_wait)
    monkeypatch.setattr(journal_writer, "sync_directory", sync_parent)
    with JournalWriter(layout) as writer:
        reader = JournalReader(writer.path, run_id=RUN_ID, attempt=writer.attempt)
        reader_thread = threading.Thread(target=poll, args=(reader,))
        writer_thread = threading.Thread(target=append, args=(writer,))
        reader_thread.start()
        try:
            assert reader_pinned.wait(5)
            writer_thread.start()
            assert writer_waiting.wait(5)
            assert not writer_done.is_set()
        finally:
            release_reader.set()
            reader_thread.join(6)
            if writer_thread.ident is not None:
                writer_thread.join(6)
        assert not reader_thread.is_alive() and not writer_thread.is_alive()
        assert failures == []
        assert writer_done.is_set()
        assert syncs == [writer.path.parent]
        writer.append("state", {})
        assert syncs == [writer.path.parent]
        events = JournalReader(
            writer.path, run_id=RUN_ID, attempt=writer.attempt
        ).read()
        assert [(event.type, event.seq) for event in events] == [
            ("hello", 0),
            ("state", 1),
        ]


def test_directory_flush_failure_poisoned_without_republishing(tmp_path, monkeypatch):
    layout = RunLayout.create(tmp_path, RUN_ID)
    calls = []

    def denied(path):
        calls.append(path)
        raise PermissionError("directory admission deadline exhausted")

    monkeypatch.setattr(journal_writer, "sync_directory", denied)
    with JournalWriter(layout) as writer:
        with pytest.raises(PermissionError, match="deadline exhausted"):
            writer.append("hello", {})
        assert not writer._entry_synced
        assert writer.seq == 1
        before = writer.path.read_bytes()
        assert decode(before).type == "hello"
        with pytest.raises(RuntimeError, match="new attempt"):
            writer.append("state", {})
    assert calls == [writer.path.parent]
    assert writer.path.read_bytes() == before
