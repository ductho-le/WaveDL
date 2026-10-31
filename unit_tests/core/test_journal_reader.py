import json
import os

import pytest

from wavedl.core.journal_reader import JournalReader, seek_to_seq
from wavedl.core.journal_writer import JournalWriter
from wavedl.core.layout import RunLayout
from wavedl.core.protocol import Envelope, ProtocolError, encode
from wavedl.core.safe_files import SafeFile


RUN_ID = "00000000-0000-4000-8000-000000000001"


def record(seq, **changes):
    values = {
        "run_id": RUN_ID,
        "attempt": 1,
        "seq": seq,
        "type": "hello" if seq == 0 else "metric",
        "ts": "2026-09-06T00:00:00Z",
        "payload": {"label": "café"},
    }
    return encode(Envelope(**(values | changes)))


def reader_for(tmp_path, data, **options):
    path = tmp_path / "events.jsonl"
    path.write_bytes(data)
    return JournalReader(path, run_id=RUN_ID, attempt=1, **options)


@pytest.mark.skipif(os.name != "posix", reason="POSIX FIFO rejection")
def test_fifo_journal_is_rejected_without_blocking(tmp_path):
    path = tmp_path / "events.jsonl"
    os.mkfifo(path)
    reader = JournalReader(path, run_id=RUN_ID, attempt=1)
    with pytest.raises((ValueError, OSError)):
        reader.read()


def test_symlink_journal_is_never_followed(tmp_path):
    target = tmp_path / "target.jsonl"
    target.write_bytes(record(0))
    path = tmp_path / "events.jsonl"
    try:
        path.symlink_to(target)
    except OSError:
        pytest.skip("Symlink creation unavailable")
    reader = JournalReader(path, run_id=RUN_ID, attempt=1)
    with pytest.raises((ValueError, OSError)):
        reader.read()


def test_seek_past_the_visible_end_keeps_the_replay_floor(tmp_path):
    """Records appended after a seek must still be measured against the cursor.

    Seeking to a cursor beyond the last visible record is the live-tailing case:
    a client resumes while the writer is still catching up. Resetting the floor
    to -1 there would replay everything the client already holds.
    """
    path = tmp_path / "events.jsonl"
    path.write_bytes(record(0))
    reader = JournalReader(path, run_id=RUN_ID, attempt=1)

    reader.seek_to_seq(5)
    path.write_bytes(record(0) + record(2) + record(5) + record(6))

    with pytest.raises(ProtocolError, match="Sequence must increase"):
        reader.read()


def test_seek_within_the_journal_resumes_above_the_cursor(tmp_path):
    """The ordinary resume, and the live tail that follows it."""
    path = tmp_path / "events.jsonl"
    path.write_bytes(record(0) + record(1) + record(2) + record(3))
    reader = JournalReader(path, run_id=RUN_ID, attempt=1)

    reader.seek_to_seq(1)

    assert [event.seq for event in reader.read()] == [2, 3]
    with path.open("ab") as stream:
        stream.write(record(4) + record(5))
    assert [event.seq for event in reader.read()] == [4, 5]


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"", []),
        (record(0), [0]),
        (record(0) + record(1), [0, 1]),
        (record(0)[:-1], []),
        (record(0) + record(1)[:-1], [0]),
    ],
)
def test_newline_commit_cases(tmp_path, data, expected):
    reader = reader_for(tmp_path, data)
    assert [event.seq for event in reader.read()] == expected
    assert reader.read() == []
    assert reader.malformed_count == 0


def test_pending_completes_on_later_reopened_poll(tmp_path):
    reader = reader_for(tmp_path, record(0) + record(1)[:-5])
    assert [event.seq for event in reader.read()] == [0]
    assert reader.read() == []
    with reader.path.open("ab") as stream:
        stream.write(record(1)[-5:])
    assert [event.seq for event in reader.read()] == [1]
    assert reader.malformed_count == 0


def test_cursor_minus_one_replays_hello_seq_zero(tmp_path):
    reader = reader_for(tmp_path, record(0) + record(1))
    seek_to_seq(reader, -1)
    events = reader.read()
    assert [(event.seq, event.type) for event in events] == [
        (0, "hello"),
        (1, "metric"),
    ]


def test_missing_cursor_sequence_keeps_later_records(tmp_path):
    reader = reader_for(tmp_path, record(0) + record(2) + record(5))
    reader.seek_to_seq(1)
    assert reader.offset == len(record(0))
    assert [event.seq for event in reader.read()] == [2, 5]


def test_crlf_offsets_are_bytes_with_multibyte_utf8(tmp_path):
    first = record(0).replace(b"\n", b"\r\n")
    rest = (record(2) + record(3)).replace(b"\n", b"\r\n")
    reader = reader_for(tmp_path, first + rest)
    reader.seek_to_seq(1)
    assert reader.offset == len(first)
    assert [event.seq for event in reader.read()] == [2, 3]


def test_schema_invalid_line_raises_shared_error_and_cannot_be_skipped(tmp_path):
    invalid = json.loads(record(0)) | {"attempt": -1}
    reader = reader_for(tmp_path, json.dumps(invalid).encode() + b"\n" + record(1))
    for _ in range(2):
        with pytest.raises(ProtocolError, match="attempt"):
            reader.read()
    assert reader.malformed_count == 0


def test_malformed_diagnostics_and_io_are_bounded(tmp_path, monkeypatch):
    reader = reader_for(
        tmp_path,
        b"{bad json and diagnostic text}\n" * 200 + record(0),
        read_bytes=47,
        max_diagnostics=3,
        diagnostic_bytes=12,
    )
    real_read = SafeFile.read
    reads = []

    def tracked_read(stream, size):
        reads.append(size)
        return real_read(stream, size)

    monkeypatch.setattr(SafeFile, "read", tracked_read)
    events = []
    for _ in range(200):
        events.extend(reader.read(max_records=1))
    assert [event.seq for event in events] == [0]
    assert reader.malformed_count == 200
    assert len(reader.diagnostics) == 3
    assert all(len(raw) <= 12 for offset, raw in reader.diagnostics)
    assert reads and max(reads) <= 47


@pytest.mark.parametrize("data", [b"x" * 512, b"x" * 512 + b"\n"])
def test_oversized_tail_or_line_fails_with_bounded_buffer(tmp_path, data):
    reader = reader_for(tmp_path, data, read_bytes=64, max_record_bytes=128)
    with pytest.raises(ProtocolError, match="bound"):
        for _ in range(4):
            reader.read()
    assert len(reader._pending) <= 128


def test_record_count_bound_does_not_drop_buffered_events(tmp_path):
    reader = reader_for(tmp_path, b"".join(record(seq) for seq in range(20)))
    batches = [reader.read(max_records=3) for _ in range(7)]
    assert all(len(batch) <= 3 for batch in batches)
    assert [event.seq for batch in batches for event in batch] == list(range(20))


def test_writer_reader_conformance_and_eof_is_not_terminal(tmp_path):
    layout = RunLayout.create(tmp_path, RUN_ID)
    with JournalWriter(layout) as writer:
        first = writer.append("hello", {})
        reader = JournalReader(writer.path, run_id=RUN_ID, attempt=writer.attempt)
        assert reader.read() == [first]
        assert reader.read() == []
        second = writer.append("metric", {"loss": float("inf")})
        assert reader.read() == [second]
        assert second.payload["loss"] is None


def test_seek_past_visible_records_keeps_incomplete_tail(tmp_path):
    reader = reader_for(tmp_path, record(0) + record(2)[:-1])
    reader.seek_to_seq(0)
    assert reader.offset == len(record(0))
    with reader.path.open("ab") as stream:
        stream.write(b"\n")
    assert [event.seq for event in reader.read()] == [2]


def test_seek_budget_and_attempt_identity(tmp_path):
    reader = reader_for(tmp_path, record(0) + record(2))
    with pytest.raises(ProtocolError, match="budget"):
        reader.seek_to_seq(10, max_scan_bytes=10)
    reader = reader_for(tmp_path, record(0, attempt=2))
    with pytest.raises(ProtocolError, match="identity"):
        reader.read()


def test_newline_terminated_oversized_record_reaches_line_check(tmp_path):
    reader = reader_for(
        tmp_path, b"x" * 512 + b"\n", read_bytes=1024, max_record_bytes=128
    )
    with pytest.raises(ProtocolError, match="Record exceeds byte bound"):
        reader.read()
    assert reader._pending == b""


def test_duplicate_after_eof_is_not_replayed(tmp_path):
    reader = reader_for(tmp_path, record(0))
    assert [event.seq for event in reader.read()] == [0]
    assert reader.read() == []
    with reader.path.open("ab") as stream:
        stream.write(record(0))
    with pytest.raises(ProtocolError, match="Sequence must increase"):
        reader.read()


def test_utf8_split_across_polls_is_not_decoded_early(tmp_path):
    data = record(0)
    split = data.index("é".encode()) + 1
    reader = reader_for(tmp_path, data[:split])
    assert reader.read() == []
    assert reader.malformed_count == 0
    with reader.path.open("ab") as stream:
        stream.write(data[split:])
    assert reader.read()[0].payload["label"] == "café"
