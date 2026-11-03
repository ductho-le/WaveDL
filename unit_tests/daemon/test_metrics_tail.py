"""Reading attempts/N/metrics.jsonl from a byte bookmark."""

import json

from wavedl_daemon.events.metrics_tail import read_tail
from wavedl_daemon.runs.model import Metric, SourceWarning, WarningCode


def line(**stats):
    return (json.dumps(stats) + "\n").encode()


def test_complete_lines_become_metrics_and_a_line_in_progress_waits(tmp_path):
    path = tmp_path / "metrics.jsonl"
    first, second = line(epoch=1, val_loss=0.5), line(epoch=2, val_loss=0.25)
    path.write_bytes(first + second + b'{"epoch": 3, "val_')
    chunk = read_tail(path, start=0)
    assert chunk.rows == (
        Metric(1, {"epoch": 1, "val_loss": 0.5}, len(first)),
        Metric(2, {"epoch": 2, "val_loss": 0.25}, len(first + second)),
    )
    assert chunk.pending == len(b'{"epoch": 3, "val_')
    assert chunk.drained
    with path.open("ab") as stream:
        stream.write(b'loss": 0.125}\n')
    resumed = read_tail(path, start=chunk.end)
    assert resumed.rows == (Metric(3, {"epoch": 3, "val_loss": 0.125}, path.stat().st_size),)
    assert resumed.pending == 0


def test_non_finite_values_become_null(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(
        b'{"epoch": 1, "val_loss": NaN, "lr": Infinity, "grad_norm": -Infinity, '
        b'"val_r2": 1e999, "MAE_Phys_P0": 0.1}\n'
    )
    (row,) = read_tail(path, start=0).rows
    assert row.stats == {
        "epoch": 1,
        "val_loss": None,
        "lr": None,
        "grad_norm": None,
        "val_r2": None,
        "MAE_Phys_P0": 0.1,
    }
    assert row.payload()["value"] is None


def test_a_malformed_line_is_consumed_as_a_warning(tmp_path):
    path = tmp_path / "metrics.jsonl"
    bad = [
        b"not json\n",
        b"[1, 2]\n",
        b'{"val_loss": 1}\n',
        b'{"epoch": 0}\n',
        b'{"epoch": true}\n',
        b'{"epoch": 1.0}\n',
        b'{"epoch": 1, "note": "text"}\n',
        b'{"epoch": 1, "lr": [1]}\n',
        b'{"epoch": 1, "epoch": 2}\n',
        b"\n",
    ]
    good = line(epoch=4, val_loss=0.5)
    path.write_bytes(b"".join(bad) + good)
    chunk = read_tail(path, start=0)
    warnings, last = chunk.rows[:-1], chunk.rows[-1]
    assert [row.code for row in warnings] == [WarningCode.MALFORMED_LINE] * len(bad)
    assert [row.source_end for row in warnings] == [
        sum(map(len, bad[: index + 1])) for index in range(len(bad))
    ]
    assert last == Metric(4, {"epoch": 4, "val_loss": 0.5}, path.stat().st_size)
    assert not chunk.sealed


def test_a_line_longer_than_the_bound_seals_the_file(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(b'{"epoch": 1, "pad": ' + b"1" * 64 + b"}\n")
    chunk = read_tail(path, start=0, limit=32)
    (row,) = chunk.rows
    assert row.code is WarningCode.LINE_TOO_LONG
    assert row.source_end == 33
    assert chunk.sealed and chunk.drained


def test_reads_stop_at_the_bound_and_report_what_is_left(tmp_path):
    path = tmp_path / "metrics.jsonl"
    rows = [line(epoch=epoch, val_loss=1.0) for epoch in range(1, 6)]
    path.write_bytes(b"".join(rows))
    first = read_tail(path, start=0, limit=len(rows[0]) * 2)
    assert [row.epoch for row in first.rows] == [1, 2]
    assert not first.drained
    second = read_tail(path, start=first.end, limit=len(rows[0]) * 2)
    assert [row.epoch for row in second.rows] == [3, 4]


def test_a_shrunken_file_is_reported_at_the_bookmark(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(line(epoch=1, val_loss=1.0))
    chunk = read_tail(path, start=500)
    assert chunk.rows == (SourceWarning(WarningCode.TRUNCATED, chunk.rows[0].message, 500),)
    assert chunk.sealed


def test_a_missing_file_reads_as_empty(tmp_path):
    chunk = read_tail(tmp_path / "metrics.jsonl", start=0)
    assert (chunk.rows, chunk.pending, chunk.drained) == ((), 0, True)
    gone = read_tail(tmp_path / "metrics.jsonl", start=10)
    assert gone.rows[0].code is WarningCode.TRUNCATED
