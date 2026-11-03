# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Byte-offset tail of ``attempts/N/metrics.jsonl``. Reads files, never SQLite.

The trainer's global rank 0 appends one ``json.dumps(epoch_stats)`` line per
epoch. Python writes non-finite floats as ``NaN``/``Infinity``, which become
None here so stored payloads stay strict JSON. The bookmark is the
``source_end`` of the last row, committed with the rows themselves.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from wavedl_daemon.backends.slurm_exit import parse_exit_record
from wavedl_daemon.runs.layout import RunPaths
from wavedl_daemon.runs.lifecycle import AttemptFiles
from wavedl_daemon.runs.model import (
    BackendKind,
    Evaluation,
    Metric,
    Operation,
    SourceWarning,
    WarningCode,
)


if TYPE_CHECKING:
    from pathlib import Path

    from wavedl_daemon.runs.lifecycle import RunHead
    from wavedl_daemon.runs.model import ExitRecord, FromFile


MAX_LINE_BYTES = 1 << 20


@dataclass(frozen=True)
class TailChunk:
    start: int  # the bookmark this chunk was read from
    rows: tuple[FromFile, ...]
    pending: int  # bytes after the last complete line, maybe a line in progress
    size: int  # file size when opened

    @property
    def end(self) -> int:
        return self.rows[-1].source_end if self.rows else self.start

    @property
    def sealed(self) -> bool:
        return any(isinstance(row, SourceWarning) and row.seals for row in self.rows)

    @property
    def drained(self) -> bool:
        """Nothing is left to read beyond ``pending``."""
        return self.sealed or self.end + self.pending >= self.size


def read_tail(path: Path, *, start: int, limit: int = MAX_LINE_BYTES) -> TailChunk:
    """Read the complete lines after ``start``, at most ``limit`` bytes of them.

    A fresh open per call gives NFS close-to-open consistency. A missing file
    reads as empty. A line in progress is left unread.
    """
    try:
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            stream.seek(min(start, size))
            data = stream.read(limit + 1)
    except FileNotFoundError:
        size, data = 0, b""
    if size < start:
        warning = SourceWarning(
            WarningCode.TRUNCATED,
            f"Metrics file shrank from {start} to {size} bytes",
            start,
        )
        return TailChunk(start, (warning,), 0, size)
    complete = data.rfind(b"\n") + 1
    if not complete and len(data) > limit:
        warning = SourceWarning(
            WarningCode.LINE_TOO_LONG,
            f"Metrics line exceeds {limit} bytes",
            start + len(data),
        )
        return TailChunk(start, (warning,), 0, size)
    rows, position = [], start
    for line in data[:complete].split(b"\n")[:-1]:
        position += len(line) + 1
        rows.append(_parse_line(line, position))
    return TailChunk(start, tuple(rows), len(data) - complete, size)


def read_attempt_files(head: RunHead) -> AttemptFiles:
    """What settle needs from the current attempt's directory."""
    paths, operation = RunPaths(head.ref.run_dir), head.ref.operation
    tail = None
    if operation is Operation.TRAIN and not head.metrics_sealed:
        tail = read_tail(paths.metrics(head.attempt), start=head.metrics_offset)
    evaluation = None
    if operation is Operation.TEST:
        evaluation = read_evaluation(paths.evaluation(head.attempt))
    record = None
    if head.ref.backend is BackendKind.SLURM:
        record = _read_exit_record(paths.exit_record(head.attempt))
    return AttemptFiles(tail, record, evaluation)


def read_evaluation(path: Path) -> Evaluation | SourceWarning:
    """``wavedl-test --metrics_json`` output: one object of numbers."""
    try:
        with path.open("rb") as stream:
            data = stream.read(MAX_LINE_BYTES + 1)
    except FileNotFoundError:
        return SourceWarning(
            WarningCode.EVALUATION_MISSING,
            "The test saved no evaluation metrics",
            None,
        )
    value = _loads(data) if len(data) <= MAX_LINE_BYTES else None
    if not _numbers(value):
        return SourceWarning(
            WarningCode.EVALUATION_MALFORMED,
            "Evaluation metrics are not one JSON object of numbers",
            None,
        )
    return Evaluation(value)


def _read_exit_record(path: Path) -> ExitRecord | None:
    try:
        with path.open("rb") as stream:
            return parse_exit_record(stream.read(64))
    except (FileNotFoundError, ValueError):
        return None


def _loads(data: bytes) -> object:
    try:
        return json.loads(
            data,
            parse_constant=lambda _: None,
            parse_float=_finite,
            object_pairs_hook=_unique,
        )
    except (ValueError, RecursionError):
        return None


def _numbers(value: object) -> bool:
    return type(value) is dict and all(
        item is None or type(item) in {int, float} for item in value.values()
    )


def _parse_line(line: bytes, end: int) -> FromFile:
    try:
        value = json.loads(
            line,
            parse_constant=lambda _: None,
            parse_float=_finite,
            object_pairs_hook=_unique,
        )
    except (ValueError, RecursionError) as exc:
        return SourceWarning(WarningCode.MALFORMED_LINE, f"Not JSON: {exc}"[:200], end)
    if (
        not _numbers(value)
        or type(value.get("epoch")) is not int
        or value["epoch"] < 1
    ):
        return SourceWarning(
            WarningCode.MALFORMED_LINE,
            "Expected an object of numbers with an integer epoch >= 1",
            end,
        )
    return Metric(value["epoch"], value, end)


def _finite(text: str) -> float | None:
    value = float(text)
    return value if math.isfinite(value) else None


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    fields = dict(pairs)
    if len(fields) != len(pairs):
        raise ValueError("Duplicate field")
    return fields
