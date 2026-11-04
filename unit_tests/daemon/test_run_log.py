"""The SQLite run log: ordered commits, head tokens, and a cache equal to the fold."""

import json
import sqlite3
from functools import reduce
from pathlib import Path

import pytest

from wavedl_daemon.events.metrics_tail import read_tail
from wavedl_daemon.runs.layout import RunPaths
from wavedl_daemon.runs.lifecycle import (
    AttemptFiles,
    IllegalTransition,
    RunRef,
    RunState,
    evolve,
    settle,
)
from wavedl_daemon.runs.model import (
    AttemptOpened,
    BackendKind,
    CancelRequested,
    Exited,
    Metric,
    Observed,
    Operation,
    Outcome,
    Phase,
    PlanId,
    RunId,
    SlurmHandle,
    SlurmObservation,
    Submitted,
    Submitting,
)
from wavedl_daemon.storage.database import UnsupportedDatabase, open_database
from wavedl_daemon.storage.ownership import DispatchRequired, acquire_dispatch_lock
from wavedl_daemon.storage.run_log import Expect, New, RunLog, Stale, encode


RUN = RunId("run-1")
SUBMIT = AttemptOpened("submit", None)
SLURM = SlurmHandle("4242", "c1", "alice")


@pytest.fixture
def state(tmp_path):
    with acquire_dispatch_lock(tmp_path / "dispatch.lock") as lock:
        yield tmp_path, lock


@pytest.fixture
def log(state):
    root, lock = state
    notified = []
    connection = open_database(root / "daemon.sqlite3", dispatch_lock=lock)
    yield RunLog(
        connection, notify=notified.append, clock=lambda: "2026-11-03T00:00:00+00:00"
    )
    connection.close()


def ref(run_id=RUN, run_dir=Path("/runs/run-1")):
    return RunRef(
        run_id, PlanId(f"plan-{run_id}"), BackendKind.SLURM, Operation.TRAIN, run_dir
    )


def start(log, run=None):
    run = run or ref()
    log.append(run.run_id, [SUBMIT], expect=New(run))
    head = log.append(run.run_id, [Submitting()], expect=Expect(revision=1))
    return log.append(
        run.run_id, [Submitted(SLURM)], expect=Expect(revision=head.revision)
    )


def history(log, run_id=RUN):
    return log.read_after(run_id, (0, -1), limit=1000)


def observation(**fields):
    values = {
        "restart_count": 0,
        "phase": Phase.RUNNING,
        "terminal": False,
        "native_state": "RUNNING",
        "reason": None,
        "exit_code": None,
        "signal": None,
        "earlier": {},
    }
    return SlurmObservation(**(values | fields))


def test_a_new_database_reopens_and_old_databases_are_refused(state):
    root, lock = state
    path = root / "daemon.sqlite3"
    open_database(path, dispatch_lock=lock).close()
    open_database(path, dispatch_lock=lock).close()
    for name, setup in {
        "unversioned.sqlite3": "CREATE TABLE runs (run_id TEXT)",
        "schema13.sqlite3": "PRAGMA user_version=13",
        "partial.sqlite3": "CREATE TABLE runs (run_id TEXT); PRAGMA user_version=1",
    }.items():
        old = sqlite3.connect(root / name)
        old.executescript(setup)
        old.close()
        with pytest.raises(UnsupportedDatabase):
            open_database(root / name, dispatch_lock=lock)


def test_opening_requires_dispatch_ownership_of_its_directory(tmp_path):
    with (
        acquire_dispatch_lock(tmp_path / "elsewhere" / "dispatch.lock") as lock,
        pytest.raises(DispatchRequired),
    ):
        open_database(tmp_path / "daemon.sqlite3", dispatch_lock=lock)


def test_the_stored_head_equals_the_fold_of_the_stored_events(log, tmp_path):
    paths = RunPaths(tmp_path / "run")
    paths.attempt(1).mkdir(parents=True)
    paths.metrics(1).write_bytes(
        b'{"epoch": 1, "val_loss": NaN}\n{"epoch": 2, "val_loss": 0.5}\n'
    )
    run = ref(run_dir=paths.run_dir)
    head = start(log, run)
    chunk = read_tail(paths.metrics(1), start=head.metrics_offset)
    head = log.append(RUN, chunk.rows, expect=Expect(offset=(1, 0)))
    head = log.append(RUN, [CancelRequested()], expect=Expect(revision=head.revision))
    stored = history(log)
    assert reduce(evolve, [item.event for item in stored], run) == head == log.head(RUN)
    assert [item.event for item in stored][3:5] == list(chunk.rows)


def test_a_slurm_requeue_replays_in_attempt_order(log, tmp_path):
    paths = RunPaths(tmp_path / "run")
    paths.attempt(1).mkdir(parents=True)
    paths.metrics(1).write_bytes(b'{"epoch": 1, "val_loss": 1.0}\n{"epoch": 2, "val_lo')
    run = ref(run_dir=paths.run_dir)
    head = start(log, run)
    seen = observation(
        restart_count=1,
        phase=Phase.QUEUED,
        native_state="PENDING",
        earlier={0: "NODE_FAIL"},
    )
    files = AttemptFiles(read_tail(paths.metrics(1), start=head.metrics_offset), None)
    head = log.append(
        RUN,
        settle(head, seen, files),
        expect=Expect(
            revision=head.revision, offset=(head.attempt, head.metrics_offset)
        ),
    )
    stored = history(log)
    assert [(item.attempt, item.seq, item.event.kind) for item in stored] == [
        (1, 0, "attempt"),
        (1, 1, "submitting"),
        (1, 2, "submitted"),
        (1, 3, "metric"),
        (1, 4, "warning"),
        (1, 5, "exited"),
        (2, 0, "attempt"),
        (2, 1, "observed"),
    ]
    assert (head.attempt, head.metrics_offset, head.stage.handle) == (2, 0, SLURM)
    assert log.read_after(RUN, (1, 4), limit=2) == stored[5:7]
    assert reduce(evolve, [item.event for item in stored], run) == head


def test_encode_is_the_sse_data_of_one_event(log):
    start(log)
    log.append(
        RUN,
        [Metric(1, {"epoch": 1, "val_loss": None}, 30)],
        expect=Expect(offset=(1, 0)),
    )
    frame = json.loads(
        encode(history(log)[-1]),
        parse_constant=lambda value: pytest.fail(f"non-finite {value}"),
    )
    assert frame == {
        "run_id": RUN,
        "attempt": 1,
        "seq": 3,
        "ts": "2026-11-03T00:00:00+00:00",
        "type": "metric",
        "payload": {
            "name": "epoch",
            "step": 1,
            "value": None,
            "unit": "",
            "metrics": {"epoch": 1, "val_loss": None},
        },
        "protocol": "wavedl-run",
        "version": 2,
    }


def test_stale_tokens_write_nothing(log):
    head = start(log)
    with pytest.raises(Stale):
        log.append(RUN, [CancelRequested()], expect=Expect(revision=head.revision - 1))
    with pytest.raises(Stale):
        log.append(RUN, [Metric(1, {"epoch": 1}, 9)], expect=Expect(offset=(1, 5)))
    with pytest.raises(Stale):
        log.append(RUN, [SUBMIT], expect=New(ref()))
    with pytest.raises(Stale):
        log.append(RunId("other"), [CancelRequested()], expect=Expect(revision=1))
    assert log.head(RUN) == head
    assert len(history(log)) == 3


def test_tail_and_lifecycle_tokens_do_not_stale_each_other(log):
    head = start(log)
    tail_token = Expect(offset=(head.attempt, head.metrics_offset))
    lifecycle_token = Expect(revision=head.revision)
    after_metric = log.append(RUN, [Metric(1, {"epoch": 1}, 9)], expect=tail_token)
    after_observed = log.append(
        RUN, [Observed(Phase.RUNNING, "RUNNING", None)], expect=lifecycle_token
    )
    assert after_metric.revision == head.revision
    assert after_observed.metrics_offset == 9
    with pytest.raises(Stale):
        log.append(RUN, [Metric(1, {"epoch": 1}, 9)], expect=tail_token)


def test_an_illegal_event_rolls_back_the_whole_append(log):
    head = start(log)
    with pytest.raises(IllegalTransition):
        log.append(
            RUN,
            [Observed(Phase.RUNNING, "RUNNING", None), Submitting()],
            expect=Expect(revision=head.revision),
        )
    with pytest.raises(IllegalTransition):
        log.append(
            RUN,
            [Exited(Outcome.REQUEUED, False)],
            expect=Expect(revision=head.revision),
        )
    assert log.head(RUN) == head
    assert len(history(log)) == 3


def test_triggers_reject_rows_out_of_commit_order(log, state):
    root, _ = state
    start(log)
    log.append(
        RUN,
        [Exited(Outcome.REQUEUED, False), AttemptOpened("requeue", 1)],
        expect=Expect(),
    )
    raw = sqlite3.connect(root / "daemon.sqlite3")

    def insert(attempt, seq, kind="observed", source_end=None):
        raw.execute(
            "INSERT INTO run_events VALUES (?, ?, ?, ?, 'ts', '{}', ?)",
            (RUN, attempt, seq, kind, source_end),
        )

    for row in [
        (1, 5),
        (2, 2),
        (2, 0),
        (4, 0, "attempt"),
        (2, 1, "attempt"),
        (2, 1, "observed", 10),
        (2, 1, "metric", 0),
    ]:
        with pytest.raises(sqlite3.IntegrityError):
            insert(*row)
    insert(2, 1)
    with pytest.raises(sqlite3.IntegrityError):
        raw.execute(
            "UPDATE run_events SET payload_json = '{}' WHERE run_id = ?", (RUN,)
        )
    with pytest.raises(sqlite3.IntegrityError):
        raw.execute("DELETE FROM run_events WHERE run_id = ?", (RUN,))
    with pytest.raises(sqlite3.IntegrityError):
        raw.execute(
            "INSERT INTO run_events VALUES ('fresh', 2, 0, 'attempt', 'ts', '{}', NULL)"
        )
    raw.rollback()
    raw.close()


def test_a_crash_between_read_and_commit_rereads_the_same_rows(state, tmp_path):
    root, lock = state
    paths = RunPaths(tmp_path / "run")
    paths.attempt(1).mkdir(parents=True)
    paths.metrics(1).write_bytes(b'{"epoch": 1}\n{"epoch": 2}\n')
    path = root / "daemon.sqlite3"
    connection = open_database(path, dispatch_lock=lock)
    head = start(RunLog(connection), ref(run_dir=paths.run_dir))
    lost = read_tail(paths.metrics(1), start=head.metrics_offset)
    connection.close()

    connection = open_database(path, dispatch_lock=lock)
    reopened = RunLog(connection)
    head = reopened.head(RUN)
    assert head.metrics_offset == 0
    chunk = read_tail(paths.metrics(1), start=head.metrics_offset)
    assert chunk == lost
    head = reopened.append(RUN, chunk.rows, expect=Expect(offset=(1, 0)))
    assert [(item.attempt, item.seq) for item in history(reopened)][3:] == [
        (1, 3),
        (1, 4),
    ]
    assert read_tail(paths.metrics(1), start=head.metrics_offset).rows == ()
    connection.close()


def test_heads_list_active_runs_and_filter_by_state(log):
    start(log)
    other = RunId("run-2")
    log.append(other, [SUBMIT], expect=New(ref(other)))
    log.append(other, [CancelRequested()], expect=Expect(revision=1))
    log.append(
        other,
        [Exited(Outcome.CANCELLED, True)],
        expect=Expect(revision=2),
    )
    assert [head.ref.run_id for head in log.heads(active=True)] == [RUN]
    assert [head.ref.run_id for head in log.heads(active=False)] == [other]
    assert [head.ref.run_id for head in log.heads(state=RunState.CANCELLED)] == [other]
    assert {head.ref.run_id for head in log.heads()} == {RUN, other}


def test_subscribers_are_notified_after_each_commit(state):
    root, lock = state
    notified = []
    connection = open_database(root / "daemon.sqlite3", dispatch_lock=lock)
    log = RunLog(connection, notify=notified.append)
    start(log)
    with pytest.raises(Stale):
        log.append(RUN, [CancelRequested()], expect=Expect(revision=0))
    assert notified == [RUN, RUN, RUN]
    connection.close()
