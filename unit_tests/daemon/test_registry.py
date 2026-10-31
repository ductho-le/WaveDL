from datetime import UTC, datetime
from pathlib import Path

import pytest

from wavedl.core.protocol import Envelope, ProtocolError, encode
from wavedl_daemon.storage.registry import Registry


RUN_ID = "00000000-0000-4000-8000-000000000001"
OTHER_ID = "00000000-0000-4000-8000-000000000002"


def event(seq, *, run_id=RUN_ID, attempt=1, type="metric", **payload):
    if type == "metric":
        payload = {"name": "loss", "value": None, "step": seq} | payload
    return Envelope(
        run_id=run_id,
        attempt=attempt,
        type=type,
        seq=seq,
        ts=datetime(2026, 9, 8, 12, seq % 60, tzinfo=UTC).isoformat(),
        payload=payload,
    )


# A writer that pauses between the two writes: the projection row has been
# written, the cursor has not. Killing it there is the only way to tell an
# atomic implementation from one that commits the projection separately.
VICTIM = """
import sys, time
sys.path.insert(0, {src!r})
from datetime import UTC, datetime
from wavedl.core.protocol import Envelope
from wavedl_daemon.storage.registry import Registry

def pause(self, run_id, attempt, seq):
    print("record written, cursor pending", flush=True)
    time.sleep(300)
Registry._write_cursor = pause

registry = Registry({path!r})
registry.apply_events({run_id!r}, 1, [Envelope({run_id!r}, 1, "hello", 0,
    datetime.now(UTC).isoformat(), {{}})])
"""


def test_projection_and_cursor_commit_atomically(tmp_path, paused_process):
    """A crash between the two writes must leave NEITHER applied."""
    path = tmp_path / "registry.db"
    Registry(path).close()  # create the schema first

    script = VICTIM.format(
        src=str(Path("src").resolve()), path=str(path), run_id=RUN_ID
    )
    with paused_process(script, "record written, cursor pending") as victim:
        victim.kill()
        victim.wait(timeout=10)

    with Registry(path) as registry:
        assert registry.get(RUN_ID) is None, "projection committed without its cursor"
        assert registry.cursor(RUN_ID, 1) == -1


def test_duplicate_apply_is_a_no_op(tmp_path):
    """Re-delivering a batch after a lost response must change nothing."""
    with Registry(tmp_path / "registry.db") as registry:
        batch = [event(0, type="hello"), event(1), event(2)]

        assert registry.apply_events(RUN_ID, 1, batch) == 2
        first = registry.get(RUN_ID)
        assert registry.apply_events(RUN_ID, 1, batch) == 2

        assert registry.get(RUN_ID) == first
        assert first.event_count == 3


def test_partial_redelivery_applies_only_what_is_new(tmp_path):
    with Registry(tmp_path / "registry.db") as registry:
        registry.apply_events(RUN_ID, 1, [event(0, type="hello"), event(1)])

        assert registry.apply_events(RUN_ID, 1, [event(1), event(2), event(3)]) == 3
        assert registry.get(RUN_ID).event_count == 4


def test_cursor_is_per_attempt_not_per_run(tmp_path):
    """A requeued Slurm job reuses its id and restarts seq at zero."""
    with Registry(tmp_path / "registry.db") as registry:
        registry.apply_events(RUN_ID, 1, [event(0, type="hello"), event(1), event(2)])

        assert registry.cursor(RUN_ID, 1) == 2
        assert registry.cursor(RUN_ID, 2) == -1

        applied = registry.apply_events(
            RUN_ID, 2, [event(0, attempt=2, type="hello"), event(1, attempt=2)]
        )

        assert applied == 1  # attempt 2's own seq, not continued from attempt 1
        assert registry.cursor(RUN_ID, 1) == 2
        assert registry.get(RUN_ID).attempt == 2


def test_events_belonging_to_another_run_are_refused(tmp_path):
    with Registry(tmp_path / "registry.db") as registry:
        with pytest.raises(ProtocolError, match="does not belong"):
            registry.apply_events(RUN_ID, 1, [event(0, run_id=OTHER_ID)])

        assert registry.get(RUN_ID) is None
        assert registry.cursor(RUN_ID, 1) == -1


def test_list_runs_filters_and_orders(tmp_path):
    with Registry(tmp_path / "registry.db") as registry:
        registry.apply_events(RUN_ID, 1, [event(0, type="hello")])
        registry.apply_events(
            OTHER_ID, 1, [event(0, run_id=OTHER_ID, type="exit", state="failed")]
        )

        assert [r.run_id for r in registry.list_runs(state="running")] == [RUN_ID]
        assert [r.run_id for r in registry.list_runs(state="failed")] == [OTHER_ID]
        assert len(registry.list_runs()) == 2
        assert registry.list_runs(limit=1) == registry.list_runs(limit=1, offset=0)


def write_journal(root, run_id, attempt, events):
    path = root / run_id / "attempts" / str(attempt) / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        for item in events:
            stream.write(encode(item))
    return path


def test_rebuild_from_journals_restores_a_lost_registry(tmp_path):
    """Losing the index costs a rebuild, not data. events.jsonl is the truth."""
    run_root = tmp_path / "runs"
    write_journal(run_root, RUN_ID, 1, [event(0, type="hello"), event(1), event(2)])
    write_journal(
        run_root,
        OTHER_ID,
        1,
        [event(0, run_id=OTHER_ID, type="hello"), event(1, run_id=OTHER_ID)],
    )

    path = tmp_path / "registry.db"
    with Registry(path) as registry:
        registry.apply_events(RUN_ID, 1, [event(0, type="hello"), event(1), event(2)])
        before = registry.get(RUN_ID)

    path.unlink()  # the index is gone
    # Recover telemetry into a new location. Normal startup at the initialized
    # intent path must refuse silent recreation once durable control exists.
    with Registry(tmp_path / "recovered-telemetry.db") as rebuilt:
        assert rebuilt.get(RUN_ID) is None

        assert rebuilt.rebuild_from_journals(run_root) == 2

        assert rebuilt.get(RUN_ID) == before
        assert rebuilt.cursor(RUN_ID, 1) == 2
        assert rebuilt.get(OTHER_ID).event_count == 2


def test_rebuild_is_idempotent_and_discards_stale_rows(tmp_path):
    run_root = tmp_path / "runs"
    write_journal(run_root, RUN_ID, 1, [event(0, type="hello"), event(1)])

    with Registry(tmp_path / "registry.db") as registry:
        registry.apply_events(OTHER_ID, 1, [event(0, run_id=OTHER_ID, type="hello")])

        registry.rebuild_from_journals(run_root)
        once = registry.list_runs()
        registry.rebuild_from_journals(run_root)

        assert registry.list_runs() == once
        assert [r.run_id for r in once] == [RUN_ID]  # the stale row is gone


def test_rebuild_keeps_going_past_a_damaged_journal_tail(tmp_path):
    run_root = tmp_path / "runs"
    damaged = write_journal(run_root, RUN_ID, 1, [event(0, type="hello"), event(1)])
    with damaged.open("ab") as stream:
        stream.write(b'{"not a valid envelope"}\n')
    write_journal(run_root, OTHER_ID, 1, [event(0, run_id=OTHER_ID, type="hello")])

    with Registry(tmp_path / "registry.db") as registry:
        registry.rebuild_from_journals(run_root)

        assert registry.get(RUN_ID).event_count == 2  # the good prefix survives
        assert registry.get(OTHER_ID) is not None  # and the next run still folds


@pytest.mark.parametrize("sequences", [[0, 0], [2, 1]])
def test_unordered_or_duplicate_batch_is_refused_atomically(tmp_path, sequences):
    with Registry(tmp_path / "registry.db") as registry:
        with pytest.raises(ProtocolError, match="Sequence must increase"):
            registry.apply_events(RUN_ID, 1, [event(seq) for seq in sequences])
        assert registry.get(RUN_ID) is None
        assert registry.cursor(RUN_ID, 1) == -1


def test_rebuild_preserves_good_prefix_before_schema_invalid_line(tmp_path):
    root = tmp_path / "runs"
    path = write_journal(root, RUN_ID, 1, [event(0), event(1)])
    with path.open("ab") as stream:
        stream.write(b'{"protocol":"wavedl-jsonl"}\n')
    with Registry(tmp_path / "registry.db") as registry:
        assert registry.rebuild_from_journals(root) == 1
        assert registry.get(RUN_ID).event_count == 2
        assert registry.cursor(RUN_ID, 1) == 1
        assert registry.rebuild_diagnostics[0][0] == path


def test_rebuild_does_not_treat_empty_bounded_read_as_eof(tmp_path, monkeypatch):
    from wavedl_daemon.storage import registry as module

    root = tmp_path / "runs"
    write_journal(root, RUN_ID, 1, [event(0), event(1)])
    original = module.JournalReader
    monkeypatch.setattr(
        module, "JournalReader", lambda *a, **kw: original(*a, **kw, read_bytes=16)
    )
    with Registry(tmp_path / "registry.db") as registry:
        assert registry.rebuild_from_journals(root) == 1
        assert registry.get(RUN_ID).event_count == 2


def test_rebuild_orders_attempts_numerically(tmp_path):
    root = tmp_path / "runs"
    write_journal(root, RUN_ID, 2, [event(0, attempt=2, type="state", state="failed")])
    write_journal(root, RUN_ID, 10, [event(0, attempt=10, type="hello")])
    with Registry(tmp_path / "registry.db") as registry:
        assert registry.rebuild_from_journals(root) == 2
        assert registry.get(RUN_ID).attempt == 10
        assert registry.get(RUN_ID).state == "running"


def test_rebuild_io_failure_restores_previous_index(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    write_journal(root, RUN_ID, 1, [event(0)])
    with Registry(tmp_path / "registry.db") as registry:
        registry.apply_events(OTHER_ID, 1, [event(0, run_id=OTHER_ID)])
        prior = registry.list_runs()

        def fail(*args):
            raise OSError("storage failed")

        monkeypatch.setattr(registry, "_write_cursor", fail)
        with pytest.raises(OSError, match="storage failed"):
            registry.rebuild_from_journals(root)
        assert registry.list_runs() == prior
        assert registry.cursor(RUN_ID, 1) == -1


def test_missing_rebuild_root_does_not_erase_the_index(tmp_path):
    with Registry(tmp_path / "registry.db") as registry:
        registry.apply_events(RUN_ID, 1, [event(0)])
        prior = registry.get(RUN_ID)
        with pytest.raises(FileNotFoundError):
            registry.rebuild_from_journals(tmp_path / "missing")
        assert registry.get(RUN_ID) == prior


def test_old_attempt_does_not_overwrite_current_runtime_projection(tmp_path):
    with Registry(tmp_path / "registry.db") as registry:
        registry.apply_events(RUN_ID, 2, [event(0, attempt=2, type="hello")])
        registry.apply_events(RUN_ID, 1, [event(0, type="exit", status="failed")])
        assert registry.get(RUN_ID).state == "running"
        assert registry.get(RUN_ID).attempt == 2
        assert registry.get(RUN_ID).event_count == 2


def test_initial_schema_is_stamped_and_future_schema_is_read_only(tmp_path):
    import sqlite3

    from wavedl_daemon.storage.schema import MigrationFailed, schema_version

    path = tmp_path / "registry.db"
    with Registry(path) as registry:
        registry.apply_events(RUN_ID, 1, [event(0)])
    with sqlite3.connect(path) as connection:
        assert schema_version(connection) == 13
        connection.execute("ALTER TABLE runs ADD COLUMN future_field TEXT")
        connection.execute("PRAGMA user_version=999")
    with pytest.raises(MigrationFailed, match="newer than supported"):
        Registry(path)
    with Registry(path, read_only=True) as registry:
        assert registry.get(RUN_ID).event_count == 1
        with pytest.raises(PermissionError, match="read-only"):
            registry.apply_events(RUN_ID, 1, [event(1)])
        with pytest.raises(PermissionError, match="read-only"):
            registry.rebuild_from_journals(tmp_path)


def test_foreign_unversioned_database_is_not_imported(tmp_path):
    import sqlite3

    from wavedl_daemon.storage.schema import MigrationFailed

    path = tmp_path / "foreign.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE tbl_task(url TEXT)")
        connection.execute("INSERT INTO tbl_task VALUES ('user data')")
    with pytest.raises(MigrationFailed, match="no legacy import"):
        Registry(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT * FROM tbl_task").fetchall() == [
            ("user data",)
        ]
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall() == [("tbl_task",)]


def test_rebuild_restores_operation_from_shared_manifest(tmp_path):
    import sys

    from wavedl.core.layout import OperationKind, RunLayout, allocate_attempt
    from wavedl.core.manifest import RuntimeSelection, build_manifest, write_manifest

    root = tmp_path / "runs"
    layout = RunLayout.create(root, RUN_ID)
    manifest = build_manifest(
        layout,
        operation=OperationKind.PLOT,
        resolved_config={},
        runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
        submission_cwd=tmp_path,
    )
    write_manifest(layout, manifest)
    attempt = allocate_attempt(layout)
    write_journal(root, RUN_ID, attempt, [event(0, type="hello")])
    with Registry(tmp_path / "registry.db") as registry:
        assert registry.rebuild_from_journals(root) == 1
        assert registry.get(RUN_ID).operation == "plot"
