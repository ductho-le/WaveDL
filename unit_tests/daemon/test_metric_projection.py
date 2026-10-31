import sqlite3

import pytest

from unit_tests.core.test_generations import setup_run
from unit_tests.core.test_metric_history import (
    RUN_ID,
    branched_events,
    causal_events,
    event,
)
from wavedl.core.generations import publish_generation
from wavedl.core.protocol import Envelope, ProtocolError, encode
from wavedl_daemon.storage.registry import Registry


def test_registry_persists_deduplicated_resumed_history(tmp_path):
    path = tmp_path / "registry.db"
    with Registry(path) as registry:
        registry.apply_events(RUN_ID, 1, [event(1), event(2), event(3)])
        resumed = [
            event(
                0,
                attempt=2,
                kind="supersede",
                source_attempt=1,
                step=1,
                generation="gen-000001",
            ),
            event(1, attempt=2, step=2, value=10.0),
        ]
        registry.apply_events(RUN_ID, 2, resumed)
        registry.apply_events(RUN_ID, 2, resumed)
        assert len(registry.metrics(RUN_ID, active_only=False)) == 4
    with Registry(path) as registry:
        assert [(r.step, r.value) for r in registry.metrics(RUN_ID)] == [
            (1, 1.0),
            (2, 10.0),
        ]
        assert registry.metrics(RUN_ID)[1].source_generation == "gen-000001"
        assert registry.metrics(RUN_ID, limit=1, offset=1)[0].attempt == 2
        assert registry._connection.execute("PRAGMA user_version").fetchone()[0] == 13


def test_registry_source_lineage_and_delayed_old_attempts(tmp_path):
    path = tmp_path / "registry.db"
    with Registry(path) as registry:
        for item in branched_events():
            registry.apply_events(RUN_ID, item.attempt, [item])
    with Registry(path) as registry:
        assert [(r.step, r.value) for r in registry.metrics(RUN_ID)] == [
            (1, 1.0),
            (2, 20.0),
            (3, 40.0),
        ]
        assert all(
            not r.active
            for r in registry.metrics(RUN_ID, active_only=False)
            if r.value == 99.0
        )


def test_source_sequence_cutoff_survives_journal_rebuild(tmp_path):
    run_root = tmp_path / "runs"
    batch = causal_events()
    for attempt in {item.attempt for item in batch}:
        journal = run_root / RUN_ID / "attempts" / f"{attempt:06d}" / "events.jsonl"
        journal.parent.mkdir(parents=True)
        journal.write_bytes(
            b"\n".join(encode(item) for item in batch if item.attempt == attempt)
            + b"\n"
        )
    with Registry(tmp_path / "registry.db") as registry:
        for item in batch:
            registry.apply_events(RUN_ID, item.attempt, [item])
        expected = registry.metrics(RUN_ID)
        assert [(r.step, r.value) for r in expected] == [(1, 1.0), (2, 1.0), (3, 20.0)]
        assert registry.rebuild_from_journals(run_root) == 3
        assert registry.metrics(RUN_ID) == expected


def test_bad_supersession_rolls_back_every_projection_and_cursor(tmp_path):
    with Registry(tmp_path / "registry.db") as registry:
        with pytest.raises(ProtocolError):
            registry.apply_events(
                RUN_ID,
                2,
                [
                    event(0, attempt=2),
                    event(
                        1,
                        attempt=2,
                        kind="supersede",
                        source_attempt=1,
                        step=99,
                        generation="gen-000001",
                    ),
                ],
            )
        assert registry.get(RUN_ID) is None
        assert registry.cursor(RUN_ID, 2) == -1
        assert registry.metrics(RUN_ID, active_only=False) == []


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"value": float("nan")},
        {"metrics": {"future": float("inf")}},
        {"step": True},
        {"metrics": {"epoch": 1.5}},
    ],
)
def test_bad_metric_rolls_back(payload, tmp_path):
    with Registry(tmp_path / "registry.db") as registry:
        bad = event(1, **payload) if payload else event(1, kind="metric-invalid")
        if not payload:
            bad = type(bad)(RUN_ID, 1, "metric", 1, bad.ts, {})
        with pytest.raises(ProtocolError):
            registry.apply_events(RUN_ID, 1, [bad])
        assert registry.cursor(RUN_ID, 1) == -1


def test_rebuild_retains_good_prefix_and_preserves_control_tables(tmp_path):
    run_root = tmp_path / "runs"
    journal = run_root / RUN_ID / "attempts" / "000001" / "events.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_bytes(encode(event(1)) + b"\n" + encode(event(2)) + b"\n")
    with Registry(tmp_path / "registry.db") as registry:
        before = registry._connection.execute("SELECT count(*) FROM plans").fetchone()[
            0
        ]
        registry.rebuild_from_journals(run_root)
        assert len(registry.metrics(RUN_ID)) == 2
        registry.rebuild_from_journals(run_root)
        assert len(registry.metrics(RUN_ID)) == 2
        assert (
            registry._connection.execute("SELECT count(*) FROM plans").fetchone()[0]
            == before
        )
        bad = event(3, kind="supersede", source_attempt=1, step=1, generation="bad")
        with journal.open("ab") as stream:
            stream.write(encode(bad) + b"\n")
        registry.rebuild_from_journals(run_root)
        assert registry.cursor(RUN_ID, 1) == 2
        assert registry.rebuild_diagnostics


def test_migration_five_backup_exists(tmp_path):
    path = tmp_path / "registry.db"
    Registry(path).close()
    backup = tmp_path / "registry.db.backup-0005_metric_projection"
    assert backup.is_file()
    with sqlite3.connect(backup) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 13


def test_publication_boundary_is_verified_not_a_journal_artifact(tmp_path):
    layout, attempt, source = setup_run(tmp_path)
    published = publish_generation(layout, source, attempt=attempt, step=8)
    with Registry(tmp_path / "registry.db") as registry:
        registry.apply_events(
            RUN_ID,
            1,
            [
                Envelope(
                    RUN_ID,
                    1,
                    "artifact",
                    0,
                    "2026-10-03T00:00:00Z",
                    {
                        "kind": "checkpoint",
                        "path": str(published.path),
                        "sha256": "a" * 64,
                        "size": 1,
                        "step": 8,
                        "generation": published.generation,
                    },
                )
            ],
        )
        marker = event(
            0,
            attempt=2,
            kind="supersede",
            source_attempt=1,
            step=8,
            generation=published.generation,
        )
        with pytest.raises(ProtocolError):
            registry.apply_events(RUN_ID, 2, [marker])
        registry.acknowledge_metric_generation(layout, published)
        registry.apply_events(RUN_ID, 2, [marker])
        (published.path / "model.bin").write_bytes(b"corrupted")
        with pytest.raises(ProtocolError):
            registry.acknowledge_metric_generation(layout, published)
