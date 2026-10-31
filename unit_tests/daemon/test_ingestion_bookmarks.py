import asyncio
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest

from wavedl.core.events import HelloPayload
from wavedl.core.journal_reader import JournalReader
from wavedl.core.protocol import Envelope, ProtocolError, encode
from wavedl_daemon.backends.contracts import Handle
from wavedl_daemon.events.broker import EventBroker
from wavedl_daemon.events.ingestion import Ingestor
from wavedl_daemon.execution.budget import BudgetExhausted, WorkBudget
from wavedl_daemon.execution.pool import BoundedThreadPool
from wavedl_daemon.runs.negotiation import NegotiationStatus
from wavedl_daemon.storage.registry import Registry


def records(run_id, count=10):
    return [
        Envelope(
            run_id,
            1,
            "metric",
            seq,
            "2026-10-04T00:00:00Z",
            {"name": "loss", "value": float(seq), "step": seq},
        )
        for seq in range(count)
    ]


class Controller:
    def accept_runtime_events(self, plan_id, events):
        return SimpleNamespace(status=NegotiationStatus.ACCEPTED, reason="")


def test_alternating_evicted_journals_reach_later_metrics(tmp_path):
    ids = [str(uuid4()), str(uuid4())]
    paths = [tmp_path / "a.jsonl", tmp_path / "b.jsonl"]
    for run_id, path in zip(ids, paths, strict=True):
        path.write_bytes(b"".join(encode(event) for event in records(run_id)))
    with (
        Registry(tmp_path / "registry.db") as registry,
        BoundedThreadPool(WorkBudget()) as pool,
    ):
        ingestor = Ingestor(
            registry,
            Controller(),
            EventBroker(),
            pool,
            max_readers=1,
            max_pages=1,
            page_size=2,
        )

        async def run():
            for _ in range(5):
                for run_id, path in zip(ids, paths, strict=True):
                    await ingestor.poll_once("plan", run_id, 1, path)
                    assert len(ingestor._readers) == 1
            assert [registry.cursor(run_id, 1) for run_id in ids] == [9, 9]
            assert len(ingestor._bookmarks) == 2
            assert all(mark.last_seq == 9 for mark in ingestor._bookmarks.values())

        asyncio.run(run())


@pytest.mark.parametrize("change", ["truncate", "replace"])
def test_restored_reader_refuses_truncated_or_replaced_source(tmp_path, change):
    run_id = str(uuid4())
    path = tmp_path / "events.jsonl"
    original = b"".join(encode(event) for event in records(run_id))
    path.write_bytes(original)
    reader = JournalReader(path, run_id=run_id, attempt=1)
    assert [event.seq for event in reader.read(max_records=2)] == [0, 1]
    mark = reader.snapshot()
    if change == "replace":
        replacement = tmp_path / "replacement"
        replacement.write_bytes(original)
        replacement.replace(path)
    else:
        path.write_bytes(encode(records(run_id)[0]))
    resumed = JournalReader(path, run_id=run_id, attempt=1)
    resumed.restore(mark)
    with pytest.raises(ProtocolError, match=r"truncated|replaced"):
        resumed.read()


def test_snapshot_rewinds_partial_tail_and_preserves_sequence_floor(tmp_path):
    run_id = str(uuid4())
    path = tmp_path / "events.jsonl"
    first, second = [encode(event) for event in records(run_id, 2)]
    path.write_bytes(first + second[:20])
    reader = JournalReader(path, run_id=run_id, attempt=1)
    assert [event.seq for event in reader.read()] == [0]
    mark = reader.snapshot()
    assert mark.offset == len(first) and mark.last_seq == 0
    with path.open("ab") as stream:
        stream.write(second[20:])
    resumed = JournalReader(path, run_id=run_id, attempt=1)
    resumed.restore(mark)
    assert [event.seq for event in resumed.read()] == [1]


def test_failed_commit_retains_only_previous_committed_bookmark(tmp_path, monkeypatch):
    run_id = str(uuid4())
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"".join(encode(event) for event in records(run_id, 6)))
    with (
        Registry(tmp_path / "registry.db") as registry,
        BoundedThreadPool(WorkBudget()) as pool,
    ):
        ingestor = Ingestor(
            registry, Controller(), EventBroker(), pool, max_pages=1, page_size=2
        )
        original = registry._write_cursor

        def fail(*args):
            raise RuntimeError("injected failed commit")

        async def run():
            assert await ingestor.poll_once("plan", run_id, 1, path) == 1
            committed = ingestor._bookmarks[(run_id, 1)]
            monkeypatch.setattr(registry, "_write_cursor", fail)
            with pytest.raises(RuntimeError):
                await ingestor.poll_once("plan", run_id, 1, path)
            assert ingestor._bookmarks[(run_id, 1)] == committed
            assert registry.cursor(run_id, 1) == 1
            assert (run_id, 1) not in ingestor._readers
            monkeypatch.setattr(registry, "_write_cursor", original)
            assert await ingestor.poll_once("plan", run_id, 1, path) == 3
            assert await ingestor.poll_once("plan", run_id, 1, path) == 5

        asyncio.run(run())


def test_resume_capacity_refuses_new_journal_without_dropping_old_progress(tmp_path):
    ids = [str(uuid4()), str(uuid4())]
    paths = [tmp_path / "a.jsonl", tmp_path / "b.jsonl"]
    for run_id, path in zip(ids, paths, strict=True):
        path.write_bytes(b"".join(encode(event) for event in records(run_id, 4)))
    with (
        Registry(tmp_path / "registry.db") as registry,
        BoundedThreadPool(WorkBudget()) as pool,
    ):
        ingestor = Ingestor(
            registry,
            Controller(),
            EventBroker(),
            pool,
            max_readers=1,
            max_bookmarks=1,
            max_pages=1,
            page_size=2,
        )

        async def run():
            assert await ingestor.poll_once("plan", ids[0], 1, paths[0]) == 1
            with pytest.raises(BudgetExhausted, match="bookmark capacity"):
                await ingestor.poll_once("plan", ids[1], 1, paths[1])
            assert await ingestor.poll_once("plan", ids[0], 1, paths[0]) == 3
            ingestor.forget(ids[0], 1)
            assert await ingestor.poll_once("plan", ids[1], 1, paths[1]) == 1

        asyncio.run(run())


@pytest.mark.parametrize("fault", ["path", "run_id", "attempt", "handle", "missing"])
def test_active_poll_refuses_unregistered_source_before_file_io(tmp_path, fault):
    """Active ingestion requires matching durable lineage before reading bytes."""
    run_id = str(uuid4())
    owned = tmp_path / "owned" / run_id
    foreign = tmp_path / "foreign" / "events.jsonl"
    foreign.parent.mkdir()
    foreign.write_bytes(b"".join(encode(event) for event in records(run_id, 2)))
    handle = Handle(
        backend="local",
        run_id=run_id,
        request_id="request",
        attempt=1,
        owner="owner",
        run_dir=str(owned),
        pid=123,
        supervisor_id="supervisor",
    )
    record = SimpleNamespace(
        run_id=run_id,
        request_id="request",
        owner="owner",
        backend="local",
        run_dir=str(owned),
        attempt=1,
        handle=handle,
    )
    if fault == "handle":
        record.handle = replace(handle, owner="foreign-owner")
    if fault == "missing":
        record = None
    requested_run = str(uuid4()) if fault == "run_id" else run_id
    requested_attempt = 2 if fault == "attempt" else 1
    path = foreign if fault == "path" else owned / "attempts" / "1" / "events.jsonl"

    class OwnedController(Controller):
        execution_store = SimpleNamespace(get=lambda _plan_id: record)

    with (
        Registry(tmp_path / "registry.db") as registry,
        BoundedThreadPool(WorkBudget()) as pool,
    ):
        ingestor = Ingestor(
            registry,
            OwnedController(),
            EventBroker(),
            pool,
        )

        async def run():
            expected_error = KeyError if fault == "missing" else ProtocolError
            with pytest.raises(expected_error):
                await ingestor.poll_once("plan", requested_run, requested_attempt, path)
            assert registry.cursor(run_id, 1) == -1
            assert registry.get(run_id) is None
            assert not ingestor._readers
            assert not ingestor._bookmarks
            assert not ingestor._resume_keys

        asyncio.run(run())


def test_active_poll_accepts_matching_durable_handle(tmp_path):
    """A registered active journal still commits telemetry through real SQLite."""
    run_id = str(uuid4())
    owned = tmp_path / "owned" / run_id
    path = owned / "attempts" / "1" / "events.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"".join(encode(event) for event in records(run_id, 2)))
    handle = Handle(
        "local", run_id, "request", 1, "owner", str(owned), raw_json='{"adopted":true}'
    )
    record = SimpleNamespace(**handle.to_dict(), handle=handle)

    class OwnedController(Controller):
        execution_store = SimpleNamespace(get=lambda _plan_id: record)

    with (
        Registry(tmp_path / "registry.db") as registry,
        BoundedThreadPool(WorkBudget()) as pool,
    ):
        ingestor = Ingestor(registry, OwnedController(), EventBroker(), pool)
        assert asyncio.run(ingestor.poll_once("plan", run_id, 1, path)) == 1
        assert registry.cursor(run_id, 1) == 1
        assert registry.get(run_id).event_count == 2


def test_historical_hello_proof_survives_compact_bookmark(tmp_path):
    run_id = str(uuid4())
    path = tmp_path / "events.jsonl"
    events = records(run_id, 3)
    hello = HelloPayload(1, 1, "test", "3.11", "cpu", ["train"], "/usr/bin/python3")
    from dataclasses import asdict

    events[0] = replace(events[0], type="hello", payload=asdict(hello))
    path.write_bytes(b"".join(encode(event) for event in events))
    reader = JournalReader(path, run_id=run_id, attempt=1, verify_private_owner=True)
    assert reader.read(max_records=1)[0].type == "hello"
    reader.historical_hello = True
    mark = reader.snapshot()
    resumed = JournalReader(path, run_id=run_id, attempt=1, verify_private_owner=True)
    resumed.restore(mark)
    assert resumed.historical_hello
    assert [event.seq for event in resumed.read()] == [1, 2]
