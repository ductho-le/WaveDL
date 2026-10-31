"""Real loopback HTTP connects runtime push to authenticated durable ingestion."""

import asyncio
import socket
import sqlite3
import sys
import threading
import time
from contextlib import closing

import pytest
import uvicorn
from starlette.responses import JSONResponse

from wavedl.core.events import ExitPayload, HelloPayload, MetricPayload
from wavedl.core.live_push import LivePushClient, LivePushConfig
from wavedl.core.manifest import read_manifest
from wavedl.journal import RunJournal, RuntimeBootstrap
from wavedl_daemon.events.ingestion import Ingestor
from wavedl_daemon.http.app import create_app
from wavedl_daemon.runs.runtime_service import HostedServices
from wavedl_daemon.storage.settings import Settings

from .test_hosted_app import Backend


@pytest.mark.parametrize("failure_point", [None, "before_commit", "after_commit"])
def test_live_push_real_http_commits_canonical_events(
    tmp_path, monkeypatch, failure_point
):
    """Controlled delivery faults recover through real owner/file/SQLite paths."""
    replay_enabled = threading.Event()
    if failure_point is None:
        replay_enabled.set()
    original_poll = Ingestor.poll_once

    async def deferred_poll(ingestor, *args, **kwargs):
        # This is a controlled polling schedule, not shared-filesystem visibility.
        if replay_enabled.is_set():
            return await original_poll(ingestor, *args, **kwargs)
        return None

    monkeypatch.setattr(Ingestor, "poll_once", deferred_poll)
    settings = Settings(tmp_path / "state", tmp_path / "runs")
    backend = Backend()
    services = HostedServices(settings, backend_factory=lambda lock: {"local": backend})
    app = create_app(
        settings,
        token="test-token",
        daemon_id="daemon",
        build_id="build",
        services=services,
    )
    pushes = 0
    suppressed_status = None

    async def delivery_boundary(scope, receive, send):
        nonlocal pushes, suppressed_status
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or not scope["path"].endswith("/events")
        ):
            await app(scope, receive, send)
            return
        pushes += 1
        if pushes != 2 or failure_point is None:
            await app(scope, receive, send)
            return
        if failure_point == "after_commit":
            messages = []

            async def capture(message):
                messages.append(message)

            await app(scope, receive, capture)
            suppressed_status = messages[0]["status"]
        await JSONResponse({"error": "controlled delivery outage"}, status_code=503)(
            scope, receive, send
        )

    async def run():
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(16)
        listener.setblocking(False)
        port = listener.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(
                delivery_boundary,
                host="127.0.0.1",
                port=port,
                log_config=None,
                access_log=False,
                lifespan="on",
            )
        )
        server.install_signal_handlers = lambda: None
        task = asyncio.create_task(server.serve(sockets=[listener]))
        client = None
        try:
            deadline = time.monotonic() + 3
            while not server.started and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            assert server.started
            payload = {
                "backend": "local",
                "operation": "train",
                "config": {"epochs": 1},
                "topology": {
                    "num_machines": 1,
                    "processes_per_machine": 1,
                    "machine_rank": 0,
                },
                "runtime": {
                    "python": sys.executable,
                    "mode": "existing",
                    "device": "cpu",
                    "environment": {},
                },
            }
            plan = await services.run_service.plan(payload, "push")
            result = await services.run_service.submit(plan.plan_id)
            deadline = time.monotonic() + 3
            while backend.handle is None and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            assert backend.handle is not None
            backend.runtime_startup()
            deadline = time.monotonic() + 3
            while result.attempt is None and time.monotonic() < deadline:
                result = await services.run_service.get(plan.run_id)
                await asyncio.sleep(0.01)
            assert result.attempt is not None
            layout = await services.run_service.layout(plan.run_id)
            manifest = read_manifest(layout.manifest)
            client = LivePushClient(
                LivePushConfig(f"http://127.0.0.1:{port}", "test-token")
            )
            bootstrap = RuntimeBootstrap(
                layout, manifest, result.attempt, plan.request_id, manifest
            )
            journal = RunJournal(
                bootstrap,
                hello=HelloPayload(
                    1, 1, "actual", "3.11", "cpu", ["train"], sys.executable
                ),
                push_client=client,
            )
            metric = journal.emit("metric", MetricPayload("loss", 0.5, 1))
            if failure_point is not None:
                deadline = time.monotonic() + 3
                while not client.counters["disabled"] and time.monotonic() < deadline:
                    await asyncio.sleep(0.01)
                assert client.counters["disabled"]
                assert client.counters["acknowledged"] == 1
                assert pushes == 2
                page = await services.event_log.read_after(plan.run_id)
                assert [event.seq for event in page] == (
                    [0, 1] if failure_point == "after_commit" else [0]
                )
                assert suppressed_status == (
                    200 if failure_point == "after_commit" else None
                )
                exit_event = journal.emit("exit", ExitPayload(0, "completed"))
                replay_enabled.set()
            deadline = time.monotonic() + 3
            expected_count = 2 if failure_point is None else 3
            while time.monotonic() < deadline:
                page = await services.event_log.read_after(plan.run_id)
                if len(page) == expected_count:
                    break
                await asyncio.sleep(0.01)
            assert [event.seq for event in page] == list(range(expected_count))
            assert page[1] == metric
            if failure_point is None:
                deadline = time.monotonic() + 3
                while (
                    client.counters["acknowledged"] < 2 and time.monotonic() < deadline
                ):
                    await asyncio.sleep(0.01)
                assert client.counters["acknowledged"] == 2, client.counters
                assert not client.counters["disabled"]
            else:
                assert page[-1] == exit_event
                assert pushes == 2, (
                    "Disabled delivery must not retry or leak later events"
                )

                def projection_snapshot():
                    uri = (settings.state_root / "registry.db").as_uri() + "?mode=ro"
                    with closing(sqlite3.connect(uri, uri=True)) as connection:
                        connection.execute("BEGIN")
                        return (
                            connection.execute(
                                "SELECT event_count,last_type,state "
                                "FROM runs WHERE run_id=?",
                                (plan.run_id,),
                            ).fetchall(),
                            connection.execute(
                                "SELECT attempt,seq FROM cursors WHERE run_id=?",
                                (plan.run_id,),
                            ).fetchall(),
                            connection.execute(
                                "SELECT attempt,seq,step,name,value,active "
                                "FROM metric_history WHERE run_id=?",
                                (plan.run_id,),
                            ).fetchall(),
                        )

                before_replay = projection_snapshot()
                assert before_replay[0] == [(3, "exit", "completed")]
                assert before_replay[1] == [(result.attempt, 2)]
                assert before_replay[2] == [(result.attempt, 1, 1, "loss", 0.5, 1)]
                # Both a pre-commit outage and lost post-commit ACK replay exactly
                # once. A subsequent duplicate push cannot advance the cursor.
                assert (
                    await services.ingestor.push(
                        plan.plan_id, plan.run_id, result.attempt, page
                    )
                    == 2
                )
                assert await services.event_log.read_after(plan.run_id) == page
                assert projection_snapshot() == before_replay
            journal.close()
            assert b"test-token" not in layout.journal(result.attempt).read_bytes()
        finally:
            if client is not None:
                client.close()
            server.should_exit = True
            await asyncio.wait_for(task, 5)
            listener.close()

    asyncio.run(run())
