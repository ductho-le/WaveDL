"""Assembled local ASGI/storage crash acceptance, not real-site acceptance.

The compute backend is controlled, but public planning/submission/cancellation,
writer receipt/startup proof, journals, SQLite, service ownership and replay are
real. No native scheduler or remote-filesystem guarantee is claimed here.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sqlite3
import sys
import time
from dataclasses import asdict
from pathlib import Path

import httpx
import pytest

from unit_tests.daemon.test_hosted_app import Backend
from wavedl.core.events import HelloPayload
from wavedl.core.layout import RunLayout
from wavedl.core.protocol import Envelope, encode
from wavedl_daemon.backends.contracts import (
    BackendObservation,
    ObservationStatus,
    SubmissionHandle,
    SubmissionObservation,
)
from wavedl_daemon.http.app import create_app
from wavedl_daemon.runs.runtime_service import HostedServices
from wavedl_daemon.storage.registry import Registry
from wavedl_daemon.storage.settings import Settings


TOKEN = "assembled-crash-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class AuditedBackend(Backend):
    """Real local writer proof, controlled compute, persistent submit audit."""

    def __init__(self, audit):
        super().__init__()
        self.audit = audit

    def submit(self, plan, **kwargs):
        handle = super().submit(plan, **kwargs)
        with self.audit.open("a") as stream:
            stream.write(json.dumps(asdict(handle), sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return handle

    def observe(self, handle):
        if isinstance(handle, SubmissionHandle):
            return SubmissionObservation(ObservationStatus.RUNNING)
        return BackendObservation(handle.attempt, ObservationStatus.RUNNING)


def payload():
    return {
        "backend": "local",
        "operation": "train",
        "config": {"epochs": 1},
        "runtime": {
            "python": sys.executable,
            "mode": "existing",
            "device": "cpu",
            "environment": {},
        },
        "topology": {
            "num_machines": 1,
            "processes_per_machine": 1,
            "machine_rank": 0,
        },
    }


def hello(run_id, attempt):
    return Envelope(
        run_id,
        attempt,
        "hello",
        0,
        "2026-10-12T00:00:00Z",
        asdict(
            HelloPayload(1, 1, "assembled", "3.11", "cpu", ["train"], sys.executable)
        ),
    )


async def crash_child(root, reverse):
    """Stop the service owner after its first uncommitted telemetry write."""
    root = Path(root)
    settings = Settings(root / "state", root / "runs")
    backend = AuditedBackend(root / "submissions.jsonl")
    services = HostedServices(settings, backend_factory=lambda lock: {"local": backend})
    app = create_app(
        settings, token=TOKEN, daemon_id="assembled", build_id="test", services=services
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://localhost:8420",
            headers=AUTH,
        ) as client,
    ):
        response = await client.post(
            "/api/runs/plan",
            json=payload(),
            headers={"Idempotency-Key": "assembled-crash"},
        )
        assert response.status_code == 200, response.text
        planned = response.json()
        run_id = planned["run_id"]
        response = await client.post("/api/runs", json={"plan_id": planned["plan_id"]})
        assert response.status_code == 202, response.text
        deadline = time.monotonic() + 5
        while not backend.submissions and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert backend.submissions == 1
        binding = backend.runtime_startup()
        while time.monotonic() < deadline:
            response = await client.get(f"/api/runs/{run_id}")
            if response.json()["attempt"] == binding.attempt:
                break
            await asyncio.sleep(0.01)
        assert response.json()["attempt"] == binding.attempt
        response = await client.post(f"/api/runs/{run_id}/cancel")
        assert response.status_code == 202, response.text
        (root / "planned.json").write_text(json.dumps(planned))
        original_record = Registry._write_record
        original_cursor = Registry._write_cursor

        def record_boundary(self, record):
            assert self._connection.in_transaction
            if reverse:
                original_cursor(self, record.run_id, record.attempt, 0)
            else:
                original_record(self, record)
            print("telemetry-write-pending-commit", flush=True)
            time.sleep(300)
            if reverse:
                original_record(self, record)

        Registry._write_record = record_boundary
        layout = RunLayout(settings.run_root / run_id, run_id)
        with layout.journal(binding.attempt).open("wb") as stream:
            stream.write(encode(hello(run_id, binding.attempt)))
            stream.flush()
            os.fsync(stream.fileno())
        # The real HostedServices pump discovers and ingests the durable journal.
        await asyncio.sleep(10)
        raise AssertionError("Hosted pump did not reach the telemetry transaction")


def durable_snapshot(path, run_id):
    """Read independent SQLite truth after process death or completed replay."""
    with sqlite3.connect(path) as connection:
        return {
            "plans": connection.execute(
                "SELECT plan_id,request_id,plan_json FROM plans"
            ).fetchall(),
            "intent": connection.execute(
                "SELECT plan_id,submission_requested,cancellation_requested "
                "FROM plan_control"
            ).fetchall(),
            "executions": connection.execute(
                "SELECT plan_id,attempt,owner,handle_json FROM execution_control"
            ).fetchall(),
            "runs": connection.execute(
                "SELECT event_count FROM runs WHERE run_id=?", (run_id,)
            ).fetchall(),
            "cursor": connection.execute(
                "SELECT attempt,seq FROM cursors WHERE run_id=?", (run_id,)
            ).fetchall(),
            "events": connection.execute(
                "SELECT attempt,seq FROM event_log WHERE run_id=?", (run_id,)
            ).fetchall(),
        }


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL and dispatch ownership")
@pytest.mark.parametrize(
    "reverse", [False, True], ids=["projection-first", "cursor-first"]
)
def test_assembled_http_crash_replay_is_durable_and_idempotent(
    tmp_path, paused_process, reverse
):
    """Both write orders roll back, restart from journal, and redeliver once."""
    project = Path(__file__).resolve().parents[2]
    script = (
        f"import sys; sys.path[:0] = {[str(project), str(project / 'src')]!r}; "
        "import asyncio; "
        "from unit_tests.daemon.test_assembled_crash_replay import crash_child; "
        f"asyncio.run(crash_child({str(tmp_path)!r}, {reverse!r}))"
    )
    with paused_process(script, "telemetry-write-pending-commit") as child:
        child.kill()
        child.wait(timeout=10)
        assert child.returncode == -signal.SIGKILL
    planned = json.loads((tmp_path / "planned.json").read_text())
    run_id = planned["run_id"]
    database = tmp_path / "state" / "registry.db"
    before = durable_snapshot(database, run_id)
    assert before["runs"] == before["cursor"] == before["events"] == []
    assert len(before["plans"]) == 1
    assert before["plans"][0][0] == planned["plan_id"]
    manifest = json.loads(json.loads(before["plans"][0][2])["manifest_json"])
    assert manifest["run_id"] == run_id
    assert before["intent"] == [(planned["plan_id"], 1, 1)]
    assert len(before["executions"]) == 1
    assert before["executions"][0][:2] == (planned["plan_id"], 1)
    audit = (tmp_path / "submissions.jsonl").read_bytes()
    assert len(audit.splitlines()) == 1

    async def restart():
        settings = Settings(tmp_path / "state", tmp_path / "runs")
        backend = AuditedBackend(tmp_path / "submissions.jsonl")
        services = HostedServices(
            settings, backend_factory=lambda lock: {"local": backend}
        )
        app = create_app(
            settings,
            token=TOKEN,
            daemon_id="assembled",
            build_id="test",
            services=services,
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://localhost:8420",
                headers=AUTH,
            ) as client,
        ):
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                page = await services.event_log.read_after(run_id)
                if page:
                    break
                await asyncio.sleep(0.01)
            assert page == [hello(run_id, 1)]
            replayed = durable_snapshot(database, run_id)
            assert replayed["runs"] == [(1,)]
            assert replayed["cursor"] == replayed["events"] == [(1, 0)]
            response = await client.get(f"/api/runs/{run_id}")
            assert response.status_code == 200, response.text
            view = response.json()
            assert view["run_id"] == run_id
            assert view["request_id"] == before["plans"][0][1]
            assert view["attempt"] == 1
            assert view["submission_requested"] is True
            assert view["cancellation_requested"] is True
            assert view["state"] == "stopping"
            # A lost HTTP ACK/repeated request retains the authoritative identity.
            response = await client.post(
                "/api/runs/plan",
                json=payload(),
                headers={"Idempotency-Key": "assembled-crash"},
            )
            assert response.status_code == 200, response.text
            assert response.json()["plan_id"] == planned["plan_id"]
            assert response.json()["run_id"] == run_id
            response = await client.post(
                "/api/runs", json={"plan_id": planned["plan_id"]}
            )
            assert response.status_code == 202, response.text
            response = await client.post(f"/api/runs/{run_id}/cancel")
            assert response.status_code == 202, response.text
            for _ in range(2):
                response = await client.post(
                    f"/api/runs/{run_id}/events",
                    json={"attempt": 1, "events": [asdict(hello(run_id, 1))]},
                )
                assert response.status_code == 200, response.text
                assert response.json() == {"attempt": 1, "seq": 0}
            assert durable_snapshot(database, run_id) == replayed
            assert replayed["plans"] == before["plans"]
            assert replayed["intent"] == before["intent"]
            assert replayed["executions"] == before["executions"]
            assert backend.submissions == 0
        assert not services.host._thread.is_alive()
        assert services.host.outstanding == 0

    asyncio.run(restart())
    assert (tmp_path / "submissions.jsonl").read_bytes() == audit
    attempts = tmp_path / "runs" / run_id / "attempts"
    assert [entry.name for entry in attempts.iterdir()] == ["1"]
