"""Assembled ASGI/storage crash acceptance with a stub Local supervisor.

Planning, submission, cancellation, the run log, the controller, the metrics
tail and replay are real. The supervisor is a test double that never runs an
interpreter, and the daemon dies by SIGKILL, not by a clean shutdown.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

from wavedl_daemon.backends.local import LocalBackend
from wavedl_daemon.http.app import create_app
from wavedl_daemon.runs.runtime_service import HostedServices
from wavedl_daemon.storage.settings import Settings
from wavedl_daemon.supervision.base import ProcessResult


TOKEN = "assembled-crash-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
SPAWNED = "training-spawned"
CLAIMED = "attempt-claimed"


class StubProcess:
    """An interpreter that never exits unless cancelled."""

    def __init__(self, pid, result=None):
        self.pid = pid
        self._done = threading.Event()
        self._result = result
        if result is not None:
            self._done.set()

    def cancel(self, grace_seconds=5, *, interrupt=False):
        self._result = ProcessResult(130 if interrupt else 143, b"", b"", cancelled=True)
        self._done.set()

    def wait(self, timeout=None):
        self._done.wait(timeout)
        return self._result


class StubSupervisor:
    """Answers the interpreter probe; audits each training spawn durably.

    ``block`` stalls the training spawn after the attempt claim, as a daemon
    that dies between the claim and recording its result would.
    """

    def __init__(self, audit, *, block=False):
        self.audit, self.block = audit, block

    def start(self, spec, *, output_limit=1048576):
        if spec.argv[1] == "-c":
            probe = {"interpreter": sys.executable, "prefix": sys.prefix, "devices": ["cpu"]}
            return StubProcess(1, ProcessResult(0, json.dumps(probe).encode(), b""))
        with self.audit.open("a") as stream:
            stream.write(json.dumps(spec.argv) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if self.block:
            print(CLAIMED, flush=True)
            time.sleep(300)
        return StubProcess(4242)


def payload():
    return {
        "backend": "local",
        "operation": "train",
        "config": {"epochs": 3},
        "runtime": {
            "python": sys.executable,
            "mode": "existing",
            "device": "cpu",
            "environment": {},
        },
        "topology": {"num_machines": 1, "processes_per_machine": 1, "machine_rank": 0},
    }


def line(epoch):
    return json.dumps({"epoch": epoch, "train_loss": 1.0 / epoch, "val_loss": float("nan")})


def assembled(root, supervisor):
    settings = Settings(root / "state", root / "runs")
    services = HostedServices(
        settings,
        backend_factory=lambda lock: {
            "local": LocalBackend(dispatch_lock=lock, supervisor=supervisor, max_processes=1)
        },
    )
    app = create_app(
        settings, token=TOKEN, daemon_id="assembled", build_id="test", services=services
    )
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://localhost:8420",
        headers=AUTH,
    )
    return settings, services, app, client


async def until(read, accept, *, timeout=10):
    deadline = time.monotonic() + timeout
    while True:
        value = await read()
        if accept(value):
            return value
        if time.monotonic() > deadline:
            raise AssertionError(f"Timed out at {value!r}")
        await asyncio.sleep(0.02)


async def plan_and_submit(client, key):
    response = await client.post(
        "/api/runs/plan", json=payload(), headers={"Idempotency-Key": key}
    )
    assert response.status_code == 200, response.text
    planned = response.json()
    response = await client.post("/api/runs", json={"plan_id": planned["plan_id"]})
    assert response.status_code == 202, response.text
    return planned


async def view(client, run_id):
    response = await client.get(f"/api/runs/{run_id}")
    assert response.status_code == 200, response.text
    return response.json()


async def crash_child(root, mode):
    """Leave durable state behind, then wait to be killed."""
    root = Path(root)
    _, services, app, client = assembled(
        root, StubSupervisor(root / "spawns.jsonl", block=mode == "claimed")
    )
    async with app.router.lifespan_context(app), client:
        planned = await plan_and_submit(client, "first")
        (root / "planned.json").write_text(json.dumps(planned))
        if mode == "claimed":
            await asyncio.sleep(300)
        run_id = planned["run_id"]
        await until(lambda: view(client, run_id), lambda run: run["state"] == "running")
        metrics = root / "runs" / run_id / "attempts" / "1" / "metrics.jsonl"
        metrics.write_text(line(1) + "\n" + line(2) + "\n")

        async def committed():
            return await services.event_log.read_after(run_id, (0, -1))

        await until(committed, lambda rows: sum(row.event.kind == "metric" for row in rows) == 2)
        # One Local slot is taken, so this run stays requested until cancelled.
        waiting = await plan_and_submit(client, "second")
        (root / "waiting.json").write_text(json.dumps(waiting))
        response = await client.post(f"/api/runs/{waiting['run_id']}/cancel")
        assert response.status_code == 202, response.text
        await until(
            lambda: view(client, waiting["run_id"]), lambda run: run["state"] == "cancelled"
        )
        print(SPAWNED, flush=True)
        await asyncio.sleep(300)


def kill_child(paused_process, root, mode, marker):
    project = Path(__file__).resolve().parents[2]
    script = (
        f"import sys; sys.path[:0] = {[str(project), str(project / 'src')]!r}; "
        "import asyncio; "
        "from unit_tests.daemon.test_assembled_crash_replay import crash_child; "
        f"asyncio.run(crash_child({str(root)!r}, {mode!r}))"
    )
    with paused_process(script, marker) as child:
        child.kill()
        child.wait(timeout=10)
        assert child.returncode == -signal.SIGKILL


def restarted(root, check):
    """Run ``check(services, client)`` against a fresh daemon on the same state."""

    async def scenario():
        _, services, app, client = assembled(root, StubSupervisor(root / "spawns.jsonl"))
        async with app.router.lifespan_context(app), client:
            await check(services, client)
        assert not services.host._thread.is_alive()
        assert services.host.outstanding == 0

    asyncio.run(scenario())


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL and dispatch ownership")
def test_live_local_run_settles_as_lost_after_draining_its_metrics(tmp_path, paused_process):
    kill_child(paused_process, tmp_path, "live", SPAWNED)
    run_id = json.loads((tmp_path / "planned.json").read_text())["run_id"]
    waiting = json.loads((tmp_path / "waiting.json").read_text())
    spawns = (tmp_path / "spawns.jsonl").read_bytes()
    assert len(spawns.splitlines()) == 1
    metrics = tmp_path / "runs" / run_id / "attempts" / "1" / "metrics.jsonl"
    # The trainer outlived the daemon: one more epoch and a write cut short.
    with metrics.open("a") as stream:
        stream.write(line(3) + "\n" + '{"epoch": 4, "train_lo')

    async def check(services, client):
        run = await until(lambda: view(client, run_id), lambda run: run["state"] != "running")
        assert run["state"] == "supervision_lost"
        assert run["uncertainty"] is True
        assert run["evidence"]["outcome"] == "lost"
        rows = await services.event_log.read_after(run_id, (0, -1))
        kinds = [row.event.kind for row in rows]
        assert kinds == [
            "attempt", "submitting", "probed", "submitted", "observed",
            "metric", "metric", "metric", "warning", "exited",
        ]
        assert [(row.attempt, row.seq) for row in rows] == [(1, seq) for seq in range(10)]
        assert [row.event.epoch for row in rows if row.event.kind == "metric"] == [1, 2, 3]
        assert rows[5].event.stats["val_loss"] is None
        assert rows[8].event.code.value == "metrics_torn_tail"
        head = await services.host.call(lambda session: session.log.head(run_id))
        assert head.metrics_offset == metrics.stat().st_size
        assert head.metrics_sealed

        # The run cancelled while it waited for a slot never started.
        other = await view(client, waiting["run_id"])
        assert other["state"] == "cancelled" and other["uncertainty"] is False
        other_rows = await services.event_log.read_after(waiting["run_id"], (0, -1))
        assert [row.event.kind for row in other_rows] == [
            "attempt", "cancel_requested", "exited",
        ]

        # A replayed plan, submit or cancel after the crash changes nothing.
        response = await client.post(
            "/api/runs/plan", json=payload(), headers={"Idempotency-Key": "first"}
        )
        assert response.json()["run_id"] == run_id
        assert (await client.post("/api/runs", json={"plan_id": run["plan_id"]})).status_code == 202
        assert (await client.post(f"/api/runs/{run_id}/cancel")).status_code == 202
        await asyncio.sleep(0.3)
        assert await services.event_log.read_after(run_id, (0, -1)) == rows

    restarted(tmp_path, check)
    assert (tmp_path / "spawns.jsonl").read_bytes() == spawns
    assert not (tmp_path / "runs" / waiting["run_id"]).exists()


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL and dispatch ownership")
def test_claimed_run_is_recovered_as_lost_and_never_spawned_twice(tmp_path, paused_process):
    kill_child(paused_process, tmp_path, "claimed", CLAIMED)
    run_id = json.loads((tmp_path / "planned.json").read_text())["run_id"]
    spawns = (tmp_path / "spawns.jsonl").read_bytes()
    assert len(spawns.splitlines()) == 1

    async def check(services, client):
        run = await until(lambda: view(client, run_id), lambda run: run["state"] != "submission_unknown")
        assert run["state"] == "supervision_lost"
        assert run["evidence"]["outcome"] == "lost"
        rows = await services.event_log.read_after(run_id, (0, -1))
        assert [row.event.kind for row in rows] == ["attempt", "submitting", "exited"]
        assert rows[-1].event.detail == "The daemon stopped before the launch was recorded"

    restarted(tmp_path, check)
    assert (tmp_path / "spawns.jsonl").read_bytes() == spawns
    attempts = tmp_path / "runs" / run_id / "attempts"
    assert [entry.name for entry in attempts.iterdir()] == ["1"]
