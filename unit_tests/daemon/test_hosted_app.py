"""The assembled HTTP app runs Local runs on the run log, with stub processes."""

import asyncio
import itertools
import json
import os
import sqlite3
import sys
import threading
import time

import httpx
import pytest

from wavedl_daemon.backends.local import LocalBackend
from wavedl_daemon.execution.budget import StorageWorkError
from wavedl_daemon.http.app import create_app
from wavedl_daemon.runs.runtime_service import HostedServices
from wavedl_daemon.storage.database import UnsupportedDatabase
from wavedl_daemon.storage.settings import Settings
from wavedl_daemon.supervision.base import ProcessResult


TOKEN = "hosted-test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class StubProcess:
    """A supervised process the test finishes; cancel exits like the launcher."""

    def __init__(self, pid, spec, result=None):
        self.pid = pid
        self.spec = spec
        self.cancels = []
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._result = None
        self._error = None
        if result is not None:
            self._finish(result)

    def _finish(self, result):
        with self._lock:
            if self._result is None:
                self._result = result
                self._done.set()

    def finish(self, returncode):
        self._finish(ProcessResult(returncode, b"", b""))

    def fail(self, cause):
        """End supervision without exit evidence, as POSIXProcess.wait does."""
        with self._lock:
            if self._result is None:
                self._error = cause
                self._done.set()

    def cancel(self, grace_seconds=5, *, interrupt=False):
        self.cancels.append((grace_seconds, interrupt))
        self._finish(ProcessResult(130 if interrupt else 143, b"", b"", cancelled=True))

    def wait(self, timeout=None):
        if not self._done.wait(timeout):
            raise RuntimeError("stub process still running")
        if self._error is not None:
            raise RuntimeError("process supervision failed") from self._error
        return self._result


class StubSupervisor:
    """Answers the interpreter probe and records every launched command."""

    def __init__(self, *, probe_returncode=0):
        self.probe_returncode = probe_returncode
        self.probes = []
        self.launches = []
        self._pids = itertools.count(4000)

    def start(self, spec, *, output_limit=1048576):
        pid = next(self._pids)
        if spec.argv[1] == "-c":
            self.probes.append(spec)
            identity = {"interpreter": sys.executable, "prefix": sys.prefix, "devices": ["cpu"]}
            return StubProcess(
                pid,
                spec,
                ProcessResult(self.probe_returncode, json.dumps(identity).encode(), b""),
            )
        process = StubProcess(pid, spec)
        self.launches.append(process)
        return process


def stub_backends(supervisor):
    return lambda lock: {"local": LocalBackend(dispatch_lock=lock, supervisor=supervisor)}


def train_payload(**config):
    return {
        "backend": "local",
        "operation": "train",
        "config": {"epochs": 1, **config},
        "runtime": {
            "python": sys.executable,
            "mode": "existing",
            "device": "cpu",
            "environment": {},
        },
        "topology": {"num_machines": 1, "processes_per_machine": 1, "machine_rank": 0},
    }


async def until(check, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if result := await check():
            return result
        await asyncio.sleep(0.02)
    pytest.fail("The hosted pump did not make bounded progress")


@pytest.mark.skipif(os.name != "posix", reason="POSIX state-root confidentiality")
def test_public_state_root_is_refused_without_chmod_or_database_write(tmp_path):
    settings = Settings(tmp_path / "state", tmp_path / "runs")
    settings.state_root.mkdir(mode=0o755)
    # A private harness umask must not turn this public-state witness into 0700.
    settings.state_root.chmod(0o755)
    services = HostedServices(settings)

    async def run():
        with pytest.raises(StorageWorkError):
            await services.start()
        await services.close()

    asyncio.run(run())
    assert settings.state_root.stat().st_mode & 0o777 == 0o755
    assert not (settings.state_root / "registry.db").exists()


def test_old_daemon_database_is_refused_and_left_untouched(tmp_path):
    settings = Settings(tmp_path / "state", tmp_path / "runs")
    settings.state_root.mkdir(mode=0o700)
    old = settings.state_root / "registry.db"
    with sqlite3.connect(old) as connection:
        connection.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY)")
        connection.execute("PRAGMA user_version=13")
    connection.close()
    services = HostedServices(settings, backend_factory=stub_backends(StubSupervisor()))

    async def run():
        with pytest.raises(StorageWorkError) as error:
            await services.start()
        await services.close()
        return error.value

    failure = asyncio.run(run())
    assert isinstance(failure.__cause__, UnsupportedDatabase)
    with sqlite3.connect(old) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (13,)
        tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        assert tables.fetchall() == [("runs",)]
    connection.close()


def test_hosted_local_run_streams_metrics_completes_and_cancels(tmp_path):
    settings = Settings(tmp_path / "state", tmp_path / "runs")
    supervisor = StubSupervisor()
    services = HostedServices(settings, backend_factory=stub_backends(supervisor))
    app = create_app(
        settings, token=TOKEN, daemon_id="host", build_id="test", services=services
    )
    assert not settings.state_root.exists()

    async def run():
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                base_url="http://localhost:8420",
                headers=AUTH,
            ) as client,
        ):
            health = await client.get("/api/health")
            assert health.status_code == 200
            assert health.json()["registered_backends"] == ["local"]
            assert health.json()["dispatch_errors"] == []
            await services.host.call(
                lambda session: session._monitor_error("test-source", "BudgetExhausted")
            )
            health = await client.get("/api/health")
            assert {
                "source": "test-source",
                "error": "BudgetExhausted",
            } in health.json()["monitoring_errors"]

            preview = await client.post(
                "/api/runs/plan", json=train_payload(), headers={"Idempotency-Key": "one"}
            )
            assert preview.status_code == 200, preview.text
            planned = preview.json()
            run_id = planned["run_id"]
            run_dir = settings.run_root / run_id
            assert "allocation_mode" not in planned["preview"]
            assert planned["preview"]["spec"]["run_directory"] == str(run_dir)
            assert planned["preview"]["argv"][-2:] == ["--output_dir", str(run_dir / "output")]
            assert not run_dir.exists()
            assert (await client.get(f"/api/runs/{run_id}")).json()["state"] == "planned"

            for _ in range(2):
                submitted = await client.post("/api/runs", json={"plan_id": planned["plan_id"]})
                assert submitted.status_code == 202, submitted.text
            assert submitted.json()["submission_requested"] is True

            async def running():
                view = (await client.get(f"/api/runs/{run_id}")).json()
                return view if view["state"] == "running" else None

            view = await until(running)
            assert view["attempt"] == 1 and view["uncertainty"] is False
            assert view["evidence"] == {
                "reason": "",
                "phase": "running",
                "native_state": None,
                "outcome": None,
                "exit_code": None,
                "signal": None,
                "attempt_exit": None,
                "restart_count": None,
            }
            assert len(supervisor.launches) == 1
            launch = supervisor.launches[0]
            metrics = run_dir / "attempts" / "1" / "metrics.jsonl"
            assert launch.spec.argv[-2:] == ("--metrics_jsonl", str(metrics))
            assert run_dir.stat().st_mode & 0o777 == 0o700

            metrics.write_text(
                json.dumps({"epoch": 1, "train_loss": 1.5, "val_loss": float("nan")}) + "\n"
            )

            async def metric():
                page = await services.event_log.read_after(run_id, (0, -1))
                return [event for event in page if event.event.kind == "metric"]

            [row] = await until(metric)
            assert (row.attempt, row.event.payload()["step"]) == (1, 1)
            assert row.event.payload()["value"] is None
            launch.finish(0)

            async def ended():
                view = (await client.get(f"/api/runs/{run_id}")).json()
                return view if view["state"] == "completed" else None

            view = await until(ended)
            assert view["evidence"]["outcome"] == "completed"
            assert view["evidence"]["exit_code"] == 0
            page = await services.event_log.read_after(run_id, (0, -1))
            assert [event.event.kind for event in page] == [
                "attempt",
                "submitting",
                "probed",
                "submitted",
                "observed",
                "metric",
                "exited",
            ]
            assert [(event.attempt, event.seq) for event in page] == [
                (1, seq) for seq in range(7)
            ]

            second = (
                await client.post(
                    "/api/runs/plan", json=train_payload(), headers={"Idempotency-Key": "two"}
                )
            ).json()
            await client.post("/api/runs", json={"plan_id": second["plan_id"]})

            async def second_running():
                view = (await client.get(f"/api/runs/{second['run_id']}")).json()
                return view if view["state"] == "running" else None

            await until(second_running)
            cancel = await client.post(f"/api/runs/{second['run_id']}/cancel")
            assert cancel.status_code == 202
            assert cancel.json()["cancellation_requested"] is True

            async def cancelled():
                view = (await client.get(f"/api/runs/{second['run_id']}")).json()
                return view if view["state"] == "cancelled" else None

            view = await until(cancelled)
            assert view["uncertainty"] is False
            assert view["evidence"]["exit_code"] == 130
            assert supervisor.launches[1].cancels == [(45.0, True)]
            listed = (await client.get("/api/runs")).json()["runs"]
            assert [item["run_id"] for item in listed] == [run_id, second["run_id"]]

    asyncio.run(run())
    assert services.host.outstanding == 0
    assert not services.host._thread.is_alive()
