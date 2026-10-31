"""F04 local assembled HTTP projection/cursor crash-and-recovery acceptance."""

import asyncio
import json
import os
import signal
import stat
import sys
import time
from dataclasses import asdict
from pathlib import Path

import httpx
import pytest

from tools.verify_registry_http_recovery_report import LIMITATION, verify_report
from unit_tests.daemon.test_hosted_app import AUTH, TOKEN, Backend
from wavedl.core.events import HelloPayload, MetricPayload
from wavedl.core.journal_writer import JournalWriter
from wavedl.core.layout import RunLayout
from wavedl.core.persistence import write_once
from wavedl.core.protocol import Envelope
from wavedl_daemon.backends.contracts import BackendObservation, ObservationStatus
from wavedl_daemon.http.app import create_app
from wavedl_daemon.runs.runtime_service import HostedServices
from wavedl_daemon.storage.registry import Registry
from wavedl_daemon.storage.settings import Settings


MARKER = "F04 transaction barrier reached"
pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="F04 requires POSIX SIGKILL, no-follow reads, and private report modes",
)


async def _wait_for(callback):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if await callback():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Hosted runtime binding timed out")


def _client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://localhost:8420",
        headers=AUTH,
    )


def _fault_victim(root, order, *, metric=False):
    settings = Settings(root / "state", root / "runs")
    backend = Backend()
    services = HostedServices(settings, backend_factory=lambda lock: {"local": backend})
    app = create_app(
        settings, token=TOKEN, daemon_id="f04", build_id="test", services=services
    )

    async def run():
        async with app.router.lifespan_context(app), _client(app) as client:
            planned = await client.post(
                "/api/runs/plan",
                headers={"Idempotency-Key": "f04-crash"},
                json={
                    "backend": "local",
                    "config": {"epochs": 1},
                    "runtime": {
                        "python": sys.executable,
                        "mode": "existing",
                        "device": "cpu",
                        "environment": {},
                    },
                    "operation": "train",
                    "topology": {
                        "num_machines": 1,
                        "processes_per_machine": 1,
                        "machine_rank": 0,
                    },
                },
            )
            assert planned.status_code == 200, planned.text
            plan = planned.json()
            submitted = await client.post(
                "/api/runs", json={"plan_id": plan["plan_id"]}
            )
            assert submitted.status_code == 202, submitted.text

            async def submitted_backend():
                return backend.submissions

            await _wait_for(submitted_backend)
            binding = backend.runtime_startup()

            async def bound():
                view = await client.get(f"/api/runs/{plan['run_id']}")
                return view.json()["attempt"] == binding.attempt

            await _wait_for(bound)

            # Freeze the background reader after binding so only the following
            # authenticated HTTP push can reach the armed transaction barrier.
            async def stop_pump(session):
                session._pump_task.cancel()
                try:
                    await session._pump_task
                except asyncio.CancelledError:
                    pass

            await services.host.call(stop_pump)
            event = Envelope(
                plan["run_id"],
                binding.attempt,
                "hello",
                0,
                "2026-10-14T00:00:00Z",
                asdict(
                    HelloPayload(
                        1, 1, "f04-runtime", "3.11", "cpu", ["train"], sys.executable
                    )
                ),
            )
            hello = None
            if metric:
                with JournalWriter(
                    RunLayout(settings.run_root / plan["run_id"], plan["run_id"]),
                    attempt=binding.attempt,
                    fsync_every=1,
                ) as writer:
                    hello = writer.append("hello", event.payload)
                    committed = await client.post(
                        f"/api/runs/{plan['run_id']}/events",
                        json={"attempt": binding.attempt, "events": [asdict(hello)]},
                    )
                    assert committed.status_code == 200, committed.text
                    assert committed.json() == {"attempt": binding.attempt, "seq": 0}
                    event = writer.append(
                        "metric",
                        asdict(
                            MetricPayload(
                                "loss",
                                0.375,
                                7,
                                "mse",
                                {
                                    "epoch": 7,
                                    "train_loss": 0.375,
                                    "val_loss": None,
                                    "mae_per_param": [0.125, None],
                                },
                            )
                        ),
                    )
            prior_count = 1 if metric else 0
            original_record = Registry._write_record

            def barrier():
                print(MARKER, flush=True)
                time.sleep(300)

            if order == "projection-cursor":

                def cursor(self, run_id, attempt, seq):
                    assert self._connection.in_transaction
                    assert self.get(run_id).event_count == prior_count + 1
                    assert self.cursor(run_id, attempt) == prior_count - 1
                    assert (
                        self._connection.execute(
                            "SELECT COUNT(*) FROM event_log WHERE run_id=?", (run_id,)
                        ).fetchone()[0]
                        == prior_count + 1
                    )
                    if metric:
                        assert len(self.metrics(run_id)) == 1
                    barrier()

                Registry._write_cursor = cursor
            else:

                def record(self, projected):
                    assert self._connection.in_transaction
                    self._write_cursor(projected.run_id, projected.attempt, prior_count)
                    assert (
                        self.cursor(projected.run_id, projected.attempt) == prior_count
                    )
                    prior = self.get(projected.run_id)
                    if metric:
                        assert prior.event_count == 1
                    else:
                        assert prior is None
                    assert (
                        self._connection.execute(
                            "SELECT COUNT(*) FROM event_log WHERE run_id=?",
                            (projected.run_id,),
                        ).fetchone()[0]
                        == prior_count + 1
                    )
                    if metric:
                        assert len(self.metrics(projected.run_id)) == 1
                    barrier()
                    original_record(self, projected)

                Registry._write_record = record
            journal = (
                settings.run_root
                / plan["run_id"]
                / "attempts"
                / str(binding.attempt)
                / "events.jsonl"
            )
            if not metric:
                with JournalWriter(
                    RunLayout(settings.run_root / plan["run_id"], plan["run_id"]),
                    attempt=binding.attempt,
                    fsync_every=1,
                ) as writer:
                    event = writer.append("hello", event.payload)
            assert journal.is_file()
            write_once(
                root / "identity.json",
                json.dumps(
                    {
                        "run_id": plan["run_id"],
                        "plan_id": plan["plan_id"],
                        "event": asdict(event),
                        **({"hello": asdict(hello)} if metric else {}),
                    }
                ).encode(),
            )
            await client.post(
                f"/api/runs/{plan['run_id']}/events",
                json={"attempt": binding.attempt, "events": [asdict(event)]},
            )

    asyncio.run(run())


@pytest.mark.parametrize("order", ["projection-cursor", "cursor-projection"])
@pytest.mark.parametrize("recovery", ["rebuild", "ingest"])
def test_registry_http_crash_rebuild_replays_exactly_once(
    tmp_path, paused_process, order, recovery
):
    """Kill real owner transaction, reopen same DB, rebuild and poll exactly once."""
    script = (
        "import sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(Path.cwd())!r}); "
        f"sys.path.insert(0, {str(Path('src').resolve())!r}); "
        "from unit_tests.daemon.test_registry_http_recovery import _fault_victim; "
        f"_fault_victim(Path({str(tmp_path)!r}), {order!r})"
    )
    with paused_process(script, MARKER) as victim:
        victim.kill()
        victim.wait(timeout=10)
        expected_returncode = -signal.SIGKILL if os.name == "posix" else 1
        assert victim.returncode == expected_returncode
        child_returncode = victim.returncode
        child_reaped = victim.poll() is not None
    identity = json.loads((tmp_path / "identity.json").read_text())
    run_id, event = identity["run_id"], Envelope(**identity["event"])
    path = tmp_path / "state" / "registry.db"
    with Registry(path) as registry:
        rolled_record = registry.get(run_id)
        rollback = {
            "event_count": 0 if rolled_record is None else rolled_record.event_count,
            "cursor": registry.cursor(run_id, event.attempt),
            "replay_rows": registry._connection.execute(
                "SELECT COUNT(*) FROM event_log WHERE run_id=?", (run_id,)
            ).fetchone()[0],
        }
        assert rolled_record is None
        assert rollback == {"event_count": 0, "cursor": -1, "replay_rows": 0}
    settings = Settings(tmp_path / "state", tmp_path / "runs")

    class RecoveryBackend(Backend):
        def observe(self, handle):
            return BackendObservation(handle.attempt, ObservationStatus.RUNNING)

    services = HostedServices(
        settings, backend_factory=lambda lock: {"local": RecoveryBackend()}
    )
    app = create_app(
        settings, token=TOKEN, daemon_id="f04", build_id="test", services=services
    )

    async def recover():
        # Pin the hosted owner session before lifespan startup. Gate only the
        # background scheduling loop, leaving actual poll_once/read/fold real.
        await services.host.start()

        async def idle_pump():
            await asyncio.Event().wait()

        await services.host.call(lambda session: setattr(session, "_pump", idle_pump))
        async with app.router.lifespan_context(app), _client(app) as client:
            denied = await client.get(
                f"/api/runs/{run_id}", headers={"Authorization": "Bearer invalid"}
            )
            assert denied.status_code == 401
            before = await services.host.call(
                lambda session: {
                    "event_count": 0
                    if session.registry.get(run_id) is None
                    else session.registry.get(run_id).event_count,
                    "cursor": session.registry.cursor(run_id, event.attempt),
                    "replay_rows": len(session.event_log.read_after(run_id)),
                }
            )
            assert before == rollback
            if recovery == "rebuild":
                await services.host.call(
                    lambda session: session.registry.rebuild_from_journals(
                        settings.run_root
                    )
                )
            snapshots = []
            for _ in range(3):
                await services.host.call(
                    lambda session: session.ingestor.poll_once(
                        identity["plan_id"],
                        run_id,
                        event.attempt,
                        settings.run_root
                        / run_id
                        / "attempts"
                        / str(event.attempt)
                        / "events.jsonl",
                    )
                )
                response = await client.get(f"/api/runs/{run_id}")
                assert response.status_code == 200, response.text
                snapshots.append(response.json())
                assert await services.event_log.read_after(run_id) == [event]
            assert snapshots == [snapshots[0]] * 3
            assert snapshots[0]["attempt"] == event.attempt
            assert snapshots[0]["state"] == "running"
            observed = await services.host.call(
                lambda session: {
                    "event_count": session.registry.get(run_id).event_count,
                    "cursor": session.registry.cursor(run_id, event.attempt),
                    "replay_rows": session.registry._connection.execute(
                        "SELECT COUNT(*) FROM event_log WHERE run_id=?", (run_id,)
                    ).fetchone()[0],
                }
            )
            assert observed == {"event_count": 1, "cursor": 0, "replay_rows": 1}
            return (
                observed,
                denied.status_code,
                response.status_code,
                len(snapshots),
                before,
            )

    observed, denied_status, auth_status, polls, before = asyncio.run(recover())
    outstanding = services.host.outstanding
    host_closed = not services.host._thread.is_alive()
    assert outstanding == 0
    assert host_closed
    evidence = {
        "schema_version": 1,
        "case": "registry-http-recovery",
        "order": order,
        "recovery": recovery,
        "run_id": run_id,
        **observed,
        "rollback": rollback,
        "before_recovery": before,
        "authenticated_status": auth_status,
        "unauthenticated_status": denied_status,
        "stable_polls": polls,
        "child_reaped": child_reaped,
        "child_returncode": child_returncode,
        "kill_platform": os.name,
        "host_closed": host_closed,
        "host_outstanding": outstanding,
        "limitation": LIMITATION,
    }
    report_root = Path(os.environ.get("WAVEDL_REGISTRY_RECOVERY_REPORT_DIR", tmp_path))
    report_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    report_info = report_root.lstat()
    assert stat.S_ISDIR(report_info.st_mode)
    assert not report_root.is_symlink()
    assert not report_info.st_mode & 0o077
    if hasattr(os, "getuid"):
        assert report_info.st_uid == os.getuid()
    report = report_root / f"registry-http-recovery-{order}-{recovery}.json"
    write_once(report, (json.dumps(evidence, indent=2) + "\n").encode())
    verify_report(report)


@pytest.mark.parametrize(
    "changed",
    [
        {"event_count": 2},
        {"cursor": -1},
        {"replay_rows": 0},
        {"stable_polls": 2},
        {"host_closed": False},
        {"child_reaped": False},
        {"authenticated_status": 401},
        {"unauthenticated_status": 200},
        {"order": "unknown"},
        {"event_count": True},
        {"limitation": "remote production acceptance"},
        {"unexpected": 0},
        {"rollback": {"event_count": False, "cursor": -1, "replay_rows": 0}},
        {"rollback": {"event_count": 0, "cursor": -1, "replay_rows": False}},
        {"before_recovery": {"event_count": False, "cursor": -1, "replay_rows": 0}},
        {"run_id": "../unsafe"},
        {"run_id": 0},
        {"host_outstanding": True},
        {"child_returncode": 0},
        {"kill_platform": "nt", "child_returncode": 1},
    ],
)
def test_registry_http_report_refuses_incomplete_evidence(tmp_path, changed):
    """The evidence gate must not accept missing or weakened acceptance."""
    report = {
        "schema_version": 1,
        "case": "registry-http-recovery",
        "order": "projection-cursor",
        "recovery": "ingest",
        "run_id": "00000000-0000-4000-8000-00000000f004",
        "event_count": 1,
        "cursor": 0,
        "replay_rows": 1,
        "rollback": {"event_count": 0, "cursor": -1, "replay_rows": 0},
        "before_recovery": {"event_count": 0, "cursor": -1, "replay_rows": 0},
        "authenticated_status": 200,
        "unauthenticated_status": 401,
        "stable_polls": 3,
        "child_reaped": True,
        "host_closed": True,
        "host_outstanding": 0,
        "child_returncode": -signal.SIGKILL if os.name == "posix" else 1,
        "kill_platform": os.name,
        "limitation": LIMITATION,
    } | changed
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(ValueError):
        verify_report(path)


def test_registry_http_report_refuses_oversized_file(tmp_path):
    path = tmp_path / "oversized.json"
    path.write_bytes(b" " * (64 * 1024 + 1))
    with pytest.raises(ValueError, match="bounded"):
        verify_report(path)


@pytest.mark.parametrize("content", [b"", b"{", b'{"case":1,"case":2}', b"{}"])
def test_registry_http_report_refuses_invalid_json(tmp_path, content):
    path = tmp_path / "invalid.json"
    path.write_bytes(content)
    path.chmod(0o600)
    with pytest.raises(ValueError):
        verify_report(path)


def test_registry_http_report_refuses_symlink(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("{}")
    target.chmod(0o600)
    path = tmp_path / "alias.json"
    path.symlink_to(target)
    with pytest.raises(ValueError):
        verify_report(path)


def test_registry_http_report_refuses_symlink_ancestor(tmp_path):
    target = tmp_path / "reports"
    target.mkdir()
    report = target / "report.json"
    report.write_text("{}")
    report.chmod(0o600)
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="ancestor"):
        verify_report(alias / "report.json")


def test_registry_http_report_public_file_is_refused(tmp_path):
    path = tmp_path / "public.json"
    path.write_text("{}")
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        verify_report(path)


def test_registry_http_report_publication_never_overwrites(tmp_path):
    path = tmp_path / "existing.json"
    write_once(path, b"original")
    with pytest.raises(FileExistsError):
        write_once(path, b"replacement")
    assert path.read_bytes() == b"original"
