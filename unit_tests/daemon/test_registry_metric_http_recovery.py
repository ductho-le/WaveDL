"""F04 metric-bearing transaction SIGKILL matrix; additive to hello-only proof."""

import asyncio
import json
import os
import signal
import stat
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from tools.verify_registry_http_recovery_report import LIMITATION
from tools.verify_registry_metric_http_recovery_report import (
    METRIC,
    snapshot,
    verify_report,
)
from unit_tests.daemon.test_hosted_app import TOKEN, Backend
from unit_tests.daemon.test_registry_http_recovery import MARKER, _client
from wavedl.core.persistence import write_once
from wavedl.core.protocol import Envelope
from wavedl_daemon.backends.contracts import BackendObservation, ObservationStatus
from wavedl_daemon.http.app import create_app
from wavedl_daemon.runs.runtime_service import HostedServices
from wavedl_daemon.storage.registry import Registry
from wavedl_daemon.storage.settings import Settings


pytestmark = pytest.mark.skipif(os.name != "posix", reason="F04 requires POSIX SIGKILL")


def _snapshot(registry, run_id, attempt):
    record = registry.get(run_id)
    rows = registry._connection.execute(
        "SELECT event_json FROM event_log WHERE run_id=? ORDER BY attempt,seq",
        (run_id,),
    ).fetchall()
    events = [json.loads(row[0]) for row in rows]
    return {
        "event_count": record.event_count,
        "cursor": registry.cursor(run_id, attempt),
        "last_type": record.last_type,
        "state": record.state,
        "events": [{"type": row["type"], "seq": row["seq"]} for row in events],
        "history": [asdict(row) for row in registry.metrics(run_id, active_only=False)],
    }


class _BoundedReplay:
    """Disconnect after two genuine replay frames, without replacing response bytes."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope.get("method") != "GET"
            or not scope["path"].endswith("/events")
        ):
            return await self.app(scope, receive, send)
        disconnected = asyncio.Event()
        initial = True
        frames = 0

        async def client_receive():
            nonlocal initial
            if initial:
                initial = False
                return await receive()
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def client_send(message):
            nonlocal frames
            if disconnected.is_set():
                return
            if message["type"] == "http.response.body":
                frames += message.get("body", b"").count(b"\ndata: ")
                if frames >= 2:
                    await send({**message, "more_body": False})
                    disconnected.set()
                    return
            await send(message)

        await self.app(scope, client_receive, client_send)


@pytest.mark.parametrize("order", ["projection-cursor", "cursor-projection"])
@pytest.mark.parametrize("recovery", ["rebuild", "ingest"])
def test_metric_transaction_sigkill_preserves_prefix_and_recovers_once(
    tmp_path, paused_process, order, recovery
):
    script = (
        "import sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(Path.cwd())!r}); "
        f"sys.path.insert(0, {str(Path('src').resolve())!r}); "
        "from unit_tests.daemon.test_registry_http_recovery import _fault_victim; "
        f"_fault_victim(Path({str(tmp_path)!r}), {order!r}, metric=True)"
    )
    with paused_process(script, MARKER) as victim:
        victim.kill()
        victim.wait(timeout=10)
        assert victim.returncode == -signal.SIGKILL
        child_returncode, child_reaped = victim.returncode, victim.poll() is not None
    identity = json.loads((tmp_path / "identity.json").read_text())
    hello, metric = Envelope(**identity["hello"]), Envelope(**identity["event"])
    run_id, attempt = metric.run_id, metric.attempt
    assert hello.seq == 0 and metric.seq == 1 and metric.payload == METRIC
    path = tmp_path / "state/registry.db"
    with Registry(path) as registry:
        rollback = _snapshot(registry, run_id, attempt)
        assert rollback == snapshot(run_id, attempt, recovered=False)
        committed = registry._connection.execute(
            "SELECT event_json FROM event_log WHERE run_id=?", (run_id,)
        ).fetchall()
        assert [json.loads(row[0]) for row in committed] == [asdict(hello)]

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
        await services.host.start()

        async def idle_pump():
            await asyncio.Event().wait()

        await services.host.call(lambda session: setattr(session, "_pump", idle_pump))
        async with (
            app.router.lifespan_context(app),
            _client(_BoundedReplay(app)) as client,
        ):
            denied = await client.get(
                f"/api/runs/{run_id}/events",
                headers={"Authorization": "Bearer invalid"},
            )
            assert denied.status_code == 401
            before = await services.host.call(
                lambda session: _snapshot(session.registry, run_id, attempt)
            )
            assert before == rollback
            if recovery == "rebuild":
                await services.host.call(
                    lambda session: session.registry.rebuild_from_journals(
                        settings.run_root
                    )
                )
            views, histories, snapshots = [], [], []
            for _ in range(3):
                await services.host.call(
                    lambda session: session.ingestor.poll_once(
                        identity["plan_id"],
                        run_id,
                        attempt,
                        settings.run_root
                        / run_id
                        / "attempts"
                        / str(attempt)
                        / "events.jsonl",
                    )
                )
                view = await client.get(f"/api/runs/{run_id}")
                assert view.status_code == 200
                assert (
                    view.json()["state"] == "running"
                    and view.json()["attempt"] == attempt
                )
                views.append(view.json())
                response = await asyncio.wait_for(
                    client.get(f"/api/runs/{run_id}/events"), timeout=5
                )
                assert response.status_code == 200
                events = [
                    json.loads(line[6:])
                    for line in response.text.splitlines()
                    if line.startswith("data: ")
                ]
                assert events == [asdict(hello), asdict(metric)]
                histories.append(events)
                assert await services.event_log.read_after(run_id) == [hello, metric]
                snapshots.append(
                    await services.host.call(
                        lambda session: _snapshot(session.registry, run_id, attempt)
                    )
                )
            assert views == [views[0]] * 3
            assert histories == [histories[0]] * 3
            assert snapshots == [snapshot(run_id, attempt, recovered=True)] * 3
            return (
                before,
                snapshots[0],
                histories[0][1]["payload"],
                response.status_code,
                denied.status_code,
            )

    before, recovered, http_metric, auth_status, denied_status = asyncio.run(recover())
    host_closed, outstanding = (
        not services.host._thread.is_alive(),
        services.host.outstanding,
    )
    assert host_closed and outstanding == 0
    with Registry(path) as registry:
        reopened = _snapshot(registry, run_id, attempt)
        assert reopened == recovered
        rows = registry._connection.execute(
            "SELECT event_json FROM event_log WHERE run_id=? ORDER BY attempt,seq",
            (run_id,),
        ).fetchall()
        assert [json.loads(row[0]) for row in rows] == [asdict(hello), asdict(metric)]
    evidence = {
        "schema_version": 1,
        "case": "registry-metric-http-recovery",
        "order": order,
        "recovery": recovery,
        "run_id": run_id,
        "attempt": attempt,
        "rollback": rollback,
        "before_recovery": before,
        "recovered": recovered,
        "reopened": reopened,
        "http_metric": http_metric,
        "hello_preserved": True,
        "authenticated_status": auth_status,
        "unauthenticated_status": denied_status,
        "stable_polls": 3,
        "child_reaped": child_reaped,
        "child_returncode": child_returncode,
        "kill_platform": os.name,
        "host_closed": host_closed,
        "host_outstanding": outstanding,
        "limitation": LIMITATION,
    }
    report_root = Path(
        os.environ.get("WAVEDL_REGISTRY_METRIC_RECOVERY_REPORT_DIR", tmp_path)
    )
    report_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = report_root.lstat()
    assert stat.S_ISDIR(info.st_mode) and not info.st_mode & 0o077
    assert info.st_uid == os.getuid()
    report = report_root / f"registry-metric-http-recovery-{order}-{recovery}.json"
    write_once(report, (json.dumps(evidence, indent=2) + "\n").encode())
    verify_report(report)


def _valid_report():
    run_id = "00000000-0000-4000-8000-00000000f004"
    report = {
        "schema_version": 1,
        "case": "registry-metric-http-recovery",
        "order": "projection-cursor",
        "recovery": "ingest",
        "run_id": run_id,
        "attempt": 1,
        "rollback": snapshot(run_id, 1, recovered=False),
        "before_recovery": snapshot(run_id, 1, recovered=False),
        "recovered": snapshot(run_id, 1, recovered=True),
        "reopened": snapshot(run_id, 1, recovered=True),
        "http_metric": METRIC,
        "hello_preserved": True,
        "authenticated_status": 200,
        "unauthenticated_status": 401,
        "stable_polls": 3,
        "child_reaped": True,
        "child_returncode": -signal.SIGKILL,
        "kill_platform": "posix",
        "host_closed": True,
        "host_outstanding": 0,
        "limitation": LIMITATION,
    }
    return deepcopy(report)


@pytest.mark.parametrize(
    "keys,value",
    [
        (("rollback", "cursor"), 1),
        (("rollback", "event_count"), 0),
        (("rollback", "events"), []),
        (("before_recovery", "history"), [{}]),
        (("recovered", "history", 0, "value"), None),
        (("reopened", "history", 0, "active"), 1),
        (("reopened", "history", 0, "seq"), True),
        (("http_metric", "metrics", "epoch"), 7.0),
        (("http_metric", "metrics", "mae_per_param", 0), True),
        (("http_metric", "metrics", "val_loss"), 0.0),
        (("hello_preserved",), False),
        (("stable_polls",), 2),
        (("child_returncode",), 0),
        (("child_reaped",), False),
        (("host_closed",), False),
        (("host_outstanding",), True),
        (("attempt",), True),
        (("order",), "unknown"),
        (("recovery",), "unknown"),
        (("case",), "registry-http-recovery"),
        (("unauthenticated_status",), 200),
        (("unexpected",), 1),
    ],
)
def test_metric_report_rejects_incomplete_or_wrongly_typed_evidence(
    tmp_path, keys, value
):
    report = _valid_report()
    target = report
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    path = tmp_path / "metric.json"
    write_once(path, json.dumps(report).encode())
    with pytest.raises(ValueError):
        verify_report(path)


def test_metric_guard_cli_requires_exact_four_case_matrix(tmp_path):
    paths = []
    for order in ("projection-cursor", "cursor-projection"):
        for recovery in ("ingest", "rebuild"):
            report = _valid_report() | {"order": order, "recovery": recovery}
            path = tmp_path / f"{order}-{recovery}.json"
            write_once(path, json.dumps(report).encode())
            verify_report(path)
            paths.append(str(path))
    command = [sys.executable, "tools/verify_registry_metric_http_recovery_report.py"]
    valid = subprocess.run(command + paths, capture_output=True, text=True, timeout=10)
    assert valid.returncode == 0, valid.stderr
    for args in (paths[:-1], paths[:-1] + [paths[0]]):
        invalid = subprocess.run(
            command + args, capture_output=True, text=True, timeout=10
        )
        assert invalid.returncode != 0


@pytest.mark.parametrize(
    "content",
    [b"", b"{", b'{"attempt":1,"attempt":2}', b"{}", b" " * 65537],
    ids=["empty", "torn", "duplicate-key", "missing-fields", "oversized"],
)
def test_metric_guard_refuses_malformed_or_oversized_report(tmp_path, content):
    path = tmp_path / "metric.json"
    write_once(path, content)
    with pytest.raises(ValueError):
        verify_report(path)


def test_metric_guard_reuses_private_no_follow_reader(tmp_path):
    path = tmp_path / "metric.json"
    write_once(path, json.dumps(_valid_report()).encode())
    linked = tmp_path / "linked.json"
    linked.symlink_to(path)
    with pytest.raises(ValueError):
        verify_report(linked)
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        verify_report(path)
