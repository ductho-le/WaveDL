"""Abrupt submission-boundary crashes through authenticated HTTP and SQLite.

Controlled scheduler effects are durable outside the daemon. This proves a
local process recovery slice, not native Slurm or real-site HPC support.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
import sqlite3
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from wavedl_daemon.backends.contracts import (
    ObservationStatus,
    SubmissionHandle,
    SubmissionObservation,
)
from wavedl_daemon.backends.slurm_plan import SlurmPlanner, SlurmSite
from wavedl_daemon.http.app import create_app
from wavedl_daemon.runs.runtime_service import HostedServices
from wavedl_daemon.storage.executions import ExecutionStore
from wavedl_daemon.storage.plans import PlanStore
from wavedl_daemon.storage.settings import Settings


TOKEN = "submission-crash-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class DurableBackend:
    """Controlled scheduler whose acceptance survives abrupt daemon death."""

    owner = "crash-owner"

    def __init__(self, root, *, crash=False, recovery="unique"):
        self.root = Path(root)
        self.crash = crash
        self.recovery = recovery
        self.submits = 0
        self.lookups = 0

    def plan(self, manifest, **kwargs):
        return SlurmPlanner(SlurmSite(partition="cpu", cluster="alpha")).plan(
            manifest, **kwargs
        )

    def submit(self, plan, *, request_id, attempt):
        assert attempt is None and plan.allocation_mode == "writer"
        self.submits += 1
        raw = {
            "cluster": "alpha",
            "job_name": f"wavedl-{request_id}",
            "comment": request_id,
            "restart_count": 0,
            "operation": plan.operation.value,
            "allocation_mode": "writer",
            "plan_sha256": hashlib.sha256(plan.to_json().encode()).hexdigest(),
        }
        handle = SubmissionHandle(
            "slurm",
            plan.run_id,
            request_id,
            self.owner,
            plan.run_dir,
            job_id="42;alpha",
            raw_json=json.dumps(raw),
        )
        # Exclusive creation catches any second backend acceptance as well as
        # counting submit calls in each independently constructed backend.
        with (self.root / "accepted.json").open("x") as stream:
            json.dump(handle.to_dict(), stream)
            stream.flush()
            os.fsync(stream.fileno())
        if self.crash:
            print("accepted-before-return", flush=True)
            threading.Event().wait(30)
            raise AssertionError("parent failed to kill the acceptance boundary")
        return handle

    def find_by_request_id(self, plan, *, request_id, attempt):
        self.lookups += 1
        assert attempt is None
        if self.recovery == "missing":
            return None
        if self.recovery == "ambiguous":
            raise ValueError("Multiple jobs match the full request identity")
        receipt = SubmissionHandle.from_dict(
            json.loads((self.root / "accepted.json").read_text())
        )
        assert (receipt.run_id, receipt.request_id) == (plan.run_id, request_id)
        return receipt

    def observe(self, handle):
        return SubmissionObservation(ObservationStatus.RUNNING)

    def discover_initial(self, handle):
        return None


def assembled_app(root, backend):
    settings = Settings(root / "state", root / "runs")
    return create_app(
        settings,
        token=TOKEN,
        daemon_id="crash",
        build_id="test",
        services=HostedServices(
            settings, backend_factory=lambda lock: {"slurm": backend}
        ),
    )


async def crash_child(root, boundary="acceptance"):
    root = Path(root)
    app = assembled_app(root, DurableBackend(root, crash=boundary == "acceptance"))

    def pause_after_commit(original):
        def committed(*args, **kwargs):
            original(*args, **kwargs)
            # These public store calls have exited their transaction contexts.
            # Pause the actual owner thread, without unwinding or graceful close.
            print(f"{boundary}-committed", flush=True)
            threading.Event().wait(30)
            raise AssertionError("parent failed to kill the committed boundary")

        return committed

    target = {
        "intent": (PlanStore, "request_submission"),
        "claim": (ExecutionStore, "claim_submission"),
        "receipt": (ExecutionStore, "attach_submission"),
    }.get(boundary)

    async def transport_app(scope, receive, send):
        async def gated_send(message):
            if (
                boundary == "receipt"
                and scope["path"] == "/api/runs"
                and scope["method"] == "POST"
                and message["type"] == "http.response.start"
            ):
                # Submission dispatch is asynchronous: normally its 202 can
                # precede acceptance. Hold that real response at the ASGI send
                # boundary so this cut proves receipt-before-ack recovery.
                await asyncio.sleep(30)
                raise AssertionError("parent failed to kill before HTTP ack")
            await send(message)

        await app(scope, receive, gated_send)

    # Install in the isolated child only; production persistence is unchanged.
    if target is not None:
        store, method = target
        patcher = patch.object(
            store, method, pause_after_commit(getattr(store, method))
        )
        patcher.start()
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=transport_app),
            base_url="http://localhost:8420",
            headers=AUTH,
        ) as client,
    ):
        payload = {
            "backend": "slurm",
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
        denied = await client.post(
            "/api/runs/plan", json=payload, headers={"Authorization": ""}
        )
        assert denied.status_code == 401
        response = await client.post(
            "/api/runs/plan", json=payload, headers={"Idempotency-Key": "crash"}
        )
        assert response.status_code == 200, response.text
        planned = response.json()
        (root / "planned.json").write_text(json.dumps(planned))
        response = await client.post("/api/runs", json={"plan_id": planned["plan_id"]})
        assert response.status_code == 202, response.text
        (root / "acknowledged").touch()
        await asyncio.sleep(30)
        raise AssertionError("parent did not interrupt the child")


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL and dispatch ownership")
@pytest.mark.parametrize(
    ("boundary", "recovery"),
    [
        ("intent", "unique"),
        ("claim", "missing"),
        ("acceptance", "unique"),
        ("acceptance", "missing"),
        ("acceptance", "ambiguous"),
        ("receipt", "unique"),
    ],
)
def test_http_acceptance_crash_same_db_recovery(
    tmp_path, paused_process, boundary, recovery
):
    """Every committed boundary survives SIGKILL without duplicate acceptance."""
    project = Path(__file__).resolve().parents[2]
    script = (
        f"import sys; sys.path[:0] = {[str(project), str(project / 'src')]!r}; "
        "import asyncio; "
        "from unit_tests.daemon.test_submission_http_crash_recovery import crash_child; "
        f"asyncio.run(crash_child({str(tmp_path)!r}, {boundary!r}))"
    )
    marker = (
        "accepted-before-return"
        if boundary == "acceptance"
        else f"{boundary}-committed"
    )
    with paused_process(script, marker) as child:
        child.kill()
        child.wait(timeout=10)
        assert child.returncode == -signal.SIGKILL
    planned = json.loads((tmp_path / "planned.json").read_text())
    accepted = tmp_path / "accepted.json"
    had_acceptance = boundary in {"acceptance", "receipt"}
    assert accepted.exists() == had_acceptance
    audit = accepted.read_bytes() if had_acceptance else None
    if boundary != "acceptance":
        assert not (tmp_path / "acknowledged").exists()
    database = tmp_path / "state" / "registry.db"
    inode = database.stat().st_ino
    with sqlite3.connect(database) as connection:
        before = connection.execute(
            "SELECT state,submission_handle_json FROM submission_control"
        ).fetchall()
        if boundary == "intent":
            assert before == []  # No allocation/control record before _ensure.
        elif boundary == "receipt":
            assert len(before) == 1 and before[0][0] == "submitted"
            assert json.loads(before[0][1]) == json.loads(audit)
        else:
            assert before == [("submitting", None)]
        assert connection.execute(
            "SELECT submission_requested FROM plan_control"
        ).fetchall() == [(1,)]
        assert connection.execute(
            "SELECT COUNT(*) FROM execution_control"
        ).fetchone() == (0,)
    assert not list((tmp_path / "runs").glob("*/attempts/[0-9]*"))

    async def reopen(*, first=False):
        backend = DurableBackend(tmp_path, recovery=recovery)
        app = assembled_app(tmp_path, backend)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://localhost:8420",
                headers=AUTH,
            ) as client,
        ):
            url = f"/api/runs/{planned['run_id']}"
            denied = await client.get(url, headers={"Authorization": ""})
            assert denied.status_code == 401
            for _ in range(200):
                response = await client.get(f"{url}/attempts")
                assert response.status_code == 200, response.text
                records = response.json()["attempts"]
                if records and records[0]["execution_state"] == (
                    "submitted" if recovery == "unique" else "unknown"
                ):
                    break
                await asyncio.sleep(0.01)
            assert len(records) == 1
            record = records[0]
            assert record["execution_state"] == (
                "submitted" if recovery == "unique" else "unknown"
            )
            receipt = record["submission_handle"]
            assert record["attempt"] is None
            if recovery == "unique":
                assert receipt["job_id"] == "42;alpha"
                assert receipt["request_id"] == planned["request_id"]
            else:
                assert receipt is None
                status = (await client.get(url)).json()
                assert status["state"] == "submission_unknown"
                assert status["uncertainty"]
            for _ in range(3):
                repeated = await client.post(
                    "/api/runs", json={"plan_id": planned["plan_id"]}
                )
                assert repeated.status_code == 202, repeated.text
                await asyncio.sleep(0.02)
            expected_lookup = boundary == "claim" or (
                boundary == "acceptance" and (first or recovery != "unique")
            )
            assert (backend.lookups > 0) == expected_lookup
            assert backend.submits == int(boundary == "intent" and first)
        assert database.stat().st_ino == inode
        # Read outside the closed service using an independent connection, so
        # HTTP views cannot pass merely by retaining an in-memory receipt.
        with sqlite3.connect(database) as connection:
            rows = connection.execute(
                "SELECT state,submission_handle_json FROM submission_control"
            ).fetchall()
            assert len(rows) == 1
            state, persisted = rows[0]
            assert state == ("submitted" if recovery == "unique" else "unknown")
            if recovery == "unique":
                assert json.loads(persisted)["job_id"] == "42;alpha"
            else:
                assert persisted is None
            assert connection.execute(
                "SELECT submission_requested FROM plan_control"
            ).fetchall() == [(1,)]
            assert connection.execute(
                "SELECT COUNT(*) FROM execution_control"
            ).fetchone() == (0,)

    # A second complete reopen proves receipts/unknown are durable, not caches.
    asyncio.run(reopen(first=True))
    asyncio.run(reopen())
    if audit is not None:
        assert accepted.read_bytes() == audit
    elif boundary == "intent":
        receipt = json.loads(accepted.read_text())
        assert receipt["request_id"] == planned["request_id"]
    else:
        assert not accepted.exists()
    assert not list((tmp_path / "runs").glob("*/attempts/[0-9]*"))
