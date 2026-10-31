"""Public cancellation at unclaimed and in-flight real Local boundaries."""

import os
import sys
import threading
import time

import pytest
from starlette.testclient import TestClient

from wavedl_daemon.backends.local import LocalBackend
from wavedl_daemon.execution.budget import WorkBudget
from wavedl_daemon.http.app import create_app
from wavedl_daemon.runs.runtime_service import HostedServices
from wavedl_daemon.storage.settings import Settings


def test_pending_and_submitting_cancel_use_real_local_receipt(tmp_path, monkeypatch):
    """Intent survives submit latency; an unclaimed cancelled run never spawns.

    The selected-python shim is a POSIX executable sentinel, not an ML runtime;
    this boundary test is skipped on Windows where its shebang is not portable.
    """
    if sys.platform == "win32":
        pytest.skip("POSIX process sentinel")
    entered = threading.Event()
    release = threading.Event()
    submitted = []
    cancel_calls = []
    spawned = tmp_path / "spawned"
    executable = tmp_path / "selected-python"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json,sys,time\n"
        "from pathlib import Path\n"
        "if sys.argv[1] == '-c':\n"
        "    print(json.dumps({'interpreter':sys.executable,"
        "'prefix':sys.prefix,'base_prefix':sys.base_prefix}))\n"
        "else:\n"
        f"    Path({str(spawned)!r}).write_text('started')\n"
        "    time.sleep(60)\n"
    )
    executable.chmod(0o700)
    original_probe = LocalBackend._probe
    original_submit = LocalBackend.submit

    def held_probe(backend, plan):
        entered.set()
        assert release.wait(10), "test did not release the actual Local probe"
        return original_probe(backend, plan)

    def recorded_submit(backend, plan, **kwargs):
        submitted.append(plan.run_id)
        return original_submit(backend, plan, **kwargs)

    original_cancel = LocalBackend.request_cancel

    def recorded_cancel(backend, handle):
        cancel_calls.append(handle.run_id)
        # Stop only after the actual submitted interpreter reached its workload,
        # so the sentinel proves a real process, not just a claimed submission.
        deadline = time.monotonic() + 5
        while not spawned.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert spawned.exists(), "submitted Local child never reached sentinel"
        return original_cancel(backend, handle)

    monkeypatch.setattr(LocalBackend, "_probe", held_probe)
    monkeypatch.setattr(LocalBackend, "submit", recorded_submit)
    monkeypatch.setattr(LocalBackend, "request_cancel", recorded_cancel)
    settings = Settings(tmp_path / "state", tmp_path / "runs")
    services = HostedServices(
        settings,
        budget=WorkBudget(max_workers=1, max_pending=0),
        backend_factory=lambda lock: {
            "local": LocalBackend(dispatch_lock=lock, cancel_grace=0.05)
        },
    )
    app = create_app(
        settings,
        token="cancel-test",
        daemon_id="test",
        build_id="test",
        services=services,
    )
    payload = {
        "backend": "local",
        "operation": "train",
        "config": {"model": "cnn", "epochs": 1, "precision": "no"},
        "runtime": {
            "python": str(executable),
            "mode": "selected",
            "device": "cpu",
            "environment": {},
        },
        "topology": {
            "num_machines": 1,
            "processes_per_machine": 1,
            "machine_rank": 0,
        },
    }
    try:
        with TestClient(
            app,
            base_url="http://localhost:8420",
            headers={"Authorization": "Bearer cancel-test"},
        ) as client:
            plans = []
            for key in ("in-flight", "unclaimed"):
                response = client.post(
                    "/api/runs/plan", json=payload, headers={"Idempotency-Key": key}
                )
                assert response.status_code == 200, response.text
                plans.append(response.json())
            urls = [f"/api/runs/{plan['run_id']}" for plan in plans]
            response = client.post("/api/runs", json={"plan_id": plans[0]["plan_id"]})
            assert response.status_code == 202, response.text
            assert entered.wait(5), "real submit did not enter the Local probe"
            response = client.post("/api/runs", json={"plan_id": plans[1]["plan_id"]})
            assert response.status_code == 202, response.text
            pending = client.get(urls[1]).json()
            assert pending["state"] == "pending" and pending["submission_requested"]
            assert submitted == [plans[0]["run_id"]]
            for url in urls:
                response = client.post(f"{url}/cancel")
                assert response.status_code == 202, response.text
                assert response.json()["cancellation_requested"]
            pending = client.get(urls[1]).json()
            assert pending["state"] == "cancelled" and not pending["uncertainty"]
            unresolved = client.get(urls[0]).json()
            assert unresolved["state"] == "submission_unknown"
            assert unresolved["uncertainty"] and unresolved["cancellation_requested"]
            assert unresolved["attempt"] is None
            assert not spawned.exists()
            release.set()
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                response = client.get(urls[0])
                assert response.status_code == 200, response.text
                if response.json()["state"] == "cancelled":
                    break
                time.sleep(0.02)
            result = response.json()
            assert result["state"] == "cancelled", result
            assert not result["uncertainty"] and result["cancellation_requested"]
            assert result["evidence"]["backend"]["status"] == "terminal"
            assert result["evidence"]["batch"]["status"] == "cancelled"
            assert result["evidence"]["runtime"]["hello_seen"] is False
            assert spawned.read_text() == "started"
            for index, url in enumerate(urls):
                lineage = client.get(f"{url}/attempts")
                assert lineage.status_code == 200, lineage.text
                lineage = lineage.json()
                assert lineage["total"] == 1 and len(lineage["attempts"]) == 1
                execution = lineage["attempts"][0]
                assert execution["attempt"] is None
                assert execution["cancellation_requested"]
                assert execution["execution_state"] == (
                    "submitted" if index == 0 else "pending"
                )
                if index == 0:
                    receipt = execution["submission_handle"]
                    assert receipt["run_id"] == plans[0]["run_id"]
                    assert receipt["pid"] > 0 and receipt["supervisor_id"]
                    assert execution["cancel_delivered"]
                    assert execution["submission_observation"]["status"] == "terminal"
                    with pytest.raises(ProcessLookupError):
                        os.kill(receipt["pid"], 0)
                else:
                    assert execution["submission_handle"] is None
                    assert not execution["cancel_delivered"]
            assert submitted == [plans[0]["run_id"]]
            assert cancel_calls == [plans[0]["run_id"]]
            assert client.get(urls[1]).json()["state"] == "cancelled"
            for url in urls:
                response = client.post(f"{url}/cancel")
                assert response.status_code == 202, response.text
                assert response.json()["state"] == "cancelled"
            assert submitted == [plans[0]["run_id"]]
            assert cancel_calls == [plans[0]["run_id"]]
    finally:
        release.set()
