"""Public cancellation of planned, pending and in-flight real Local runs."""

import json
import os
import sqlite3
import sys
import threading
import time

import pytest
from starlette.testclient import TestClient

from wavedl_daemon.backends.local import LocalBackend
from wavedl_daemon.execution.budget import WorkBudget
from wavedl_daemon.http.app import create_app
from wavedl_daemon.runs.runtime_service import DATABASE_FILE, HostedServices
from wavedl_daemon.storage.settings import Settings


pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX selected-python sentinel"
)


def events(settings, run_id):
    with sqlite3.connect(settings.state_root / DATABASE_FILE) as connection:
        return [
            (row[0], json.loads(row[1]))
            for row in connection.execute(
                "SELECT type, payload_json FROM run_events WHERE run_id = ? "
                "ORDER BY attempt, seq",
                (run_id,),
            )
        ]


def wait_for(client, url, state):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = client.get(url)
        assert response.status_code == 200, response.text
        if response.json()["state"] == state:
            return response.json()
        time.sleep(0.02)
    raise AssertionError(f"{url} never reached {state}: {response.json()}")


def test_cancel_at_each_stage_of_a_real_local_run(tmp_path, monkeypatch):
    """A planned run ignores cancel, a pending one ends without a spawn, and an
    in-flight one is stopped once its launch is recorded."""
    entered = threading.Event()
    release = threading.Event()
    spawned = []
    executable = tmp_path / "selected-python"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json,signal,sys,time\n"
        "if sys.argv[1] == '-c':\n"
        "    print(json.dumps({'interpreter':sys.executable,"
        "'prefix':sys.prefix,'devices':['cpu']}))\n"
        "else:\n"
        "    signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        f"    open({str(tmp_path / 'ready')!r}, 'w').close()\n"
        "    time.sleep(60)\n"
    )
    executable.chmod(0o700)
    original_probe = LocalBackend._probe
    original_spawn = LocalBackend.spawn
    original_cancel = LocalBackend.cancel

    def held_probe(backend, plan, cwd, environment):
        entered.set()
        assert release.wait(10), "test did not release the actual Local probe"
        return original_probe(backend, plan, cwd, environment)

    def recorded_spawn(backend, plan, attempt):
        spawned.append(plan.run_id)
        return original_spawn(backend, plan, attempt)

    def ready_cancel(backend, handle):
        # A SIGINT before the stand-in ignores it would end the run without SIGKILL.
        deadline = time.monotonic() + 10
        while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        return original_cancel(backend, handle)

    monkeypatch.setattr(LocalBackend, "_probe", held_probe)
    monkeypatch.setattr(LocalBackend, "cancel", ready_cancel)
    monkeypatch.setattr(LocalBackend, "spawn", recorded_spawn)
    settings = Settings(tmp_path / "state", tmp_path / "runs")
    services = HostedServices(
        settings,
        budget=WorkBudget(max_workers=1, max_pending=0),
        backend_factory=lambda lock: {
            "local": LocalBackend(dispatch_lock=lock, max_processes=1, cancel_grace=0.05)
        },
    )
    app = create_app(
        settings, token="cancel-test", daemon_id="test", build_id="test", services=services
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
        "topology": {"num_machines": 1, "processes_per_machine": 1, "machine_rank": 0},
    }
    try:
        with TestClient(
            app,
            base_url="http://localhost:8420",
            headers={"Authorization": "Bearer cancel-test"},
        ) as client:
            plans = []
            for key in ("in-flight", "pending", "planned"):
                response = client.post(
                    "/api/runs/plan", json=payload, headers={"Idempotency-Key": key}
                )
                assert response.status_code == 200, response.text
                plans.append(response.json())
            in_flight, pending, planned = (
                f"/api/runs/{plan['run_id']}" for plan in plans
            )

            response = client.post(f"{planned}/cancel")
            assert response.status_code == 202, response.text
            view = response.json()
            assert view["state"] == "planned" and not view["cancellation_requested"]
            assert events(settings, plans[2]["run_id"]) == []

            response = client.post("/api/runs", json={"plan_id": plans[0]["plan_id"]})
            assert response.status_code == 202, response.text
            assert entered.wait(5), "submit did not reach the Local probe"
            response = client.post("/api/runs", json={"plan_id": plans[1]["plan_id"]})
            assert response.status_code == 202, response.text
            # One Local process slot is taken, so the second run waits unclaimed.
            time.sleep(0.3)
            view = client.get(pending).json()
            assert view["state"] == "pending" and view["submission_requested"]

            for url in (in_flight, pending):
                response = client.post(f"{url}/cancel")
                assert response.status_code == 202, response.text
                assert response.json()["cancellation_requested"]
            view = wait_for(client, pending, "cancelled")
            assert not view["uncertainty"] and view["evidence"]["outcome"] == "cancelled"
            assert [kind for kind, _ in events(settings, plans[1]["run_id"])] == [
                "attempt",
                "cancel_requested",
                "exited",
            ]
            view = client.get(in_flight).json()
            assert view["state"] == "submission_unknown" and view["uncertainty"]
            assert view["cancellation_requested"]

            release.set()
            view = wait_for(client, in_flight, "cancelled")
            assert not view["uncertainty"]
            assert view["evidence"]["outcome"] == "cancelled"
            assert view["evidence"]["signal"] == 9
            history = events(settings, plans[0]["run_id"])
            assert [kind for kind, _ in history] == [
                "attempt",
                "submitting",
                "cancel_requested",
                "probed",
                "submitted",
                "observed",
                "cancel_delivered",
                "exited",
            ]
            pid = dict(history)["submitted"]["handle"]["pid"]
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)

            for url in (in_flight, pending):
                response = client.post(f"{url}/cancel")
                assert response.status_code == 202, response.text
                assert response.json()["state"] == "cancelled"
            assert len(events(settings, plans[0]["run_id"])) == len(history)
            assert spawned == [plans[0]["run_id"]]
            runs = settings.run_root
            assert not (runs / plans[1]["run_id"]).exists()
            assert not (runs / plans[2]["run_id"]).exists()
    finally:
        release.set()
