"""Default writer Local runs through the actual production HTTP app.

Training, supervision, writer binding, provenance, journal ingestion and
artifacts are real; no test-only allocation policy overrides are installed.
"""

import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from starlette.testclient import TestClient

from wavedl_daemon.backends.local import LocalBackend
from wavedl_daemon.http.app import create_app
from wavedl_daemon.runs.runtime_service import HostedServices
from wavedl_daemon.storage.settings import Settings


class DisconnectAfterExit:
    """Test transport disconnects an infinite SSE response after its real exit."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].endswith("/events"):
            return await self.app(scope, receive, send)
        disconnected = asyncio.Event()
        received_request = False

        async def client_receive():
            nonlocal received_request
            if not received_request:
                received_request = True
                return await receive()
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def client_send(message):
            if disconnected.is_set():
                return
            if message["type"] == "http.response.body":
                body = message.get("body", b"")
                if b'"type":"exit"' in body:
                    message = {**message, "more_body": False}
                    await send(message)
                    disconnected.set()
                    return
            await send(message)

        await self.app(scope, client_receive, client_send)


def test_actual_cpu_writer_training_public_http(tmp_path, monkeypatch):
    """No caller attempt, daemon allocation, synthetic events or fake receipts."""
    pytest.importorskip("torch")
    pytest.importorskip("accelerate")

    def daemon_must_not_allocate(*args, **kwargs):
        raise AssertionError("Writer mode must not allocate an attempt in the daemon")

    monkeypatch.setattr(
        "wavedl_daemon.runs.control.dispatch.allocate_attempt", daemon_must_not_allocate
    )
    generator = np.random.default_rng(19)
    data = tmp_path / "data.npz"
    np.savez(
        data,
        X=generator.normal(size=(20, 32)).astype("float32"),
        y=generator.normal(size=(20, 1)).astype("float32"),
    )
    settings = Settings(tmp_path / "state", tmp_path / "runs")
    services = HostedServices(
        settings,
        backend_factory=lambda lock: {"local": LocalBackend(dispatch_lock=lock)},
    )
    app = create_app(
        settings,
        token="writer-test",
        daemon_id="test",
        build_id="test",
        services=services,
    )
    source = Path(__file__).resolve().parents[2] / "src"
    payload = {
        "backend": "local",
        "operation": "train",
        "config": {
            "model": "cnn",
            "epochs": 1,
            "data_path": str(data),
            "precision": "no",
            "workers": 0,
            "batch_size": 8,
        },
        "runtime": {
            "python": sys.executable,
            "mode": "existing",
            "device": "cpu",
            "environment": {
                "PYTHONPATH": str(source),
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        },
        "topology": {"num_machines": 1, "processes_per_machine": 1, "machine_rank": 0},
    }
    with TestClient(
        DisconnectAfterExit(app),
        base_url="http://localhost:8420",
        headers={"Authorization": "Bearer writer-test"},
    ) as client:
        response = client.post(
            "/api/runs/plan", json=payload, headers={"Idempotency-Key": "cpu-writer"}
        )
        assert response.status_code == 200, response.text
        planned = response.json()
        assert planned["preview"]["allocation_mode"] == "writer"
        run_url = f"/api/runs/{planned['run_id']}"
        assert client.get(f"{run_url}/attempts").json() == {"attempts": [], "total": 0}
        response = client.post("/api/runs", json={"plan_id": planned["plan_id"]})
        assert response.status_code == 202, response.text
        assert response.json()["attempt"] is None
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            response = client.get(run_url)
            assert response.status_code == 200, response.text
            if response.json()["state"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.05)
        view = response.json()
        assert view["state"] == "completed", response.text
        assert view["attempt"] == 1 and not view["uncertainty"]
        evidence = client.get(f"{run_url}/attempts")
        assert evidence.status_code == 200, evidence.text
        active = evidence.json()["attempts"][0]
        assert active["attempt"] == 1 and active["current"]
        assert active["runtime"]["device"] == "cpu"
        assert active["handle"]["pid"] > 0 and active["handle"]["supervisor_id"]
        response = client.get(f"{run_url}/events")
        assert response.status_code == 200, response.text
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert events[0]["type"] == "hello" and events[-1]["type"] == "exit"
        assert all(event["attempt"] == 1 for event in events)
        assert events[-1]["payload"]["status"] == "completed"
        listing = client.get(f"{run_url}/artifacts")
        assert listing.status_code == 200, listing.text
        generations = listing.json()["generations"]
        assert generations and all(item["attempt"] == 1 for item in generations)
        generation = next(
            item for item in generations if "pytorch_model.bin" in item["files"]
        )
        metadata = generation["files"]["pytorch_model.bin"]
        checkpoint = client.get(
            f"{run_url}/artifacts/{generation['generation']}/pytorch_model.bin"
        )
        assert checkpoint.status_code == 200, checkpoint.text
        assert len(checkpoint.content) == metadata["size"] > 0
        assert hashlib.sha256(checkpoint.content).hexdigest() == metadata["sha256"]
    assert not services.host._thread.is_alive()
