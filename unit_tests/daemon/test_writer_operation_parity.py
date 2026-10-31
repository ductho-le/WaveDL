"""Actual CPU writer train/test/plot acceptance through the public HTTP API."""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from starlette.testclient import TestClient

from wavedl_daemon.backends.contracts import Topology, UnsupportedPlan
from wavedl_daemon.backends.local import LocalBackend
from wavedl_daemon.http.app import create_app
from wavedl_daemon.runs.runtime_service import HostedServices
from wavedl_daemon.storage.settings import Settings

from .test_local_planning import manifest
from .test_writer_http_acceptance import DisconnectAfterExit


@pytest.fixture
def writer_cpu_http(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("accelerate")

    def forbidden(*args, **kwargs):
        raise AssertionError("The daemon must not allocate writer attempts")

    monkeypatch.setattr(
        "wavedl_daemon.runs.control.dispatch.allocate_attempt", forbidden
    )
    generator = np.random.default_rng(31)
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
        token="parity-test",
        daemon_id="test",
        build_id="test",
        services=services,
    )
    runtime = {
        "python": sys.executable,
        "mode": "existing",
        "device": "cpu",
        "environment": {
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MPLBACKEND": "Agg",
        },
    }
    with TestClient(
        DisconnectAfterExit(app),
        base_url="http://localhost:8420",
        headers={"Authorization": "Bearer parity-test"},
    ) as client:
        yield client, settings, runtime, data
    assert not services.host._thread.is_alive()


def submit_operation(context, operation, config, *, state="completed", ranks=1):
    client, settings, runtime, _ = context
    response = client.post(
        "/api/runs/plan",
        json={
            "backend": "local",
            "operation": operation,
            "config": config,
            "runtime": runtime,
            "topology": {
                "num_machines": 1,
                "processes_per_machine": ranks,
                "machine_rank": 0,
            },
        },
        headers={"Idempotency-Key": f"{operation}-{time.monotonic_ns()}"},
    )
    assert response.status_code == 200, response.text
    planned = response.json()
    assert planned["preview"]["allocation_mode"] == "writer"
    url = f"/api/runs/{planned['run_id']}"
    assert client.get(f"{url}/attempts").json() == {"attempts": [], "total": 0}
    response = client.post("/api/runs", json={"plan_id": planned["plan_id"]})
    assert response.status_code == 202, response.text
    assert response.json()["attempt"] is None
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        response = client.get(url)
        assert response.status_code == 200, response.text
        if response.json()["state"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.05)
    root = settings.run_root / planned["run_id"]
    diagnostics = "\n".join(
        path.read_text(errors="replace") for path in root.rglob("*.log")
    )
    assert response.json()["state"] == state, response.text + diagnostics
    assert response.json()["attempt"] == 1
    evidence = client.get(f"{url}/attempts").json()["attempts"]
    assert len(evidence) == 1 and evidence[0]["attempt"] == 1
    assert evidence[0]["runtime"]["device"] == "cpu"
    assert evidence[0]["runtime"]["interpreter"] == sys.executable
    response = client.get(f"{url}/events")
    assert response.status_code == 200, response.text
    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[0]["type"] == "hello"
    assert events[0]["payload"]["device"] == "cpu"
    assert events[0]["payload"]["interpreter"] == sys.executable
    assert events[-1]["type"] == "exit"
    assert events[-1]["payload"]["status"] == state
    assert all(event["attempt"] == 1 for event in events)
    assert len([event for event in events if event["type"] == "hello"]) == 1
    assert list(root.rglob("events.jsonl")) == [root / "attempts/1/events.jsonl"]
    assert [path.name for path in (root / "attempts").iterdir()] == ["1"]
    return url, root, events


def verify_artifacts(client, url):
    response = client.get(f"{url}/artifacts")
    assert response.status_code == 200, response.text
    generations = response.json()["generations"]
    for generation in generations:
        assert generation["attempt"] == 1
        for name, record in generation["files"].items():
            response = client.get(f"{url}/artifacts/{generation['generation']}/{name}")
            assert response.status_code == 200, response.text
            assert len(response.content) == record["size"] > 0
            assert hashlib.sha256(response.content).hexdigest() == record["sha256"]
            if name.endswith(".png"):
                assert response.content.startswith(b"\x89PNG\r\n\x1a\n")
    return generations


def test_actual_writer_cpu_train_test_plot_published_generation(writer_cpu_http):
    """Inference consumes immutable published weights, not legacy checkpoints."""
    client, _, _, data = writer_cpu_http
    train_url, train_root, _ = submit_operation(
        writer_cpu_http,
        "train",
        {
            "model": "cnn",
            "epochs": 1,
            "data_path": str(data),
            "precision": "no",
            "workers": 0,
            "batch_size": 8,
        },
    )
    generations = verify_artifacts(client, train_url)
    best = next(
        generation
        for generation in generations
        if json.loads(
            (
                train_root
                / "checkpoints"
                / generation["generation"]
                / "checkpoint_selection.json"
            ).read_text()
        )["kind"]
        == "best"
    )
    checkpoint = train_root / "checkpoints" / best["generation"]
    # Fail loudly if inference ever falls back to the old, mutable location.
    legacy = train_root / "best_checkpoint"
    assert not legacy.is_symlink()
    legacy.mkdir(exist_ok=True)
    for weights in legacy.glob("*model*"):
        if weights.is_file():
            weights.write_bytes(b"not a checkpoint")
    for operation in ("test", "plot"):
        url, _, events = submit_operation(
            writer_cpu_http,
            operation,
            {
                "checkpoint": str(checkpoint),
                "data_path": str(data),
                "batch_size": 8,
                "workers": 0,
                "save_predictions": True,
                "plot_format": ["png"],
            },
        )
        assert any(
            str(checkpoint) in event["payload"].get("message", "")
            for event in events
            if event["type"] == "log"
        )
        metrics = [event for event in events if event["type"] == "metric"]
        assert len(metrics) == 1
        assert metrics[0]["payload"]["metrics"]["samples"] == 20
        outputs = verify_artifacts(client, url)
        assert len(outputs) == 1
        names = set(outputs[0]["files"])
        assert "predictions.csv" in names
        if operation == "plot":
            assert len([name for name in names if name.endswith(".png")]) == 9


@pytest.mark.parametrize("operation", ["test", "plot"])
def test_actual_writer_inference_setup_failure_is_terminal(writer_cpu_http, operation):
    """Missing input checkpoint fails with the writer's real error/exit journal."""
    client, settings, _, data = writer_cpu_http
    url, _, events = submit_operation(
        writer_cpu_http,
        operation,
        {"checkpoint": str(settings.run_root / "missing"), "data_path": str(data)},
        state="failed",
    )
    errors = [event for event in events if event["type"] == "error"]
    assert errors and errors[0]["payload"]["code"] == "OPERATION_FAILED"
    assert events[-1]["payload"]["code"] == 1
    assert not verify_artifacts(client, url)


def test_actual_writer_plot_invalid_configuration_is_terminal(writer_cpu_http):
    """A real invalid plot/export combination reports the runtime's failure."""
    client, settings, _, data = writer_cpu_http
    url, _, events = submit_operation(
        writer_cpu_http,
        "plot",
        {
            "checkpoint": str(settings.run_root / "unused"),
            "data_path": str(data),
            "export": "onnx",
        },
        state="failed",
    )
    errors = [event for event in events if event["type"] == "error"]
    assert errors and "Plot operations cannot export" in errors[0]["payload"]["message"]
    assert not any(event["type"] == "metric" for event in events)
    assert not verify_artifacts(client, url)


def test_actual_writer_two_rank_cpu_training(writer_cpu_http):
    """Two actual Gloo ranks share one writer allocation and complete save proof."""
    torch = pytest.importorskip("torch")
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_gloo_available()
    ):
        pytest.skip("Native CPU distributed acceptance requires PyTorch Gloo")
    client, _, _, data = writer_cpu_http
    url, root, events = submit_operation(
        writer_cpu_http,
        "train",
        {
            "model": "cnn",
            "epochs": 1,
            "data_path": str(data),
            "precision": "no",
            "workers": 0,
            "batch_size": 8,
            "save_every": 1,
        },
        ranks=2,
    )
    assert (
        "Training runtime rank 1: cpu" in (root / "attempts/1/rank/1.log").read_text()
    )
    assert not any(
        "Training runtime rank 1:" in event["payload"].get("message", "")
        for event in events
    )
    assert len([event for event in events if event["type"] == "metric"]) == 1
    proof = list(root.rglob("local_identity.json"))
    assert proof == [root / "attempts/1/local_identity.json"]
    startup = list(root.rglob("startup.json"))
    assert len(startup) == 1
    assert json.loads(startup[0].read_text()) == json.loads(proof[0].read_text())
    generations = verify_artifacts(client, url)
    assert len(generations) == 2
    assert all(
        {"random_states_0.pkl", "random_states_1.pkl"} <= set(generation["files"])
        for generation in generations
    )
    assert len([event for event in events if event["type"] == "artifact"]) == 2


def test_cpu_topology_process_budget_is_enforced_before_spawn(tmp_path, monkeypatch):
    from wavedl.launcher import parse_args

    with pytest.raises(UnsupportedPlan, match="at most 64"):
        LocalBackend().plan(manifest(tmp_path), topology=Topology(1, 65))
    monkeypatch.setattr(sys, "argv", ["wavedl-train", "--cpu", "--num_gpus", "65"])
    with pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2
