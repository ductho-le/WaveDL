"""Native two-rank CPU acceptance, with bounded process-tree supervision."""

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch.distributed as dist

from wavedl.core.protocol import decode
from wavedl.runtime import wait_for_workers
from wavedl_daemon.execution.budget import WorkBudget
from wavedl_daemon.execution.commands import run_command
from wavedl_daemon.supervision.base import ProcessSpec


@pytest.mark.parametrize(
    "device,initialized,expected",
    [("cpu", True, "cpu"), ("cpu", False, "accelerate"), ("cuda", True, "accelerate")],
)
def test_worker_barrier_preserves_device_and_fake_contracts(
    monkeypatch, device, initialized, expected
):
    """CPU distributed barriers omit device IDs; other waits retain Accelerate."""
    calls = []
    monkeypatch.setattr(dist, "is_initialized", lambda: initialized)
    monkeypatch.setattr(dist, "barrier", lambda: calls.append("cpu"))
    accelerator = SimpleNamespace(
        device=SimpleNamespace(type=device),
        wait_for_everyone=lambda: calls.append("accelerate"),
    )
    wait_for_workers(accelerator)
    assert calls == [expected]


@pytest.mark.integration
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Native CPU distributed acceptance requires PyTorch Gloo support",
)
def test_actual_two_rank_cpu_training_publishes_complete_generations(
    tmp_path, monkeypatch
):
    """Both ranks finish saving before rank zero publishes causal checkpoints."""
    # Direct launch must allocate its own identity, not inherit another run.
    for name in ("WAVEDL_RUN_ID", "WAVEDL_REQUEST_ID", "WAVEDL_ATTEMPT"):
        monkeypatch.delenv(name, raising=False)
    source = Path(__file__).resolve().parents[1] / "src"
    data = tmp_path / "data.npz"
    generator = np.random.default_rng(8)
    np.savez(
        data,
        X=generator.normal(size=(20, 32)).astype("float32"),
        y=generator.normal(size=(20, 1)).astype("float32"),
    )
    output = tmp_path / "run"
    command = (
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "--local_addr=127.0.0.1",
        "--start-method=spawn",
        "--run-path",
        str(source / "wavedl" / "train.py"),
        "--data_path",
        str(data),
        "--output_dir",
        str(output),
        "--model",
        "cnn",
        "--epochs",
        "1",
        "--workers",
        "0",
        "--batch_size",
        "8",
        "--precision",
        "no",
        "--save_every",
        "1",
    )
    environment = {
        "ACCELERATE_USE_CPU": "true",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONPATH": str(source)
        + (
            os.pathsep + os.environ["PYTHONPATH"]
            if os.environ.get("PYTHONPATH")
            else ""
        ),
    }
    # run_command selects native POSIX process groups or Windows Jobs and waits
    # for actual tree termination on deadline; it also bounds drained diagnostics.
    result = run_command(
        ProcessSpec(command, tmp_path, environment),
        WorkBudget(timeout_seconds=60, max_output_bytes=128 * 1024),
    )
    diagnostics = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    assert not result.timed_out, diagnostics
    assert result.returncode == 0, diagnostics

    assert list(output.rglob("manifest.json")) == [output / "manifest.json"]
    attempts = list((output / "attempts").iterdir())
    assert [attempt.name for attempt in attempts] == ["1"]
    journals = list(output.rglob("events.jsonl"))
    assert journals == [attempts[0] / "events.jsonl"]
    journal = [decode(line) for line in journals[0].read_bytes().splitlines()]
    manifest = json.loads((output / "manifest.json").read_text())
    assert all(event.run_id == manifest["run_id"] for event in journal)
    assert all(event.attempt == 1 for event in journal)
    assert [event.seq for event in journal] == list(range(len(journal)))
    hello = [event for event in journal if event.type == "hello"]
    assert len(hello) == 1
    assert hello[0].payload["device"] == "cpu"
    assert hello[0].payload["interpreter"] == sys.executable
    assert hello[0].payload["allocated_devices"] == []
    exits = [event for event in journal if event.type == "exit"]
    assert len(exits) == 1 and exits[0].payload["status"] == "completed"
    assert exits[0].payload["code"] == 0
    assert journal[-1] == exits[0]
    rank_log = attempts[0] / "rank/1.log"
    assert "Training runtime rank 1: cpu" in rank_log.read_text()
    assert not any(
        "Training runtime rank 1:" in event.payload.get("message", "")
        for event in journal
    )
    metrics = [event for event in journal if event.type == "metric"]
    assert len(metrics) == 1 and metrics[0].payload["step"] == 1
    assert diagnostics.count("##METRICS##") == 1

    generations = sorted((output / "checkpoints").glob("gen-*"))
    artifacts = [event for event in journal if event.type == "artifact"]
    assert len(generations) == len(artifacts) == 2
    assert {event.payload["label"] for event in artifacts} == {"best", "periodic"}
    for generation, artifact in zip(generations, artifacts, strict=True):
        marker = generation / "COMPLETE.json"
        completion = json.loads(marker.read_text())
        assert completion["attempt"] == 1 and completion["step"] == 1
        assert completion["run_id"] == manifest["run_id"]
        assert artifact.payload["generation"] == generation.name
        assert artifact.payload["path"] == marker.relative_to(output).as_posix()
        assert (
            artifact.payload["sha256"]
            == hashlib.sha256(marker.read_bytes()).hexdigest()
        )
        assert artifact.seq < exits[0].seq
        # Both independent rank RNG saves must already be included in COMPLETE,
        # not copied later into an already-published generation.
        assert {"random_states_0.pkl", "random_states_1.pkl"} <= set(
            completion["files"]
        )
        for name, record in completion["files"].items():
            member = generation / name
            content = member.read_bytes()
            assert len(content) == record["size"]
            assert hashlib.sha256(content).hexdigest() == record["sha256"]
            assert member.stat().st_mtime_ns <= marker.stat().st_mtime_ns
        selection = json.loads((generation / "checkpoint_selection.json").read_text())
        assert selection["kind"] == artifact.payload["label"]
        source_seq = selection["source_seq"]
        assert type(source_seq) is int
        assert metrics[0].seq <= source_seq < artifact.seq
        assert journal[source_seq].seq == source_seq
