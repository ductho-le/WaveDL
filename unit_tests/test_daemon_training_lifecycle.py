"""Actual launcher, CPU training, durable provenance and published checkpoints."""

import sys
import time
from pathlib import Path
from uuid import uuid4

import numpy as np

from wavedl.core.generations import published_generations
from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.manifest import RuntimeSelection, build_manifest
from wavedl.core.protocol import decode
from wavedl_daemon.backends.contracts import ObservationStatus
from wavedl_daemon.backends.local import LocalBackend
from wavedl_daemon.runs.controller import SubmissionController
from wavedl_daemon.runs.reducer import RunState
from wavedl_daemon.storage.executions import ExecutionStore
from wavedl_daemon.storage.ownership import acquire_dispatch_lock
from wavedl_daemon.storage.plans import PlanStore
from wavedl_daemon.storage.registry import Registry
from wavedl_daemon.storage.runtime_provenance import RuntimeProvenanceStore


def test_controller_actual_cpu_training_publishes_and_negotiates(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src"
    data = tmp_path / "data.npz"
    generator = np.random.default_rng(9)
    np.savez(
        data,
        X=generator.normal(size=(20, 32)).astype("float32"),
        y=generator.normal(size=(20, 1)).astype("float32"),
    )
    layout = RunLayout(tmp_path / "runs" / str(uuid4()), str(uuid4()))
    # The directory need not be named after the run, but identity must be stable.
    manifest = build_manifest(
        layout,
        operation=OperationKind.TRAIN,
        resolved_config={
            "model": "cnn",
            "data_path": str(data),
            "epochs": 1,
            "workers": 0,
            "batch_size": 8,
            "precision": "no",
            "save_every": 1,
        },
        runtime=RuntimeSelection(
            sys.executable,
            "existing",
            "cpu",
            {"PYTHONPATH": str(source), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        ),
        submission_cwd=tmp_path,
    )
    database = tmp_path / "registry.db"
    with Registry(database):
        pass
    with (
        acquire_dispatch_lock(tmp_path / "dispatch.lock") as lock,
        PlanStore(database, dispatch_lock=lock) as plans,
        ExecutionStore(database, dispatch_lock=lock) as executions,
        RuntimeProvenanceStore(database, dispatch_lock=lock) as provenance,
    ):
        backend = LocalBackend(dispatch_lock=lock)
        controller = SubmissionController(
            plans,
            executions,
            {"local": backend},
            lock,
            provenance_store=provenance,
            handshake_timeout_seconds=45,
        )
        saved = plans.put(
            backend.plan(manifest),
            idempotency_key="actual-cpu",
            request_payload={"epochs": 1},
        )
        try:
            controller.request_submit(saved.plan_id)
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                controller.reconcile_once()
                record = executions.get(saved.plan_id)
                if (
                    record.latest_observation
                    and record.latest_observation.status == ObservationStatus.TERMINAL
                    and provenance.get(saved.plan_id)
                ):
                    break
                time.sleep(0.01)
            else:
                raise AssertionError(f"CPU execution did not complete: {record}")
            assert record.latest_observation.exit_code == 0, record.latest_observation
            events = [
                decode(line)
                for line in layout.journal(record.attempt).read_bytes().splitlines()
            ]
            assert events[0].type == "hello"
            assert events[-1].payload["status"] == "completed"
            assert controller.derive(saved.plan_id, events).state == RunState.COMPLETED
            assert provenance.get(saved.plan_id).hello.interpreter == sys.executable
            generations = published_generations(layout)
            assert len(generations) == 2
            assert all(
                "scaler.pkl" in item.files and "training_history.csv" in item.files
                for item in generations
            )
            assert len([event for event in events if event.type == "artifact"]) == 2
            with Registry(database, dispatch_lock=lock) as registry:
                registry.apply_events(layout.run_id, record.attempt, events)
                assert len(registry.metrics(layout.run_id)) == 1
                assert registry.metrics(layout.run_id)[0].step == 1
        finally:
            controller.close()
            backend.close()
