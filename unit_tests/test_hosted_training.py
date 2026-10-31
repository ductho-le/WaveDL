"""Public ASGI workflows exercise real CPU runtime and genuine CLI adoption."""

import asyncio
import os
import shutil
import sys
import time
import venv
from pathlib import Path

import httpx
import numpy as np
import pytest

from wavedl.core.manifest import read_manifest
from wavedl_daemon.backends.local import LocalBackend
from wavedl_daemon.execution.budget import WorkBudget
from wavedl_daemon.execution.commands import run_command
from wavedl_daemon.http.app import create_app
from wavedl_daemon.runs.runtime_service import HostedServices
from wavedl_daemon.storage.settings import Settings
from wavedl_daemon.supervision.base import ProcessSpec


def data_file(tmp_path):
    generator = np.random.default_rng(19)
    path = tmp_path / "data.npz"
    np.savez(
        path,
        X=generator.normal(size=(20, 32)).astype("float32"),
        y=generator.normal(size=(20, 1)).astype("float32"),
    )
    return path


@pytest.mark.parametrize(
    "explicit_source,python_kind",
    [(False, "native"), (True, "native"), (False, "alias"), (False, "venv")],
)
def test_public_http_actual_cpu_training(tmp_path, explicit_source, python_kind):
    selected_python = sys.executable
    if python_kind == "alias":
        installed_alias = shutil.which(
            f"python{sys.version_info.major}.{sys.version_info.minor}"
        )
        if installed_alias and os.path.samefile(installed_alias, sys.executable):
            # Homebrew canonicalizes this installed selected path at CPython
            # startup, reproducing the original browser-authored failure.
            selected_python = installed_alias
        else:
            alias = tmp_path / "selected-python"
            alias.symlink_to(sys.executable)
            selected_python = str(alias)
    elif python_kind == "venv":
        root = tmp_path / "selected-venv"
        venv.EnvBuilder(
            system_site_packages=True, with_pip=False, symlinks=os.name != "nt"
        ).create(root)
        selected_python = str(
            root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        )
    data = data_file(tmp_path)
    settings = Settings(tmp_path / "state", tmp_path / "runs")
    services = HostedServices(settings)
    app = create_app(
        settings,
        token="test-token",
        daemon_id="daemon",
        build_id="build",
        services=services,
    )
    source = Path(__file__).resolve().parents[1] / "src"

    async def run():
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://localhost:8420",
                headers={"Authorization": "Bearer test-token"},
            ) as client,
        ):
            payload = {
                "backend": "local",
                "operation": "train",
                "topology": {
                    "num_machines": 1,
                    "processes_per_machine": 1,
                    "machine_rank": 0,
                },
                "config": {
                    "model": "cnn",
                    "epochs": 1,
                    "data_path": str(data),
                    "precision": "no",
                    "workers": 0,
                    "batch_size": 8,
                },
                "runtime": {
                    "python": selected_python,
                    "mode": "existing",
                    "device": "cpu",
                    "environment": {
                        "PYTHONPATH": str(source),
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                    },
                },
            }
            response = await client.post(
                "/api/runs/plan", json=payload, headers={"Idempotency-Key": "cpu"}
            )
            assert response.status_code == 200, response.text
            preview = response.json()
            response = await client.post(
                "/api/runs", json={"plan_id": preview["plan_id"]}
            )
            assert response.status_code == 202, response.text
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                response = await client.get(f"/api/runs/{preview['run_id']}")
                assert response.status_code == 200, response.text
                if response.json()["state"] in {"completed", "failed"}:
                    break
                await asyncio.sleep(0.05)
            assert response.json()["state"] == "completed", response.text
            rows = await services.event_log.read_after(preview["run_id"], limit=256)
            assert rows[0].type == "hello" and rows[-1].type == "exit"
            assert rows[-1].payload["status"] == "completed"
            approved_manifest = read_manifest(
                settings.run_root / preview["run_id"] / "manifest.json"
            )
            assert approved_manifest.runtime.python == selected_python
            assert rows[0].payload["prefix"]
            if python_kind == "venv":
                assert rows[0].payload["prefix"] == str(tmp_path / "selected-venv")
                assert rows[0].payload["prefix"] != rows[0].payload["base_prefix"]
            if python_kind != "native":
                # Focus the alias regressions on real train→HTTP/event ingestion;
                # native cases below retain full test/plot workflow coverage.
                return
            listing = await client.get(f"/api/runs/{preview['run_id']}/artifacts")
            assert listing.status_code == 200, listing.text
            assert listing.json()["generations"]
            training_dir = str(settings.run_root / preview["run_id"])
            selected_generation = listing.json()["generations"][-1]["generation"]
            for operation in ("test", "plot"):
                payload["operation"] = operation
                payload["config"] = {
                    "checkpoint": training_dir,
                    "data_path": str(data),
                    "save_predictions": True,
                    "batch_size": 8,
                    "workers": 0,
                }
                if explicit_source:
                    payload["config"].pop("checkpoint")
                    payload["checkpoint_source"] = {
                        "run_id": preview["run_id"],
                        "generation": selected_generation,
                    }
                # Plotting must not select an incomplete host LaTeX install.
                payload["runtime"]["environment"]["PATH"] = os.defpath
                response = await client.post(
                    "/api/runs/plan",
                    json=payload,
                    headers={"Idempotency-Key": operation},
                )
                assert response.status_code == 200, response.text
                operation_plan = response.json()
                if explicit_source:
                    import json

                    approved = json.loads(operation_plan["preview"]["manifest_json"])
                    assert approved["checkpoint_source"]["run_id"] == preview["run_id"]
                    assert (
                        approved["checkpoint_source"]["generation"]
                        == selected_generation
                    )
                    assert (
                        approved["resolved_config"]["checkpoint"]
                        == f"{training_dir}/checkpoints/{selected_generation}"
                    )
                    replay = await client.post(
                        "/api/runs/plan",
                        json=payload,
                        headers={"Idempotency-Key": operation},
                    )
                    assert replay.json() == operation_plan
                response = await client.post(
                    "/api/runs", json={"plan_id": operation_plan["plan_id"]}
                )
                assert response.status_code == 202, response.text
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    response = await client.get(f"/api/runs/{operation_plan['run_id']}")
                    if response.json()["state"] in {"completed", "failed"}:
                        break
                    await asyncio.sleep(0.05)
                assert response.json()["state"] == "completed", response.text
                rows = await services.event_log.read_after(
                    operation_plan["run_id"], limit=256
                )
                assert rows[0].type == "hello" and rows[-1].type == "exit"
                assert rows[0].payload["device"] == "cpu"
                artifacts = await client.get(
                    f"/api/runs/{operation_plan['run_id']}/artifacts"
                )
                assert artifacts.status_code == 200, artifacts.text
                generations = artifacts.json()["generations"]
                assert generations
                artifact = next(row for row in rows if row.type == "artifact")
                prefix = "checkpoints/"
                relative = artifact.payload["path"].removeprefix(prefix)
                downloaded = await client.get(
                    f"/api/runs/{operation_plan['run_id']}/artifacts/{relative}"
                )
                assert downloaded.status_code == 200, downloaded.text
                assert len(downloaded.content) == artifact.payload["size"]
                if operation == "plot":
                    assert any(
                        row.type == "artifact" and row.payload["kind"] == "plot"
                        for row in rows
                    )
                # Backend capacity is released only when actual collection has
                # completed; an HTTP completed state must reflect that boundary.
            assert (
                read_manifest(Path(training_dir) / "manifest.json").operation.value
                == "train"
            )

    asyncio.run(run())


def test_http_adopts_actual_standalone_launcher_without_claiming_supervision(tmp_path):
    data = data_file(tmp_path)
    output = tmp_path / "standalone"
    source = Path(__file__).resolve().parents[1] / "src"
    environment = dict(os.environ)
    environment.update(PYTHONPATH=str(source), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = run_command(
        ProcessSpec(
            (
                sys.executable,
                "-m",
                "wavedl.launcher",
                "--cpu",
                "--mixed_precision",
                "no",
                "--model",
                "cnn",
                "--data_path",
                str(data),
                "--output_dir",
                str(output),
                "--epochs",
                "1",
                "--workers",
                "0",
                "--batch_size",
                "8",
            ),
            tmp_path,
            environment,
        ),
        WorkBudget(timeout_seconds=60),
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    manifest = read_manifest(output / "manifest.json")
    settings = Settings(tmp_path / "state", tmp_path / "managed")
    services = HostedServices(settings)
    app = create_app(
        settings,
        token="test-token",
        daemon_id="daemon",
        build_id="build",
        services=services,
    )

    async def run():
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://localhost:8420",
                headers={"Authorization": "Bearer test-token"},
            ) as client,
        ):
            response = await client.post(
                "/api/runs/adopt",
                json={"path": str(output)},
                headers={"Idempotency-Key": "external"},
            )
            assert response.status_code == 200, response.text
            assert response.json()["run_id"] == manifest.run_id
            assert (
                response.json()["state"] == "supervision_lost"
                and response.json()["uncertainty"]
            )
            deadline = time.monotonic() + 5
            rows = []
            while time.monotonic() < deadline:
                rows = await services.event_log.read_after(manifest.run_id)
                if rows and rows[-1].type == "exit":
                    break
                await asyncio.sleep(0.01)
            assert rows[0].type == "hello" and rows[-1].payload["status"] == "completed"
            view = await client.get(f"/api/runs/{manifest.run_id}")
            assert view.json()["state"] == "supervision_lost"
            handles = await services.host.call(
                lambda session: (
                    session.runs.execution_store.get(response.json()["plan_id"]).handle
                )
            )
            assert handles.pid is None and handles.supervisor_id is None

    asyncio.run(run())


@pytest.mark.parametrize("cancel_grace", [0, 5])
def test_public_http_cancels_actual_cpu_tree_without_requiring_runtime_exit(
    tmp_path, cancel_grace
):
    """Owned terminal cancellation is proof even if SIGTERM cuts off the journal."""
    data = data_file(tmp_path)
    settings = Settings(tmp_path / "state", tmp_path / "runs")
    # Immediate escalation exercises missing runtime-exit evidence; the normal
    # grace also permits a genuine runtime cancellation exit to be published.
    services = HostedServices(
        settings,
        backend_factory=lambda lock: {
            "local": LocalBackend(dispatch_lock=lock, cancel_grace=cancel_grace)
        },
    )
    app = create_app(
        settings,
        token="test-token",
        daemon_id="daemon",
        build_id="build",
        services=services,
    )
    source = Path(__file__).resolve().parents[1] / "src"

    async def run():
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://localhost:8420",
                headers={"Authorization": "Bearer test-token"},
            ) as client,
        ):
            response = await client.post(
                "/api/runs/plan",
                headers={"Idempotency-Key": "cancel-cpu"},
                json={
                    "backend": "local",
                    "operation": "train",
                    "topology": {
                        "num_machines": 1,
                        "processes_per_machine": 1,
                        "machine_rank": 0,
                    },
                    "config": {
                        "model": "cnn",
                        "epochs": 100000,
                        "patience": 100000,
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
                },
            )
            assert response.status_code == 200, response.text
            plan = response.json()
            run_id = plan["run_id"]
            response = await client.post("/api/runs", json={"plan_id": plan["plan_id"]})
            assert response.status_code == 202, response.text
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                rows = await services.event_log.read_after(run_id, limit=256)
                if any(row.type == "metric" for row in rows):
                    break
                await asyncio.sleep(0.05)
            assert any(row.type == "hello" for row in rows)
            assert any(row.type == "metric" for row in rows), rows
            response = await client.post(f"/api/runs/{run_id}/cancel")
            assert response.status_code == 202, response.text
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                response = await client.get(f"/api/runs/{run_id}")
                assert response.status_code == 200, response.text
                if response.json()["state"] in {
                    "cancelled",
                    "completed",
                    "failed",
                    "terminal_unknown",
                }:
                    break
                await asyncio.sleep(0.05)
            final = response.json()
            assert final["state"] == "cancelled", final
            assert not final["uncertainty"]
            assert final["cancellation_requested"]
            assert final["evidence"]["backend"]["native_state"] == "CANCELLED"
            assert final["evidence"]["batch"]["status"] == "cancelled"
            # The HTTP outcome derives from an owned, reaped process tree. No
            # synthetic runtime exit is inserted to make cancellation succeed.
            rows = await services.event_log.read_after(run_id, limit=256)
            exits = [row for row in rows if row.type == "exit"]
            if cancel_grace == 0:
                assert not exits, rows
                assert final["evidence"]["runtime"]["outcome"] is None
            else:
                assert all(row.payload["status"] == "cancelled" for row in exits)

    asyncio.run(run())
