"""Authenticated hosted custom registration survives immutable-source consumers."""

import asyncio
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

from wavedl.core.checkpoint_selection import read_selection_metadata
from wavedl.core.generations import published_generations
from wavedl.core.layout import RunLayout
from wavedl.core.manifest import read_manifest
from wavedl_daemon.http.app import create_app
from wavedl_daemon.runs.runtime_service import HostedServices
from wavedl_daemon.storage.settings import Settings


pytestmark = pytest.mark.integration


@pytest.mark.parametrize("operations", [("test", "plot"), ("export",)])
def test_authenticated_hosted_custom_training_and_source_consumers(
    tmp_path, operations
):
    """Real owner-dispatched CPU children import only explicitly approved code."""
    if "export" in operations:
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

    project = tmp_path / "submission project"
    project.mkdir()
    import_log = project / "imports.jsonl"
    module = project / "custom affine.py"
    module.write_text(
        "import json, math, os\n"
        "from pathlib import Path\n"
        "from torch import nn\n"
        "from wavedl.models import BaseModel, register_model\n"
        f"with Path({str(import_log)!r}).open('a') as stream:\n"
        "    stream.write(json.dumps({'pid': os.getpid(), 'cwd': os.getcwd()}) + '\\n')\n"
        "@register_model('hosted_custom_affine')\n"
        "class Affine(BaseModel):\n"
        "    def __init__(self, in_shape, out_size, **kwargs):\n"
        "        super().__init__(in_shape, out_size)\n"
        "        self.linear = nn.Linear(math.prod(in_shape), out_size)\n"
        "    def forward(self, inputs):\n"
        "        return self.linear(inputs.flatten(1))\n"
    )
    rng = np.random.default_rng(29)
    inputs = rng.normal(size=(16, 32)).astype("float32")
    targets = np.stack((inputs[:, 0] + 0.2, inputs[:, 1] * 0.5), axis=1)
    data = project / "data.npz"
    np.savez(data, X=inputs, y=targets)
    settings = Settings(tmp_path / "state", tmp_path / "runs")
    services = HostedServices(settings)
    token = "temporary-custom-workflow-token"
    app = create_app(
        settings,
        token=token,
        daemon_id="before-hosted-ownership",
        build_id="custom-acceptance",
        services=services,
    )
    runtime = {
        "python": sys.executable,
        "mode": "existing",
        "device": "cpu",
        "environment": {
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PATH": os.defpath,
        },
    }

    def imports():
        return (
            [json.loads(line) for line in import_log.read_text().splitlines()]
            if import_log.exists()
            else []
        )

    async def run():
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://localhost:8420",
                headers={"Authorization": f"Bearer {token}"},
            ) as client,
        ):
            health = await client.get("/api/health")
            assert health.status_code == 200, health.text
            daemon_id = health.json()["daemon_id"]
            assert daemon_id != "before-hosted-ownership"
            client.headers["X-WaveDL-Daemon-ID"] = daemon_id
            denied = await client.get(
                "/api/runs", headers={"Authorization": "Bearer wrong-token"}
            )
            assert denied.status_code == 401
            foreign = await client.get(
                "/api/runs", headers={"X-WaveDL-Daemon-ID": "foreign-daemon"}
            )
            assert foreign.status_code == 409

            async def approve(payload, key):
                before = imports()
                response = await client.post(
                    "/api/runs/plan",
                    json=payload,
                    headers={"Idempotency-Key": key},
                )
                assert response.status_code == 200, response.text
                plan = response.json()
                replay = await client.post(
                    "/api/runs/plan",
                    json=payload,
                    headers={"Idempotency-Key": key},
                )
                assert replay.status_code == 200 and replay.json() == plan
                assert imports() == before, "Planning must not execute custom code"
                approved = json.loads(plan["preview"]["manifest_json"])
                assert approved["resolved_config"]["import_modules"] == [str(module)]
                assert approved["working_directory"] != str(project)
                assert "--import" in plan["preview"]["argv"]
                return plan, approved

            async def submit(plan):
                response = await client.post(
                    "/api/runs", json={"plan_id": plan["plan_id"]}
                )
                assert response.status_code == 202, response.text
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline:
                    response = await client.get(f"/api/runs/{plan['run_id']}")
                    assert response.status_code == 200, response.text
                    if response.json()["state"] in {"completed", "failed"}:
                        break
                    await asyncio.sleep(0.05)
                layout = RunLayout(settings.run_root / plan["run_id"], plan["run_id"])
                rows = await services.event_log.read_after(plan["run_id"], limit=256)
                logs = list(layout.run_dir.glob("attempts/*/*.log"))
                assert response.json()["state"] == "completed", (
                    response.text,
                    [(str(path), path.read_text()[-4000:]) for path in logs],
                )
                assert rows[0].type == "hello" and rows[-1].type == "exit"
                assert rows[0].payload["device"] == "cpu"
                assert rows[-1].payload["status"] == "completed"
                generations = published_generations(layout)
                assert generations, "Completion must include verified publication"
                listing = await client.get(f"/api/runs/{plan['run_id']}/artifacts")
                assert listing.status_code == 200, listing.text
                assert {
                    item["generation"] for item in listing.json()["generations"]
                } == {item.generation for item in generations}
                imported = imports()[-1]
                assert imported["pid"] != os.getpid()
                assert imported["cwd"] == str(layout.run_dir)
                return layout, generations, rows

            payload = {
                "backend": "local",
                "operation": "train",
                "submission_cwd": str(project),
                "topology": {
                    "num_machines": 1,
                    "processes_per_machine": 1,
                    "machine_rank": 0,
                },
                "runtime": runtime,
                "config": {
                    "model": "hosted_custom_affine",
                    "import_modules": [module.name],
                    "data_path": str(data),
                    "epochs": 1,
                    "precision": "no",
                    "workers": 0,
                    "batch_size": 4,
                },
            }
            training_plan, _approved = await approve(payload, "custom-train")
            assert not import_log.exists()
            training_layout, training_generations, _rows = await submit(training_plan)
            best = next(
                item
                for item in training_generations
                if (read_selection_metadata(item) or {}).get("kind") == "best"
            )
            immutable_files = {
                name: (best.path / name).read_bytes() for name in best.files
            }
            immutable_marker = (best.path / "COMPLETE.json").read_bytes()
            source = {"run_id": training_plan["run_id"], "generation": best.generation}
            consumer_predictions = {}

            for operation in operations:
                config = {
                    "data_path": str(data),
                    "import_modules": [module.name],
                    "workers": 0,
                    "batch_size": 4,
                    "save_predictions": True,
                }
                if operation == "export":
                    config["export"] = "onnx"
                consumer = payload | {
                    "operation": "plot" if operation == "plot" else "test",
                    "config": config,
                    "checkpoint_source": source,
                }
                plan, approved = await approve(consumer, f"custom-{operation}")
                assert approved["checkpoint_source"]["run_id"] == source["run_id"]
                assert approved["checkpoint_source"]["generation"] == best.generation
                assert (
                    approved["checkpoint_source"]["manifest_sha256"]
                    == best.manifest_sha256
                )
                assert approved["checkpoint_source"]["complete_sha256"] == (
                    hashlib.sha256(immutable_marker).hexdigest()
                )
                assert approved["resolved_config"]["checkpoint"] == str(best.path)
                layout, generations, rows = await submit(plan)
                assert layout.run_dir != training_layout.run_dir
                assert (
                    read_manifest(layout.run_dir / "manifest.json").checkpoint_source
                    == approved["checkpoint_source"]
                )
                artifact_rows = [row.payload for row in rows if row.type == "artifact"]
                assert artifact_rows
                downloads = {}
                for generation in generations:
                    for name, record in generation.files.items():
                        response = await client.get(
                            f"/api/runs/{plan['run_id']}/artifacts/"
                            f"{generation.generation}/{name}"
                        )
                        assert response.status_code == 200, response.text
                        assert len(response.content) == record.size
                        assert (
                            hashlib.sha256(response.content).hexdigest()
                            == record.sha256
                        )
                        downloads[name] = response.content
                prediction_csv = downloads["predictions.csv"]
                assert prediction_csv.splitlines()[0].decode() == (
                    "True_P0,True_P1,Pred_P0,Pred_P1,"
                    "Error_P0,AbsError_P0,Error_P1,AbsError_P1"
                )
                predictions = np.loadtxt(
                    io.BytesIO(prediction_csv), delimiter=",", skiprows=1
                )
                assert predictions.shape == (16, 8)
                assert np.isfinite(predictions).all()
                np.testing.assert_allclose(predictions[:, :2], targets, atol=1e-6)
                for parameter in range(2):
                    error = predictions[:, parameter + 2] - targets[:, parameter]
                    np.testing.assert_allclose(
                        predictions[:, 4 + 2 * parameter], error, atol=1e-6
                    )
                    np.testing.assert_allclose(
                        predictions[:, 5 + 2 * parameter], np.abs(error), atol=1e-6
                    )
                consumer_predictions[operation] = predictions[:, 2:4]
                if operation == "export":
                    import onnx
                    import onnxruntime as ort

                    exported = next(
                        value
                        for name, value in downloads.items()
                        if name.endswith(".onnx")
                    )
                    onnx.checker.check_model(onnx.load_model_from_string(exported))
                    session = ort.InferenceSession(
                        exported, providers=["CPUExecutionProvider"]
                    )
                    result = session.run(
                        None, {session.get_inputs()[0].name: inputs[:, None, :]}
                    )[0]
                    assert result.shape == targets.shape and np.isfinite(result).all()
                    np.testing.assert_allclose(
                        result, predictions[:, 2:4], rtol=1e-5, atol=1e-5
                    )
                else:
                    if operation == "plot":
                        images = [
                            value
                            for name, value in downloads.items()
                            if name.endswith(".png")
                        ]
                        assert len(images) == 10
                        for value in images:
                            with Image.open(io.BytesIO(value)) as image:
                                assert image.format == "PNG"
                                image.load()
                                assert image.width > 0 and image.height > 0
                assert (best.path / "COMPLETE.json").read_bytes() == immutable_marker
                assert all(
                    (best.path / name).read_bytes() == value
                    for name, value in immutable_files.items()
                )
                assert best.generation in {
                    item.generation for item in published_generations(training_layout)
                }
            if set(operations) == {"test", "plot"}:
                np.testing.assert_array_equal(
                    consumer_predictions["test"], consumer_predictions["plot"]
                )
            recorded = imports()
            assert len({item["pid"] for item in recorded}) == len(operations) + 1
            assert len({item["cwd"] for item in recorded}) == len(operations) + 1
            assert (
                read_manifest(training_layout.run_dir / "manifest.json").operation.value
                == "train"
            )
        assert services.host.outstanding == 0
        assert services.pool.outstanding == 0

    asyncio.run(run())
