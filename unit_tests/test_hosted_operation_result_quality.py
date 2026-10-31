"""Public HTTP evaluation failures preserve published training generations."""

import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pytest

from wavedl_daemon.http.app import create_app
from wavedl_daemon.runs.runtime_service import HostedServices
from wavedl_daemon.storage.settings import Settings


def _write_dataset(path: Path, *, kind: str) -> Path:
    if kind == "empty":
        x = np.empty((0, 32), dtype="float32")
    elif kind in {"nan", "inf"}:
        x = np.full((4, 32), np.nan if kind == "nan" else np.inf, dtype="float32")
    else:
        generator = np.random.default_rng(31)
        x = generator.normal(size=(20, 32)).astype("float32")
        y = generator.normal(size=(20, 1)).astype("float32")
    if kind == "training":
        np.savez(path, X=x, y=y)
    else:
        np.savez(path, X=x)
    return path


def _tree_digest(root: Path) -> dict[str, tuple[int, str]]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            result[str(path.relative_to(root))] = (
                path.stat().st_size,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
    return result


@pytest.mark.parametrize("failure_kind", ["nan", "inf", "empty"])
def test_public_http_evaluation_failure_preserves_training_publication(
    tmp_path, failure_kind
):
    training_data = _write_dataset(tmp_path / "training.npz", kind="training")
    evaluation_data = _write_dataset(tmp_path / "evaluation.npz", kind=failure_kind)
    settings = Settings(tmp_path / "state", tmp_path / "runs")
    services = HostedServices(settings)
    source = Path(__file__).resolve().parents[1] / "src"
    app = create_app(
        settings,
        token="test-token",
        daemon_id="quality-daemon",
        build_id="quality-build",
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
            train_payload = {
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
                    "data_path": str(training_data),
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
            }
            response = await client.post(
                "/api/runs/plan",
                json=train_payload,
                headers={"Idempotency-Key": "quality-train"},
            )
            assert response.status_code == 200, response.text
            training_plan = response.json()
            assert (
                await client.post(
                    "/api/runs", json={"plan_id": training_plan["plan_id"]}
                )
            ).status_code == 202
            training_view = await _wait_for_terminal(client, training_plan["run_id"])
            assert training_view["state"] == "completed", training_view
            assert training_view["uncertainty"] is False
            training_events = await services.event_log.read_after(
                training_plan["run_id"], limit=256
            )
            assert training_events[0].type == "hello"
            assert training_events[0].payload["device"] == "cpu"
            assert training_events[0].payload["interpreter"] == sys.executable
            assert training_events[-1].type == "exit"
            assert training_events[-1].payload["status"] == "completed"
            assert training_events[-1].payload["code"] == 0

            listing = await client.get(f"/api/runs/{training_plan['run_id']}/artifacts")
            assert listing.status_code == 200, listing.text
            generations = listing.json()["generations"]
            assert generations
            training_root = settings.run_root / training_plan["run_id"]
            best = [
                item
                for item in generations
                if json.loads(
                    (
                        training_root
                        / "checkpoints"
                        / item["generation"]
                        / "checkpoint_selection.json"
                    ).read_text()
                )["kind"]
                == "best"
            ]
            assert len(best) == 1
            selected = best[0]["generation"]
            assert any(
                name.endswith("COMPLETE.json") for name in _tree_digest(training_root)
            )
            before = _tree_digest(training_root)

            evaluation_payload = {
                **train_payload,
                "operation": "test",
                "config": {
                    "data_path": str(evaluation_data),
                    "save_predictions": True,
                    "batch_size": 8,
                    "workers": 0,
                },
                "checkpoint_source": {
                    "run_id": training_plan["run_id"],
                    "generation": selected,
                },
            }
            response = await client.post(
                "/api/runs/plan",
                json=evaluation_payload,
                headers={"Idempotency-Key": f"quality-eval-{failure_kind}"},
            )
            assert response.status_code == 200, response.text
            evaluation_plan = response.json()
            assert evaluation_plan["plan_id"] != training_plan["plan_id"]
            approved = json.loads(evaluation_plan["preview"]["manifest_json"])
            assert approved["checkpoint_source"]["run_id"] == training_plan["run_id"]
            assert approved["checkpoint_source"]["generation"] == selected
            assert approved["checkpoint_source"]["files"] == best[0]["files"]
            assert (
                approved["checkpoint_source"]["complete_sha256"]
                == hashlib.sha256(
                    (
                        training_root / "checkpoints" / selected / "COMPLETE.json"
                    ).read_bytes()
                ).hexdigest()
            )
            assert approved["resolved_config"]["checkpoint"] == str(
                training_root / "checkpoints" / selected
            )
            response = await client.post(
                "/api/runs", json={"plan_id": evaluation_plan["plan_id"]}
            )
            assert response.status_code == 202, response.text
            evaluation_view = await _wait_for_terminal(
                client, evaluation_plan["run_id"]
            )
            assert evaluation_view["state"] == "failed", evaluation_view
            assert evaluation_view["uncertainty"] is False
            assert evaluation_view["submission_requested"] is True
            assert evaluation_view["cancellation_requested"] is False
            evaluation_root = settings.run_root / evaluation_plan["run_id"]
            assert (
                json.loads((evaluation_root / "manifest.json").read_text()) == approved
            )

            rows = await services.event_log.read_after(
                evaluation_plan["run_id"], limit=256
            )
            assert sum(row.type == "exit" for row in rows) == 1
            assert rows[-1].type == "exit"
            assert rows[-1].payload["status"] == "failed"
            assert rows[-1].payload["code"] != 0
            assert rows[0].type == "hello"
            assert rows[0].payload["device"] == "cpu"
            assert rows[0].payload["interpreter"] == sys.executable
            assert any(
                row.type == "error" and row.payload["code"] == "OPERATION_FAILED"
                for row in rows
            )
            operation_error = next(row for row in rows if row.type == "error")
            assert (
                # The loader validates inputs before the model runs.
                "non-empty array"
                if failure_kind == "empty"
                else "only finite real values"
            ) in operation_error.payload["message"]
            assert not any(row.type == "artifact" for row in rows)
            assert not list(evaluation_root.rglob("COMPLETE.json"))
            assert not list(evaluation_root.rglob("*.csv"))
            assert not (
                await client.get(f"/api/runs/{evaluation_plan['run_id']}/artifacts")
            ).json()["generations"]
            assert _tree_digest(training_root) == before

            attempts = await client.get(
                f"/api/runs/{evaluation_plan['run_id']}/attempts"
            )
            assert attempts.status_code == 200, attempts.text
            assert attempts.json()["total"] == 1
            evidence = attempts.json()["attempts"][0]
            assert evidence["attempt"] == 1
            assert evidence["latest_observation"]["exit_code"] != 0
            assert evidence["runtime"]["interpreter"] == sys.executable
            for plan in (training_plan, evaluation_plan):
                record, history_count = await services.host.call(
                    lambda session, plan_id=plan["plan_id"]: (
                        session.runs.execution_store.get(plan_id),
                        session.runs.execution_store.history_count(plan_id),
                    )
                )
                assert record.attempt == 1
                assert record.handle.backend == "local"
                assert history_count == 0
                assert record.latest_observation.exit_code == (
                    0 if plan == training_plan else 1
                )
            response = await client.get("/api/runs")
            assert response.status_code == 200, response.text
            assert {item["run_id"] for item in response.json()["runs"]} == {
                training_plan["run_id"],
                evaluation_plan["run_id"],
            }

    asyncio.run(run())


async def _wait_for_terminal(client, run_id: str) -> dict:
    deadline = time.monotonic() + 60
    response = None
    while time.monotonic() < deadline:
        response = await client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200, response.text
        view = response.json()
        if view["state"] in {"completed", "failed"}:
            return view
        await asyncio.sleep(0.05)
    raise AssertionError(f"run did not reach terminal state: {response.text}")
