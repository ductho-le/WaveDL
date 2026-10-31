"""Real SQLite and ASGI exercise immutable previews and durable control intent."""

import asyncio
import hashlib
import json
import sqlite3
import sys
import threading
import time
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi.routing import APIRoute

from wavedl_daemon.backends.contracts import (
    ObservationStatus,
    Plan,
    SubmissionHandle,
    SubmissionObservation,
)
from wavedl_daemon.backends.local import LocalBackend
from wavedl_daemon.execution.budget import BudgetExhausted
from wavedl_daemon.http.app import create_app
from wavedl_daemon.http.runs import lifecycle_routes
from wavedl_daemon.runs.controller import SubmissionController
from wavedl_daemon.runs.service import RunService
from wavedl_daemon.storage.executions import ExecutionState, ExecutionStore
from wavedl_daemon.storage.ownership import acquire_dispatch_lock
from wavedl_daemon.storage.plans import PlanStore, _canonical_request
from wavedl_daemon.storage.registry import Registry
from wavedl_daemon.storage.settings import Settings


AUTH = {"Authorization": "Bearer lifecycle-test-token"}


@pytest.mark.parametrize(
    "option,value",
    [
        ("resume", "/private/secret-checkpoint"),
        ("save_every", 1),
        ("save_every", 0),
        ("fresh", True),
        ("compile", True),
        ("wandb", True),
        ("wandb_watch", True),
        ("project_name", "private-project"),
        ("run_name", "private-run"),
    ],
)
@pytest.mark.parametrize("backend_name", ["local", "slurm"])
def test_cv_unsupported_plan_refused_without_persistence(
    environment, option, value, backend_name
):
    app, service, _, plans, _, backend = environment
    body = payload()
    body["backend"] = backend_name
    body["config"].update(cv=2, **{option: value})
    for _ in range(2):
        with pytest.raises(ValueError, match=f"--{option}") as error:
            service.plan(body, "cv-refused")
        assert "private" not in str(error.value)
        response = request(app, "POST", "/api/runs/plan", body=body, key="cv-refused")
        assert response.status_code == 400
        assert response.json() == {"error": "Bad request"}
        assert "private" not in response.text
        assert plans.list_plans() == []
        assert plans.get_by_key("cv-refused") is None
    assert backend.jobs == []


@pytest.mark.parametrize("backend_name", ["local", "slurm"])
def test_one_fold_cv_refused_before_plan_persistence(environment, backend_name):
    app, service, _, plans, _, backend = environment
    body = payload()
    body["backend"] = backend_name
    body["config"]["cv"] = 1
    for _ in range(2):
        with pytest.raises(ValueError, match="at least two folds"):
            service.plan(body, "invalid-fold-count")
        response = request(
            app, "POST", "/api/runs/plan", body=body, key="invalid-fold-count"
        )
        assert response.status_code == 400
        assert response.json() == {"error": "Bad request"}
        assert plans.list_plans() == []
        assert plans.get_by_key("invalid-fold-count") is None
    assert backend.jobs == []


@pytest.mark.parametrize(
    "options,flag",
    [
        ({"optimizer": "adamw", "nesterov": True}, "--optimizer"),
        ({"optimizer": "sgd", "nesterov": True, "momentum": 0}, "--momentum"),
        ({"optimizer": "sgd", "nesterov": True, "momentum": -0.1}, "--momentum"),
        ({"scheduler": "plateau", "scheduler_factor": -0.1}, "--scheduler_factor"),
        ({"scheduler": "plateau", "scheduler_factor": 1}, "--scheduler_factor"),
        ({"scheduler": "plateau", "scheduler_factor": 2}, "--scheduler_factor"),
    ],
)
@pytest.mark.parametrize("backend_name", ["local", "slurm"])
def test_cv_invalid_dependent_options_refused_before_plan(
    environment, options, flag, backend_name
):
    app, service, _, plans, _, backend = environment
    body = payload()
    body["backend"] = backend_name
    body["config"].update(cv=2, **options)
    for _ in range(2):
        with pytest.raises(ValueError, match=flag):
            service.plan(body, "cv-invalid-dependent")
        response = request(
            app, "POST", "/api/runs/plan", body=body, key="cv-invalid-dependent"
        )
        assert response.status_code == 400
        assert response.json() == {"error": "Bad request"}
        assert plans.list_plans() == []
        assert plans.get_by_key("cv-invalid-dependent") is None
        assert not service.settings.run_root.exists()
    assert backend.jobs == []


@pytest.mark.parametrize("factor", [0, 0.5])
@pytest.mark.parametrize("backend_name", ["local", "slurm"])
def test_cv_valid_dependent_options_preserved_in_http_plan(
    environment, factor, backend_name
):
    app, _, _, plans, _, backend = environment
    body = payload()
    body["backend"] = backend_name
    body["config"].update(
        cv=2,
        optimizer="sgd",
        nesterov=True,
        momentum=0.8,
        scheduler="plateau",
        scheduler_factor=factor,
    )
    response = request(
        app, "POST", "/api/runs/plan", body=body, key="cv-valid-dependent"
    )
    assert response.status_code == 200
    resolved = plans.get(response.json()["plan_id"]).plan.manifest.resolved_config
    assert all(resolved[name] == value for name, value in body["config"].items())
    assert len(plans.list_plans()) == 1
    assert backend.jobs == []


@pytest.mark.parametrize("cache_validate", ["sha256", "fast", "size"])
def test_cv_plan_preserves_supported_options_and_inapplicable_cache(
    environment, cache_validate
):
    _, service, _, plans, _, _ = environment
    body = payload()
    body["config"].update(
        cv=2,
        loss="weighted_mse",
        loss_weights="1,2",
        optimizer="sgd",
        nesterov=True,
        scheduler="multistep",
        milestones="1,2",
        grad_accum_steps=2,
        precision="no",
        pretrained=True,
        deterministic=True,
        constraint=["y0-y1"],
        constraint_weight=[0.2],
        cv_stratify=True,
        cache_validate=cache_validate,
    )
    response = service.plan(body, "cv-supported")
    resolved = response.preview.manifest.resolved_config
    assert all(resolved[name] == value for name, value in body["config"].items())
    assert resolved["save_every"] == 50
    assert len(plans.list_plans()) == 1


class FakeBackend:
    owner = "test-user"

    def __init__(self, database):
        self.database = database
        self.jobs = []
        self.lost_ack = True
        self.cancel_count = 0

    def plan(self, manifest, *, topology, allocation_mode="reserved"):
        return Plan(
            "slurm",
            json.dumps(asdict(manifest)),
            script="#!/bin/sh\nexit 0\n",
            topology=topology,
            environment=tuple(sorted(manifest.runtime.environment.items())),
            allocation_mode=allocation_mode,
        )

    def submit(self, plan, *, request_id, attempt):
        assert plan.allocation_mode == "writer" and attempt is None
        # A second connection proves intent/claim was committed before work.
        with sqlite3.connect(self.database) as connection:
            row = connection.execute(
                "SELECT c.submission_requested,s.state FROM plans p JOIN plan_control c USING(plan_id) JOIN submission_control s USING(plan_id) WHERE p.request_id=?",
                (request_id,),
            ).fetchone()
            assert row == (1, "submitting")
        handle = SubmissionHandle(
            "slurm",
            plan.run_id,
            request_id,
            self.owner,
            plan.run_dir,
            job_id="123;alpha",
            raw_json=json.dumps(
                {
                    "cluster": "alpha",
                    "comment": request_id,
                    "job_name": f"wavedl-{request_id}",
                    "restart_count": 0,
                    "operation": plan.operation.value,
                    "allocation_mode": "writer",
                    "plan_sha256": hashlib.sha256(plan.to_json().encode()).hexdigest(),
                }
            ),
        )
        self.jobs.append(handle)
        if self.lost_ack:
            self.lost_ack = False
            raise TimeoutError("Accepted, response lost")
        return handle

    def find_by_request_id(self, plan, *, request_id, attempt):
        return self.jobs[0] if self.jobs else None

    def request_cancel(self, handle):
        self.cancel_count += 1

    def observe(self, handle):
        return SubmissionObservation(ObservationStatus.RUNNING)

    def discover_initial(self, handle):
        return None

    def collect_submission(self, handle):
        return None


@pytest.fixture
def environment(tmp_path):
    database = tmp_path / "registry.db"
    settings = Settings(tmp_path / "state", tmp_path / "runs")
    backend = FakeBackend(database)
    with (
        acquire_dispatch_lock(tmp_path / "dispatch.lock") as lock,
        Registry(database, dispatch_lock=lock),
        PlanStore(database, dispatch_lock=lock) as plans,
        ExecutionStore(database, dispatch_lock=lock) as executions,
    ):
        backends = {"slurm": backend, "local": LocalBackend()}
        controller = SubmissionController(plans, executions, backends, lock)
        service = RunService(settings, plans, executions, controller, backends)
        app = create_app(
            settings, token="lifecycle-test-token", daemon_id="test", build_id="test"
        )
        app.router.routes.extend(lifecycle_routes(service))
        yield app, service, controller, plans, executions, backend
        controller.close()


def payload():
    return {
        "backend": "slurm",
        "operation": "train",
        "config": {"epochs": 2},
        "runtime": {
            "python": sys.executable,
            "mode": "existing",
            "device": "cpu",
            "environment": {},
        },
        "topology": {"num_machines": 1, "processes_per_machine": 1, "machine_rank": 0},
    }


def test_registered_checkpoint_source_plan_and_idempotent_replay(environment, tmp_path):
    from wavedl.core.checkpoint_source import capture_checkpoint_source
    from wavedl.core.generations import publish_generation
    from wavedl.core.layout import RunLayout, allocate_attempt
    from wavedl.core.manifest import write_manifest

    _, service, _, _, _, _ = environment
    request_payload = payload()
    request_payload["config"]["model"] = "cnn"
    training = service.plan(request_payload, "source")
    layout = RunLayout.create(service.settings.run_root, training.run_id)
    write_manifest(layout, training.preview.manifest)
    staging = tmp_path / "source-stage"
    staging.mkdir()
    for name in ("model_weights.pth", "training_meta.pkl", "scaler.pkl"):
        (staging / name).write_bytes(b"published-state")
    (staging / "checkpoint_selection.json").write_text(
        json.dumps({"schema_version": 1, "kind": "best"})
    )
    generation = publish_generation(
        layout, staging, attempt=allocate_attempt(layout), step=1
    )
    request_payload["operation"] = "test"
    request_payload["config"] = {"data_path": str(tmp_path / "data.npz")}
    request_payload["checkpoint_source"] = {
        "run_id": training.run_id,
        "generation": generation.generation,
    }
    captured, name, digest = service.checkpoint_selection(request_payload, "inference")
    pin = capture_checkpoint_source(captured, name, manifest_sha256=digest)
    inference = service.plan(request_payload, "inference", checkpoint_selection=pin)
    assert inference.preview.manifest.checkpoint_source == pin
    assert inference.preview.manifest.resolved_config["checkpoint"] == str(
        generation.path
    )
    (generation.path / "model_weights.pth").write_bytes(b"changed-after-plan")
    assert service.plan(request_payload, "inference") == inference
    altered = deepcopy(request_payload)
    altered["checkpoint_source"]["generation"] = "gen-999999"
    with pytest.raises(ValueError, match="Key is bound"):
        service.plan(altered, "inference")
    for invalid in (
        request_payload | {"operation": "train"},
        request_payload | {"config": request_payload["config"] | {"checkpoint": None}},
        request_payload | {"config": request_payload["config"] | {"model": "wrong"}},
        request_payload
        | {
            "checkpoint_source": {
                "run_id": str(uuid4()),
                "generation": generation.generation,
            }
        },
    ):
        with pytest.raises((ValueError, KeyError)):
            service.checkpoint_selection(invalid, "invalid-source")


def request(app, method, path, *, body=None, key=None, headers=None):
    async def run():
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://localhost:8420"
        ) as client:
            selected = dict(AUTH if headers is None else headers)
            if key is not None:
                selected["Idempotency-Key"] = key
            return await client.request(method, path, json=body, headers=selected)

    return asyncio.run(run())


def tick_until(controller, predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        controller.reconcile_once()
        if predicate():
            return
        time.sleep(0.005)
    pytest.fail("Controller did not make bounded progress")


def test_foreign_origin_cancel_does_not_mutate_then_authorized_cancel_succeeds(
    environment,
):
    app, _, _, plans, _, _ = environment
    preview = request(
        app, "POST", "/api/runs/plan", body=payload(), key="origin-cancel"
    ).json()
    path = f"/api/runs/{preview['run_id']}/cancel"
    foreign = request(app, "POST", path, headers={"Origin": "http://localhost:8421"})
    assert foreign.status_code == 403
    assert not plans.get(preview["plan_id"]).cancellation_requested
    authorized = request(app, "POST", path)
    assert authorized.status_code == 202


def test_preview_uuid_owned_pure_and_idempotent(environment, monkeypatch):
    app, service, _, plans, _, backend = environment
    monkeypatch.setenv("SECRET_RUNTIME_TOKEN", "do-not-inherit")
    body = payload()
    response = request(app, "POST", "/api/runs/plan", body=body, key="first")
    assert response.status_code == 200
    value = response.json()
    UUID(value["run_id"])
    manifest = plans.get(value["plan_id"]).plan.manifest
    assert manifest.run_directory == str(service.settings.run_root / value["run_id"])
    assert manifest.resolved_config["output_dir"] == manifest.working_directory
    assert manifest.resolved_config["precision"] == "bf16"
    assert "list_models" not in manifest.resolved_config
    assert manifest.runtime.environment == {}
    assert not service.settings.run_root.exists()
    monkeypatch.setattr(
        backend, "plan", lambda *args, **kwargs: pytest.fail("Retry replanned")
    )
    assert (
        request(app, "POST", "/api/runs/plan", body=body, key="first").json() == value
    )
    body["config"]["epochs"] = 3
    assert (
        request(app, "POST", "/api/runs/plan", body=body, key="first").status_code
        == 409
    )


def test_lost_ack_submit_retry_get_list_and_cancel_keep_uncertainty(environment):
    app, _, controller, _, executions, backend = environment
    preview = request(app, "POST", "/api/runs/plan", body=payload(), key="lost").json()
    response = request(app, "POST", "/api/runs", body={"plan_id": preview["plan_id"]})
    assert response.status_code == 202
    assert response.json()["attempt"] is None
    tick_until(
        controller,
        lambda: executions.get(preview["plan_id"]).submission_handle is not None,
    )
    for _ in range(3):
        assert (
            request(
                app, "POST", "/api/runs", body={"plan_id": preview["plan_id"]}
            ).status_code
            == 202
        )
    assert len(backend.jobs) == 1
    assert executions.get(preview["plan_id"]).attempt is None
    page = request(app, "GET", f"/api/runs/{preview['run_id']}/attempts").json()
    assert page["total"] == 1
    assert page["attempts"][0]["attempt"] is None
    assert page["attempts"][0]["handle"] is None
    assert page["attempts"][0]["submission_handle"] == backend.jobs[0].to_dict()
    cancel = request(app, "POST", f"/api/runs/{preview['run_id']}/cancel")
    assert cancel.status_code == 202
    assert cancel.json()["cancellation_requested"] is True
    assert cancel.json()["state"] != "cancelled"
    get = request(app, "GET", f"/api/runs/{preview['run_id']}").json()
    assert get["run_id"] == preview["run_id"]
    assert get["uncertainty"] is True
    assert request(app, "GET", "/api/runs").json()["runs"] == [get]


def test_cancel_before_handle_persists_without_submitting(environment):
    app, _, controller, plans, executions, backend = environment
    preview = request(
        app, "POST", "/api/runs/plan", body=payload(), key="cancel"
    ).json()
    response = request(app, "POST", f"/api/runs/{preview['run_id']}/cancel")
    assert response.status_code == 202
    assert plans.get(preview["plan_id"]).cancellation_requested
    assert response.json()["state"] == "cancelled"
    assert response.json()["uncertainty"] is False
    source = executions.get(preview["plan_id"])
    assert source.state == ExecutionState.PENDING and source.handle is None
    assert source.attempt is None
    page = request(app, "GET", f"/api/runs/{preview['run_id']}/attempts").json()
    assert page["total"] == 1
    assert page["attempts"][0]["attempt"] is None
    assert page["attempts"][0]["handle"] is None
    assert page["attempts"][0]["submission_handle"] is None
    run_dir = Path(plans.get(preview["plan_id"]).plan.run_dir)
    assert list((run_dir / "attempts").iterdir()) == []
    # A queued submit arriving after durable cancellation cannot claim a job.
    late = request(app, "POST", "/api/runs", body={"plan_id": preview["plan_id"]})
    assert late.status_code == 202
    assert late.json()["state"] == "cancelled"
    for _ in range(3):
        controller.reconcile_once()
    assert backend.jobs == []
    assert backend.cancel_count == 0
    assert executions.get(preview["plan_id"]).handle is None
    current = request(app, "GET", f"/api/runs/{preview['run_id']}").json()
    assert current["state"] == "cancelled" and current["uncertainty"] is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda body: body.update(output_dir="/tmp/foreign"),
        lambda body: body["config"].update(output_dir="/tmp/foreign"),
        lambda body: body["config"].update(config="private.yaml"),
        lambda body: body["config"].update(list_models=True),
        lambda body: body["config"].update(loss="invalid"),
        lambda body: body["config"].update(epochs=True),
        lambda body: body["runtime"].update(extra="secret"),
        lambda body: body["runtime"].update(mode="invalid"),
        lambda body: body["runtime"].update(environment={"BAD=KEY": "value"}),
        lambda body: body["runtime"].update(python="x" * 8193),
        lambda body: body["topology"].update(processes_per_machine=True),
        lambda body: body["topology"].update(num_machines=1025),
        lambda body: body["topology"].update(extra=1),
    ],
)
def test_strict_bounded_payload_rejection(environment, mutate):
    app, service, _, plans, _, _ = environment
    body = deepcopy(payload())
    mutate(body)
    response = request(app, "POST", "/api/runs/plan", body=body, key="bad")
    assert response.status_code == 400
    assert "secret" not in response.text
    assert plans.get_by_key("bad") is None
    assert not service.settings.run_root.exists()


def test_http_requires_key_auth_and_distinguishes_unknown_run(environment):
    app, *_ = environment
    assert request(app, "POST", "/api/runs/plan", body=payload()).status_code == 400
    assert (
        request(
            app, "POST", "/api/runs/plan", body=payload(), key="key", headers={}
        ).status_code
        == 401
    )
    assert request(app, "GET", f"/api/runs/{uuid4()}").status_code == 404
    assert request(app, "GET", "/api/runs/not-a-uuid").status_code == 400
    assert request(app, "GET", "/api/runs?limit=1001").status_code == 400
    assert (
        request(
            app, "POST", "/api/runs", body={"plan_id": str(uuid4()), "output": "bad"}
        ).status_code
        == 400
    )


def test_service_rejects_sql_thread_handoff(environment):
    _, service, *_ = environment
    errors = []

    def other_thread():
        try:
            service.plan(payload(), "other")
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=other_thread)
    thread.start()
    thread.join()
    assert errors == ["Run service must remain on its store owner thread"]


def test_local_plan_uses_effective_shared_defaults(environment):
    app, *_ = environment
    body = payload()
    body["backend"] = "local"
    response = request(app, "POST", "/api/runs/plan", body=body, key="local")
    assert response.status_code == 200
    assert "--list_models" not in response.json()["preview"]["argv"]
    assert response.json()["preview"]["allocation_mode"] == "writer"


def test_existing_reserved_key_replays_immutable_legacy_preview(
    environment, monkeypatch
):
    app, service, _, plans, _, backend = environment
    body = payload()
    fresh = service.plan(body, "writer-source")
    reserved = replace(fresh.preview, allocation_mode="reserved")
    old = plans.put(reserved, idempotency_key="legacy", request_payload=body)
    before = old.plan_json
    monkeypatch.setattr(
        backend, "plan", lambda *args, **kwargs: pytest.fail("Legacy retry replanned")
    )
    response = request(app, "POST", "/api/runs/plan", body=body, key="legacy")
    assert response.status_code == 200
    assert response.json()["plan_id"] == old.plan_id
    assert response.json()["preview"] == reserved.to_dict()
    assert "allocation_mode" not in response.json()["preview"]
    assert plans.get(old.plan_id).plan_json == before
    assert plans.get(old.plan_id).plan.allocation_mode == "reserved"


def test_http_rejects_backend_that_ignores_writer_allocation(environment, monkeypatch):
    app, _, _, plans, _, backend = environment
    original_plan = backend.plan

    def ignore_writer(manifest, *, topology, allocation_mode):
        del allocation_mode
        return original_plan(manifest, topology=topology)

    monkeypatch.setattr(backend, "plan", ignore_writer)
    response = request(app, "POST", "/api/runs/plan", body=payload(), key="ignored")
    assert response.status_code == 400
    assert plans.get_by_key("ignored") is None


@pytest.mark.parametrize(
    "error",
    [
        BudgetExhausted("private path"),
        PermissionError("secret.sqlite"),
        sqlite3.OperationalError("database full: secret path"),
        OSError("private credential path"),
    ],
)
def test_storage_capacity_failures_are_visible_redacted_503(
    environment, monkeypatch, error
):
    app, service, *_ = environment

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(service, "plan", fail)
    response = request(app, "POST", "/api/runs/plan", body=payload(), key="unavailable")
    assert response.status_code == 503
    assert "private" not in response.text
    assert "secret" not in response.text


def test_unknown_programming_failure_is_500_not_bad_request(environment, monkeypatch):
    app, service, *_ = environment

    def fail(*args, **kwargs):
        raise AttributeError("private implementation failure")

    monkeypatch.setattr(service, "get", fail)
    response = request(app, "GET", f"/api/runs/{uuid4()}")
    assert response.status_code == 500
    assert "private" not in response.text


def test_foreign_registry_run_requires_verified_adoption(environment, tmp_path):
    app, service, *_ = environment
    foreign_id = str(uuid4())
    with Registry(tmp_path / "foreign" / "registry.db") as registry:
        with sqlite3.connect(registry.path) as connection:
            connection.execute(
                "INSERT INTO runs(run_id, operation) VALUES (?, 'train')", (foreign_id,)
            )
        service.registry = registry
        response = request(app, "POST", f"/api/runs/{foreign_id}/cancel")
        assert response.status_code == 400
        assert request(app, "GET", f"/api/runs/{uuid4()}").status_code == 404


def test_list_includes_plans_and_filters_before_offset(environment):
    app, *_ = environment
    first = request(app, "POST", "/api/runs/plan", body=payload(), key="first").json()
    second = request(app, "POST", "/api/runs/plan", body=payload(), key="second").json()
    third = request(app, "POST", "/api/runs/plan", body=payload(), key="third").json()
    request(app, "POST", f"/api/runs/{second['run_id']}/cancel")
    response = request(app, "GET", "/api/runs?state=planned&limit=1&offset=1")
    assert response.status_code == 200
    remaining = response.json()["runs"]
    # Persisted previews are stable oldest-first; offset counts matching states.
    assert remaining[0]["run_id"] == third["run_id"]
    assert remaining[0]["run_id"] != first["run_id"]
    assert remaining[0]["run_id"] != second["run_id"]
    assert remaining[0]["state"] == "planned"


def test_duplicate_key_header_is_refused(environment):
    app, *_ = environment

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost:8420"
        ) as client:
            return await client.post(
                "/api/runs/plan",
                json=payload(),
                headers=[
                    ("authorization", AUTH["Authorization"]),
                    ("idempotency-key", "one"),
                    ("idempotency-key", "two"),
                ],
            )

    assert asyncio.run(run()).status_code == 400


def test_runs_carry_creation_time_and_approved_inputs(environment):
    app, *_ = environment
    planned = request(app, "POST", "/api/runs/plan", body=payload(), key="named").json()
    run = request(app, "GET", f"/api/runs/{planned['run_id']}").json()
    listed = request(app, "GET", "/api/runs").json()["runs"]
    assert run["created_at"].endswith("+00:00")
    assert run["summary"]["epochs"] == 2
    assert run["summary"]["source_run_id"] is None
    assert set(run["summary"]) == {
        "name",
        "model",
        "data_path",
        "epochs",
        "cv",
        "source_run_id",
    }
    assert [item["summary"] for item in listed] == [run["summary"]]


def test_plan_replay_survives_model_round_trip(environment):
    app, _, _, plans, _, _ = environment
    body = payload()
    body["config"] = {
        "epochs": 2,
        "lr": 0.001,
        "weight_decay": 1.0,
        "pretrained": True,
    }
    body["runtime"]["environment"] = {"B": "2", "A": "1"}

    first = request(app, "POST", "/api/runs/plan", body=body, key="model-round-trip")
    replay = request(app, "POST", "/api/runs/plan", body=body, key="model-round-trip")

    assert first.status_code == replay.status_code == 200
    assert replay.json()["plan_id"] == first.json()["plan_id"]
    assert plans.get_by_key("model-round-trip").request_payload_json == (
        _canonical_request(body)
    )


@pytest.mark.parametrize(
    "field",
    ["storage_map", "submission_cwd", "checkpoint_source"],
)
def test_plan_optional_fields_reject_explicit_null(environment, field):
    app, _, _, plans, _, _ = environment
    body = payload()
    body[field] = None

    response = request(app, "POST", "/api/runs/plan", body=body, key="null")

    assert response.status_code == 400
    assert response.json() == {"error": "Bad request"}
    assert plans.get_by_key("null") is None


def test_plan_rejects_lax_json(environment):
    app, *_ = environment

    async def post_raw(content):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://localhost:8420",
        ) as client:
            return await client.post(
                "/api/runs/plan",
                content=content,
                headers=AUTH | {"Content-Type": "application/json"},
            )

    body = json.dumps(payload(), separators=(",", ":"))
    duplicate_backend = body.replace(
        '{"backend":"slurm",', '{"backend":"slurm","backend":"slurm",', 1
    )
    non_finite = body.replace('"epochs":2', '"epochs":NaN', 1)
    oversized_body = payload()
    oversized_body["config"]["padding"] = "x" * 65537
    oversized = json.dumps(oversized_body, separators=(",", ":"))

    duplicate_response, non_finite_response, oversized_response = asyncio.run(
        _gather_raw(
            post_raw(duplicate_backend),
            post_raw(non_finite),
            post_raw(oversized),
        )
    )

    assert duplicate_response.status_code == 400
    assert non_finite_response.status_code == 400
    assert oversized_response.status_code == 413


async def _gather_raw(*requests_to_run):
    return await asyncio.gather(*requests_to_run)


def test_runs_queries_reject_unknown_and_repeated_keys(environment):
    app, *_ = environment
    planned = request(app, "POST", "/api/runs/plan", body=payload(), key="queries")
    run_id = planned.json()["run_id"]

    responses = [
        request(app, "GET", "/api/runs?limit=1&limit=2"),
        request(app, "GET", "/api/runs?page=1"),
        request(app, "GET", f"/api/runs/{run_id}/attempts?offset=0&offset=1"),
        request(app, "GET", "/api/runs?state=bogus"),
    ]

    assert [response.status_code for response in responses] == [400] * 4
    assert all(response.json() == {"error": "Bad request"} for response in responses)


def test_runs_responses_are_no_store(environment):
    app, *_ = environment
    planned = request(app, "POST", "/api/runs/plan", body=payload(), key="no-store")
    assert planned.status_code == 200
    assert planned.headers["cache-control"] == "no-store"
    plan = planned.json()

    submitted = request(app, "POST", "/api/runs", body={"plan_id": plan["plan_id"]})
    got = request(app, "GET", f"/api/runs/{plan['run_id']}")
    listed = request(app, "GET", "/api/runs")
    cancelled = request(app, "POST", f"/api/runs/{plan['run_id']}/cancel")
    attempts = request(app, "GET", f"/api/runs/{plan['run_id']}/attempts")

    assert submitted.status_code == 202
    assert cancelled.status_code == 202
    assert [
        response.headers.get("cache-control")
        for response in (submitted, got, listed, cancelled, attempts)
    ] == ["no-store"] * 5


def test_lifecycle_routes_are_api_routes(environment):
    _, service, *_ = environment

    assert all(isinstance(route, APIRoute) for route in lifecycle_routes(service))


def test_lifecycle_get_routes_support_head(environment):
    app, *_ = environment
    planned = request(app, "POST", "/api/runs/plan", body=payload(), key="head")
    run_id = planned.json()["run_id"]

    list_response = request(app, "HEAD", "/api/runs")
    run_response = request(app, "HEAD", f"/api/runs/{run_id}")

    assert list_response.status_code == 200
    assert list_response.content == b""
    assert run_response.status_code == 200
    assert run_response.content == b""


def test_plan_request_round_trip_preserves_json_values_and_omissions():
    from pydantic import ValidationError

    from wavedl_daemon.http.schemas.runs import PlanRequest

    body = payload()
    body["config"] = {
        "epochs": 2,
        "lr": 0.001,
        "weight_decay": 1.0,
        "pretrained": True,
        "nullable": None,
        "nested": {
            "enabled": False,
            "count": 1,
            "ratio": 1.0,
            "items": [None, True, 1, 0.001],
        },
    }
    body["runtime"]["environment"] = {"B": "2", "A": "1"}
    body["topology"] = {}
    body["storage_map"] = [{"host_root": "/host", "execution_root": "/exec"}]
    body["submission_cwd"] = "/work"
    body["checkpoint_source"] = {
        "run_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "generation": "gen-123456",
    }

    def reverse_object_keys(value):
        if isinstance(value, dict):
            return {
                key: reverse_object_keys(item)
                for key, item in reversed(list(value.items()))
            }
        if isinstance(value, list):
            return [reverse_object_keys(item) for item in value]
        return value

    reversed_body = reverse_object_keys(body)
    model = PlanRequest.model_validate(reversed_body)
    dumped = model.model_dump(mode="json", exclude_unset=True)

    assert _canonical_request(dumped) == _canonical_request(reversed_body)
    assert dumped["topology"] == {}
    assert "storage_map" in dumped
    assert "submission_cwd" in dumped
    assert dumped["checkpoint_source"]["generation"] == "gen-123456"
    assert type(dumped["checkpoint_source"]["run_id"]) is str
    assert type(model.config["epochs"]) is int
    assert type(model.config["lr"]) is float
    assert type(model.config["weight_decay"]) is float
    assert model.config["pretrained"] is True
    assert model.config["nullable"] is None
    assert model.config["nested"]["enabled"] is False
    assert type(model.config["nested"]["count"]) is int
    assert type(model.config["nested"]["ratio"]) is float

    bad_topology = deepcopy(body)
    bad_topology["topology"] = {"num_machines": True}
    with pytest.raises(ValidationError):
        PlanRequest.model_validate(bad_topology)


def test_uppercase_run_uuid_is_bad_request(environment):
    app, *_ = environment
    planned = request(app, "POST", "/api/runs/plan", body=payload(), key="uppercase")
    run_id = planned.json()["run_id"].upper()

    response = request(app, "GET", f"/api/runs/{run_id}")

    assert response.status_code == 400
    assert response.json() == {"error": "Bad request"}


def test_lifecycle_response_models_preserve_service_bodies(environment):
    app, service, _, _, _, _ = environment
    body = payload()

    planned = request(app, "POST", "/api/runs/plan", body=body, key="body-equality")
    run_id = planned.json()["run_id"]
    plan_service_body = service.plan(body, "body-equality").to_dict()
    run_service_body = service.get(run_id).to_dict()
    list_service_body = {"runs": [item.to_dict() for item in service.list()]}
    attempts_service_body = service.attempts(run_id)

    got = request(app, "GET", f"/api/runs/{run_id}")
    listed = request(app, "GET", "/api/runs")
    attempts = request(app, "GET", f"/api/runs/{run_id}/attempts")

    assert planned.json() == plan_service_body
    assert got.json() == run_service_body
    assert listed.json() == list_service_body
    assert attempts.json() == attempts_service_body


def test_run_wire_contract_has_no_placeholder_defaults():
    from wavedl_daemon.http.schemas.runs import PlanRequest, RunResponse

    plan = PlanRequest.model_json_schema()
    assert set(plan["required"]) == {
        "backend",
        "operation",
        "config",
        "runtime",
        "topology",
    }
    for field in ("storage_map", "submission_cwd", "checkpoint_source"):
        assert "default" not in plan["properties"][field]
    run = RunResponse.model_json_schema()
    assert set(run["required"]) == set(run["properties"])


def test_response_models_preserve_populated_evidence_bodies(environment):
    app, service, controller, _, executions, _ = environment
    preview = request(
        app, "POST", "/api/runs/plan", body=payload(), key="evidence-bodies"
    ).json()
    submitted = request(app, "POST", "/api/runs", body={"plan_id": preview["plan_id"]})
    assert submitted.status_code == 202
    tick_until(
        controller,
        lambda: executions.get(preview["plan_id"]).submission_handle is not None,
    )
    run_id = preview["run_id"]
    got = request(app, "GET", f"/api/runs/{run_id}")
    attempts = request(app, "GET", f"/api/runs/{run_id}/attempts")
    # The Starlette routes serialized service dicts with JSONResponse (json.dumps).
    expected_run = json.loads(json.dumps(service.get(run_id).to_dict()))
    assert expected_run["evidence"] is not None
    assert got.json() == expected_run
    assert attempts.json() == json.loads(json.dumps(service.attempts(run_id)))
    assert attempts.json()["attempts"]
