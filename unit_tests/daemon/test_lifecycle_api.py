"""Real SQLite and ASGI exercise immutable plans and the run log behind them."""

import asyncio
import json
import sqlite3
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute

from wavedl_daemon.backends.contracts import CheckpointSource
from wavedl_daemon.backends.local import LocalBackend
from wavedl_daemon.backends.slurm import SlurmBackend
from wavedl_daemon.backends.slurm_plan import SlurmSite
from wavedl_daemon.execution.budget import BudgetExhausted, WorkBudget
from wavedl_daemon.execution.pool import BoundedThreadPool
from wavedl_daemon.http.app import create_app
from wavedl_daemon.http.runs import lifecycle_routes
from wavedl_daemon.runs.controller import RunController
from wavedl_daemon.runs.service import RunService
from wavedl_daemon.storage.database import open_database
from wavedl_daemon.storage.ownership import acquire_dispatch_lock
from wavedl_daemon.storage.plans import PlanStore, canonical_request
from wavedl_daemon.storage.run_log import RunLog
from wavedl_daemon.storage.settings import Settings

from .test_hosted_app import StubSupervisor, train_payload


AUTH = {"Authorization": "Bearer lifecycle-test-token"}


@dataclass(frozen=True)
class Environment:
    app: FastAPI
    service: RunService
    controller: RunController
    plans: PlanStore
    log: RunLog
    local: LocalBackend
    supervisor: StubSupervisor


@pytest.fixture
def environment(tmp_path):
    settings = Settings(tmp_path / "state", tmp_path / "runs")
    settings.validate()
    supervisor = StubSupervisor()
    with acquire_dispatch_lock(settings.state_root / "dispatch.lock") as lock:
        connection = open_database(settings.state_root / "registry.db", dispatch_lock=lock)
        log, plans = RunLog(connection), PlanStore(connection)
        local = LocalBackend(dispatch_lock=lock, supervisor=supervisor)
        slurm = SlurmBackend(
            SlurmSite("debug", cluster="alpha"), None, owner="tester", dispatch_lock=lock
        )
        service = RunService(settings, plans, log, {"local": local, "slurm": slurm})
        pool = BoundedThreadPool(WorkBudget())
        controller = RunController(log, plans, pool, local=local)
        app = create_app(
            settings, token="lifecycle-test-token", daemon_id="test", build_id="test"
        )
        app.router.routes.extend(lifecycle_routes(service))
        try:
            yield Environment(app, service, controller, plans, log, local, supervisor)
        finally:
            local.close()
            pool.shutdown()
            connection.close()


def payload():
    return train_payload(epochs=2)


def all_plans(plans):
    return plans.page(limit=1000, offset=0)


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


def tick_until(environment, run_id, predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        environment.controller.tick()
        view = environment.service.get(run_id)
        if predicate(view):
            return view
        time.sleep(0.005)
    pytest.fail("Controller did not make bounded progress")


def kinds(environment, run_id):
    return [event.event.kind for event in environment.log.read_after(run_id, (0, -1))]


def planned_and_submitted(environment, key):
    plan = request(environment.app, "POST", "/api/runs/plan", body=payload(), key=key)
    assert plan.status_code == 200, plan.text
    plan = plan.json()
    submitted = request(environment.app, "POST", "/api/runs", body={"plan_id": plan["plan_id"]})
    assert submitted.status_code == 202, submitted.text
    return plan


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
def test_cv_unsupported_plan_refused_without_persistence(environment, option, value):
    app, service, plans = environment.app, environment.service, environment.plans
    body = payload()
    body["config"].update(cv=2, **{option: value})
    for _ in range(2):
        with pytest.raises(ValueError, match=f"--{option}") as error:
            service.plan(body, "cv-refused")
        assert "private" not in str(error.value)
        response = request(app, "POST", "/api/runs/plan", body=body, key="cv-refused")
        assert response.status_code == 400
        assert response.json() == {"error": "Bad request"}
        assert "private" not in response.text
        assert all_plans(plans) == []
        assert plans.get_by_key("cv-refused") is None


def test_one_fold_cv_refused_before_plan_persistence(environment):
    app, service, plans = environment.app, environment.service, environment.plans
    body = payload()
    body["config"]["cv"] = 1
    for _ in range(2):
        with pytest.raises(ValueError, match="at least two folds"):
            service.plan(body, "invalid-fold-count")
        response = request(
            app, "POST", "/api/runs/plan", body=body, key="invalid-fold-count"
        )
        assert response.status_code == 400
        assert response.json() == {"error": "Bad request"}
        assert all_plans(plans) == []
        assert plans.get_by_key("invalid-fold-count") is None


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
def test_cv_invalid_dependent_options_refused_before_plan(environment, options, flag):
    app, service, plans = environment.app, environment.service, environment.plans
    body = payload()
    body["config"].update(cv=2, **options)
    for _ in range(2):
        with pytest.raises(ValueError, match=flag):
            service.plan(body, "cv-invalid-dependent")
        response = request(
            app, "POST", "/api/runs/plan", body=body, key="cv-invalid-dependent"
        )
        assert response.status_code == 400
        assert response.json() == {"error": "Bad request"}
        assert all_plans(plans) == []
        assert plans.get_by_key("cv-invalid-dependent") is None
        assert not any(service.settings.run_root.iterdir())


@pytest.mark.parametrize("factor", [0, 0.5])
def test_cv_valid_dependent_options_preserved_in_http_plan(environment, factor):
    app, plans = environment.app, environment.plans
    body = payload()
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
    resolved = plans.get(response.json()["plan_id"]).plan.spec.resolved_config
    assert all(resolved[name] == value for name, value in body["config"].items())
    assert len(all_plans(plans)) == 1


@pytest.mark.parametrize("cache_validate", ["sha256", "fast", "size"])
def test_cv_plan_preserves_supported_options_and_inapplicable_cache(
    environment, cache_validate
):
    service, plans = environment.service, environment.plans
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
    resolved = response.preview.spec.resolved_config
    assert all(resolved[name] == value for name, value in body["config"].items())
    assert resolved["save_every"] == 50
    assert len(all_plans(plans)) == 1


def test_slurm_plans_preview_their_batch_script(environment):
    body = train_payload(epochs=2, model="cnn", data_path="data.npz") | {"backend": "slurm"}
    response = request(environment.app, "POST", "/api/runs/plan", body=body, key="slurm")
    assert response.status_code == 200, response.text
    preview = response.json()["preview"]
    assert preview["target"] == "slurm" and preview["argv"] == []
    assert f"#SBATCH --job-name=wavedl-{response.json()['run_id']}" in preview["script"]
    assert len(all_plans(environment.plans)) == 1


def ended_train_run(environment, key, *, returncode=0):
    run_id = planned_and_submitted(environment, key)["run_id"]
    tick_until(environment, run_id, lambda view: view.state == "running")
    environment.supervisor.launches[-1].finish(returncode)
    tick_until(environment, run_id, lambda view: view.state in {"completed", "failed"})
    return run_id


def from_checkpoint(source, tmp_path, checkpoint="best_checkpoint", **fields):
    return payload() | {
        "operation": "test",
        "config": {"data_path": str(tmp_path / "data.npz"), "model": "cnn"},
        "checkpoint_source": {"run_id": source, "checkpoint": checkpoint},
    } | fields


def test_a_test_reads_an_ended_runs_checkpoint_in_place(environment, tmp_path):
    source = ended_train_run(environment, "source")
    checkpoint = environment.service.settings.run_root / source / "output" / "best_checkpoint"
    checkpoint.mkdir(parents=True)
    response = request(
        environment.app, "POST", "/api/runs/plan", body=from_checkpoint(source, tmp_path), key="test"
    )
    assert response.status_code == 200, response.text
    plan = environment.plans.get(response.json()["plan_id"]).plan
    assert plan.spec.checkpoint_source == CheckpointSource(
        source, "best_checkpoint", str(checkpoint)
    )
    assert f"--checkpoint={checkpoint.resolve()}" in plan.argv
    run_id = response.json()["run_id"]
    assert environment.service.get(run_id).summary["source_run_id"] == source

    checkpoint.rmdir()
    submitted = request(environment.app, "POST", "/api/runs", body={"plan_id": response.json()["plan_id"]})
    assert submitted.status_code == 202
    view = tick_until(environment, run_id, lambda view: view.state == "failed")
    assert view.evidence["reason"] == f"The checkpoint best_checkpoint of run {source} is gone"
    assert len(environment.supervisor.launches) == 1


def test_a_slurm_test_gets_the_checkpoint_through_the_storage_map(environment, tmp_path):
    source = ended_train_run(environment, "source")
    run_root = environment.service.settings.run_root
    (run_root / source / "output" / "epoch_3_checkpoint").mkdir(parents=True)
    body = from_checkpoint(
        source,
        tmp_path,
        "epoch_3_checkpoint",
        backend="slurm",
        storage_map=[{"host_root": str(run_root), "execution_root": "/cluster/runs"}],
    )
    response = request(environment.app, "POST", "/api/runs/plan", body=body, key="slurm")
    assert response.status_code == 200, response.text
    spec = response.json()["preview"]["spec"]
    executed = f"/cluster/runs/{source}/output/epoch_3_checkpoint"
    assert spec["resolved_config"]["checkpoint"] == executed
    assert executed in response.json()["preview"]["script"]


def test_checkpoint_sources_are_refused_unless_ended_and_present(environment, tmp_path):
    completed = ended_train_run(environment, "completed")
    (environment.service.settings.run_root / completed / "output" / "best_checkpoint").mkdir(
        parents=True
    )
    (environment.service.settings.run_root / completed / "output" / "epoch_1_checkpoint").symlink_to(
        "best_checkpoint", target_is_directory=True
    )
    failed = ended_train_run(environment, "failed", returncode=1)
    (environment.service.settings.run_root / failed / "output" / "best_checkpoint").mkdir(
        parents=True
    )
    running = planned_and_submitted(environment, "running")["run_id"]
    tick_until(environment, running, lambda view: view.state == "running")
    planned = request(environment.app, "POST", "/api/runs/plan", body=payload(), key="planned")
    count = len(all_plans(environment.plans))
    refused = [
        from_checkpoint(running, tmp_path),
        from_checkpoint(failed, tmp_path),
        from_checkpoint(planned.json()["run_id"], tmp_path),
        from_checkpoint(str(uuid4()), tmp_path),
        from_checkpoint(completed, tmp_path, "interrupted_checkpoint"),
        from_checkpoint(completed, tmp_path, "epoch_1_checkpoint"),
        from_checkpoint(completed, tmp_path, "../best_checkpoint"),
        from_checkpoint(completed, tmp_path, "gen-000001"),
        from_checkpoint(completed, tmp_path, operation="train"),
        from_checkpoint(completed, tmp_path)
        | {"config": {"data_path": str(tmp_path / "d.npz"), "checkpoint": "/x"}},
    ]
    for index, body in enumerate(refused):
        response = request(environment.app, "POST", "/api/runs/plan", body=body, key=f"r{index}")
        assert response.status_code == 400, (index, response.text)
    assert len(all_plans(environment.plans)) == count


def test_foreign_origin_cancel_does_not_mutate_then_authorized_cancel_succeeds(
    environment,
):
    plan = planned_and_submitted(environment, "origin-cancel")
    path = f"/api/runs/{plan['run_id']}/cancel"
    foreign = request(
        environment.app, "POST", path, headers={"Origin": "http://localhost:8421"}
    )
    assert foreign.status_code == 403
    assert not environment.log.head(plan["run_id"]).cancel_requested
    authorized = request(environment.app, "POST", path)
    assert authorized.status_code == 202
    assert authorized.json()["cancellation_requested"] is True


def test_preview_uuid_owned_pure_and_idempotent(environment, monkeypatch):
    app, service, plans = environment.app, environment.service, environment.plans
    monkeypatch.setenv("SECRET_RUNTIME_TOKEN", "do-not-inherit")
    body = payload()
    response = request(app, "POST", "/api/runs/plan", body=body, key="first")
    assert response.status_code == 200
    value = response.json()
    UUID(value["run_id"])
    plan = plans.get(value["plan_id"]).plan
    run_dir = service.settings.run_root / value["run_id"]
    assert plan.spec.run_directory == str(run_dir)
    assert plan.argv[-2:] == ("--output_dir", str(run_dir / "output"))
    assert "output_dir" not in plan.spec.resolved_config
    assert plan.spec.resolved_config["precision"] == "bf16"
    assert "list_models" not in plan.spec.resolved_config
    assert plan.spec.runtime.environment == {}
    assert value["preview"] == plan.to_dict()
    assert not run_dir.exists()
    monkeypatch.setattr(
        environment.local, "plan", lambda *args, **kwargs: pytest.fail("Retry replanned")
    )
    assert request(app, "POST", "/api/runs/plan", body=body, key="first").json() == value
    body["config"]["epochs"] = 3
    assert request(app, "POST", "/api/runs/plan", body=body, key="first").status_code == 409


def test_repeated_submit_starts_one_attempt_and_one_process(environment):
    plan = planned_and_submitted(environment, "repeat")
    run_id = plan["run_id"]
    for _ in range(3):
        again = request(environment.app, "POST", "/api/runs", body={"plan_id": plan["plan_id"]})
        assert again.status_code == 202
        assert again.json()["state"] == "pending"
    assert kinds(environment, run_id) == ["attempt"]
    view = tick_until(environment, run_id, lambda view: view.state == "running")
    for _ in range(5):
        environment.controller.tick()
    assert view.attempt == 1 and view.uncertainty is False
    assert len(environment.supervisor.launches) == 1
    run_dir = environment.service.settings.run_root / run_id
    assert environment.supervisor.launches[0].spec.argv[-2:] == (
        "--metrics_jsonl",
        str(run_dir / "attempts" / "1" / "metrics.jsonl"),
    )
    environment.supervisor.launches[0].finish(0)
    view = tick_until(environment, run_id, lambda view: view.state == "completed")
    assert view.uncertainty is False
    assert view.evidence["outcome"] == "completed"
    assert kinds(environment, run_id) == [
        "attempt",
        "submitting",
        "probed",
        "submitted",
        "observed",
        "exited",
    ]


def test_cancel_of_a_planned_run_changes_nothing(environment):
    plan = request(
        environment.app, "POST", "/api/runs/plan", body=payload(), key="planned"
    ).json()
    response = request(environment.app, "POST", f"/api/runs/{plan['run_id']}/cancel")
    assert response.status_code == 202
    assert response.json()["state"] == "planned"
    assert response.json()["cancellation_requested"] is False
    assert environment.log.head(plan["run_id"]) is None


def test_cancel_before_spawn_ends_without_a_process(environment):
    plan = planned_and_submitted(environment, "cancel")
    run_id = plan["run_id"]
    response = request(environment.app, "POST", f"/api/runs/{run_id}/cancel")
    assert response.status_code == 202
    assert response.json()["cancellation_requested"] is True
    assert response.json()["state"] == "pending"
    view = tick_until(environment, run_id, lambda view: view.state == "cancelled")
    assert view.uncertainty is False
    assert kinds(environment, run_id) == ["attempt", "cancel_requested", "exited"]
    # A submit arriving after the cancel cannot start anything.
    late = request(environment.app, "POST", "/api/runs", body={"plan_id": plan["plan_id"]})
    assert late.status_code == 202 and late.json()["state"] == "cancelled"
    for _ in range(3):
        environment.controller.tick()
    assert environment.supervisor.launches == []
    assert not (environment.service.settings.run_root / run_id).exists()


def test_cancel_of_a_running_process_interrupts_and_ends_cancelled(environment):
    run_id = planned_and_submitted(environment, "running-cancel")["run_id"]
    tick_until(environment, run_id, lambda view: view.state == "running")
    response = request(environment.app, "POST", f"/api/runs/{run_id}/cancel")
    assert response.json()["state"] == "stopping"
    view = tick_until(environment, run_id, lambda view: view.state == "cancelled")
    assert environment.supervisor.launches[0].cancels == [(45.0, True)]
    assert view.uncertainty is False
    assert view.evidence["exit_code"] == 130
    assert kinds(environment, run_id)[-3:] == [
        "cancel_requested",
        "cancel_delivered",
        "exited",
    ]


def test_a_supervision_failure_is_an_unconfirmed_ending(environment, tmp_path):
    run_id = planned_and_submitted(environment, "escaped")["run_id"]
    tick_until(environment, run_id, lambda view: view.state == "running")
    (environment.service.settings.run_root / run_id / "output" / "best_checkpoint").mkdir(
        parents=True
    )
    environment.supervisor.launches[0].fail(
        TimeoutError("an escaped descendant may still be alive")
    )
    view = tick_until(environment, run_id, lambda view: view.state == "terminal_unknown")
    assert view.uncertainty is True
    assert "an escaped descendant may still be alive" in view.evidence["reason"]
    assert (view.evidence["exit_code"], view.evidence["signal"]) == (None, None)
    response = request(
        environment.app, "POST", "/api/runs/plan", body=from_checkpoint(run_id, tmp_path), key="t"
    )
    assert response.status_code == 400, response.text


def test_a_refused_interpreter_probe_fails_the_run_without_launching(environment):
    environment.supervisor.probe_returncode = 1
    run_id = planned_and_submitted(environment, "probe")["run_id"]
    view = tick_until(environment, run_id, lambda view: view.state == "failed")
    assert view.uncertainty is False
    assert "probe refused execution" in view.evidence["reason"]
    assert environment.supervisor.launches == []


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
    body = deepcopy(payload())
    mutate(body)
    response = request(environment.app, "POST", "/api/runs/plan", body=body, key="bad")
    assert response.status_code == 400
    assert "secret" not in response.text
    assert environment.plans.get_by_key("bad") is None
    assert not any(environment.service.settings.run_root.iterdir())


def test_http_requires_key_auth_and_distinguishes_unknown_run(environment):
    app = environment.app
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
    service = environment.service
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
    response = request(environment.app, "POST", "/api/runs/plan", body=payload(), key="local")
    assert response.status_code == 200
    assert "--list_models" not in response.json()["preview"]["argv"]
    assert "allocation_mode" not in response.json()["preview"]


def test_the_attempts_route_is_gone(environment):
    plan = request(environment.app, "POST", "/api/runs/plan", body=payload(), key="gone")
    response = request(environment.app, "GET", f"/api/runs/{plan.json()['run_id']}/attempts")
    assert response.status_code == 404


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
    app, service = environment.app, environment.service

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(service, "plan", fail)
    response = request(app, "POST", "/api/runs/plan", body=payload(), key="unavailable")
    assert response.status_code == 503
    assert "private" not in response.text
    assert "secret" not in response.text


def test_unknown_programming_failure_is_500_not_bad_request(environment, monkeypatch):
    app, service = environment.app, environment.service

    def fail(*args, **kwargs):
        raise AttributeError("private implementation failure")

    monkeypatch.setattr(service, "get", fail)
    response = request(app, "GET", f"/api/runs/{uuid4()}")
    assert response.status_code == 500
    assert "private" not in response.text


def test_list_pages_plans_and_filters_before_offset(environment):
    app = environment.app
    first = request(app, "POST", "/api/runs/plan", body=payload(), key="first").json()
    second = request(app, "POST", "/api/runs/plan", body=payload(), key="second").json()
    third = request(app, "POST", "/api/runs/plan", body=payload(), key="third").json()
    request(app, "POST", "/api/runs", body={"plan_id": second["plan_id"]})
    response = request(app, "GET", "/api/runs?state=planned&limit=1&offset=1")
    assert response.status_code == 200
    # Plans list oldest first; the offset counts matching runs only.
    assert [run["run_id"] for run in response.json()["runs"]] == [third["run_id"]]
    pending = request(app, "GET", "/api/runs?state=pending").json()["runs"]
    assert [run["run_id"] for run in pending] == [second["run_id"]]
    paged = request(app, "GET", "/api/runs?limit=2&offset=1").json()["runs"]
    assert [run["run_id"] for run in paged] == [second["run_id"], third["run_id"]]
    assert first["run_id"] not in [run["run_id"] for run in paged]


def test_duplicate_key_header_is_refused(environment):
    app = environment.app

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
    app = environment.app
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
    app, plans = environment.app, environment.plans
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
    assert plans.get_by_key("model-round-trip").request_json == canonical_request(body)


@pytest.mark.parametrize(
    "field",
    ["storage_map", "submission_cwd", "checkpoint_source"],
)
def test_plan_optional_fields_reject_explicit_null(environment, field):
    app, plans = environment.app, environment.plans
    body = payload()
    body[field] = None

    response = request(app, "POST", "/api/runs/plan", body=body, key="null")

    assert response.status_code == 400
    assert response.json() == {"error": "Bad request"}
    assert plans.get_by_key("null") is None


def test_plan_rejects_lax_json(environment):
    app = environment.app

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
        '{"backend":"local",', '{"backend":"local","backend":"local",', 1
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
    app = environment.app
    planned = request(app, "POST", "/api/runs/plan", body=payload(), key="queries")
    run_id = planned.json()["run_id"]

    responses = [
        request(app, "GET", "/api/runs?limit=1&limit=2"),
        request(app, "GET", "/api/runs?page=1"),
        request(app, "GET", "/api/runs?offset=0&offset=1"),
        request(app, "GET", "/api/runs?state=bogus"),
    ]

    assert [response.status_code for response in responses] == [400] * 4
    assert all(response.json() == {"error": "Bad request"} for response in responses)


def test_runs_responses_are_no_store(environment):
    app = environment.app
    planned = request(app, "POST", "/api/runs/plan", body=payload(), key="no-store")
    assert planned.status_code == 200
    assert planned.headers["cache-control"] == "no-store"
    plan = planned.json()

    submitted = request(app, "POST", "/api/runs", body={"plan_id": plan["plan_id"]})
    got = request(app, "GET", f"/api/runs/{plan['run_id']}")
    listed = request(app, "GET", "/api/runs")
    cancelled = request(app, "POST", f"/api/runs/{plan['run_id']}/cancel")

    assert submitted.status_code == 202
    assert cancelled.status_code == 202
    assert [
        response.headers.get("cache-control")
        for response in (submitted, got, listed, cancelled)
    ] == ["no-store"] * 4


def test_lifecycle_routes_are_api_routes(environment):
    routes = lifecycle_routes(environment.service)
    assert all(isinstance(route, APIRoute) for route in routes)


def test_lifecycle_get_routes_support_head(environment):
    app = environment.app
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
        "checkpoint": "best_checkpoint",
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

    assert canonical_request(dumped) == canonical_request(reversed_body)
    assert dumped["topology"] == {}
    assert "storage_map" in dumped
    assert "submission_cwd" in dumped
    assert dumped["checkpoint_source"]["checkpoint"] == "best_checkpoint"
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
    app = environment.app
    planned = request(app, "POST", "/api/runs/plan", body=payload(), key="uppercase")
    run_id = planned.json()["run_id"].upper()

    response = request(app, "GET", f"/api/runs/{run_id}")

    assert response.status_code == 400
    assert response.json() == {"error": "Bad request"}


def test_lifecycle_response_models_preserve_service_bodies(environment):
    app, service = environment.app, environment.service
    body = payload()

    planned = request(app, "POST", "/api/runs/plan", body=body, key="body-equality")
    run_id = planned.json()["run_id"]
    plan_service_body = service.plan(body, "body-equality").to_dict()
    run_service_body = service.get(run_id).to_dict()
    list_service_body = {"runs": [item.to_dict() for item in service.list()]}

    got = request(app, "GET", f"/api/runs/{run_id}")
    listed = request(app, "GET", "/api/runs")

    assert planned.json() == plan_service_body
    assert got.json() == run_service_body
    assert listed.json() == list_service_body


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
    run_id = planned_and_submitted(environment, "evidence-bodies")["run_id"]
    tick_until(environment, run_id, lambda view: view.state == "running")
    got = request(environment.app, "GET", f"/api/runs/{run_id}")
    expected = json.loads(json.dumps(environment.service.get(run_id).to_dict()))
    assert got.json() == expected
    assert set(expected["evidence"]) == {
        "reason",
        "phase",
        "native_state",
        "outcome",
        "exit_code",
        "signal",
        "attempt_exit",
        "restart_count",
    }
    assert expected["evidence"]["phase"] == "running"
