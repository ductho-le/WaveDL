import asyncio
import getpass
import os
import sqlite3
import sys
import threading
from contextlib import ExitStack
from types import SimpleNamespace

import httpx
import pytest

from unit_tests.daemon.conftest import route_app
from wavedl.core.events import HelloPayload
from wavedl.core.manifest import RuntimeSelection
from wavedl.journal import RunJournal, bootstrap_run
from wavedl_daemon.execution.budget import WorkBudget
from wavedl_daemon.execution.pool import BoundedThreadPool
from wavedl_daemon.http.adoption import adoption_routes
from wavedl_daemon.runs.adoption_service import VerifiedAdoptionService
from wavedl_daemon.runs.controller import SubmissionController
from wavedl_daemon.storage.adoptions import AdoptionRegistrationStore
from wavedl_daemon.storage.executions import ExecutionState, ExecutionStore
from wavedl_daemon.storage.ownership import acquire_dispatch_lock
from wavedl_daemon.storage.plans import IdempotencyConflict, PlanStore
from wavedl_daemon.storage.registry import Registry
from wavedl_daemon.storage.runtime_provenance import RuntimeProvenanceStore


@pytest.fixture
def adoption_service(tmp_path):
    output = tmp_path / "standalone"
    output.mkdir(mode=0o700)
    bootstrap = bootstrap_run(
        output,
        resolved_config={"epochs": 1},
        runtime=RuntimeSelection(sys.executable, "selected", "cpu", {}),
        environment={},
    )
    with RunJournal(
        bootstrap,
        hello=HelloPayload(1, 1, "actual", "3.11", "cpu", ["train"], sys.executable),
    ) as journal:
        journal.emit("exit", {"code": 0, "status": "completed"})

    class NoExecution:
        owner = getpass.getuser()

        def submit(self, *args, **kwargs):
            pytest.fail("adoption must never submit")

        def cancel(self, *args, **kwargs):
            pytest.fail("adoption must never cancel")

    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    with ExitStack() as stack:
        lock = stack.enter_context(acquire_dispatch_lock(state / "dispatch.lock"))
        registry = stack.enter_context(
            Registry(state / "registry.db", dispatch_lock=lock)
        )
        plans = stack.enter_context(PlanStore(registry.path, dispatch_lock=lock))
        executions = stack.enter_context(
            ExecutionStore(registry.path, dispatch_lock=lock)
        )
        provenance = stack.enter_context(
            RuntimeProvenanceStore(registry.path, dispatch_lock=lock)
        )
        pool = stack.enter_context(BoundedThreadPool(WorkBudget()))
        backends = {"local": NoExecution()}
        controller = SubmissionController(
            plans, executions, backends, lock, provenance_store=provenance
        )
        stack.callback(controller.close)
        service = VerifiedAdoptionService(
            SimpleNamespace(),
            plans,
            executions,
            provenance,
            registry,
            controller,
            pool,
            backends,
        )
        yield service, bootstrap


def test_atomic_adoption_preserves_real_identity_and_never_claims_terminal_backend(
    adoption_service,
):
    service, bootstrap = adoption_service
    response = asyncio.run(
        service.adopt(str(bootstrap.layout.run_dir), idempotency_key="adopt-once")
    )
    record = service.plans.get(response.plan_id)
    execution = service.executions.get(response.plan_id)
    assert record.request_id == bootstrap.request_id
    assert record.plan.manifest == bootstrap.request_manifest
    assert execution.state == ExecutionState.LOCAL_LOST
    assert execution.handle.pid is execution.handle.supervisor_id is None
    assert execution.observation is None
    assert service.provenance.get(response.plan_id).event.type == "hello"
    assert response.state == "supervision_lost" and response.uncertainty
    assert service.controller.derive(response.plan_id).state == "supervision_lost"
    service.controller.reconcile_once()
    replay = asyncio.run(
        service.adopt(str(bootstrap.layout.run_dir), idempotency_key="adopt-once")
    )
    assert replay == response
    with pytest.raises(IdempotencyConflict):
        asyncio.run(
            service.adopt(str(bootstrap.layout.run_dir), idempotency_key="foreign-key")
        )


def test_mid_registration_failure_rolls_back_every_dispatchable_record(
    adoption_service, monkeypatch
):
    service, bootstrap = adoption_service
    original = AdoptionRegistrationStore._insert_execution

    def fail(*args):
        original(*args)
        raise sqlite3.OperationalError("injected storage failure")

    monkeypatch.setattr(AdoptionRegistrationStore, "_insert_execution", fail)
    with pytest.raises(sqlite3.OperationalError):
        asyncio.run(
            service.adopt(str(bootstrap.layout.run_dir), idempotency_key="rollback")
        )
    assert service.plans.get_by_key("rollback") is None
    assert service.executions.list_pending() == []
    with sqlite3.connect(service.plans.path) as connection:
        for table in (
            "plans",
            "plan_control",
            "execution_control",
            "runtime_provenance",
        ):
            assert (
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
            )
    service.controller.reconcile_once()


def test_registration_failure_closes_all_retained_files(adoption_service, monkeypatch):
    service, bootstrap = adoption_service
    from wavedl_daemon.runs import adoption_service as module

    original = module.verify_standalone_run
    proofs = []

    def verify(*args, **kwargs):
        result = original(*args, **kwargs)
        proofs.append(result)
        return result

    def fail(*args):
        raise sqlite3.OperationalError("injected storage failure")

    monkeypatch.setattr(module, "verify_standalone_run", verify)
    monkeypatch.setattr(AdoptionRegistrationStore, "_insert_execution", fail)
    with pytest.raises(sqlite3.OperationalError):
        asyncio.run(
            service.adopt(str(bootstrap.layout.run_dir), idempotency_key="close")
        )
    assert proofs
    for resource in proofs[0]._directories:
        with pytest.raises(ValueError):
            resource.guard()


def test_sql_registration_owner_thread_and_file_verification_worker(
    adoption_service, monkeypatch
):
    service, bootstrap = adoption_service
    from wavedl_daemon.runs import adoption_service as module

    real_verify, real_register = (
        module.verify_standalone_run,
        AdoptionRegistrationStore.register_adoption,
    )
    main = threading.get_ident()

    def verify(*args, **kwargs):
        assert threading.get_ident() != main
        return real_verify(*args, **kwargs)

    def register(*args, **kwargs):
        assert threading.get_ident() == main
        return real_register(*args, **kwargs)

    monkeypatch.setattr(module, "verify_standalone_run", verify)
    monkeypatch.setattr(AdoptionRegistrationStore, "register_adoption", register)
    asyncio.run(service.adopt(str(bootstrap.layout.run_dir), idempotency_key="owner"))


def test_changed_metadata_before_registration_is_refused_without_database_rows(
    adoption_service, monkeypatch
):
    service, bootstrap = adoption_service
    from wavedl_daemon.runs import adoption_service as module

    real_verify = module.verify_standalone_run
    blocked = []
    original_bytes = bootstrap.layout.manifest.read_bytes()

    def verify(*args, **kwargs):
        proof = real_verify(*args, **kwargs)
        path = bootstrap.layout.manifest
        raw = path.read_bytes()
        if os.name == "nt":
            with pytest.raises(PermissionError) as denied:
                path.unlink()
            assert denied.value.winerror == 32
            assert path.read_bytes() == raw
            blocked.append(True)
            return proof
        path.unlink()
        path.write_bytes(raw)
        return proof

    monkeypatch.setattr(module, "verify_standalone_run", verify)
    if os.name == "nt":
        asyncio.run(
            service.adopt(str(bootstrap.layout.run_dir), idempotency_key="changed")
        )
        assert blocked == [True]
        assert service.plans.get_by_key("changed") is not None
        assert bootstrap.layout.manifest.read_bytes() == original_bytes
    else:
        with pytest.raises((ValueError, OSError)):
            asyncio.run(
                service.adopt(str(bootstrap.layout.run_dir), idempotency_key="changed")
            )
        assert service.plans.get_by_key("changed") is None


def test_adoption_http_shape_idempotency_and_no_filesystem_disclosure(adoption_service):
    service, bootstrap = adoption_service

    async def check():
        app = route_app(routes=adoption_routes(service))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            body = {"path": str(bootstrap.layout.run_dir)}
            assert (await client.post("/api/runs/adopt", json=body)).status_code == 400
            invalid = await client.post(
                "/api/runs/adopt",
                json=body | {"pid": 123},
                headers={"Idempotency-Key": "http"},
            )
            assert invalid.status_code == 400
            good = await client.post(
                "/api/runs/adopt", json=body, headers={"Idempotency-Key": "http"}
            )
            assert good.status_code == 200
            assert good.json()["request_id"] == bootstrap.request_id
            assert str(bootstrap.layout.run_dir) not in good.text
            conflict = await client.post(
                "/api/runs/adopt", json=body, headers={"Idempotency-Key": "other"}
            )
            assert conflict.status_code == 409
            assert str(bootstrap.layout.run_dir) not in conflict.text

    asyncio.run(check())
