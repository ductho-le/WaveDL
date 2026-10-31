"""Unbound native submissions never acquire a runtime identity from telemetry."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from wavedl.core.layout import OperationKind
from wavedl.core.protocol import Envelope, ProtocolError
from wavedl_daemon.backends.contracts import (
    BackendObservation,
    Handle,
    ObservationStatus,
    SubmissionHandle,
    SubmissionObservation,
)
from wavedl_daemon.events.ingestion import Ingestor
from wavedl_daemon.runs.reducer import (
    PublicationEvidence,
    RunIntent,
    RunState,
    SubmissionState,
    reduce,
)
from wavedl_daemon.runs.runtime_service import ServiceSession
from wavedl_daemon.runs.service import RunService
from wavedl_daemon.storage.executions import ExecutionState


RUN_ID = "00000000-0000-0000-0000-000000000001"
RECEIPT = SubmissionHandle(
    "slurm",
    RUN_ID,
    RUN_ID,
    "owner",
    str(Path("/unused").absolute()),
    job_id="123;cluster",
    raw_json=json.dumps(
        {
            "cluster": "cluster",
            "comment": RUN_ID,
            "job_name": f"wavedl-{RUN_ID}",
            "restart_count": 0,
        }
    ),
)
INTENT = RunIntent(
    RUN_ID, OperationKind.TRAIN, None, "slurm", SubmissionState.SUBMITTED, RECEIPT
)


@pytest.mark.parametrize(
    ("status", "values", "expected"),
    [
        (ObservationStatus.QUEUED, {}, RunState.QUEUED),
        (ObservationStatus.RUNNING, {}, RunState.RUNNING),
        (ObservationStatus.TERMINAL, {"exit_code": 0}, RunState.TERMINAL_UNKNOWN),
        (ObservationStatus.TERMINAL, {"exit_code": 1}, RunState.FAILED),
        (ObservationStatus.TERMINAL, {"native_state": "CANCELLED"}, RunState.CANCELLED),
    ],
)
def test_unbound_native_lifecycle_ignores_runtime_and_publication(
    status, values, expected
):
    events = [
        Envelope(
            RUN_ID,
            1,
            "exit",
            1,
            "2026-10-01T00:00:00Z",
            {"status": "completed", "code": 0},
        )
    ]
    native = SubmissionObservation(status, **values)
    derived = reduce(
        INTENT,
        events,
        native,
        last_observation=native,
        publication=PublicationEvidence(RUN_ID, 1),
    )
    assert derived.state == expected
    assert derived.backend is native and derived.last_observation is native
    assert derived.runtime.outcome is None and not derived.runtime.hello_seen
    assert derived.publication is None


def test_unbound_rejects_bound_handles_and_evidence():
    bound = Handle(
        "slurm",
        RUN_ID,
        "request",
        1,
        "owner",
        str(Path("/unused").absolute()),
        job_id="123",
    )
    with pytest.raises(ValueError):
        replace(INTENT, handle=bound)
    with pytest.raises(ValueError):
        replace(INTENT, attempt=1)
    derived = reduce(INTENT, (), BackendObservation(1, ObservationStatus.RUNNING))
    assert derived.backend is None
    assert derived.state == RunState.OBSERVATION_UNAVAILABLE
    derived = reduce(
        replace(INTENT, handle=None, submission=SubmissionState.PENDING), (), None
    )
    assert derived.state == RunState.PENDING


def test_unbound_ingestion_rejects_before_runtime_or_journal_io(tmp_path):
    record = SimpleNamespace(run_id=RUN_ID, attempt=None)
    controller = Mock()
    controller.execution_store.get.return_value = record
    registry = Mock()
    pool = Mock()
    ingestor = Ingestor(registry, controller, Mock(), pool)
    for attempt in (None, 1):
        with pytest.raises(ProtocolError, match="no bound runtime attempt"):
            ingestor.push("plan", RUN_ID, attempt, [])
        with pytest.raises(ProtocolError, match="no bound runtime attempt"):
            ingestor.reject_runtime_events("plan", RUN_ID, attempt, "bad")
        with pytest.raises(ProtocolError, match="no bound runtime attempt"):
            asyncio.run(
                ingestor.poll_once("plan", RUN_ID, attempt, tmp_path / "events.jsonl")
            )
    registry.cursor.assert_not_called()
    pool.submit.assert_not_called()
    controller.accept_runtime_events.assert_not_called()
    controller.reject_runtime_events.assert_not_called()


def test_public_unbound_view_and_attempt_page_do_not_query_runtime_stores():
    native = SubmissionObservation(ObservationStatus.RUNNING)
    execution = SimpleNamespace(
        attempt=None,
        state=ExecutionState.SUBMITTED,
        handle=None,
        observation=None,
        latest_observation=None,
        submission_handle=RECEIPT,
        submission_observation=native,
        latest_submission_observation=native,
        native_identity=None,
        cancellation_requested=False,
        cancel_delivered=False,
    )
    plan = SimpleNamespace(
        run_id=RUN_ID,
        target="slurm",
        operation=OperationKind.TRAIN,
        manifest=SimpleNamespace(resolved_config={}, checkpoint_source=None),
    )
    record = SimpleNamespace(
        plan_id="plan",
        request_id=RUN_ID,
        plan=plan,
        submission_requested=True,
        cancellation_requested=False,
        created_at="2026-10-29T00:00:00+00:00",
    )
    store, controller, registry, provenance, collections = [Mock() for _ in range(5)]
    store.get.return_value = execution
    store.history_count.return_value = 0
    store.history.return_value = []
    controller.derive.return_value = reduce(INTENT, (), native)
    service = RunService(
        None,
        Mock(),
        store,
        controller,
        {},
        registry,
        provenance,
        collections=collections,
    )
    response = service._view(record)
    assert response.attempt is None and response.state == "running"
    assert response.evidence["backend"]["status"] == "running"
    registry.lifecycle_events.assert_not_called()
    service._owned = lambda run_id: record
    page = service.attempts(RUN_ID)
    assert page["attempts"][0]["attempt"] is None
    assert page["attempts"][0]["submission_handle"] == RECEIPT.to_dict()
    provenance.get.assert_not_called()
    collections.get.assert_not_called()
    store.requeue_binding.assert_not_called()


def test_runtime_pump_skips_unbound_even_if_a_foreign_handle_exists(monkeypatch):
    session = ServiceSession(None)
    session.controller = Mock()
    session.runs = Mock()
    session.runs.plan_store.list_pending.return_value = [
        SimpleNamespace(plan_id="plan")
    ]
    session.runs.execution_store.get.return_value = SimpleNamespace(
        attempt=None, handle=RECEIPT, run_id=RUN_ID
    )
    session.pool = Mock()
    session.ingestor = Mock()

    async def stop_after_lap(delay):
        raise asyncio.CancelledError

    monkeypatch.setattr(
        "wavedl_daemon.runs.runtime_service.asyncio.sleep", stop_after_lap
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(session._pump())
    session.controller.reconcile_once.assert_called_once()
    session.pool.submit.assert_not_called()
    session.ingestor.poll_once.assert_not_called()
    assert session.errors == {}
