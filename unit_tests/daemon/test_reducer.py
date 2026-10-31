"""Attempt-scoped truth table; no ML, Qt, subprocess or filesystem needed."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from wavedl.core.layout import OperationKind
from wavedl.core.protocol import Envelope, ProtocolError
from wavedl_daemon.backends.contracts import (
    BackendObservation,
    Handle,
    ObservationStatus,
    OperationOutcome,
    OutcomeStatus,
)
from wavedl_daemon.runs.reducer import (
    PublicationEvidence,
    RunIntent,
    RunState,
    SubmissionState,
    reduce,
)


RUN_ID = "00000000-0000-0000-0000-000000000001"
OTHER_ID = "00000000-0000-0000-0000-000000000002"
HANDLE = Handle(
    "slurm",
    RUN_ID,
    "request",
    1,
    "owner",
    str(Path("/unused").absolute()),
    job_id="123",
)
INTENT = RunIntent(
    RUN_ID, OperationKind.TRAIN, 1, "slurm", SubmissionState.SUBMITTED, HANDLE
)


def event(kind="exit", seq=1, attempt=1, run_id=RUN_ID, **payload):
    return Envelope(run_id, attempt, kind, seq, "2026-10-01T00:00:00Z", payload)


def exited(status="completed", code=0, **kwargs):
    return event(status=status, code=code, **kwargs)


def observed(status="terminal", **kwargs):
    return BackendObservation(1, ObservationStatus(status), **kwargs)


@pytest.mark.parametrize(
    ("events", "observation", "expected", "confidence"),
    [
        ([exited()], None, RunState.OBSERVATION_UNAVAILABLE, "unconfirmed"),
        ([exited()], observed(exit_code=0), RunState.COMPLETED, "confirmed"),
        ([exited()], observed(exit_code=2), RunState.FAILED, "confirmed"),
        (
            [],
            observed("error", detail="controller timeout"),
            RunState.OBSERVATION_UNAVAILABLE,
            "unconfirmed",
        ),
        ([], observed("absent"), RunState.TERMINAL_UNKNOWN, "unconfirmed"),
        ([], observed("queued"), RunState.QUEUED, "confirmed"),
        ([], observed("held"), RunState.HELD, "confirmed"),
        ([], None, RunState.OBSERVATION_UNAVAILABLE, "unconfirmed"),
        ([], observed("supervision_lost"), RunState.SUPERVISION_LOST, "unconfirmed"),
        ([exited("failed", 1)], observed(exit_code=0), RunState.FAILED, "confirmed"),
        ([exited()], observed("absent"), RunState.TERMINAL_UNKNOWN, "unconfirmed"),
        (
            [exited("failed", 1)],
            observed("absent"),
            RunState.TERMINAL_UNKNOWN,
            "unconfirmed",
        ),
        ([], observed(exit_code=0), RunState.TERMINAL_UNKNOWN, "unconfirmed"),
        ([], observed(exit_code=1), RunState.FAILED, "confirmed"),
        ([], observed("running"), RunState.RUNNING, "confirmed"),
        ([exited()], observed("running"), RunState.RUNNING, "unconfirmed"),
        (
            [],
            observed(native_state="CANCELLED", exit_code=0),
            RunState.CANCELLED,
            "confirmed",
        ),
        (
            [exited()],
            observed(native_state="UNKNOWN"),
            RunState.TERMINAL_UNKNOWN,
            "unconfirmed",
        ),
        ([exited()], observed(signal=9), RunState.FAILED, "confirmed"),
        (
            [exited("cancelled", 130)],
            observed(exit_code=0),
            RunState.CANCELLED,
            "confirmed",
        ),
    ],
)
def test_truth_table(events, observation, expected, confidence):
    result = reduce(INTENT, iter(events), observation)
    assert result.state is expected
    assert result.confidence == confidence
    assert result.reason


def test_cancelled_unclaimed_pending_execution_is_confirmed_without_runtime_or_backend():
    intent = RunIntent(RUN_ID, OperationKind.TRAIN, 1, "local", cancel_requested=True)
    derived = reduce(intent, [], None)
    assert derived.state == RunState.CANCELLED
    assert derived.confidence == "confirmed"
    assert "before submission" in derived.reason
    assert derived.runtime.outcome is None and derived.backend is None


@pytest.mark.parametrize(
    "submission",
    [SubmissionState.SUBMITTING, SubmissionState.UNKNOWN, SubmissionState.SUBMITTED],
)
def test_cancel_intent_without_handle_after_claim_is_never_confirmed_cancel(submission):
    intent = RunIntent(
        RUN_ID,
        OperationKind.TRAIN,
        1,
        "slurm",
        submission=submission,
        cancel_requested=True,
    )
    derived = reduce(intent, [], None)
    assert derived.state != RunState.CANCELLED
    assert derived.confidence == "unconfirmed"


@pytest.mark.parametrize(
    "events",
    [
        [event("state", state="running")],
        [exited("cancelled", 130)],
    ],
)
def test_pending_cancel_with_runtime_evidence_does_not_fabricate_prelaunch_cancel(
    events,
):
    intent = RunIntent(RUN_ID, OperationKind.TRAIN, 1, "local", cancel_requested=True)
    derived = reduce(intent, events, None)
    assert derived.state == RunState.OBSERVATION_UNAVAILABLE
    assert derived.confidence == "unconfirmed"


def test_cancelled_submitted_execution_with_absent_backend_stays_uncertain():
    derived = reduce(replace(INTENT, cancel_requested=True), [], observed("absent"))
    assert derived.state == RunState.TERMINAL_UNKNOWN
    assert derived.confidence == "unconfirmed"


@pytest.mark.parametrize(
    "submission", [SubmissionState.SUBMITTING, SubmissionState.UNKNOWN]
)
def test_ambiguous_submission_never_implies_retry(submission):
    intent = replace(INTENT, handle=None, submission=submission, cancel_requested=True)
    assert reduce(intent, [], None).state is RunState.SUBMISSION_UNKNOWN


def test_pending_and_submitted_are_not_running():
    pending = replace(INTENT, handle=None, submission=SubmissionState.PENDING)
    assert reduce(pending, [], None).state is RunState.PENDING
    assert reduce(INTENT, [], None).state is RunState.OBSERVATION_UNAVAILABLE
    assert reduce(pending, [exited()], None).state is RunState.OBSERVATION_UNAVAILABLE


def test_cancel_intent_not_cancel_evidence():
    intent = replace(INTENT, cancel_requested=True)
    assert reduce(intent, [], observed("running")).state is RunState.STOPPING
    assert reduce(intent, [exited()], observed(exit_code=0)).state is RunState.COMPLETED
    assert reduce(intent, [], observed(exit_code=1)).state is RunState.FAILED
    assert reduce(intent, [], None).state is RunState.OBSERVATION_UNAVAILABLE


@pytest.mark.parametrize("status", ["queued", "held"])
def test_queued_and_held_never_start_handshake_deadline(status):
    intent = replace(INTENT, handshake_timed_out=True)
    assert reduce(intent, [], observed(status)).state.value == status


def test_running_handshake_deadline_is_uncertainty_not_execution_failure():
    intent = replace(INTENT, handshake_timed_out=True)
    assert (
        reduce(intent, [], observed("running")).state
        is RunState.OBSERVATION_UNAVAILABLE
    )
    hello = event(
        "hello",
        protocol_min=1,
        protocol_max=1,
        runtime_version="1",
        python="3.11",
        device="cpu",
        capabilities=[],
        interpreter="python",
    )
    assert reduce(intent, [hello], observed("running")).state is RunState.RUNNING


def test_requeued_attempt_excludes_all_stale_evidence():
    intent = replace(INTENT, attempt=2, handle=replace(HANDLE, attempt=2))
    publication = PublicationEvidence(RUN_ID, 1, ("gen-000001",))
    previous = observed(exit_code=0)
    result = reduce(
        intent, [exited()], previous, publication=publication, last_observation=previous
    )
    assert result.state is RunState.OBSERVATION_UNAVAILABLE
    assert result.runtime.outcome is None
    assert result.backend is None
    assert result.last_observation is None
    assert result.publication is None
    current = BackendObservation(2, ObservationStatus.RUNNING)
    assert reduce(intent, [exited("failed", 1)], current).state is RunState.RUNNING


def test_foreign_run_cannot_contaminate_active_attempt():
    result = reduce(
        INTENT, [exited("failed", 1, run_id=OTHER_ID)], observed(exit_code=0)
    )
    assert result.state is RunState.TERMINAL_UNKNOWN


@pytest.mark.parametrize("field", ["batch", "stage_out"])
def test_independent_collection_or_wrapper_failure_preserves_runtime(field):
    failure = OperationOutcome(OutcomeStatus.FAILED, 2, detail="collection failed")
    observation = observed(exit_code=0, **{field: failure})
    result = reduce(INTENT, [exited()], observation)
    assert result.state is RunState.FAILED
    assert result.runtime.outcome.status is OutcomeStatus.SUCCESS
    assert getattr(result, field) is failure
    assert result.backend.exit_code == 0


def test_unknown_collection_outcome_prevents_settled_success():
    observation = observed(
        exit_code=0, stage_out=OperationOutcome(OutcomeStatus.UNKNOWN)
    )
    assert reduce(INTENT, [exited()], observation).state is RunState.TERMINAL_UNKNOWN


@pytest.mark.parametrize("termination", [{"exit_code": 143}, {"signal": 15}])
def test_owned_native_cancellation_keeps_raw_termination_without_runtime_exit(
    termination,
):
    observation = observed(
        native_state="CANCELLED",
        batch=OperationOutcome(OutcomeStatus.CANCELLED, **termination),
        **termination,
    )
    derived = reduce(INTENT, [], observation)
    assert derived.state is RunState.CANCELLED
    assert derived.confidence == "confirmed"
    assert derived.runtime.outcome is None
    assert derived.backend is observation


@pytest.mark.parametrize("field", ["batch", "stage_out", "runtime"])
def test_independent_failure_overrides_confirmed_native_cancellation(field):
    failure = OperationOutcome(OutcomeStatus.FAILED, 7)
    observation = observed(
        native_state="CANCELLED",
        exit_code=143,
        batch=failure
        if field == "batch"
        else OperationOutcome(OutcomeStatus.CANCELLED, 143),
        stage_out=failure if field == "stage_out" else None,
    )
    derived = reduce(
        INTENT, [exited("failed", 7)] if field == "runtime" else [], observation
    )
    assert derived.state is RunState.FAILED
    assert derived.confidence == "confirmed"


@pytest.mark.parametrize("native_state", [None, "FAILED"])
def test_failed_native_state_with_cancelled_batch_is_conflicting_not_cancelled(
    native_state,
):
    observation = observed(
        native_state=native_state,
        exit_code=143,
        batch=OperationOutcome(OutcomeStatus.CANCELLED, 143),
    )
    derived = reduce(replace(INTENT, cancel_requested=True), [], observation)
    assert derived.state is RunState.TERMINAL_UNKNOWN
    assert derived.confidence == "unconfirmed"


def test_runtime_failure_resolves_failed_native_and_cancelled_batch_conflict():
    observation = observed(
        native_state="FAILED",
        exit_code=143,
        batch=OperationOutcome(OutcomeStatus.CANCELLED, 143),
    )
    derived = reduce(INTENT, [exited("failed", 7)], observation)
    assert derived.state is RunState.FAILED
    assert derived.runtime.outcome.status is OutcomeStatus.FAILED


def test_publication_is_not_execution_confirmation():
    publication = PublicationEvidence(RUN_ID, 1, ("gen-000001",))
    result = reduce(INTENT, [exited()], observed("absent"), publication=publication)
    assert result.state is RunState.TERMINAL_UNKNOWN
    assert result.publication is publication


def test_previous_success_is_diagnostic_not_current_observation():
    previous = observed(exit_code=0)
    result = reduce(INTENT, [exited()], observed("error"), last_observation=previous)
    assert result.state is RunState.OBSERVATION_UNAVAILABLE
    assert result.last_observation is previous


def test_local_supervision_loss_never_adopts_stored_pid():
    handle = replace(
        HANDLE, backend="local", job_id=None, pid=123, supervisor_id="old-owner"
    )
    intent = replace(INTENT, backend="local", handle=handle)
    result = reduce(intent, [exited()], observed("supervision_lost"))
    assert result.state is RunState.SUPERVISION_LOST
    assert result.runtime.outcome.status is OutcomeStatus.SUCCESS


def test_local_supervision_loss_preserves_durable_failed_runtime_evidence():
    """A lost Local owner remains uncertain even when the journal records failure."""
    handle = replace(
        HANDLE, backend="local", job_id=None, pid=123, supervisor_id="old-owner"
    )
    intent = replace(INTENT, backend="local", handle=handle)
    result = reduce(
        intent,
        [
            event("state", state="failed", reason="[Errno 32] Broken pipe"),
            exited("failed", 1, seq=2, reason="[Errno 32] Broken pipe"),
        ],
        observed("supervision_lost", detail="persisted PID is not process ownership"),
    )
    assert result.state is RunState.SUPERVISION_LOST
    assert result.confidence == "unconfirmed"
    assert result.runtime.outcome.status is OutcomeStatus.FAILED
    assert result.runtime.outcome.exit_code == 1


def test_conflicting_runtime_exit_evidence_stays_explicit():
    result = reduce(
        INTENT, [exited(seq=1), exited("failed", 1, seq=2)], observed(exit_code=0)
    )
    assert result.state is RunState.TERMINAL_UNKNOWN
    assert result.runtime.conflicting_exits


def test_success_status_nonzero_code_cannot_complete():
    result = reduce(INTENT, [exited(code=7)], observed(exit_code=0))
    assert result.state is RunState.TERMINAL_UNKNOWN
    assert result.runtime.conflicting_exits


def test_duplicates_idempotent_and_state_events_do_not_confirm_exit():
    one = exited()
    assert reduce(INTENT, [one], observed(exit_code=0)) == reduce(
        INTENT, [one, one], observed(exit_code=0)
    )
    state = event("state", state="completed")
    assert (
        reduce(INTENT, [state], observed(exit_code=0)).state
        is RunState.TERMINAL_UNKNOWN
    )


def test_seq_not_input_order_or_wall_clock_selects_latest_state():
    first = event("state", seq=2, state="running")
    second = replace(event("state", seq=3, state="stopping"), ts="2025-01-01T00:00:00Z")
    result = reduce(INTENT, [second, first], observed("running"))
    assert result.runtime.state == "stopping"
    assert result.runtime.state_seq == 3


def test_inputs_and_result_are_immutable_without_mutating_events():
    one = exited()
    snapshot = dict(one.payload)
    result = reduce(INTENT, [one], observed(exit_code=0))
    assert one.payload == snapshot
    with pytest.raises(FrozenInstanceError):
        INTENT.cancel_requested = True
    with pytest.raises(FrozenInstanceError):
        result.runtime.state = "failed"
    with pytest.raises(ValueError, match="active execution"):
        replace(INTENT, handle=replace(HANDLE, attempt=2))
    with pytest.raises(ValueError, match="immutable"):
        PublicationEvidence(RUN_ID, 1, ["mutable"])


def test_invalid_relevant_payload_rejected_but_stale_payload_ignored():
    invalid = event(code="0", status="completed")
    with pytest.raises(ProtocolError):
        reduce(INTENT, [invalid], observed(exit_code=0))
    assert (
        reduce(INTENT, [replace(invalid, attempt=2)], observed(exit_code=0)).state
        is RunState.TERMINAL_UNKNOWN
    )


@pytest.mark.parametrize("operation", list(OperationKind))
def test_lifecycle_is_shared_by_train_test_and_plot(operation):
    assert (
        reduce(
            replace(INTENT, operation=operation), [exited()], observed(exit_code=0)
        ).state
        is RunState.COMPLETED
    )
