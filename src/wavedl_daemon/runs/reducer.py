# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Pure reconciliation of durable intent, runtime, backend and publication.

Backend lifecycle confirms termination, not training success. Runtime success
requires terminal backend success; either independent failure makes the run
failed. Cancellation intent is not termination evidence; the sole exception is
an unclaimed PENDING execution, whose claim-before-side-effect fence proves that
submission never began. Missing/error/lost
observations retain uncertainty, even with a runtime exit or usable publication.
Events are streamed in constant space and compared only by attempt-local seq;
timestamps never determine causality. Callers supply backend snapshots already
scoped to the run whose handle they observed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from wavedl.core.events import ExitPayload, HelloPayload, StatePayload
from wavedl.core.layout import OperationKind
from wavedl_daemon.backends.contracts import (
    BackendObservation,
    Handle,
    ObservationStatus,
    OperationOutcome,
    OutcomeStatus,
    SubmissionHandle,
    SubmissionObservation,
)


if TYPE_CHECKING:
    from collections.abc import Iterable

    from wavedl.core.protocol import Envelope


class SubmissionState(StrEnum):
    PENDING = "pending"
    SUBMITTING = "submitting"
    UNKNOWN = "unknown"
    SUBMITTED = "submitted"


class RunState(StrEnum):
    PENDING = "pending"
    SUBMISSION_UNKNOWN = "submission_unknown"
    SUBMITTED = "submitted"
    QUEUED = "queued"
    HELD = "held"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TERMINAL_UNKNOWN = "terminal_unknown"
    OBSERVATION_UNAVAILABLE = "observation_unavailable"
    SUPERVISION_LOST = "supervision_lost"


@dataclass(frozen=True)
class RunIntent:
    run_id: str
    operation: OperationKind
    attempt: int | None
    backend: str
    submission: SubmissionState = SubmissionState.PENDING
    handle: Handle | SubmissionHandle | None = None
    cancel_requested: bool = False
    handshake_timed_out: bool = False

    def __post_init__(self) -> None:
        UUID(self.run_id)
        if self.attempt is not None and (
            type(self.attempt) is not int or self.attempt < 1
        ):
            raise ValueError("attempt must be a positive integer")
        if not isinstance(self.operation, OperationKind):
            raise ValueError("operation must be OperationKind")
        if not isinstance(self.submission, SubmissionState):
            raise ValueError("submission must be SubmissionState")
        if self.backend not in {"local", "slurm"}:
            raise ValueError("backend must be local or slurm")
        if (
            type(self.cancel_requested) is not bool
            or type(self.handshake_timed_out) is not bool
        ):
            raise ValueError("intent flags must be booleans")
        if self.handle is not None and (
            not isinstance(
                self.handle, SubmissionHandle if self.attempt is None else Handle
            )
            or self.handle.run_id != self.run_id
            or (self.attempt is not None and self.handle.attempt != self.attempt)
            or self.handle.backend != self.backend
        ):
            raise ValueError("handle does not identify the active execution")


@dataclass(frozen=True)
class PublicationEvidence:
    """Already verified publication identity; this reducer performs no I/O."""

    run_id: str
    attempt: int
    generations: tuple[str, ...] = ()
    detail: str = ""

    def __post_init__(self) -> None:
        UUID(self.run_id)
        if type(self.attempt) is not int or self.attempt < 1:
            raise ValueError("attempt must be a positive integer")
        if not isinstance(self.generations, tuple) or not all(
            isinstance(item, str) for item in self.generations
        ):
            raise ValueError("generations must be an immutable tuple of strings")


@dataclass(frozen=True)
class RuntimeEvidence:
    hello_seen: bool = False
    state: str | None = None
    state_seq: int = -1
    exit_seq: int = -1
    outcome: OperationOutcome | None = None
    conflicting_exits: bool = False


@dataclass(frozen=True)
class DerivedState:
    state: RunState
    confidence: Literal["confirmed", "unconfirmed"]
    reason: str
    runtime: RuntimeEvidence
    backend: BackendObservation | SubmissionObservation | None
    last_observation: BackendObservation | SubmissionObservation | None
    publication: PublicationEvidence | None
    batch: OperationOutcome | None
    stage_out: OperationOutcome | None


def _runtime(intent: RunIntent, events: Iterable[Envelope]) -> RuntimeEvidence:
    if intent.attempt is None:
        return RuntimeEvidence()
    hello = False
    state = None
    state_seq = exit_seq = -1
    outcome = None
    conflict = False
    for event in events:
        if event.run_id != intent.run_id or event.attempt != intent.attempt:
            continue
        if event.type == "hello":
            HelloPayload.from_dict(event.payload)
            hello = True
        elif event.type == "state":
            payload = StatePayload.from_dict(event.payload)
            if event.seq > state_seq:
                state, state_seq = payload.state, event.seq
        elif event.type == "exit":
            payload = ExitPayload.from_dict(event.payload)
            status = {
                "completed": OutcomeStatus.SUCCESS,
                "failed": OutcomeStatus.FAILED,
                "cancelled": OutcomeStatus.CANCELLED,
            }[payload.status]
            if status == OutcomeStatus.SUCCESS and payload.code != 0:
                status = OutcomeStatus.UNKNOWN
                conflict = True
            candidate = OperationOutcome(status, payload.code, detail=payload.message)
            if outcome is not None and (
                candidate.status != outcome.status
                or candidate.exit_code != outcome.exit_code
            ):
                conflict = True
            if event.seq > exit_seq:
                outcome, exit_seq = candidate, event.seq
    return RuntimeEvidence(hello, state, state_seq, exit_seq, outcome, conflict)


def _backend_outcome(
    observation: BackendObservation | SubmissionObservation,
) -> OutcomeStatus:
    native = (observation.native_state or "").upper().split("+", 1)[0]
    if native.startswith("CANCELLED"):
        return OutcomeStatus.CANCELLED
    if (
        observation.signal is not None
        or observation.exit_code not in {None, 0}
        or native in {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL"}
    ):
        return OutcomeStatus.FAILED
    if observation.exit_code == 0 or native == "COMPLETED":
        return OutcomeStatus.SUCCESS
    return OutcomeStatus.UNKNOWN


def reduce(
    intent: RunIntent,
    events: Iterable[Envelope],
    observation: BackendObservation | SubmissionObservation | None,
    *,
    publication: PublicationEvidence | None = None,
    last_observation: BackendObservation | SubmissionObservation | None = None,
) -> DerivedState:
    """Derive state without mutation, I/O, journal sorting or clock reads.

    ``handshake_timed_out`` is an externally measured deadline, meaningful only
    for an observed running execution; queued/held/submission time is excluded.
    Historical backend evidence is diagnostic, never a replacement for a fresh
    observation. Runtime state events cannot substitute for an exit event.
    """
    runtime = _runtime(intent, events)

    def scoped(value):
        if intent.attempt is None:
            return value if isinstance(value, SubmissionObservation) else None
        return (
            value
            if isinstance(value, BackendObservation) and value.attempt == intent.attempt
            else None
        )

    observation = scoped(observation)
    last_observation = scoped(last_observation)
    if publication is not None and (
        intent.attempt is None
        or publication.run_id != intent.run_id
        or publication.attempt != intent.attempt
    ):
        publication = None
    batch = observation.batch if observation else None
    stage_out = observation.stage_out if observation else None

    def result(state: RunState, reason: str, confirmed: bool = False) -> DerivedState:
        return DerivedState(
            state,
            "confirmed" if confirmed else "unconfirmed",
            reason,
            runtime,
            observation,
            last_observation,
            publication,
            batch,
            stage_out,
        )

    has_handle = intent.handle is not None
    if not has_handle and intent.submission in {
        SubmissionState.SUBMITTING,
        SubmissionState.UNKNOWN,
    }:
        return result(RunState.SUBMISSION_UNKNOWN, "Backend acceptance is unresolved")
    if observation is None:
        if (
            intent.submission == SubmissionState.PENDING
            and not has_handle
            and not runtime.hello_seen
            and runtime.state is None
            and runtime.outcome is None
        ):
            if (
                intent.cancel_requested
                and last_observation is None
                and publication is None
            ):
                return result(
                    RunState.CANCELLED, "Cancelled before submission was claimed", True
                )
            return result(RunState.PENDING, "Submission has not started", True)
        return result(
            RunState.OBSERVATION_UNAVAILABLE, "No current backend observation"
        )
    lifecycle = observation.status
    if lifecycle == ObservationStatus.SUPERVISION_LOST:
        return result(
            RunState.SUPERVISION_LOST,
            "Supervisor ownership was lost; PID adoption is unsafe",
        )
    if lifecycle == ObservationStatus.ERROR:
        return result(
            RunState.OBSERVATION_UNAVAILABLE,
            observation.detail or "Backend observation failed",
        )
    if lifecycle == ObservationStatus.ABSENT:
        return result(
            RunState.TERMINAL_UNKNOWN,
            "Backend absent; termination and runtime exit remain unconfirmed",
        )
    if lifecycle in {ObservationStatus.QUEUED, ObservationStatus.HELD}:
        state = (
            RunState.QUEUED if lifecycle == ObservationStatus.QUEUED else RunState.HELD
        )
        return result(
            state, "Backend allocation is not executing", runtime.outcome is None
        )
    if lifecycle == ObservationStatus.RUNNING:
        if intent.cancel_requested:
            return result(
                RunState.STOPPING,
                "Cancellation requested; termination not observed",
                True,
            )
        if intent.handshake_timed_out and not runtime.hello_seen:
            return result(
                RunState.OBSERVATION_UNAVAILABLE,
                "Running backend has no runtime handshake before deadline",
            )
        return result(
            RunState.RUNNING,
            "Backend executing; runtime exit awaits confirmation"
            if runtime.outcome
            else "Backend execution observed",
            runtime.outcome is None,
        )

    backend_status = _backend_outcome(observation)
    independent_failure = any(
        item is not None and item.status == OutcomeStatus.FAILED
        for item in (batch, stage_out)
    )
    if independent_failure:
        return result(
            RunState.FAILED,
            "Backend, batch or stage-out failure; runtime outcome preserved independently",
            True,
        )
    if backend_status == OutcomeStatus.FAILED:
        if (
            batch is not None
            and batch.status == OutcomeStatus.CANCELLED
            and not (
                runtime.outcome is not None
                and runtime.outcome.status == OutcomeStatus.FAILED
                and not runtime.conflicting_exits
            )
        ):
            return result(
                RunState.TERMINAL_UNKNOWN,
                "Backend failure contradicts batch cancellation evidence",
            )
        return result(
            RunState.FAILED,
            "Backend failure; runtime outcome preserved independently",
            True,
        )
    if runtime.conflicting_exits:
        return result(RunState.TERMINAL_UNKNOWN, "Conflicting runtime exit evidence")
    if runtime.outcome is not None and runtime.outcome.status == OutcomeStatus.FAILED:
        return result(
            RunState.FAILED, "Runtime failed despite terminal backend wrapper", True
        )
    if backend_status == OutcomeStatus.CANCELLED:
        return result(RunState.CANCELLED, "Backend cancellation observed", True)
    if (
        runtime.outcome is not None
        and runtime.outcome.status == OutcomeStatus.CANCELLED
    ):
        return result(
            RunState.CANCELLED,
            "Runtime cancellation confirmed by backend termination",
            True,
        )
    if backend_status != OutcomeStatus.SUCCESS:
        return result(
            RunState.TERMINAL_UNKNOWN, "Terminal backend has no definitive outcome"
        )
    if runtime.outcome is None or runtime.outcome.status != OutcomeStatus.SUCCESS:
        return result(
            RunState.TERMINAL_UNKNOWN,
            "Successful wrapper lacks runtime completion evidence",
        )
    if any(
        item is not None and item.status != OutcomeStatus.SUCCESS
        for item in (batch, stage_out)
    ):
        return result(
            RunState.TERMINAL_UNKNOWN,
            "Independent batch or stage-out outcome is not successful",
        )
    return result(
        RunState.COMPLETED,
        "Runtime completion and successful backend termination agree",
        True,
    )
