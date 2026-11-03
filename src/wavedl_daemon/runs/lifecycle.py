# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""The run head, its one transition table, and the pure control law.

``evolve`` is the only place a position ``(attempt, seq)`` is assigned. An
AttemptOpened starts the next attempt at seq 0 and every other event takes the
next seq of the current attempt, so nothing is ever appended to an older
attempt and commit order equals ``(attempt, seq)`` order.

Two independent tokens guard concurrent work. ``revision`` moves only with
lifecycle events and ``metrics_offset`` only with rows read from the metrics
file, so a tail append never makes a backend result stale, or the reverse.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING

from wavedl_daemon.runs.model import (
    AttemptOpened,
    BackendKind,
    CancelDelivered,
    CancelRequested,
    Exited,
    LocalExit,
    Metric,
    NotOwned,
    ObservationFailed,
    ObservationLost,
    Observed,
    Operation,
    Outcome,
    Phase,
    Probed,
    SlurmObservation,
    SourceWarning,
    Submitted,
    Submitting,
    WarningCode,
)


if TYPE_CHECKING:
    from pathlib import Path

    from wavedl_daemon.events.metrics_tail import TailChunk
    from wavedl_daemon.runs.model import (
        ExitRecord,
        FromFile,
        Handle,
        Lifecycle,
        PlanId,
        RunEvent,
        RunId,
    )


@dataclass(frozen=True)
class RunRef:
    """What a run is, fixed at submit. A run with no events yet is just this."""

    run_id: RunId
    plan_id: PlanId
    backend: BackendKind
    operation: Operation
    run_dir: Path


@dataclass(frozen=True)
class Requested:
    pass


@dataclass(frozen=True)
class Claimed:
    pass


@dataclass(frozen=True)
class Live:
    handle: Handle
    phase: Phase | None = None  # None until this attempt's first Observed
    native_state: str | None = None
    reason: str | None = None
    lost: str | None = None  # ObservationLost detail, cleared by the next Observed
    cancel_delivered: bool = False


@dataclass(frozen=True)
class Ended:
    exit: Exited
    handle: Handle | None  # kept so a requeue carries it into the next attempt


Stage = Requested | Claimed | Live | Ended


@dataclass(frozen=True)
class RunHead:
    """The fold of one run's events, cached in ``run_heads`` by the append.

    ``seq`` is the last seq of ``attempt``. ``metrics_offset`` is the
    ``source_end`` of the attempt's last row read from its metrics file.
    """

    ref: RunRef
    attempt: int
    seq: int
    revision: int
    stage: Stage
    restart_count: int | None
    cancel_requested: bool
    metrics_offset: int
    metrics_sealed: bool


class IllegalTransition(ValueError):
    pass


def evolve(head: RunHead | RunRef, event: RunEvent) -> RunHead:
    if isinstance(head, RunRef):
        if event != AttemptOpened("submit", None):
            raise IllegalTransition(f"A run must start with a submit, not {event}")
        return RunHead(head, 1, 0, 1, Requested(), None, False, 0, False)
    after = _transition(head, event)
    if isinstance(event, AttemptOpened):
        return replace(after, attempt=head.attempt + 1, seq=0, revision=head.revision + 1)
    revision = head.revision if isinstance(event, Metric | SourceWarning) else head.revision + 1
    return replace(after, seq=head.seq + 1, revision=revision)


def _transition(head: RunHead, event: RunEvent) -> RunHead:
    match head.stage, event:
        case Ended(exit=ending, handle=handle), AttemptOpened(cause="requeue") if (
            ending.outcome is Outcome.REQUEUED and event.restart_count == head.attempt
        ):
            return replace(
                head,
                stage=Live(handle),
                restart_count=event.restart_count,
                metrics_offset=0,
                metrics_sealed=False,
            )
        case Requested(), Submitting() if not head.cancel_requested:
            return replace(head, stage=Claimed())
        case Claimed(), Probed():
            return head
        case Claimed(), Submitted(handle=handle) if handle.backend is head.ref.backend:
            return replace(head, stage=Live(handle))
        case Live() as live, Observed():
            stage = replace(
                live,
                phase=event.phase,
                native_state=event.native_state,
                reason=event.reason,
                lost=None,
            )
            return replace(head, stage=stage)
        case Live() as live, ObservationLost():
            return replace(head, stage=replace(live, lost=event.detail))
        case Requested() | Claimed() | Live(), CancelRequested() if not head.cancel_requested:
            return replace(head, cancel_requested=True)
        case Live(cancel_delivered=False) as live, CancelDelivered() if head.cancel_requested:
            return replace(head, stage=replace(live, cancel_delivered=True))
        case Requested(), Exited(outcome=Outcome.CANCELLED) if head.cancel_requested:
            return _end(head, event, None)
        case Claimed(), Exited(outcome=Outcome.FAILED | Outcome.UNKNOWN | Outcome.LOST):
            return _end(head, event, None)
        case Live(handle=handle), Exited() if (
            event.outcome is not Outcome.REQUEUED or head.ref.backend is BackendKind.SLURM
        ):
            return _end(head, event, handle)
        case Claimed() | Live(), Metric() | SourceWarning() if _reads_forward(head, event):
            sealed = isinstance(event, SourceWarning) and event.seals
            return replace(head, metrics_offset=event.source_end, metrics_sealed=sealed)
    raise IllegalTransition(
        f"{type(event).__name__} is illegal in {type(head.stage).__name__} "
        f"at {head.attempt}:{head.seq}"
    )


def _end(head: RunHead, event: Exited, handle: Handle | None) -> RunHead:
    return replace(head, stage=Ended(event, handle), metrics_sealed=True)


def _reads_forward(head: RunHead, event: FromFile) -> bool:
    if head.metrics_sealed:
        return False
    if isinstance(event, SourceWarning) and event.code is WarningCode.TRUNCATED:
        return event.source_end == head.metrics_offset
    return event.source_end > head.metrics_offset


class RunState(StrEnum):
    """The browser's status vocabulary."""

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
class RunStatus:
    state: RunState
    confirmed: bool
    reason: str


_PHASE_STATES = {
    None: RunState.SUBMITTED,
    Phase.QUEUED: RunState.QUEUED,
    Phase.HELD: RunState.HELD,
    Phase.RUNNING: RunState.RUNNING,
}
_OUTCOME_STATES = {
    Outcome.COMPLETED: RunState.COMPLETED,
    Outcome.FAILED: RunState.FAILED,
    Outcome.CANCELLED: RunState.CANCELLED,
    Outcome.REQUEUED: RunState.QUEUED,
    Outcome.UNKNOWN: RunState.TERMINAL_UNKNOWN,
    Outcome.LOST: RunState.SUPERVISION_LOST,
}


def status(head: RunHead) -> RunStatus:
    match head.stage:
        case Requested():
            return RunStatus(RunState.PENDING, True, "")
        case Claimed():
            return RunStatus(RunState.SUBMISSION_UNKNOWN, False, "")
        case Live(lost=str() as detail):
            return RunStatus(RunState.OBSERVATION_UNAVAILABLE, False, detail)
        case Live(phase=Phase.RUNNING) if head.cancel_requested:
            return RunStatus(RunState.STOPPING, True, "")
        case Live(phase=phase, reason=reason):
            return RunStatus(_PHASE_STATES[phase], True, reason or "")
        case Ended(exit=ending):
            return RunStatus(_OUTCOME_STATES[ending.outcome], ending.confirmed, ending.detail)
    raise AssertionError(head.stage)


_SLURM_FAILURES = frozenset(
    {
        "FAILED",
        "TIMEOUT",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "BOOT_FAIL",
        "DEADLINE",
        "PREEMPTED",
        "REVOKED",
    }
)


def slurm_outcome(native_state: str | None, record: ExitRecord | None) -> tuple[Outcome, bool]:
    """Outcome and whether it is confirmed, for a job Slurm reports terminal."""
    words = (native_state or "").split()
    state = words[0].rstrip("+").upper() if words else ""
    if state == "CANCELLED":
        return Outcome.CANCELLED, True
    if state in _SLURM_FAILURES:
        return Outcome.FAILED, True
    if state == "COMPLETED" and record is not None:
        return (Outcome.COMPLETED if record.succeeded else Outcome.FAILED), True
    return Outcome.UNKNOWN, False


class WorkKind(StrEnum):
    SPAWN = "spawn"
    SBATCH = "sbatch"
    RECOVER = "recover"
    CANCEL = "cancel"
    TAIL = "tail"
    SETTLE = "settle"


@dataclass(frozen=True)
class Work:
    """One blocking unit for the pool, with the head tokens it was read under."""

    kind: WorkKind
    run_id: RunId
    revision: int
    attempt: int
    offset: int


@dataclass(frozen=True)
class Append:
    """Lifecycle events to commit now under the head's revision.

    ``then`` starts only after that commit, so a claim always lands before the
    side effect it fences.
    """

    events: tuple[Lifecycle, ...]
    then: WorkKind | None = None


Step = Work | Append | None

_SUBMIT = {BackendKind.LOCAL: WorkKind.SPAWN, BackendKind.SLURM: WorkKind.SBATCH}


def _work(head: RunHead, kind: WorkKind) -> Work:
    return Work(kind, head.ref.run_id, head.revision, head.attempt, head.metrics_offset)


def next_work(head: RunHead, *, local_poll: LocalExit | NotOwned | None) -> Step:
    """What to do for a head with no work in flight.

    A Claimed head with nothing in flight was left by a crash between the
    claim and its reply, so it gets RECOVER. ``local_poll`` is None for a
    running Local process and for every Slurm head, whose observations come
    in batches through ``react``.
    """
    match head.stage:
        case Requested() if head.cancel_requested:
            ending = Exited(Outcome.CANCELLED, True, detail="Cancelled before submission")
            return Append((ending,))
        case Requested():
            return Append((Submitting(),), then=_SUBMIT[head.ref.backend])
        case Claimed():
            return _work(head, WorkKind.RECOVER)
        case Live() if local_poll is not None:
            return _work(head, WorkKind.SETTLE)
        case Live(cancel_delivered=False) if head.cancel_requested:
            return _work(head, WorkKind.CANCEL)
        case Live() if head.ref.operation is Operation.TRAIN and not head.metrics_sealed:
            return _work(head, WorkKind.TAIL)
    return None


def react(head: RunHead, observation: SlurmObservation | ObservationFailed) -> Step:
    """Fold one fleet observation of a Live Slurm head. Unchanged appends nothing."""
    live = head.stage
    if not isinstance(live, Live):
        return None
    match observation:
        case ObservationFailed(detail=detail):
            return None if live.lost is not None else Append((ObservationLost(detail),))
        case SlurmObservation(terminal=True):
            return _work(head, WorkKind.SETTLE)
        case SlurmObservation(restart_count=restarts) if restarts >= head.attempt:
            return _work(head, WorkKind.SETTLE)
        case SlurmObservation(phase=None):
            return None
        case SlurmObservation(phase=phase, native_state=native) if (
            live.lost is None and (phase, native) == (live.phase, live.native_state)
        ):
            return None
    return Append((Observed(observation.phase, observation.native_state, observation.reason),))


@dataclass(frozen=True)
class AttemptFiles:
    tail: TailChunk | None  # None when the run has no metrics file to drain
    exit_record: ExitRecord | None  # Slurm only


def settle(
    head: RunHead,
    cause: LocalExit | NotOwned | SlurmObservation,
    files: AttemptFiles,
) -> tuple[RunEvent, ...]:
    """End the current attempt once its metrics file is drained.

    Until then it returns only the rows read so far, and the next settle
    continues from the new bookmark. The writer is dead by now, so bytes left
    without a newline are a torn line, not a line in progress. A requeue opens
    the next attempt in the same append.
    """
    tail = files.tail
    if tail is None or head.metrics_sealed:
        rows: tuple[FromFile, ...] = ()
    elif not tail.drained:
        return tail.rows
    elif tail.pending and not tail.sealed:
        torn = SourceWarning(
            WarningCode.TORN_TAIL,
            f"{tail.pending} bytes after the last complete metrics line",
            tail.size,
        )
        rows = (*tail.rows, torn)
    else:
        rows = tail.rows
    return (*rows, *_ending(head, cause, files.exit_record))


def _ending(
    head: RunHead,
    cause: LocalExit | NotOwned | SlurmObservation,
    record: ExitRecord | None,
) -> tuple[Lifecycle, ...]:
    match cause:
        case LocalExit(returncode=0):
            return (Exited(Outcome.COMPLETED, True, exit_code=0),)
        case LocalExit(returncode=code, signal=signal):
            outcome = Outcome.CANCELLED if head.cancel_requested else Outcome.FAILED
            return (Exited(outcome, True, exit_code=code, signal=signal),)
        case NotOwned():
            detail = "The daemon restarted and no longer supervises this process"
            return (Exited(Outcome.LOST, False, detail=detail),)
        case SlurmObservation(restart_count=restarts) if restarts >= head.attempt:
            ending = Exited(
                Outcome.REQUEUED,
                False,
                native_state=cause.earlier.get(head.attempt - 1),
                attempt_exit=record,
                detail="Requeued by Slurm",
            )
            opened = AttemptOpened("requeue", head.attempt)
            if restarts > head.attempt or cause.phase is None:
                return (ending, opened)
            return (ending, opened, Observed(cause.phase, cause.native_state, cause.reason))
        case SlurmObservation():
            outcome, confirmed = slurm_outcome(cause.native_state, record)
            return (
                Exited(
                    outcome,
                    confirmed,
                    exit_code=cause.exit_code,
                    signal=cause.signal,
                    native_state=cause.native_state,
                    attempt_exit=record,
                ),
            )
    raise AssertionError(cause)
