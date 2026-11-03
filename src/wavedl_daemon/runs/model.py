# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Daemon-authored run events, and the handles and observations they carry.

The daemon is the only author of a run's history. Each event class names its
wire ``kind``, and ``payload``/``parse_event`` are the only translation to and
from JSON, shared by the run log's rows and the SSE frames that replay them.
``source_end`` is a column, not payload, so byte offsets never reach a browser.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Literal, NewType


if TYPE_CHECKING:
    from collections.abc import Mapping


RunId = NewType("RunId", str)
PlanId = NewType("PlanId", str)


class Operation(StrEnum):
    TRAIN = "train"
    TEST = "test"
    PLOT = "plot"


class BackendKind(StrEnum):
    LOCAL = "local"
    SLURM = "slurm"


@dataclass(frozen=True)
class LocalHandle:
    backend: ClassVar[BackendKind] = BackendKind.LOCAL
    pid: int
    supervisor_id: str  # random per spawn; only the spawning daemon process owns it


@dataclass(frozen=True)
class SlurmHandle:
    backend: ClassVar[BackendKind] = BackendKind.SLURM
    job_id: str
    cluster: str
    owner: str


Handle = LocalHandle | SlurmHandle
_HANDLES: dict[str, type[Handle]] = {
    BackendKind.LOCAL: LocalHandle,
    BackendKind.SLURM: SlurmHandle,
}


def handle_payload(handle: Handle) -> dict[str, object]:
    return {"backend": handle.backend.value, **asdict(handle)}


def parse_handle(payload: Mapping[str, object]) -> Handle:
    fields = dict(payload)
    return _HANDLES[str(fields.pop("backend"))](**fields)


class Phase(StrEnum):
    QUEUED = "queued"
    HELD = "held"
    RUNNING = "running"


class Outcome(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REQUEUED = "requeued"  # never a run's last word: the next attempt opens with it
    UNKNOWN = "unknown"
    LOST = "lost"  # a Local supervisor is gone after a daemon restart


@dataclass(frozen=True)
class ExitRecord:
    """The exit codes bash writes to ``attempts/N/exit`` at the end of a job."""

    batch: int
    training: int
    stage_out: int

    @property
    def succeeded(self) -> bool:
        return (self.batch, self.training, self.stage_out) == (0, 0, 0)


class _Event:
    kind: ClassVar[str]

    def payload(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, object], source_end: int | None):
        return cls(**payload)


@dataclass(frozen=True)
class AttemptOpened(_Event):
    """Always seq 0 of its attempt, and the only event that moves the attempt."""

    kind: ClassVar[str] = "attempt"
    cause: Literal["submit", "requeue"]
    restart_count: int | None  # the Slurm restart that produced it; None on submit


@dataclass(frozen=True)
class Submitting(_Event):
    """The claim, committed before any side effect (mkdir and spawn, or sbatch)."""

    kind: ClassVar[str] = "submitting"


@dataclass(frozen=True)
class Probed(_Event):
    """The Local pre-spawn probe: the only interpreter provenance the daemon keeps."""

    kind: ClassVar[str] = "probed"
    python: str
    interpreter: str
    prefix: str
    devices: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload, source_end):
        return cls(**{**payload, "devices": tuple(payload["devices"])})


@dataclass(frozen=True)
class Submitted(_Event):
    kind: ClassVar[str] = "submitted"
    handle: Handle

    def payload(self):
        return {"handle": handle_payload(self.handle)}

    @classmethod
    def from_payload(cls, payload, source_end):
        return cls(parse_handle(payload["handle"]))


@dataclass(frozen=True)
class Observed(_Event):
    kind: ClassVar[str] = "observed"
    phase: Phase
    native_state: str | None
    reason: str | None

    @classmethod
    def from_payload(cls, payload, source_end):
        return cls(**{**payload, "phase": Phase(payload["phase"])})


@dataclass(frozen=True)
class ObservationLost(_Event):
    kind: ClassVar[str] = "observation_lost"
    detail: str


@dataclass(frozen=True)
class CancelRequested(_Event):
    kind: ClassVar[str] = "cancel_requested"


@dataclass(frozen=True)
class CancelDelivered(_Event):
    """SIGTERM reached the Local group, or scancel ran."""

    kind: ClassVar[str] = "cancel_delivered"


@dataclass(frozen=True)
class Exited(_Event):
    """Ends its attempt. Only the next attempt's AttemptOpened may follow."""

    kind: ClassVar[str] = "exited"
    outcome: Outcome
    confirmed: bool
    exit_code: int | None = None
    signal: int | None = None
    native_state: str | None = None
    attempt_exit: ExitRecord | None = None
    detail: str = ""

    @classmethod
    def from_payload(cls, payload, source_end):
        record = payload["attempt_exit"]
        return cls(
            **{
                **payload,
                "outcome": Outcome(payload["outcome"]),
                "attempt_exit": None if record is None else ExitRecord(**record),
            }
        )


@dataclass(frozen=True)
class Metric(_Event):
    """One metrics.jsonl line. Non-finite values are already None."""

    kind: ClassVar[str] = "metric"
    epoch: int
    stats: Mapping[str, float | int | None]
    source_end: int  # byte offset just past this line's newline

    def payload(self):
        return {
            "name": "epoch",
            "step": self.epoch,
            "value": self.stats.get("val_loss"),
            "unit": "",
            "metrics": dict(self.stats),
        }

    @classmethod
    def from_payload(cls, payload, source_end):
        return cls(payload["step"], payload["metrics"], source_end)


@dataclass(frozen=True)
class Evaluation(_Event):
    """The metrics ``wavedl-test`` saved to ``attempts/N/evaluation.json``.

    It is read once at settle, so it has no bookmark. On the wire it is a
    ``metric`` named ``evaluation``, which is what the browser reads.
    """

    kind: ClassVar[str] = "metric"
    name: ClassVar[str] = "evaluation"
    metrics: Mapping[str, float | int | None]

    def payload(self):
        return {
            "name": self.name,
            "step": 0,
            "value": self.metrics.get("r2_score"),
            "unit": "",
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_payload(cls, payload, source_end):
        return cls(payload["metrics"])


class WarningCode(StrEnum):
    MALFORMED_LINE = "metrics_malformed_line"
    TORN_TAIL = "metrics_torn_tail"
    TRUNCATED = "metrics_truncated"
    LINE_TOO_LONG = "metrics_line_too_long"
    EVALUATION_MISSING = "evaluation_missing"
    EVALUATION_MALFORMED = "evaluation_malformed"


_SEALING = frozenset(
    {WarningCode.TORN_TAIL, WarningCode.TRUNCATED, WarningCode.LINE_TOO_LONG}
)


@dataclass(frozen=True)
class SourceWarning(_Event):
    kind: ClassVar[str] = "warning"
    code: WarningCode
    message: str
    source_end: int | None  # None for a file read once at settle, not tailed

    @property
    def seals(self) -> bool:
        """Nothing more is read from the attempt's file after this warning."""
        return self.code in _SEALING

    def payload(self):
        return {"code": self.code.value, "message": self.message}

    @classmethod
    def from_payload(cls, payload, source_end):
        return cls(WarningCode(payload["code"]), payload["message"], source_end)


Lifecycle = (
    AttemptOpened
    | Submitting
    | Probed
    | Submitted
    | Observed
    | ObservationLost
    | CancelRequested
    | CancelDelivered
    | Exited
)
FromFile = Metric | Evaluation | SourceWarning
RunEvent = Lifecycle | FromFile

_EVENTS: dict[str, type[RunEvent]] = {
    event.kind: event
    for event in (
        AttemptOpened,
        Submitting,
        Probed,
        Submitted,
        Observed,
        ObservationLost,
        CancelRequested,
        CancelDelivered,
        Exited,
        Metric,
        SourceWarning,
    )
}


def parse_event(kind: str, payload: dict[str, object], source_end: int | None) -> RunEvent:
    if kind == Evaluation.kind and payload.get("name") == Evaluation.name:
        return Evaluation.from_payload(payload, source_end)
    return _EVENTS[kind].from_payload(payload, source_end)


@dataclass(frozen=True)
class StoredEvent:
    run_id: RunId
    attempt: int
    seq: int
    ts: str  # daemon clock at append, never a trainer clock
    event: RunEvent


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class LocalExit:
    returncode: int | None
    signal: int | None


@dataclass(frozen=True)
class SupervisionFailed:
    """Supervision ended without exit evidence; the process tree may live on."""

    detail: str


@dataclass(frozen=True)
class NotOwned:
    """The handle's supervisor does not exist in this daemon process."""


LocalPoll = LocalExit | SupervisionFailed | NotOwned


@dataclass(frozen=True)
class SlurmObservation:
    restart_count: int
    phase: Phase | None  # None when terminal or not yet visible
    terminal: bool
    native_state: str | None
    reason: str | None
    exit_code: int | None
    signal: int | None
    earlier: Mapping[int, str]  # restart -> native state, from sacct --duplicates


@dataclass(frozen=True)
class ObservationFailed:
    detail: str
