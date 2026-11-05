"""
Pydantic wire schemas for the run lifecycle endpoints.
Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import MISSING, Field, JsonValue

from wavedl_daemon.http.schemas.common import WireModel
from wavedl_daemon.runs.lifecycle import RunState


RunStateFilter = Literal["planned", *(state.value for state in RunState)]


class RuntimeRequest(WireModel):
    """Caller-selected interpreter, mode, device, and explicit environment."""

    python: str
    mode: str
    device: str
    environment: dict[str, str]


class TopologyRequest(WireModel):
    """Optional topology overrides; omitted values retain service defaults."""

    num_machines: int = 1
    processes_per_machine: int = 1
    machine_rank: int = 0


class StorageMappingRequest(WireModel):
    """Host-to-execution filesystem mapping supplied with a plan request."""

    host_root: str
    execution_root: str


class CheckpointSourceRequest(WireModel):
    """A checkpoint directory in an ended training run's ``output/``."""

    run_id: str
    checkpoint: str


class PlanRequest(WireModel):
    """Strict wire form for a plan request, leaving domain checks to RunService."""

    backend: str
    operation: str
    config: dict[str, JsonValue]
    runtime: RuntimeRequest
    topology: TopologyRequest
    # MISSING allows omission while still refusing an explicit null.
    storage_map: list[StorageMappingRequest] | MISSING = MISSING
    submission_cwd: str | MISSING = MISSING
    checkpoint_source: CheckpointSourceRequest | MISSING = MISSING


class SubmitRequest(WireModel):
    """Strict wire form for a plan submission request."""

    plan_id: str


class PlanView(WireModel):
    """Public immutable preview returned by plan creation."""

    plan_id: str
    run_id: str
    request_id: str
    preview: dict[str, JsonValue]


class ExitRecordView(WireModel):
    """The Slurm batch, training and stage-out exit codes of an attempt."""

    batch: int
    training: int
    stage_out: int


class RunEvidence(WireModel):
    """How the run stands beyond its state: the head's phase or its exit."""

    reason: str
    phase: Literal["queued", "held", "running"] | None
    native_state: str | None
    outcome: (
        Literal["completed", "failed", "cancelled", "requeued", "unknown", "lost"]
        | None
    )
    exit_code: int | None
    signal: int | None
    attempt_exit: ExitRecordView | None
    restart_count: int | None


class RunProgressView(WireModel):
    epoch: Annotated[int, Field(ge=0, le=9007199254740991)]
    total_epochs: Annotated[int, Field(gt=0, le=9007199254740991)]


class RunResponse(WireModel):
    """Lifecycle view retaining the existing daemon response field types."""

    plan_id: str
    run_id: str
    request_id: str
    backend: str
    operation: str
    attempt: int | None
    state: RunStateFilter
    uncertainty: bool
    submission_requested: bool
    cancellation_requested: bool
    evidence: RunEvidence | None
    created_at: str | None
    summary: dict[str, JsonValue] | None
    progress: RunProgressView | None


class RunList(WireModel):
    """Page of lifecycle views."""

    runs: list[RunResponse]
