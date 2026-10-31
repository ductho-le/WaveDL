# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Pydantic wire schemas for the run lifecycle endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import MISSING, JsonValue

from wavedl_daemon.http.schemas.common import WireModel
from wavedl_daemon.runs.reducer import RunState


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
    """Run and published generation selected as an inference source."""

    run_id: str
    generation: str


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
    # Evidence contains tuples and StrEnums; object values preserve its JSON encoding.
    evidence: dict[str, object] | None
    created_at: str | None
    summary: dict[str, JsonValue] | None


class RunList(WireModel):
    """Page of lifecycle views."""

    runs: list[RunResponse]


class AttemptPage(WireModel):
    """Page of historical and current attempt evidence."""

    # Attempt records may contain tuples and StrEnums that JSON-encode natively.
    attempts: list[dict[str, object]]
    total: int
