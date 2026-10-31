# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Dependency-light, immutable planning and backend evidence contracts.

Backend evidence is not an application verdict. Runtime, wrapper/batch and
stage-out outcomes must remain independent until the reducer combines them.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from uuid import UUID

from wavedl.core.layout import OperationKind
from wavedl.core.manifest import Manifest, RuntimeSelection, StorageMapping
from wavedl.core.scheduler_attempt import SchedulerIdentity


class UnsupportedPlan(ValueError):
    """The target cannot faithfully express the requested execution."""


def _reject_constant(value: str):
    raise ValueError(f"Non-finite JSON constant: {value}")


def _json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _load(value: str) -> dict:
    result = json.loads(value, parse_constant=_reject_constant)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    _json(result)
    return result


def _load_submission_evidence(value: str) -> dict:
    if type(value) is not str or len(value.encode()) > 262144:
        raise ValueError("Submission evidence exceeds byte bound")

    def unique(items):
        fields = {}
        for key, item in items:
            if key in fields:
                raise ValueError("Duplicate submission evidence field")
            fields[key] = item
        return fields

    result = json.loads(
        value, object_pairs_hook=unique, parse_constant=_reject_constant
    )
    if type(result) is not dict:
        raise ValueError("Submission evidence must be an object")
    return result


def _scheduler_identity(data: dict) -> SchedulerIdentity:
    if type(data) is not dict or set(data) != {
        "run_id",
        "request_id",
        "cluster",
        "job_id",
        "owner",
        "restart_count",
    }:
        raise ValueError("Invalid scheduler identity fields")
    return SchedulerIdentity(**data)


def _outcome(data: dict) -> OperationOutcome:
    if type(data) is not dict:
        raise ValueError("Outcome must be an object")
    return OperationOutcome(**(data | {"status": OutcomeStatus(data["status"])}))


def manifest_from_dict(data: dict) -> Manifest:
    return Manifest(
        **(
            data
            | {
                "operation": OperationKind(data["operation"]),
                "runtime": RuntimeSelection(**data["runtime"]),
                "storage_map": tuple(
                    StorageMapping(**item) for item in data["storage_map"]
                ),
                "path_fields": tuple(data["path_fields"]),
            }
        )
    )


def _positive(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _strings(values: tuple[str, ...], name: str) -> None:
    if not isinstance(values, tuple) or not all(
        isinstance(value, str) and "\0" not in value for value in values
    ):
        raise ValueError(f"{name} must be an immutable tuple of strings")


@dataclass(frozen=True)
class Topology:
    num_machines: int = 1
    processes_per_machine: int = 1
    machine_rank: int = 0

    def __post_init__(self):
        _positive(self.num_machines, "num_machines")
        _positive(self.processes_per_machine, "processes_per_machine")
        if type(self.machine_rank) is not int or not (
            0 <= self.machine_rank < self.num_machines
        ):
            raise ValueError("machine_rank must identify an allocated machine")


@dataclass(frozen=True)
class Plan:
    target: str
    manifest_json: str
    argv: tuple[str, ...] = ()
    script: str | None = None
    topology: Topology = Topology()
    resources: tuple[tuple[str, str | int | float | bool | None], ...] = ()
    environment: tuple[tuple[str, str], ...] = ()
    warnings: tuple[str, ...] = ()
    allocation_mode: str = "reserved"

    def __post_init__(self):
        if type(self.allocation_mode) is not str or self.allocation_mode not in {
            "reserved",
            "writer",
        }:
            raise ValueError("Invalid attempt allocation mode")
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("A target is required")
        manifest = manifest_from_dict(_load(self.manifest_json))
        object.__setattr__(self, "manifest_json", _json(manifest.to_dict()))
        _strings(self.argv, "argv")
        _strings(self.warnings, "warnings")
        if self.script is not None and (
            not isinstance(self.script, str) or not self.script
        ):
            raise ValueError("script must be a nonempty string")
        if bool(self.argv) == bool(self.script):
            raise ValueError("Exactly one of argv and script must be specified")
        if not isinstance(self.topology, Topology):
            raise ValueError("topology must be a Topology")
        for name, pairs in (
            ("resources", self.resources),
            ("environment", self.environment),
        ):
            if not isinstance(pairs, tuple) or not all(
                isinstance(pair, tuple)
                and len(pair) == 2
                and isinstance(pair[0], str)
                and pair[0]
                for pair in pairs
            ):
                raise ValueError(f"{name} must contain immutable named pairs")
            if len(dict(pairs)) != len(pairs):
                raise ValueError(f"Duplicate {name} key")
        if not all(
            type(value) in {str, int, float, bool, type(None)}
            for _, value in self.resources
        ):
            raise ValueError("Resources must contain JSON scalars")
        if not all(
            isinstance(value, str) and "\0" not in key + value
            for key, value in self.environment
        ):
            raise ValueError("Environment must contain strings without NUL")
        if dict(self.environment) != manifest.runtime.environment:
            raise ValueError("Plan environment differs from selected runtime")
        _json(self.to_dict())

    @property
    def manifest(self) -> Manifest:
        """Return a fresh snapshot so nested mutable core objects cannot alter plans."""
        return manifest_from_dict(_load(self.manifest_json))

    @property
    def operation(self) -> OperationKind:
        return self.manifest.operation

    @property
    def run_id(self) -> str:
        return self.manifest.run_id

    @property
    def run_dir(self) -> str:
        return self.manifest.run_directory

    @property
    def working_directory(self) -> str:
        return self.manifest.working_directory

    def to_dict(self) -> dict:
        values = asdict(self)
        if self.allocation_mode == "reserved":
            # Approved legacy preview bytes and hashes are authoritative. Adding
            # an optional internal default must never rewrite those snapshots.
            values.pop("allocation_mode")
        return json.loads(_json(values))

    def to_json(self) -> str:
        return _json(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> Plan:
        if not isinstance(data, dict):
            raise ValueError("Plan must be an object")
        for name in ("argv", "warnings", "resources", "environment"):
            if not isinstance(data.get(name), list):
                raise ValueError(f"{name} must be a JSON array")
        for name in ("resources", "environment"):
            if not all(
                isinstance(pair, list) and len(pair) == 2 for pair in data[name]
            ):
                raise ValueError(f"{name} must contain JSON pairs")
        return cls(
            **(
                data
                | {
                    "topology": Topology(**data["topology"]),
                    "argv": tuple(data["argv"]),
                    "warnings": tuple(data["warnings"]),
                    "resources": tuple(tuple(pair) for pair in data["resources"]),
                    "environment": tuple(tuple(pair) for pair in data["environment"]),
                }
            )
        )

    @classmethod
    def from_json(cls, value: str) -> Plan:
        return cls.from_dict(_load(value))


def _native_handle(value, *, submission=False):
    """Shared receipt checks; new submissions additionally verify native identity."""
    if submission and type(value.run_id) is not str:
        raise ValueError("Submission run identity must be text")
    UUID(value.run_id)
    for name in ("backend", "request_id", "owner", "run_dir"):
        item = getattr(value, name)
        if not isinstance(item, str) or not item or "\0" in item:
            raise ValueError(f"Invalid {name}")
    if not Path(value.run_dir).is_absolute():
        raise ValueError("Handle run directory must be absolute")
    if value.pid is not None:
        _positive(value.pid, "pid")
        if not value.supervisor_id:
            raise ValueError("A PID alone is not durable process identity")
    for item in (value.job_id, value.supervisor_id):
        if item is not None and (not isinstance(item, str) or not item):
            raise ValueError("Invalid backend identity")
    if submission:
        if type(value.raw_json) is not str or len(value.raw_json.encode()) > 262144:
            raise ValueError("Native submission receipt exceeds byte bound")

        def pairs(items):
            fields = {}
            for key, item in items:
                if key in fields:
                    raise ValueError("Duplicate native receipt field")
                fields[key] = item
            return fields

        json.loads(
            value.raw_json, object_pairs_hook=pairs, parse_constant=_reject_constant
        )
    from wavedl.core.runtime_identity import load_runtime_identity_json

    load_runtime_identity_json(value.raw_json)
    raw = _load(value.raw_json)
    if submission:
        for item in (value.run_id, value.request_id):
            if str(UUID(item)) != item:
                raise ValueError("Submission identity requires canonical UUIDs")
        for item in (
            value.backend,
            value.owner,
            value.run_dir,
            value.job_id,
            value.supervisor_id,
        ):
            if item is not None and (
                len(item) > 8192
                or any(ord(char) < 32 or ord(char) == 127 for char in item)
            ):
                raise ValueError("Invalid native submission identity")
        if len(_json(raw).encode()) > 262144:
            raise ValueError("Native submission receipt exceeds byte bound")
        if value.backend == "slurm":
            if value.pid is not None or value.supervisor_id is not None:
                raise ValueError(
                    "Scheduler receipt cannot claim local process identity"
                )
            job, separator, cluster = (value.job_id or "").partition(";")
            if (
                not separator
                or re.fullmatch(r"[1-9][0-9]{0,18}", job) is None
                or int(job) > 2**63 - 1
                or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", cluster) is None
            ):
                raise ValueError("Invalid scheduler submission identity")
            if (
                raw.get("cluster") != cluster
                or raw.get("comment") != value.request_id
                or raw.get("job_name") != f"wavedl-{value.request_id}"
            ):
                raise ValueError(
                    "Scheduler receipt differs from native request identity"
                )
            restart = raw.get("restart_count")
            if type(restart) is not int or not 0 <= restart <= 2**63 - 1:
                raise ValueError("Invalid scheduler submission generation")
        elif value.backend == "local":
            if value.pid is None or not value.supervisor_id or value.job_id is not None:
                raise ValueError(
                    "Local submission requires supervised process identity"
                )
        else:
            raise ValueError("Unsupported native submission backend")
    object.__setattr__(value, "raw_json", _json(raw))


@dataclass(frozen=True)
class Handle:
    backend: str
    run_id: str
    request_id: str
    attempt: int
    owner: str
    run_dir: str
    job_id: str | None = None
    pid: int | None = None
    supervisor_id: str | None = None
    raw_json: str = "{}"

    def __post_init__(self):
        _positive(self.attempt, "attempt")
        _native_handle(self)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return _json(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> Handle:
        return cls(**data)

    @classmethod
    def from_json(cls, value: str) -> Handle:
        return cls.from_dict(_load(value))


@dataclass(frozen=True)
class SubmissionHandle:
    """Verified native acceptance with no authority to invent a runtime attempt."""

    backend: str
    run_id: str
    request_id: str
    owner: str
    run_dir: str
    job_id: str | None = None
    pid: int | None = None
    supervisor_id: str | None = None
    raw_json: str = "{}"

    def __post_init__(self):
        _native_handle(self, submission=True)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return _json(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> SubmissionHandle:
        if type(data) is not dict:
            raise ValueError("Submission receipt must be an object")
        return cls(**data)

    @classmethod
    def from_json(cls, value: str) -> SubmissionHandle:
        return cls.from_dict(_load(value))

    def bind_attempt(self, attempt: int) -> Handle:
        """Attach an already-proved writer reservation; performs no filesystem IO."""
        return Handle(**self.to_dict(), attempt=attempt)


class ObservationStatus(StrEnum):
    QUEUED = "queued"
    HELD = "held"
    RUNNING = "running"
    TERMINAL = "terminal"
    ABSENT = "absent"
    ERROR = "error"
    SUPERVISION_LOST = "supervision_lost"


class OutcomeStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class OperationOutcome:
    status: OutcomeStatus
    exit_code: int | None = None
    signal: int | None = None
    detail: str = ""

    def __post_init__(self):
        if not isinstance(self.status, OutcomeStatus):
            raise ValueError("Invalid outcome status")
        if not isinstance(self.detail, str):
            raise ValueError("Outcome detail must be a string")
        _exit_identity(self.exit_code, self.signal)
        if self.status == OutcomeStatus.SUCCESS and (
            self.exit_code not in {None, 0} or self.signal is not None
        ):
            raise ValueError("Success cannot carry a failure exit or signal")


def _exit_identity(exit_code: int | None, signal: int | None) -> None:
    if exit_code is not None and type(exit_code) is not int:
        raise ValueError("exit_code must be an integer")
    if signal is not None:
        _positive(signal, "signal")


@dataclass(frozen=True)
class BackendObservation:
    attempt: int
    status: ObservationStatus
    native_state: str | None = None
    exit_code: int | None = None
    signal: int | None = None
    detail: str = ""
    batch: OperationOutcome | None = None
    stage_out: OperationOutcome | None = None

    def __post_init__(self):
        _positive(self.attempt, "attempt")
        _observation(self)


def _observation(value):
    if not isinstance(value.status, ObservationStatus):
        raise ValueError("Invalid observation status")
    if not isinstance(value.detail, str) or (
        value.native_state is not None and not isinstance(value.native_state, str)
    ):
        raise ValueError("Observation metadata must contain strings")
    if any(
        item is not None and not isinstance(item, OperationOutcome)
        for item in (value.batch, value.stage_out)
    ):
        raise ValueError("Invalid independent outcome evidence")
    _exit_identity(value.exit_code, value.signal)
    if value.status != ObservationStatus.TERMINAL and (
        value.exit_code is not None or value.signal is not None
    ):
        raise ValueError("Only terminal backend evidence carries an exit")


@dataclass(frozen=True)
class SubmissionObservation:
    """Native submission lifecycle evidence, distinct from runtime execution."""

    status: ObservationStatus
    native_state: str | None = None
    exit_code: int | None = None
    signal: int | None = None
    detail: str = ""
    batch: OperationOutcome | None = None
    stage_out: OperationOutcome | None = None
    native_identity: SchedulerIdentity | None = None

    def __post_init__(self):
        _observation(self)
        if self.native_identity is not None and not isinstance(
            self.native_identity, SchedulerIdentity
        ):
            raise ValueError("Native identity must be a SchedulerIdentity")
        if len(_json(self.to_dict()).encode()) > 262144:
            raise ValueError("Submission evidence exceeds byte bound")

    def to_dict(self) -> dict:
        values = asdict(self)
        if self.native_identity is None:
            values.pop("native_identity")
        return json.loads(_json(values))

    def to_json(self) -> str:
        return _json(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> SubmissionObservation:
        if type(data) is not dict:
            raise ValueError("Submission observation must be an object")
        values = dict(data)
        values["status"] = ObservationStatus(values["status"])
        for name in ("batch", "stage_out"):
            if values.get(name) is not None:
                values[name] = _outcome(values[name])
        if values.get("native_identity") is not None:
            values["native_identity"] = _scheduler_identity(values["native_identity"])
        return cls(**values)

    @classmethod
    def from_json(cls, value: str) -> SubmissionObservation:
        return cls.from_dict(_load_submission_evidence(value))

    def bind_attempt(self, attempt: int) -> BackendObservation:
        return BackendObservation(
            attempt,
            self.status,
            self.native_state,
            self.exit_code,
            self.signal,
            self.detail,
            self.batch,
            self.stage_out,
        )


@dataclass(frozen=True)
class OperationResult:
    operation: OperationKind
    runtime: OperationOutcome
    batch: OperationOutcome
    stage_out: OperationOutcome

    def __post_init__(self):
        if not isinstance(self.operation, OperationKind) or not all(
            isinstance(value, OperationOutcome)
            for value in (self.runtime, self.batch, self.stage_out)
        ):
            raise ValueError("Operation result requires typed independent outcomes")


@dataclass(frozen=True)
class SubmissionCollectionResult:
    """Generation-scoped collection evidence before a runtime attempt is bound."""

    identity: SchedulerIdentity
    plan_sha256: str
    manifest_sha256: str
    result: OperationResult
    observation: SubmissionObservation

    def __post_init__(self):
        if not isinstance(self.identity, SchedulerIdentity):
            raise ValueError("Collection requires a SchedulerIdentity")
        for value in (self.plan_sha256, self.manifest_sha256):
            if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("Collection requires lowercase SHA-256 hashes")
        if not isinstance(self.result, OperationResult):
            raise ValueError("Collection requires an OperationResult")
        if (
            not isinstance(self.observation, SubmissionObservation)
            or self.observation.status != ObservationStatus.TERMINAL
            or self.observation.native_identity != self.identity
        ):
            raise ValueError("Collection requires matching terminal native evidence")
        runtime = self.result.runtime
        if (
            runtime.status != OutcomeStatus.UNKNOWN
            or runtime.exit_code is not None
            or runtime.signal is not None
        ):
            raise ValueError("Unbound submission cannot claim a runtime outcome")
        if len(self.to_json().encode()) > 262144:
            raise ValueError("Submission evidence exceeds byte bound")

    def to_dict(self) -> dict:
        return json.loads(_json(asdict(self)))

    def to_json(self) -> str:
        return _json(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> SubmissionCollectionResult:
        if type(data) is not dict or set(data) != {
            "identity",
            "plan_sha256",
            "manifest_sha256",
            "result",
            "observation",
        }:
            raise ValueError("Invalid submission collection fields")
        result = data["result"]
        if type(result) is not dict or set(result) != {
            "operation",
            "runtime",
            "batch",
            "stage_out",
        }:
            raise ValueError("Invalid collection operation result fields")
        return cls(
            identity=_scheduler_identity(data["identity"]),
            plan_sha256=data["plan_sha256"],
            manifest_sha256=data["manifest_sha256"],
            result=OperationResult(
                operation=OperationKind(result["operation"]),
                runtime=_outcome(result["runtime"]),
                batch=_outcome(result["batch"]),
                stage_out=_outcome(result["stage_out"]),
            ),
            observation=SubmissionObservation.from_dict(data["observation"]),
        )

    @classmethod
    def from_json(cls, value: str) -> SubmissionCollectionResult:
        return cls.from_dict(_load_submission_evidence(value))


class Backend(Protocol):
    name: str

    def plan(
        self,
        manifest: Manifest,
        *,
        topology: Topology = Topology(),
        arguments: tuple[str, ...] = (),
        allocation_mode: str = "reserved",
    ) -> Plan: ...
    def submit(
        self, plan: Plan, *, request_id: str, attempt: int | None
    ) -> Handle | SubmissionHandle: ...
    def observe(
        self, handle: Handle | SubmissionHandle
    ) -> BackendObservation | SubmissionObservation: ...
    def request_cancel(self, handle: Handle | SubmissionHandle) -> None: ...
    def collect(self, handle: Handle | SubmissionHandle) -> OperationResult: ...
    def collect_submission(
        self, handle: SubmissionHandle
    ) -> SubmissionCollectionResult | None: ...
