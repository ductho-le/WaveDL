"""
The run spec, its topology, and the interface a backend implements.
Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Protocol

from wavedl_daemon.runs.layout import RunPaths
from wavedl_daemon.runs.model import Operation, canonical_json


if TYPE_CHECKING:
    from collections.abc import Mapping


class UnsupportedPlan(ValueError):
    """The backend cannot run this request as approved."""


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
class Runtime:
    """The interpreter, device and explicit environment a run executes with."""

    python: str
    mode: str
    device: str
    environment: Mapping[str, str] = field(default_factory=dict)
    container: str | None = None

    def __post_init__(self):
        if self.mode not in {"existing", "selected", "bundled", "container"}:
            raise ValueError("Invalid runtime mode")
        if self.device not in {"cpu", "mps", "cuda", "auto"}:
            raise ValueError("Invalid device")
        if not isinstance(self.python, str) or not self.python:
            raise ValueError("Runtime interpreter is required")
        if not isinstance(self.environment, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.environment.items()
        ):
            raise ValueError("Runtime environment must contain strings")
        if self.container is not None and not isinstance(self.container, str):
            raise ValueError("Invalid container")
        object.__setattr__(self, "environment", dict(self.environment))


@dataclass(frozen=True)
class StorageMapping:
    """A daemon-side root and where the execution host sees it."""

    host_root: str
    execution_root: str

    def __post_init__(self):
        if not Path(self.host_root).is_absolute() or not (
            Path(self.execution_root).is_absolute()
            or PurePosixPath(self.execution_root).is_absolute()
        ):
            raise ValueError("Storage mapping roots must be absolute")


@dataclass(frozen=True)
class CheckpointSource:
    """A checkpoint in an ended training run's ``output/``, read in place.

    ``path`` is the daemon's path to it; the trainer gets the mapped path in
    ``resolved_config["checkpoint"]``.
    """

    run_id: str
    checkpoint: str
    path: str


@dataclass(frozen=True)
class RunSpec:
    """The run identity and resolved inputs a plan approved.

    ``run_directory`` is the daemon's path to the run. ``working_directory`` is
    the same directory as the execution host sees it through ``storage_map``.
    Paths in ``resolved_config`` are already absolute and execution-side.
    """

    run_id: str
    operation: Operation
    resolved_config: Mapping[str, object]
    runtime: Runtime
    submission_directory: str
    run_directory: str
    working_directory: str
    storage_map: tuple[StorageMapping, ...] = ()
    checkpoint_source: CheckpointSource | None = None

    def __post_init__(self):
        if not isinstance(self.operation, Operation):
            raise ValueError("Invalid operation")
        for path in (self.submission_directory, self.run_directory):
            if not Path(path).is_absolute():
                raise ValueError("Run directories must be absolute")
        if not (
            Path(self.working_directory).is_absolute()
            or PurePosixPath(self.working_directory).is_absolute()
        ):
            raise ValueError("Run directories must be absolute")
        if not (
            Path(self.runtime.python).is_absolute()
            or PurePosixPath(self.runtime.python).is_absolute()
        ):
            raise ValueError("Runtime interpreter must be resolved")
        if not isinstance(self.resolved_config, dict):
            raise ValueError("resolved_config must be an object")
        object.__setattr__(
            self, "resolved_config", json.loads(canonical_json(self.resolved_config))
        )

    @property
    def paths(self) -> RunPaths:
        return RunPaths(Path(self.run_directory))

    def execution_path(self, path: Path) -> str:
        """A path under the run directory, as the execution host sees it."""
        relative = path.relative_to(self.run_directory)
        if PurePosixPath(self.working_directory).is_absolute():
            return str(PurePosixPath(self.working_directory).joinpath(*relative.parts))
        return str(Path(self.working_directory) / relative)

    def to_dict(self) -> dict:
        return json.loads(canonical_json(asdict(self)))

    @classmethod
    def from_dict(cls, data: dict) -> RunSpec:
        return cls(
            **data
            | {
                "operation": Operation(data["operation"]),
                "runtime": Runtime(**data["runtime"]),
                "storage_map": tuple(
                    StorageMapping(**item) for item in data["storage_map"]
                ),
                "checkpoint_source": (
                    None
                    if data["checkpoint_source"] is None
                    else CheckpointSource(**data["checkpoint_source"])
                ),
            }
        )


Scalar = str | int | float | bool | None


@dataclass(frozen=True)
class Plan:
    """An immutable preview: what the backend will run for one spec."""

    target: str
    spec: RunSpec
    argv: tuple[str, ...] = ()
    script: str | None = None
    topology: Topology = Topology()
    resources: tuple[tuple[str, Scalar], ...] = ()
    environment: tuple[tuple[str, str], ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("A target is required")
        _strings(self.argv, "argv")
        _strings(self.warnings, "warnings")
        if self.script is not None and (
            not isinstance(self.script, str) or not self.script
        ):
            raise ValueError("script must be a nonempty string")
        if bool(self.argv) == bool(self.script):
            raise ValueError("Exactly one of argv and script must be specified")
        for name, pairs in (
            ("resources", self.resources),
            ("environment", self.environment),
        ):
            if not isinstance(pairs, tuple) or not all(
                isinstance(pair, tuple) and len(pair) == 2 and pair[0] for pair in pairs
            ):
                raise ValueError(f"{name} must contain immutable named pairs")
            if len(dict(pairs)) != len(pairs):
                raise ValueError(f"Duplicate {name} key")
        if dict(self.environment) != self.spec.runtime.environment:
            raise ValueError("Plan environment differs from selected runtime")

    @property
    def run_id(self) -> str:
        return self.spec.run_id

    @property
    def operation(self) -> Operation:
        return self.spec.operation

    def to_dict(self) -> dict:
        return json.loads(self.to_json())

    def to_json(self) -> str:
        return canonical_json({**asdict(self), "spec": self.spec.to_dict()})

    @classmethod
    def from_dict(cls, data: dict) -> Plan:
        return cls(
            **data
            | {
                "spec": RunSpec.from_dict(data["spec"]),
                "topology": Topology(**data["topology"]),
                "argv": tuple(data["argv"]),
                "warnings": tuple(data["warnings"]),
                "resources": tuple(tuple(pair) for pair in data["resources"]),
                "environment": tuple(tuple(pair) for pair in data["environment"]),
            }
        )

    @classmethod
    def from_json(cls, value: str) -> Plan:
        return cls.from_dict(json.loads(value))


class Planner(Protocol):
    """Renders a spec into a plan without side effects."""

    name: str

    def plan(self, spec: RunSpec, *, topology: Topology) -> Plan: ...
