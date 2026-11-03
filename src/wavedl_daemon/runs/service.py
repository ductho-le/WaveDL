# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Plan, submit, cancel and read runs on the thread that owns the database.

Planning commits an immutable plan and nothing else. Submit and cancel append
one event to the run log; the controller does the rest. A run's state is
always ``status`` of its head, never stored separately.
"""

from __future__ import annotations

import argparse
import os
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from wavedl.core.inference_cli import inference_defaults, validate_import_modules
from wavedl.core.layout import RunLayout
from wavedl.core.training_cli import (
    build_training_parser,
    training_defaults,
    validate_cv_options,
)
from wavedl_daemon.backends.contracts import (
    Plan,
    Runtime,
    RunSpec,
    StorageMapping,
    Topology,
    UnsupportedPlan,
)
from wavedl_daemon.backends.local import _TEST_FLAGS, _TEST_VALUES
from wavedl_daemon.execution.budget import BudgetExhausted
from wavedl_daemon.runs.lifecycle import Ended, Live, RunRef, RunState, status
from wavedl_daemon.runs.model import (
    AttemptOpened,
    BackendKind,
    CancelRequested,
    Operation,
    PlanId,
    RunId,
)
from wavedl_daemon.storage.plans import IdempotencyConflict, canonical_request
from wavedl_daemon.storage.run_log import Expect, New, Stale


if TYPE_CHECKING:
    from collections.abc import Mapping

    from wavedl_daemon.backends.contracts import Planner
    from wavedl_daemon.runs.lifecycle import RunHead
    from wavedl_daemon.storage.plans import PlanRecord, PlanStore
    from wavedl_daemon.storage.run_log import RunLog
    from wavedl_daemon.storage.settings import Settings


# Paths a request may give relative to its submission directory.
PATH_FIELDS = (
    "data_path",
    "config",
    "resume",
    "checkpoint",
    "constraint_file",
    "export_path",
)


def _uuid(value: str) -> None:
    if type(value) is not str or str(UUID(value)) != value:
        raise ValueError("Identity must be a canonical UUID")


@dataclass(frozen=True)
class PlanResponse:
    plan_id: str
    run_id: str
    request_id: str
    preview: Plan

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "run_id": self.run_id,
            "request_id": self.request_id,
            "preview": self.preview.to_dict(),
        }


@dataclass(frozen=True)
class RunResponse:
    plan_id: str
    run_id: str
    request_id: str
    backend: str
    operation: str
    attempt: int | None
    state: str
    uncertainty: bool
    submission_requested: bool
    cancellation_requested: bool
    evidence: dict | None = None
    created_at: str | None = None
    summary: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _run_summary(plan: Plan) -> dict:
    """Identify a run for people: the approved inputs, never live state."""
    config = plan.spec.resolved_config

    def text(name: str) -> str | None:
        value = config.get(name)
        return value if type(value) is str and value else None

    def count(name: str, minimum: int) -> int | None:
        value = config.get(name)
        return value if type(value) is int and value >= minimum else None

    return {
        "name": text("run_name"),
        "model": text("model"),
        "data_path": text("data_path"),
        "epochs": count("epochs", 1),
        "cv": count("cv", 2),
        "source_run_id": None,
    }


def _evidence(head: RunHead) -> dict:
    """What the browser shows about how the run stands, beyond its state."""
    phase = native_state = ending = None
    match head.stage:
        case Live(phase=phase, native_state=native_state):
            pass
        case Ended(exit=ending):
            native_state = ending.native_state
    return {
        "reason": status(head).reason,
        "phase": None if phase is None else phase.value,
        "native_state": native_state,
        "outcome": None if ending is None else ending.outcome.value,
        "exit_code": None if ending is None else ending.exit_code,
        "signal": None if ending is None else ending.signal,
        "attempt_exit": (
            None if ending is None or ending.attempt_exit is None
            else asdict(ending.attempt_exit)
        ),
        "restart_count": head.restart_count,
    }


def build_spec(
    run_dir: Path,
    *,
    operation: Operation,
    resolved_config: dict[str, object],
    runtime: Runtime,
    submission_cwd: Path,
    storage_map: tuple[StorageMapping, ...] = (),
) -> RunSpec:
    """Resolve request paths against the submission directory, then map them.

    Data paths are made absolute with symlinks dereferenced. The interpreter
    is only normalized: a virtualenv's ``bin/python`` is a symlink to the base
    interpreter, and following it would run without the venv's packages.
    """
    base = submission_cwd.resolve()
    mappings = tuple(
        sorted(
            (
                StorageMapping(str(Path(item.host_root).resolve()), item.execution_root)
                for item in storage_map
            ),
            key=lambda item: len(item.host_root),
            reverse=True,
        )
    )
    if len({item.host_root for item in mappings}) != len(mappings):
        raise ValueError("Duplicate storage mapping")

    def mapped(host: Path) -> str:
        for mapping in mappings:
            try:
                relative = host.relative_to(Path(mapping.host_root))
            except ValueError:
                continue
            return str(Path(mapping.execution_root) / relative)
        return str(host)

    def resolve(value: str) -> str:
        return mapped((base / Path(value).expanduser()).resolve())

    config = dict(resolved_config)
    if "import_modules" in config:
        config["import_modules"] = validate_import_modules(
            [
                resolve(item) if item.endswith(".py") else item
                for item in validate_import_modules(config["import_modules"])
            ]
        )
    for name in PATH_FIELDS:
        if config.get(name) is not None:
            if not isinstance(config[name], str) or not config[name]:
                raise ValueError(f"Invalid path: {name}")
            config[name] = resolve(config[name])
    python = Path(runtime.python).expanduser()
    return RunSpec(
        run_id=run_dir.name,
        operation=operation,
        resolved_config=config,
        runtime=Runtime(
            mapped(Path(os.path.normpath(base / python))),
            runtime.mode,
            runtime.device,
            dict(runtime.environment),
            runtime.container,
        ),
        submission_directory=str(base),
        run_directory=str(run_dir),
        working_directory=resolve(str(run_dir)),
        storage_map=mappings,
    )


def _object(value: object, *, allowed: set[str], required: set[str] = frozenset()):
    if type(value) is not dict or set(value) - allowed or required - set(value):
        raise ValueError("Invalid request fields")
    return value


def _bounded(value: object, depth: int = 0) -> None:
    if depth > 8:
        raise ValueError("Request nesting exceeds bound")
    if type(value) is str:
        if len(value) > 8192 or "\0" in value:
            raise ValueError("Invalid string")
    elif type(value) in {dict, list}:
        if len(value) > 128:
            raise ValueError("Collection exceeds bound")
        if type(value) is dict:
            for key, child in value.items():
                _bounded(key, depth + 1)
                _bounded(child, depth + 1)
        else:
            for child in value:
                _bounded(child, depth + 1)
    elif type(value) in {int, float} and abs(value) > 10**12:
        raise ValueError("Number exceeds bound")


def _training_config(config: dict) -> dict:
    parser = build_training_parser()
    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest
        not in {
            "help",
            "list_models",
            "config",
            "output_dir",
            "allow_pickle",
            "selection_metric",
        }
    }
    if set(config) - set(actions):
        raise ValueError("Unknown or daemon-owned config option")
    result = training_defaults()
    # Pickle loading executes dataset code; selection_metric is HPO-only.
    for name in (
        "list_models",
        "config",
        "output_dir",
        "allow_pickle",
        "selection_metric",
    ):
        result.pop(name, None)
    for name, value in config.items():
        action = actions[name]
        if isinstance(action, argparse._StoreTrueAction):
            if type(value) is not bool:
                raise ValueError("Flag must be boolean")
            result[name] = value
            continue
        if value is None:
            if action.default is not None:
                raise ValueError("Option cannot be null")
            result[name] = None
            continue
        multiple = action.nargs == "+"
        if multiple:
            if type(value) is not list or not value:
                # Empty constraints/imports mean no flags, as in CLI defaults.
                if value == [] and name in {"constraint", "import_modules"}:
                    result[name] = []
                    continue
                raise ValueError("Option requires a nonempty list")
            values = value
        else:
            values = [value]
        normalized = []
        for item in values:
            if action.type is int and type(item) is not int:
                raise ValueError("Option requires an integer")
            if action.type is float and type(item) not in {int, float}:
                raise ValueError("Option requires a number")
            if action.type is str and type(item) is not str:
                raise ValueError("Option requires a string")
            item = action.type(item) if action.type else item
            if action.choices is not None and item not in action.choices:
                raise ValueError("Invalid option choice")
            normalized.append(item)
        result[name] = normalized if multiple else normalized[0]
    positive = {"batch_size", "epochs", "grad_accum_steps", "cv_bins", "step_size"}
    nonnegative = {
        "patience",
        "save_every",
        "cv",
        "warmup_epochs",
        "scheduler_patience",
    }
    if any(result[name] < 1 for name in positive) or any(
        result[name] < 0 for name in nonnegative
    ):
        raise ValueError("Invalid config range")
    if result["workers"] < -1 or result["lr"] <= 0 or result["weight_decay"] < 0:
        raise ValueError("Invalid config range")
    validate_cv_options(result)
    return result


def _test_config(config: dict) -> dict:
    if set(config) - (_TEST_VALUES | _TEST_FLAGS) or {
        "export_path",
        "device",
        "operation",
    } & set(config):
        raise ValueError("Unknown or caller-owned output option")
    integers = {"input_channels", "batch_size", "workers", "export_opset", "out_size"}
    lists = {"param_names", "plot_format"}
    choices = {"format": {"auto", "npz", "mat", "hdf5"}, "export": {"onnx"}}
    for name, value in config.items():
        if name == "import_modules":
            validate_import_modules(value)
        elif name in _TEST_FLAGS:
            if type(value) is not bool:
                raise ValueError("Flag must be boolean")
        elif value is None and name not in {"checkpoint", "data_path"}:
            continue
        elif name in integers:
            if type(value) is not int or value < (0 if name == "workers" else 1):
                raise ValueError("Invalid integer option")
        elif name in lists:
            if (
                type(value) is not list
                or not value
                or any(type(item) is not str for item in value)
            ):
                raise ValueError("Invalid list option")
        elif type(value) is not str or not value:
            raise ValueError("Invalid string option")
        if name in choices and value is not None and value not in choices[name]:
            raise ValueError("Invalid option choice")
    result = inference_defaults()
    # Pickle loading executes code from the dataset; only direct CLI use may opt in.
    for name in ("operation", "device", "output_dir", "export_path", "allow_pickle"):
        result.pop(name, None)
    result.update(config)
    if not result["checkpoint"] or not result["data_path"]:
        raise ValueError("Checkpoint and dataset are required")
    if set(result["plot_format"]) - {"png", "pdf", "svg", "eps", "tiff"}:
        raise ValueError("Unsupported plot format")
    return result


class RunService:
    """All calls stay on the thread that opened the database."""

    def __init__(
        self,
        settings: Settings,
        plans: PlanStore,
        log: RunLog,
        backends: Mapping[str, Planner],
    ) -> None:
        self.settings = settings
        self.plans = plans
        self.log = log
        self.backends = dict(backends)
        self._thread = threading.get_ident()

    def _guard(self) -> None:
        if threading.get_ident() != self._thread:
            raise RuntimeError("Run service must remain on its store owner thread")

    def plan(self, payload: dict, idempotency_key: str) -> PlanResponse:
        self._guard()
        if type(idempotency_key) is not str or not 1 <= len(idempotency_key) <= 256:
            raise ValueError("A bounded idempotency key is required")
        _bounded(idempotency_key)
        canonical = canonical_request(payload)
        _bounded(payload)
        if len(canonical.encode()) > 65536:
            raise ValueError("Request exceeds bound")
        existing = self.plans.get_by_key(idempotency_key)
        if existing is not None:
            if existing.request_json != canonical:
                raise IdempotencyConflict("Key is bound to another request")
            return self._preview(existing)
        body = _object(
            payload,
            allowed={
                "backend",
                "operation",
                "config",
                "runtime",
                "topology",
                "storage_map",
                "submission_cwd",
                "checkpoint_source",
            },
            required={"backend", "operation", "config", "runtime", "topology"},
        )
        if "checkpoint_source" in body:
            raise UnsupportedPlan(
                "Testing from a run's checkpoint is being rebuilt; "
                "give the checkpoint path instead"
            )
        backend_name = body["backend"]
        if type(backend_name) is not str or backend_name not in self.backends:
            raise ValueError("Unknown backend")
        operation = Operation(body["operation"])
        runtime = _object(
            body["runtime"],
            allowed={"python", "mode", "device", "environment"},
            required={"python", "mode", "device", "environment"},
        )
        environment = runtime["environment"]
        if type(environment) is not dict or any(
            type(key) is not str
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None
            or type(value) is not str
            for key, value in environment.items()
        ):
            raise ValueError("Invalid explicit runtime environment")
        if any(key.startswith("WAVEDL_") for key in environment):
            raise ValueError(
                "Daemon runtime identity and credentials are not caller-owned"
            )
        selected = Runtime(**runtime)
        topology = Topology(
            **_object(
                body["topology"],
                allowed={"num_machines", "processes_per_machine", "machine_rank"},
            )
        )
        if topology.num_machines * topology.processes_per_machine > 1024:
            raise ValueError("Topology exceeds bound")
        config = _object(
            body["config"],
            allowed=set(body["config"]) if type(body["config"]) is dict else set(),
        )
        config = (
            _training_config(config)
            if operation is Operation.TRAIN
            else _test_config(config)
        )
        if operation is Operation.PLOT:
            config["plot"] = True
        cwd = body.get("submission_cwd", str(self.settings.run_root))
        if type(cwd) is not str or not cwd or not Path(cwd).is_absolute():
            raise ValueError("Submission directory must be absolute")
        storage = body.get("storage_map", [])
        if type(storage) is not list:
            raise ValueError("Storage map must be a list")
        mappings = tuple(
            StorageMapping(
                **_object(
                    item,
                    allowed={"host_root", "execution_root"},
                    required={"host_root", "execution_root"},
                )
            )
            for item in storage
        )
        spec = build_spec(
            self.settings.run_root / str(uuid4()),
            operation=operation,
            resolved_config=config,
            runtime=selected,
            submission_cwd=Path(cwd),
            storage_map=mappings,
        )
        plan = self.backends[backend_name].plan(spec, topology=topology)
        if plan.target != backend_name or plan.spec != spec:
            raise ValueError("Backend changed the approved execution identity")
        record = self.plans.put(
            plan, idempotency_key=idempotency_key, request_payload=payload
        )
        return self._preview(record)

    @staticmethod
    def _preview(record: PlanRecord) -> PlanResponse:
        return PlanResponse(record.plan_id, record.run_id, record.request_id, record.plan)

    def _owned(self, run_id: str) -> PlanRecord:
        _uuid(run_id)
        record = self.plans.by_run(run_id)
        if record is None:
            raise KeyError(run_id)
        return record

    def submit(self, plan_id: str) -> RunResponse:
        self._guard()
        _uuid(plan_id)
        record = self.plans.get(plan_id)
        if record is None:
            raise KeyError(plan_id)
        run_id = RunId(record.run_id)
        if self.log.head(run_id) is None:
            ref = RunRef(
                run_id,
                PlanId(record.plan_id),
                BackendKind(record.plan.target),
                record.plan.operation,
                record.plan.spec.paths.run_dir,
            )
            try:
                self.log.append(run_id, [AttemptOpened("submit", None)], expect=New(ref))
            except Stale:
                pass  # a concurrent submit already started it
        return self._view(record)

    def cancel(self, run_id: str) -> RunResponse:
        self._guard()
        record = self._owned(run_id)
        head = self.log.head(RunId(run_id))
        if head is not None and not head.cancel_requested and not isinstance(
            head.stage, Ended
        ):
            self.log.append(
                head.ref.run_id, [CancelRequested()], expect=Expect(revision=head.revision)
            )
        return self._view(record)

    def get(self, run_id: str) -> RunResponse:
        self._guard()
        return self._view(self._owned(run_id))

    def layout(self, run_id: str) -> RunLayout:
        """Resolve persisted identity on the owner thread, without file probing."""
        self._guard()
        record = self._owned(run_id)
        return RunLayout(record.plan.spec.paths.run_dir, record.run_id)

    def _view(self, record: PlanRecord) -> RunResponse:
        plan = record.plan
        head = self.log.head(RunId(record.run_id))
        if head is None:
            state, uncertainty, evidence = "planned", True, None
        else:
            current = status(head)
            state, uncertainty = current.state.value, not current.confirmed
            evidence = _evidence(head)
        return RunResponse(
            record.plan_id,
            record.run_id,
            record.request_id,
            plan.target,
            plan.operation.value,
            None if head is None else head.attempt,
            state,
            uncertainty,
            head is not None,
            head is not None and head.cancel_requested,
            evidence,
            record.created_at,
            _run_summary(plan),
        )

    def list(
        self, limit: int = 50, offset: int = 0, state: str | None = None
    ) -> list[RunResponse]:
        self._guard()
        if (
            type(limit) is not int
            or not 1 <= limit <= 100
            or type(offset) is not int
            or not 0 <= offset <= 10000
        ):
            raise ValueError("Invalid page bounds")
        if state is not None and state not in {
            "planned",
            *(item.value for item in RunState),
        }:
            raise ValueError("Invalid state filter")
        if state is None:
            return [
                self._view(record)
                for record in self.plans.page(limit=limit, offset=offset)
            ]
        # A bounded scan keeps state filtering from monopolizing the owner thread.
        matches = [
            view
            for view in map(self._view, self.plans.page(limit=1000, offset=0))
            if view.state == state
        ]
        if self.plans.page(limit=1, offset=1000):
            raise BudgetExhausted("Derived-state list scan capacity exhausted")
        return matches[offset : offset + limit]
