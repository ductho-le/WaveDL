# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Typed lifecycle facade over already-open, owner-thread stores.

Planning is pure apart from committing an immutable SQLite preview. The
controller publishes run files and commits submission/cancellation intent before
it schedules external work. This module never starts a server or creates stores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from wavedl.core.inference_cli import inference_defaults, validate_import_modules
from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.manifest import (
    RuntimeSelection,
    StorageMapping,
    _json_bytes,
    build_manifest,
)
from wavedl.core.training_cli import (
    build_training_parser,
    training_defaults,
    validate_cv_options,
)
from wavedl_daemon.backends.contracts import Plan, Topology, UnsupportedPlan
from wavedl_daemon.backends.local import _TEST_FLAGS, _TEST_VALUES
from wavedl_daemon.execution.budget import BudgetExhausted
from wavedl_daemon.runs.reducer import RunState
from wavedl_daemon.storage.executions import native_runtime_pending
from wavedl_daemon.storage.plans import IdempotencyConflict, _canonical_request


if TYPE_CHECKING:
    from collections.abc import Mapping

    from wavedl_daemon.backends.contracts import Backend
    from wavedl_daemon.runs.controller import SubmissionController
    from wavedl_daemon.storage.executions import ExecutionStore
    from wavedl_daemon.storage.plans import PlanRecord, PlanStore
    from wavedl_daemon.storage.registry import Registry
    from wavedl_daemon.storage.runtime_provenance import RuntimeProvenanceStore
    from wavedl_daemon.storage.settings import Settings


class ForeignRun(ValueError):
    """Standalone/foreign runs cannot be controlled without verified adoption."""


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
    manifest = plan.manifest
    config = manifest.resolved_config

    def text(name: str) -> str | None:
        value = config.get(name)
        return value if type(value) is str and value else None

    def count(name: str, minimum: int) -> int | None:
        value = config.get(name)
        return value if type(value) is int and value >= minimum else None

    source = manifest.checkpoint_source
    return {
        "name": text("run_name"),
        "model": text("model"),
        "data_path": text("data_path"),
        "epochs": count("epochs", 1),
        "cv": count("cv", 2),
        "source_run_id": source["run_id"] if source is not None else None,
    }


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
    """All calls stay on the thread that opened the SQLite/controller objects."""

    def __init__(
        self,
        settings: Settings,
        plan_store: PlanStore,
        execution_store: ExecutionStore,
        controller: SubmissionController,
        backends: Mapping[str, Backend],
        registry: Registry | None = None,
        provenance: RuntimeProvenanceStore | None = None,
        *,
        collections=None,
    ) -> None:
        self.settings = settings
        self.plan_store = plan_store
        self.execution_store = execution_store
        self.controller = controller
        self.backends = dict(backends)
        self.registry = registry
        self.provenance = provenance
        self.collections = collections
        self._thread = threading.get_ident()

    def _guard(self) -> None:
        if threading.get_ident() != self._thread:
            raise RuntimeError("Run service must remain on its store owner thread")

    def checkpoint_selection(self, payload: dict, idempotency_key: str):
        """Capture registered ownership without opening source files."""
        self._guard()
        if type(idempotency_key) is not str or not 1 <= len(idempotency_key) <= 256:
            raise ValueError("A bounded idempotency key is required")
        _bounded(idempotency_key)
        _bounded(payload)
        if len(_canonical_request(payload).encode()) > 65536:
            raise ValueError("Request exceeds bound")
        if self.plan_store.get_by_key(idempotency_key) is not None:
            return None
        if type(payload) is not dict or "checkpoint_source" not in payload:
            return None
        if payload.get("operation") not in {"test", "plot"}:
            raise ValueError("Checkpoint sources are inference-only")
        source = _object(
            payload["checkpoint_source"],
            allowed={"run_id", "generation"},
            required={"run_id", "generation"},
        )
        config = payload.get("config")
        if type(config) is not dict or "checkpoint" in config:
            raise ValueError("Checkpoint source and path are mutually exclusive")
        record = self._owned(source["run_id"])
        if record.plan.operation != OperationKind.TRAIN:
            raise ValueError("Checkpoint source must be a training run")
        source_model = record.plan.manifest.resolved_config.get("model")
        if config.get("model") is not None and config["model"] != source_model:
            raise ValueError("Inference model differs from source training model")
        return (
            RunLayout(Path(record.plan.run_dir), record.plan.run_id),
            source["generation"],
            hashlib.sha256(_json_bytes(record.plan.manifest.to_dict())).hexdigest(),
        )

    def plan(
        self, payload: dict, idempotency_key: str, *, checkpoint_selection=None
    ) -> PlanResponse:
        self._guard()
        if type(idempotency_key) is not str or not 1 <= len(idempotency_key) <= 256:
            raise ValueError("A bounded idempotency key is required")
        _bounded(idempotency_key)
        canonical = _canonical_request(payload)
        _bounded(payload)
        if len(canonical.encode()) > 65536:
            raise ValueError("Request exceeds bound")
        existing = self.plan_store.get_by_key(idempotency_key)
        if existing is not None:
            if existing.request_payload_json != canonical:
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
        backend_name = body["backend"]
        if type(backend_name) is not str or backend_name not in self.backends:
            raise ValueError("Unknown backend")
        operation = OperationKind(body["operation"])
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
        selected = RuntimeSelection(**runtime)
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
        source = self.checkpoint_selection(payload, idempotency_key)
        if source is not None:
            from wavedl.core.checkpoint_source import validate_checkpoint_source

            pin = validate_checkpoint_source(checkpoint_selection)
            if (pin["run_id"], pin["generation"], pin["manifest_sha256"]) != (
                source[0].run_id,
                source[1],
                source[2],
            ):
                raise ValueError("Checkpoint selection differs from registered source")
            config = config | {"checkpoint": str(source[0].checkpoints / source[1])}
        config = (
            _training_config(config)
            if operation == OperationKind.TRAIN
            else _test_config(config)
        )
        if operation != OperationKind.TRAIN:
            config.update(operation=operation.value, device=selected.device)
            if operation == OperationKind.PLOT:
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
        run_id = str(uuid4())
        manifest = build_manifest(
            RunLayout(self.settings.run_root / run_id, run_id),
            operation=operation,
            resolved_config=config,
            runtime=selected,
            submission_cwd=Path(cwd),
            storage_map=mappings,
            checkpoint_source=pin if source is not None else None,
        )
        # Every newly-created lifecycle preview is intentionally unbound.  The
        # native scheduler/worker must establish the first execution attempt
        # only after accepting the immutable writer plan.  Existing plans are
        # returned above unchanged, preserving legacy reserved allocations on
        # idempotent replay.
        plan = self.backends[backend_name].plan(
            manifest, topology=topology, allocation_mode="writer"
        )
        if plan.allocation_mode != "writer":
            raise UnsupportedPlan(
                "Backend did not honor the required writer allocation mode"
            )
        if plan.target != backend_name or plan.manifest != manifest:
            raise ValueError("Backend changed the approved execution identity")
        record = self.plan_store.put(
            plan, idempotency_key=idempotency_key, request_payload=payload
        )
        return self._preview(record)

    @staticmethod
    def _preview(record: PlanRecord) -> PlanResponse:
        return PlanResponse(
            record.plan_id, record.plan.run_id, record.request_id, record.plan
        )

    def _owned(self, run_id: str) -> PlanRecord:
        _uuid(run_id)
        record = self.plan_store.get_by_run_id(run_id)
        if record is None:
            if self.registry is not None and self.registry.get(run_id) is not None:
                raise ForeignRun("Run requires verified adoption")
            raise KeyError(run_id)
        return record

    def submit(self, plan_id: str) -> RunResponse:
        self._guard()
        _uuid(plan_id)
        record = self.plan_store.get(plan_id)
        if record is None:
            raise KeyError(plan_id)
        self.controller.request_submit(plan_id)
        return self._view(self.plan_store.get(plan_id))

    def cancel(self, run_id: str) -> RunResponse:
        self._guard()
        record = self._owned(run_id)
        self.controller.request_cancel(record.plan_id)
        return self._view(self.plan_store.get(record.plan_id))

    def get(self, run_id: str) -> RunResponse:
        self._guard()
        return self._view(self._owned(run_id))

    def layout(self, run_id: str) -> RunLayout:
        """Resolve persisted identity on the owner thread, without file probing."""
        self._guard()
        plan = self._owned(run_id).plan
        return RunLayout(Path(plan.run_dir), plan.run_id)

    def attempts(self, run_id: str, *, limit=50, offset=0) -> dict:
        """Expose archived and current evidence without conflating generations."""
        self._guard()
        if (
            type(limit) is not int
            or not 1 <= limit <= 100
            or type(offset) is not int
            or not 0 <= offset <= 10000
        ):
            raise ValueError("Invalid attempt page bounds")
        plan = self._owned(run_id)
        current = self.execution_store.get(plan.plan_id)
        if current is None:
            return {"attempts": [], "total": 0}
        count = self.execution_store.history_count(plan.plan_id)
        records = self.execution_store.history(plan.plan_id, limit=limit, offset=offset)
        if offset <= count < offset + limit:
            records.append(current)
        result = []
        for record in records:
            runtime = (
                self.provenance.get(plan.plan_id, record.attempt)
                if self.provenance and record.attempt is not None
                else None
            )
            binding = (
                self.execution_store.requeue_binding(plan.plan_id, record.attempt)
                if record.attempt is not None
                else None
            )
            collection = (
                self.collections.get(plan.plan_id, record.attempt)
                if self.collections and record.attempt is not None
                else None
            )
            submission_collection = None
            if self.collections:
                identity = record.native_identity
                if identity is None and record.submission_observation is not None:
                    identity = record.submission_observation.native_identity
                if identity is not None:
                    submission_collection = self.collections.get_submission_generation(
                        plan.plan_id, identity
                    )
            result.append(
                {
                    "attempt": record.attempt,
                    "current": record.attempt == current.attempt
                    and not native_runtime_pending(current),
                    "last_runtime": record.attempt == current.attempt,
                    "native_pending": native_runtime_pending(current),
                    "execution_state": record.state.value,
                    "native_identity": asdict(record.native_identity)
                    if record.native_identity is not None
                    else None,
                    "handle": record.handle.to_dict() if record.handle else None,
                    "submission_handle": record.submission_handle.to_dict()
                    if record.submission_handle
                    else None,
                    "submission_observation": asdict(record.submission_observation)
                    if record.submission_observation
                    else None,
                    "latest_submission_observation": asdict(
                        record.latest_submission_observation
                    )
                    if record.latest_submission_observation
                    else None,
                    "observation": asdict(record.observation)
                    if record.observation
                    else None,
                    "latest_observation": asdict(record.latest_observation)
                    if record.latest_observation
                    else None,
                    "cancellation_requested": record.cancellation_requested,
                    "cancel_delivered": record.cancel_delivered,
                    "runtime": asdict(runtime.hello) if runtime else None,
                    "scheduler_binding": asdict(binding) if binding else None,
                    "collection": asdict(collection.result) if collection else None,
                    "submission_collection": (
                        submission_collection.proposal.to_dict()
                        if submission_collection
                        else None
                    ),
                }
            )
        page = {"attempts": result, "total": count + 1}
        if (
            len(json.dumps(page, allow_nan=False, separators=(",", ":")).encode())
            > 1024 * 1024
        ):
            raise BudgetExhausted("Attempt evidence page exceeds byte bound")
        return page

    def _view(self, record: PlanRecord) -> RunResponse:
        execution = self.execution_store.get(record.plan_id)
        plan = record.plan
        state, uncertainty = "planned", True
        evidence = None
        if execution is not None:
            events = ()
            if self.registry is not None and execution.attempt is not None:
                events = self.registry.lifecycle_events(plan.run_id, execution.attempt)
            derived = self.controller.derive(record.plan_id, events)
            state, uncertainty = derived.state.value, derived.confidence != "confirmed"
            evidence = {
                "native_pending": native_runtime_pending(execution),
                "native_identity": asdict(execution.native_identity)
                if execution.native_identity is not None
                else None,
                "current_runtime_attempt": None
                if native_runtime_pending(execution)
                else execution.attempt,
                "last_runtime_attempt": execution.attempt,
                "confidence": derived.confidence,
                "reason": derived.reason,
                "runtime": asdict(derived.runtime),
                "backend": asdict(derived.backend)
                if derived.backend is not None
                else None,
                "last_observation": asdict(derived.last_observation)
                if derived.last_observation is not None
                else None,
                "batch": asdict(derived.batch) if derived.batch is not None else None,
                "stage_out": asdict(derived.stage_out)
                if derived.stage_out is not None
                else None,
            }
        return RunResponse(
            record.plan_id,
            plan.run_id,
            record.request_id,
            plan.target,
            plan.operation.value,
            execution.attempt if execution else None,
            state,
            uncertainty,
            record.submission_requested,
            record.cancellation_requested,
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
                for record in self.plan_store.list_plans(limit=limit, offset=offset)
            ]
        # State is derived, never trusted from the journal projection. A bounded
        # page scan prevents state filtering from monopolizing the owner thread.
        result, skipped = [], 0
        for record in self.plan_store.list_plans(limit=1000, offset=0):
            view = self._view(record)
            if view.state == state:
                if skipped < offset:
                    skipped += 1
                    continue
                result.append(view)
                if len(result) == limit:
                    return result
        if self.plan_store.list_plans(limit=1, offset=1000):
            raise BudgetExhausted("Derived-state list scan capacity exhausted")
        return result
