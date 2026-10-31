"""Dependency-light runtime bootstrap and rank-aware protocol emission.

Launchers bootstrap once before starting distributed workers and propagate
``RuntimeBootstrap.environment``. Workers must never allocate their own identity.
Existing standalone output directories remain in place; only new telemetry files
are exclusively created. Storage failures propagate to the runtime caller.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from wavedl.core.events import PAYLOADS, HelloPayload, Payload
from wavedl.core.journal_writer import JournalWriter
from wavedl.core.layout import OperationKind, RunLayout, allocate_attempt
from wavedl.core.manifest import (
    Manifest,
    RuntimeSelection,
    StorageMapping,
    build_manifest,
    describes_layout,
    write_manifest,
)
from wavedl.core.persistence import write_once
from wavedl.core.protocol import ProtocolError, finite_or_none, strict_loads
from wavedl.core.safe_files import read_regular_file
from wavedl.core.scheduler_attempt import (
    SchedulerBinding,
    read_scheduler_binding,
    scheduler_environment_identity,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from wavedl.core.live_push import LivePushClient
    from wavedl.core.protocol import Envelope


def _uuid(value: str, name: str) -> str:
    if str(UUID(value)) != value:
        raise ValueError(f"{name} must be a canonical UUID")
    return value


def _read_object(path: Path, *, limit: int = 1024 * 1024) -> dict[str, object]:
    """Bound reads and reject symlinks, oversized files, and blocking FIFOs."""
    value = strict_loads(read_regular_file(path, limit=limit))
    if not isinstance(value, dict):
        raise ValueError("Runtime metadata must be an object")
    return value


def _read_manifest(path: Path) -> Manifest:
    return _manifest_from_data(_read_object(path))


def _manifest_from_data(data) -> Manifest:
    if not isinstance(data, dict):
        raise ValueError("Runtime manifest must be an object")
    return Manifest(
        **data
        | {
            "operation": OperationKind(data["operation"]),
            "runtime": RuntimeSelection(**data["runtime"]),
            "storage_map": tuple(
                StorageMapping(**item) for item in data["storage_map"]
            ),
            "path_fields": tuple(data["path_fields"]),
        }
    )


@dataclass(frozen=True)
class RuntimeBootstrap:
    layout: RunLayout
    manifest: Manifest
    attempt: int
    request_id: str
    request_manifest: Manifest

    @property
    def environment(self) -> dict[str, str]:
        return {
            "WAVEDL_RUN_ID": self.layout.run_id,
            "WAVEDL_REQUEST_ID": self.request_id,
            "WAVEDL_ATTEMPT": str(self.attempt),
        }


def bootstrap_run(
    output_dir: Path,
    *,
    resolved_config: dict[str, object],
    runtime: RuntimeSelection,
    operation: OperationKind = OperationKind.TRAIN,
    submission_cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
    is_main_process: bool = True,
    rank: int = 0,
    approved_defaults: Mapping[str, object] | None = None,
) -> RuntimeBootstrap:
    """Reserve standalone execution, or validate a launcher's supplied attempt.

    A prior standalone manifest is immutable and reopening reserves a new
    attempt. An injected identity must already have a reserved attempt. The
    launcher calls this before spawning ranks, so followers only read identity.
    Requeue callers must supply a newly reserved attempt; an old attempt's
    restart counter cannot be reused. Injected callers may add only unchanged
    defaults from an explicitly supplied, trusted parser-default snapshot.
    """
    env = os.environ if environment is None else environment
    if (
        type(is_main_process) is not bool
        or type(rank) is not int
        or rank < 0
        or is_main_process != (rank == 0)
    ):
        raise ValueError("Main-process identity must match global rank zero")
    keys = ("WAVEDL_RUN_ID", "WAVEDL_REQUEST_ID", "WAVEDL_ATTEMPT")
    supplied = [key in env for key in keys]
    if any(supplied) and not all(supplied):
        raise ValueError("Incomplete launcher runtime identity")
    output_dir = output_dir.resolve()
    configured_output = resolved_config.get("output_dir")
    if configured_output is not None:
        if not isinstance(configured_output, str) or not configured_output:
            raise ValueError("Invalid runtime output directory")
        base = submission_cwd or Path.cwd()
        if (base / Path(configured_output).expanduser()).resolve() != output_dir:
            raise ValueError("Configured output/runtime directory mismatch")
    if all(supplied):
        run_id = _uuid(env[keys[0]], keys[0])
        request_id = _uuid(env[keys[1]], keys[1])
        raw_attempt = env[keys[2]]
        if not raw_attempt.isdecimal() or str(int(raw_attempt)) != raw_attempt:
            raise ValueError("Invalid WAVEDL_ATTEMPT")
        attempt = int(raw_attempt)
        layout = RunLayout(output_dir, run_id)
        manifest = _read_manifest(layout.manifest)
        if not describes_layout(manifest, layout) or manifest.operation != operation:
            raise ValueError("Manifest/runtime identity mismatch")
        if not layout.attempt_dir(attempt).is_dir():
            raise ValueError("Launcher attempt has not been reserved")
    else:
        if not is_main_process or int(env.get("WORLD_SIZE", "1")) > 1:
            raise ValueError("Distributed runtime requires launcher identity")
        output_dir.mkdir(parents=True, exist_ok=True)
        if (output_dir / "manifest.json").exists():
            manifest = _read_manifest(output_dir / "manifest.json")
            layout = RunLayout(output_dir, manifest.run_id)
            if (
                not describes_layout(manifest, layout)
                or manifest.operation != operation
            ):
                raise ValueError("Manifest/runtime identity mismatch")
        else:
            layout = RunLayout(output_dir, str(uuid4()))
            manifest = build_manifest(
                layout,
                operation=operation,
                resolved_config=resolved_config,
                runtime=runtime,
                submission_cwd=submission_cwd or Path.cwd(),
            )
            write_manifest(layout, manifest)
        request_id = str(uuid4())
        attempt = allocate_attempt(layout)
    restart = env.get("SLURM_RESTART_COUNT", "0")
    if not restart.isdecimal():
        raise ValueError("Invalid SLURM_RESTART_COUNT")
    identity = {
        "run_id": layout.run_id,
        "request_id": request_id,
        "attempt": attempt,
        "restart_count": int(restart),
    }
    scheduler = scheduler_environment_identity(env)
    if scheduler is not None:
        expected, initial_attempt = scheduler
        binding = read_scheduler_binding(
            output_dir, expected, initial_attempt=initial_attempt
        )
        if (
            binding.attempt != attempt
            or binding.run_id != layout.run_id
            or binding.request_id != request_id
        ):
            raise ValueError("Runtime identity differs from scheduler startup binding")
        identity["scheduler_binding"] = asdict(binding)
    identity_path = layout.attempt_dir(attempt) / "runtime_identity.json"
    request_path = layout.attempt_dir(attempt) / "runtime_request.json"
    if not all(supplied):
        request_manifest = build_manifest(
            layout,
            operation=operation,
            resolved_config=resolved_config,
            runtime=runtime,
            submission_cwd=submission_cwd or Path.cwd(),
            storage_map=manifest.storage_map,
            path_fields=manifest.path_fields,
        )
        write_once(
            request_path,
            json.dumps(request_manifest.to_dict(), allow_nan=False).encode() + b"\n",
        )
    elif os.path.lexists(request_path):
        request_manifest = _read_manifest(request_path)
        if (
            not describes_layout(request_manifest, layout)
            or request_manifest.operation != operation
        ):
            raise ValueError("Attempt request/runtime identity mismatch")
    else:
        if scheduler is not None and scheduler[1] is None:
            raise ValueError("Scheduler writer runtime request has not been published")
        request_manifest = manifest
    if all(supplied):
        defaults = {} if approved_defaults is None else approved_defaults
        for name in resolved_config.keys() - request_manifest.resolved_config.keys():
            if name not in defaults or json.dumps(
                resolved_config[name], sort_keys=True, allow_nan=False
            ) != json.dumps(defaults[name], sort_keys=True, allow_nan=False):
                raise ValueError(f"Unapproved runtime config extension: {name}")
        actual = build_manifest(
            layout,
            operation=operation,
            resolved_config=resolved_config,
            runtime=runtime,
            submission_cwd=submission_cwd or Path.cwd(),
            path_fields=request_manifest.path_fields,
        )
        for name, expected in request_manifest.resolved_config.items():
            if name not in actual.resolved_config or json.dumps(
                actual.resolved_config[name], sort_keys=True, allow_nan=False
            ) != json.dumps(expected, sort_keys=True, allow_nan=False):
                raise ValueError(
                    f"Runtime config differs from approved request: {name}"
                )
        if actual.runtime != request_manifest.runtime:
            raise ValueError("Runtime selection differs from approved request")
    if is_main_process and not identity_path.exists():
        write_once(identity_path, json.dumps(identity).encode() + b"\n")
    receipt = _read_object(identity_path, limit=4096)
    if scheduler is not None:
        value = receipt.get("scheduler_binding")
        if not isinstance(value, dict) or set(value) != set(
            SchedulerBinding.__dataclass_fields__
        ):
            raise ValueError("Runtime scheduler identity fields are invalid")
        SchedulerBinding(**value)
    if receipt.keys() != identity.keys() or any(
        type(receipt[name]) is not type(value) or receipt[name] != value
        for name, value in identity.items()
    ):
        raise ValueError("Attempt identity mismatch; reserve a new requeue attempt")
    return RuntimeBootstrap(layout, manifest, attempt, request_id, request_manifest)


class RunJournal:
    """One runtime writer, mandatory real hello, rank-only diagnostics elsewhere."""

    def __init__(
        self,
        bootstrap: RuntimeBootstrap,
        *,
        hello: HelloPayload,
        is_main_process: bool = True,
        rank: int = 0,
        fsync_every: int = 1,
        push_client: LivePushClient | None = None,
    ):
        if (
            type(is_main_process) is not bool
            or type(rank) is not int
            or rank < 0
            or is_main_process != (rank == 0)
        ):
            raise ValueError("Main-process identity must match global rank zero")
        checked = HelloPayload.from_dict(asdict(hello))
        if not checked.protocol_min <= 1 <= checked.protocol_max:
            raise ProtocolError("Runtime does not support protocol version 1")
        if not checked.runtime_version or not checked.python or not checked.interpreter:
            raise ProtocolError("Runtime hello requires actual execution provenance")
        self.bootstrap = bootstrap
        self.rank = rank
        self._writer = None
        self._rank_stream = None
        self._terminal = False
        self._push_client = push_client if is_main_process else None
        if is_main_process:
            try:
                self._writer = JournalWriter(
                    bootstrap.layout, attempt=bootstrap.attempt, fsync_every=fsync_every
                )
                first = self._writer.append("hello", asdict(checked))
                # The handshake must be durable even under batched metric fsync.
                self._writer.flush()
                self._push(first)
            except BaseException:
                if self._writer is not None:
                    self._writer.close()
                if self._push_client is not None:
                    try:
                        self._push_client.close()
                    except Exception:
                        pass
                raise
        else:
            path = bootstrap.layout.attempt_dir(bootstrap.attempt) / "rank"
            path.mkdir(exist_ok=True)
            self._rank_stream = (path / f"{rank}.log").open("x", encoding="utf-8")

    @property
    def last_sequence(self) -> int | None:
        """Last included event sequence, for causal checkpoint lineage cutoffs."""
        return self._writer.seq - 1 if self._writer is not None else None

    def _push(self, envelope):
        if self._push_client is not None:
            try:
                self._push_client.submit(envelope)
            except Exception:
                # An optional delivery bug cannot invalidate a durable append.
                try:
                    self._push_client.close()
                except Exception:
                    pass
                self._push_client = None

    def emit(
        self, event_type: str, payload: Payload | dict[str, object]
    ) -> Envelope | None:
        if self._terminal:
            raise RuntimeError("Runtime journal is closed or terminal")
        if event_type == "hello":
            raise ProtocolError("Hello is emitted exactly once at startup")
        schema = PAYLOADS.get(event_type)
        if schema is None:
            raise ProtocolError(f"Unknown runtime event: {event_type}")
        raw = asdict(payload) if isinstance(payload, Payload) else payload
        checked = schema.from_dict(finite_or_none(raw))
        if self._writer is None:
            return None
        result = self._writer.append(event_type, asdict(checked))
        self._push(result)
        if event_type == "exit":
            self._terminal = True
        return result

    def log(self, message: str, *, stream: str = "stdout", level: str = "info"):
        if self._terminal:
            raise RuntimeError("Runtime journal is closed or terminal")
        payload = {"message": message, "stream": stream, "level": level}
        PAYLOADS["log"].from_dict(payload)
        if self._rank_stream is not None:
            self._rank_stream.write(message + "\n")
            self._rank_stream.flush()
            os.fsync(self._rank_stream.fileno())
            return None
        return self.emit("log", payload)

    def close(self) -> None:
        try:
            if self._writer is not None:
                self._writer.close()
            if self._rank_stream is not None:
                self._rank_stream.close()
        finally:
            self._terminal = True
            if self._push_client is not None:
                try:
                    self._push_client.close()
                except Exception:
                    pass

    def __enter__(self) -> RunJournal:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
