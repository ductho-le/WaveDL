# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Bounded Local process ownership; saved PIDs never authorize signalling.

Supervisors own pipe draining and tree cleanup. This adapter retains a bounded
number of wait futures, not one new observation thread per polling request.
POSIX containment requires descendants to remain in the inherited group; native
Windows Job semantics and Linux behavior require their respective OS CI gates.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from wavedl.core.layout import RunLayout
from wavedl.core.local_attempt import LocalStartupIdentity, read_local_binding
from wavedl.core.persistence import write_once
from wavedl.core.runtime_identity import (
    checked_python_identity,
    load_runtime_identity_json,
)
from wavedl.core.submission import (
    create_submission,
    identity_for_submission,
    publish_submission_file,
)
from wavedl_daemon.execution.budget import BudgetExhausted
from wavedl_daemon.storage.ownership import require_dispatch
from wavedl_daemon.supervision.base import (
    ProcessResult,
    ProcessSpec,
    validate_output_limit,
    validate_seconds,
)

from .contracts import (
    BackendObservation,
    Handle,
    ObservationStatus,
    OperationOutcome,
    OperationResult,
    OutcomeStatus,
    Plan,
    SubmissionHandle,
    SubmissionObservation,
    UnsupportedPlan,
    _json,
    manifest_from_dict,
)


if TYPE_CHECKING:
    from wavedl_daemon.storage.ownership import DispatchLock
    from wavedl_daemon.supervision.base import SupervisedProcess, Supervisor


class SupervisionLost(RuntimeError):
    """No live in-memory process ownership exists for this handle."""


def reserved_local_environment(key: str) -> bool:
    """Ambient launcher/scheduler identity must not become Local authority."""
    name = key.upper()
    return name.startswith(
        (
            "WAVEDL_",
            "SLURM_",
            "SBATCH_",
            "SRUN_",
            "SCANCEL_",
            "SQUEUE_",
            "SACCT_",
            "SCONTROL_",
        )
    ) or name in {
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    }


def local_environment(plan: Plan) -> dict[str, str]:
    """Preserve ordinary inherited paths and only explicitly approved controls."""
    if any(reserved_local_environment(key) for key, _ in plan.environment):
        raise UnsupportedPlan("Local environment cannot override execution identity")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not reserved_local_environment(key)
    }
    environment.update(plan.environment)
    return environment


@dataclass(frozen=True)
class _Owned:
    handle: Handle | SubmissionHandle
    plan_json: str
    plan: Plan
    process: SupervisedProcess
    result: Future[ProcessResult]
    runtime_handle: Handle | None = None


class LocalExecution:
    def __init__(
        self,
        *,
        dispatch_lock: DispatchLock | None = None,
        owner: str | None = None,
        max_processes: int = 2,
        max_handles: int = 128,
        output_limit: int = 1048576,
        probe_timeout: float = 10,
        cancel_grace: float = 0.2,
        supervisor: Supervisor | None = None,
        live_push=None,
    ):
        for value in (max_processes, max_handles):
            if type(value) is not int or value < 1:
                raise ValueError("admission limits must be positive integers")
        validate_output_limit(output_limit)
        validate_seconds(probe_timeout, "probe_timeout")
        validate_seconds(cancel_grace, "cancel_grace")
        if probe_timeout == 0:
            raise ValueError("probe_timeout must be positive")
        if owner is not None and (
            not isinstance(owner, str) or not owner or "\0" in owner
        ):
            raise ValueError("owner must be a nonempty string without NUL")
        if supervisor is None:
            if os.name == "nt":
                from wavedl_daemon.supervision.windows import WindowsSupervisor

                supervisor = WindowsSupervisor()
            else:
                from wavedl_daemon.supervision.posix import POSIXSupervisor

                supervisor = POSIXSupervisor()
        self.dispatch_lock = dispatch_lock
        if live_push is not None:
            from wavedl.core.live_push import LivePushConfig

            if not isinstance(live_push, LivePushConfig):
                raise ValueError("Live push requires a validated ephemeral config")
        self._live_push = live_push
        self._pid = os.getpid()
        self.owner = owner or secrets.token_urlsafe(32)
        self._supervisor = supervisor
        self._max_processes = max_processes
        self._max_handles = max_handles
        self._output_limit = output_limit
        self._probe_timeout = probe_timeout
        self._cancel_grace = cancel_grace
        self._lock = threading.RLock()
        self._records: dict[tuple[str, int | None], _Owned] = {}
        self._waiters = ThreadPoolExecutor(
            max_workers=max_processes, thread_name_prefix="local-wait"
        )
        self._closed = False

    def _verify(self, plan: Plan, attempt: int | None) -> None:
        # Re-render from the immutable manifest, never current user config.
        expected = self.plan(plan.manifest, topology=plan.topology)
        if (plan.target, plan.argv, plan.script, plan.resources, plan.environment) != (
            expected.target,
            expected.argv,
            expected.script,
            expected.resources,
            expected.environment,
        ):
            raise UnsupportedPlan("Local plan differs from its resolved snapshot")
        layout = RunLayout(Path(plan.run_dir), plan.run_id)
        if plan.allocation_mode == "writer":
            if attempt is not None:
                raise ValueError("Writer submission cannot reserve an attempt")
        elif attempt is None or not layout.attempt_dir(attempt).is_dir():
            raise ValueError("attempt must be reserved before submission")
        # Bound both retained snapshots and reads before parsing disk evidence.
        if len(plan.to_json().encode("utf-8")) > 1048576:
            raise BudgetExhausted("Local plan exceeds retained snapshot byte budget")
        with layout.manifest.open("rb") as stream:
            payload = stream.read(1048577)
        if len(payload) > 1048576:
            raise BudgetExhausted("Local manifest exceeds byte budget")
        if (
            _json(manifest_from_dict(json.loads(payload)).to_dict())
            != plan.manifest_json
        ):
            raise ValueError("on-disk manifest differs from immutable plan")
        command_bytes = sum(len(arg.encode("utf-8")) + 1 for arg in plan.argv)
        command_bytes += sum(
            len(key.encode("utf-8")) + len(value.encode("utf-8")) + 2
            for key, value in plan.environment
        )
        if command_bytes > 65536:
            raise BudgetExhausted("Local command exceeds byte budget")

    def _probe(self, plan: Plan) -> dict[str, str]:
        device = plan.manifest.runtime.device
        code = (
            "import sys,torch\n"
            "if sys.version_info < (3,11): raise RuntimeError('Python 3.11+ required')\n"
            + (
                "if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')\n"
                f"if torch.cuda.device_count() < {plan.topology.processes_per_machine}: raise RuntimeError('CUDA allocation insufficient')\n"
                f"for i in range({plan.topology.processes_per_machine}): torch.empty(1,device='cuda:'+str(i))\n"
                if device == "cuda"
                else "torch.empty(1,device='cpu')\n"
            )
            + (
                "if not torch.distributed.is_available() or not torch.distributed.is_gloo_available(): raise RuntimeError('CPU distributed training requires Gloo')\n"
                if device == "cpu" and plan.topology.processes_per_machine > 1
                else ""
            )
            + "import json\nprint(json.dumps({'interpreter':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix},sort_keys=True))"
        )
        process = self._supervisor.start(
            ProcessSpec(
                (plan.manifest.runtime.python, "-c", code),
                Path(plan.working_directory),
                local_environment(plan),
                inherit_environment=False,
            ),
            output_limit=32768,
        )
        try:
            result = process.wait(timeout=self._probe_timeout)
        except RuntimeError as error:
            raise UnsupportedPlan(
                "selected interpreter/device supervision failed; no fallback"
            ) from error
        if (
            result.returncode != 0
            or result.timed_out
            or result.stdout_truncated
            or result.stderr_truncated
        ):
            raise UnsupportedPlan(
                "selected interpreter/device probe refused execution; no fallback"
            )
        try:
            return checked_python_identity(load_runtime_identity_json(result.stdout))
        except (ValueError, TypeError) as error:
            raise UnsupportedPlan(
                "selected interpreter probe identity is invalid; no fallback"
            ) from error

    def _attestation(self, plan, request_id, supervisor_id, identity):
        return {
            "schema_version": 1,
            "selected_python": plan.manifest.runtime.python,
            "plan_sha256": hashlib.sha256(plan.to_json().encode()).hexdigest(),
            "run_id": plan.run_id,
            "request_id": request_id,
            "owner": self.owner,
            "supervisor_id": supervisor_id,
            "identity": identity,
        }

    def submit(
        self, plan: Plan, *, request_id: str, attempt: int | None
    ) -> Handle | SubmissionHandle:
        require_dispatch(self.dispatch_lock)
        if self._pid != os.getpid():
            raise SupervisionLost("backend process ownership cannot survive fork")
        if plan.allocation_mode == "writer":
            return self._submit_writer(plan, request_id=request_id, attempt=attempt)
        # Validate identity before any subprocess, including the capability probe.
        candidate = Handle(
            "local", plan.run_id, request_id, attempt, self.owner, plan.run_dir
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("Local backend is closed")
            key = (request_id, attempt)
            previous = self._records.get(key)
            if previous is not None:
                if previous.plan_json != plan.to_json():
                    raise ValueError(
                        "submission identity is already bound to another plan"
                    )
                return previous.handle
            if any(
                record.handle.run_id == candidate.run_id
                and record.handle.attempt == attempt
                for record in self._records.values()
            ):
                raise ValueError("run attempt already belongs to another request")
            reservation = (
                RunLayout(Path(plan.run_dir), plan.run_id).attempt_dir(attempt)
                / "local-submission.json"
            )
            try:
                reservation.lstat()
            except FileNotFoundError:
                pass
            else:
                raise SupervisionLost(
                    "Local submission reservation exists without live ownership; never resubmit"
                )
            if (
                len(self._records) >= self._max_handles
                or sum(not record.result.done() for record in self._records.values())
                >= self._max_processes
            ):
                raise BudgetExhausted(
                    "Local process/retained-handle admission exhausted"
                )
            self._verify(plan, attempt)
            probe_identity = self._probe(plan)
            # Probe latency is not permission to act with relinquished dispatch.
            require_dispatch(self.dispatch_lock)
            self._verify(plan, attempt)
            # Durable no-replace reservation precedes the external side effect.
            # A crash or spawn failure after publication is uncertain acceptance,
            # never permission to retry this attempt. No PID is adopted from disk.
            supervisor_id = secrets.token_urlsafe(32)
            attestation = self._attestation(
                plan, request_id, supervisor_id, probe_identity
            )
            candidate = Handle(
                **(
                    candidate.to_dict()
                    | {
                        "raw_json": _json({"runtime_attestation": attestation}),
                        "supervisor_id": supervisor_id,
                    }
                )
            )
            identity = candidate.to_dict() | {
                "plan_sha256": hashlib.sha256(
                    plan.to_json().encode("utf-8")
                ).hexdigest()
            }
            try:
                write_once(reservation, (_json(identity) + "\n").encode("utf-8"))
            except FileExistsError as error:
                raise SupervisionLost(
                    "Local submission reservation already exists; never resubmit"
                ) from error
            environment = local_environment(plan)
            # Credentials exist only in the child environment. They never enter
            # an approved/persisted plan, reservation, manifest or argv.
            if self._live_push is not None:
                environment.update(
                    WAVEDL_DAEMON_URL=self._live_push.url,
                    WAVEDL_DAEMON_TOKEN=self._live_push.token,
                )
            environment.update(
                WAVEDL_RUN_ID=plan.run_id,
                WAVEDL_REQUEST_ID=request_id,
                WAVEDL_ATTEMPT=str(attempt),
                WAVEDL_OWNER=self.owner,
            )
            process = self._supervisor.start(
                ProcessSpec(
                    plan.argv,
                    Path(plan.working_directory),
                    environment,
                    inherit_environment=False,
                ),
                output_limit=self._output_limit,
            )
            handle = Handle(
                "local",
                plan.run_id,
                request_id,
                attempt,
                self.owner,
                plan.run_dir,
                pid=process.pid,
                supervisor_id=supervisor_id,
                raw_json=candidate.raw_json,
            )
            try:
                future = self._waiters.submit(process.wait)
            except BaseException:
                process.cancel(grace_seconds=0)
                process.wait()
                raise
            self._records[key] = _Owned(handle, plan.to_json(), plan, process, future)
            return handle

    def _submit_writer(self, plan, *, request_id, attempt):
        if attempt is not None:
            raise ValueError("Writer submission must remain unbound")
        digest = hashlib.sha256(plan.to_json().encode()).hexdigest()
        # Establish request/manifest identity before any capability probe or
        # runtime spawn. This read neither creates a submission nor an attempt.
        identity = identity_for_submission(
            Path(plan.run_dir),
            request_id=request_id,
            backend="local",
            owner=self.owner,
            plan_sha256=digest,
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("Local backend is closed")
            key = (request_id, None)
            previous = self._records.get(key)
            if previous is not None:
                if previous.plan_json != plan.to_json():
                    raise ValueError("Submission belongs to another immutable plan")
                return previous.handle
            namespace = Path(plan.run_dir) / "submissions" / request_id
            if os.path.lexists(namespace):
                raise SupervisionLost(
                    "Local submission exists without live ownership; never resubmit"
                )
            if any(
                record.handle.run_id == plan.run_id for record in self._records.values()
            ):
                raise ValueError("Run already belongs to a Local submission")
            if (
                len(self._records) >= self._max_handles
                or sum(not record.result.done() for record in self._records.values())
                >= self._max_processes
            ):
                raise BudgetExhausted(
                    "Local process/retained-handle admission exhausted"
                )
            self._verify(plan, None)
            probe_identity = self._probe(plan)
            require_dispatch(self.dispatch_lock)
            self._verify(plan, None)
            supervisor_id = secrets.token_urlsafe(32)
            attestation = self._attestation(
                plan, request_id, supervisor_id, probe_identity
            )
            # The exclusive namespace is the pre-launch no-resubmit fence. A
            # crash during any subsequent publication leaves it claimed.
            try:
                create_submission(Path(plan.run_dir), identity)
            except FileExistsError as error:
                raise SupervisionLost("Local submission is already reserved") from error
            publish_submission_file(
                Path(plan.run_dir), identity, "plan.json", plan.to_json().encode()
            )
            publish_submission_file(
                Path(plan.run_dir),
                identity,
                "runtime-probe.json",
                _json(attestation).encode(),
            )
            publish_submission_file(
                Path(plan.run_dir),
                identity,
                "local-submission.json",
                (
                    _json(
                        {
                            "schema_version": 1,
                            "run_id": plan.run_id,
                            "request_id": request_id,
                            "owner": self.owner,
                            "supervisor_id": supervisor_id,
                            "plan_sha256": digest,
                            "runtime_probe_sha256": hashlib.sha256(
                                _json(attestation).encode()
                            ).hexdigest(),
                        }
                    )
                    + "\n"
                ).encode(),
            )
            environment = local_environment(plan)
            if self._live_push is not None:
                environment.update(
                    WAVEDL_DAEMON_URL=self._live_push.url,
                    WAVEDL_DAEMON_TOKEN=self._live_push.token,
                )
            environment.update(
                WAVEDL_RUN_ID=plan.run_id,
                WAVEDL_REQUEST_ID=request_id,
                WAVEDL_OWNER=self.owner,
                WAVEDL_SUPERVISOR_ID=supervisor_id,
                WAVEDL_PLAN_SHA256=digest,
                WAVEDL_ATTEMPT_ALLOCATION="writer",
            )
            process = self._supervisor.start(
                ProcessSpec(
                    (
                        plan.manifest.runtime.python,
                        "-u",
                        "-m",
                        "wavedl.local_startup",
                        plan.working_directory,
                    ),
                    Path(plan.working_directory),
                    environment,
                    inherit_environment=False,
                ),
                output_limit=self._output_limit,
            )
            handle = SubmissionHandle(
                "local",
                plan.run_id,
                request_id,
                self.owner,
                plan.run_dir,
                pid=process.pid,
                supervisor_id=supervisor_id,
                raw_json=_json({"runtime_attestation": attestation}),
            )
            try:
                future = self._waiters.submit(process.wait)
            except BaseException:
                process.cancel(grace_seconds=0)
                process.wait()
                raise
            self._records[key] = _Owned(handle, plan.to_json(), plan, process, future)
            return handle

    def discover_initial(self, handle: SubmissionHandle):
        """Verify startup proof only under the exact current supervisor receipt."""
        record = self._owned(handle)
        if record is None:
            raise SupervisionLost("Cannot bind a persisted or foreign Local PID")
        digest = hashlib.sha256(record.plan_json.encode()).hexdigest()
        submission = identity_for_submission(
            Path(record.plan.run_dir),
            request_id=handle.request_id,
            backend="local",
            owner=handle.owner,
            plan_sha256=digest,
        )
        identity = LocalStartupIdentity(
            run_id=submission.run_id,
            request_id=submission.request_id,
            owner=submission.owner,
            supervisor_id=handle.supervisor_id,
            run_directory=submission.run_directory,
            manifest_sha256=submission.manifest_sha256,
            plan_sha256=digest,
        )
        try:
            binding = read_local_binding(Path(handle.run_dir), identity)
        except FileNotFoundError:
            return None
        runtime_handle = handle.bind_attempt(binding.attempt)
        with self._lock:
            current = self._records.get((handle.request_id, None))
            if (
                current is None
                or current.handle != handle
                or current.plan_json != record.plan_json
                or current.runtime_handle not in {None, runtime_handle}
            ):
                raise SupervisionLost("Local submission ownership changed")
            from dataclasses import replace

            self._records[(handle.request_id, None)] = replace(
                current, runtime_handle=runtime_handle
            )
        return binding

    def _owned(self, handle: Handle | SubmissionHandle) -> _Owned | None:
        if (
            not isinstance(handle, (Handle, SubmissionHandle))
            or handle.backend != "local"
        ):
            raise ValueError("not a Local handle")
        if self._pid != os.getpid():
            return None
        with self._lock:
            record = self._records.get(
                (handle.request_id, getattr(handle, "attempt", None))
            )
            if record is None and isinstance(handle, Handle):
                record = self._records.get((handle.request_id, None))
            if record is not None and handle not in {
                record.handle,
                record.runtime_handle,
            }:
                raise ValueError("foreign or altered Local handle")
            return record

    def process_result(self, handle: Handle | SubmissionHandle) -> ProcessResult | None:
        record = self._owned(handle)
        if record is None or not record.result.done():
            return None
        return record.result.result()

    def observe(
        self, handle: Handle | SubmissionHandle
    ) -> BackendObservation | SubmissionObservation:
        record = self._owned(handle)

        def observation(status, **values):
            if isinstance(handle, SubmissionHandle):
                return SubmissionObservation(status, **values)
            return BackendObservation(handle.attempt, status, **values)

        if record is None:
            return observation(
                ObservationStatus.SUPERVISION_LOST,
                detail="persisted PID is not process ownership",
            )
        if not record.result.done():
            return observation(ObservationStatus.RUNNING)
        try:
            result = record.result.result()
        except Exception as error:
            return observation(
                ObservationStatus.ERROR,
                detail=f"supervision failed: {error}",
            )
        code = result.returncode
        outcome = OperationOutcome(
            OutcomeStatus.CANCELLED
            if result.cancelled and not result.timed_out
            else OutcomeStatus.SUCCESS
            if code == 0
            else OutcomeStatus.FAILED,
            exit_code=code if code >= 0 else None,
            signal=-code if code < 0 else None,
        )
        return observation(
            ObservationStatus.TERMINAL,
            # This is evidence from the owned supervisor after tree termination,
            # not the controller's persisted cancellation intent. Keep the raw
            # termination code/signal alongside the native lifecycle so 143 or
            # SIGTERM is not mistaken for an independent wrapper failure.
            native_state="CANCELLED"
            if result.cancelled and not result.timed_out
            else None,
            exit_code=outcome.exit_code,
            signal=outcome.signal,
            batch=outcome,
        )

    def request_cancel(self, handle: Handle) -> None:
        require_dispatch(self.dispatch_lock)
        record = self._owned(handle)
        if record is None:
            raise SupervisionLost("refusing to signal a persisted/reused PID")
        record.process.cancel(grace_seconds=self._cancel_grace)

    def collect(self, handle: Handle) -> OperationResult:
        record = self._owned(handle)
        if record is None:
            raise SupervisionLost("cannot collect evidence without owned supervision")
        observation = self.observe(handle)
        unknown = OperationOutcome(OutcomeStatus.UNKNOWN)
        return OperationResult(
            record.plan.operation, unknown, observation.batch or unknown, unknown
        )

    def release(self, handle: Handle) -> None:
        """Free terminal diagnostics after the caller durably commits evidence.

        The immutable attempt reservation remains as the no-resubmit fence.
        This method cannot verify the controller's persistence boundary; callers
        must never release before recording the terminal observation durably.
        """
        require_dispatch(self.dispatch_lock)
        with self._lock:
            record = self._owned(handle)
            if record is None:
                raise SupervisionLost("cannot release a handle without live ownership")
            if not record.result.done():
                raise RuntimeError("cannot release active Local supervision")
            del self._records[
                (record.handle.request_id, getattr(record.handle, "attempt", None))
            ]

    def close(self) -> None:
        require_dispatch(self.dispatch_lock)
        with self._lock:
            self._closed = True
            for record in self._records.values():
                if not record.result.done():
                    record.process.cancel(grace_seconds=self._cancel_grace)
        self._waiters.shutdown(wait=True)
