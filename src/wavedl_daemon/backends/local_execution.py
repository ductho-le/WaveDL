# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Bounded Local process ownership; saved PIDs never authorize signalling.

Supervisors own pipe draining and tree cleanup, and live only in this daemon
process. A handle whose supervisor is gone, after a restart or a fork, is
``NotOwned``: its PID is never adopted or signalled.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from wavedl_daemon.execution.budget import BudgetExhausted
from wavedl_daemon.runs.layout import RESULT_OPTIONS
from wavedl_daemon.runs.model import LocalExit, LocalHandle, NotOwned, Probed
from wavedl_daemon.storage.ownership import require_dispatch
from wavedl_daemon.supervision.base import (
    ProcessResult,
    ProcessSpec,
    validate_output_limit,
    validate_seconds,
)

from .contracts import UnsupportedPlan


if TYPE_CHECKING:
    from wavedl_daemon.storage.ownership import DispatchLock
    from wavedl_daemon.supervision.base import SupervisedProcess, Supervisor

    from .contracts import Plan


# The launcher forwards the stop signal to training, which runs in its own
# session, and kills that session 30 s later. Upstream training saves
# interrupted_checkpoint only on SIGINT, so cancel interrupts rather than
# terminates, and SIGKILL to the launcher waits until its grace is over: a
# kill before then would orphan the training session.
CANCEL_GRACE_SECONDS = 45.0


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


def attempt_argv(plan: Plan, attempt: int) -> tuple[str, ...]:
    """The approved command plus where this attempt's results go."""
    if plan.operation not in RESULT_OPTIONS:
        return plan.argv
    option, name = RESULT_OPTIONS[plan.operation]
    spec = plan.spec
    return (*plan.argv, option, spec.execution_path(spec.paths.attempt(attempt) / name))


def _claim(plan: Plan, attempt: int) -> Path:
    """Create the run directories and claim ``attempts/N`` exclusively."""
    paths = plan.spec.paths
    for directory in (paths.run_dir, paths.output, paths.attempt(attempt).parent):
        directory.mkdir(mode=0o700, exist_ok=True)
    try:
        paths.attempt(attempt).mkdir(mode=0o700)
    except FileExistsError:
        raise SupervisionLost(
            f"Attempt {attempt} was already claimed; it is never started twice"
        ) from None
    return paths.run_dir


def _probe_code(plan: Plan) -> str:
    device = plan.spec.runtime.device
    processes = plan.topology.processes_per_machine
    checks = (
        "if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')\n"
        f"if torch.cuda.device_count() < {processes}: "
        "raise RuntimeError('CUDA allocation insufficient')\n"
        f"for i in range({processes}): torch.empty(1,device='cuda:'+str(i))\n"
        f"devices=[torch.cuda.get_device_name(i) for i in range({processes})]\n"
        if device == "cuda"
        else "torch.empty(1,device='cpu')\ndevices=['cpu']\n"
    )
    if device == "cpu" and processes > 1:
        checks += (
            "if not torch.distributed.is_available() or not "
            "torch.distributed.is_gloo_available(): "
            "raise RuntimeError('CPU distributed training requires Gloo')\n"
        )
    return (
        "import json,sys,torch\n"
        "if sys.version_info < (3,11): raise RuntimeError('Python 3.11+ required')\n"
        + checks
        + "print(json.dumps({'interpreter':sys.executable,'prefix':sys.prefix,"
        "'devices':devices}))"
    )


def _probed(python: str, stdout: bytes) -> Probed:
    value = json.loads(stdout)
    if type(value) is not dict or set(value) != {"interpreter", "prefix", "devices"}:
        raise ValueError("Invalid probe fields")
    interpreter, prefix, devices = value["interpreter"], value["prefix"], value["devices"]
    if not all(
        type(item) is str and os.path.isabs(item) for item in (interpreter, prefix)
    ) or not (type(devices) is list and all(type(item) is str for item in devices)):
        raise ValueError("Invalid probe identity")
    return Probed(python, interpreter, prefix, tuple(devices))


@dataclass(frozen=True)
class _Owned:
    handle: LocalHandle
    process: SupervisedProcess
    result: Future[ProcessResult]
    attempt: tuple[str, int]  # (run_id, attempt)


class LocalExecution:
    def __init__(
        self,
        *,
        dispatch_lock: DispatchLock | None = None,
        max_processes: int = 2,
        output_limit: int = 1048576,
        probe_timeout: float = 10,
        cancel_grace: float = CANCEL_GRACE_SECONDS,
        supervisor: Supervisor | None = None,
    ):
        if type(max_processes) is not int or max_processes < 1:
            raise ValueError("max_processes must be a positive integer")
        validate_output_limit(output_limit)
        validate_seconds(probe_timeout, "probe_timeout")
        validate_seconds(cancel_grace, "cancel_grace")
        if probe_timeout == 0:
            raise ValueError("probe_timeout must be positive")
        if supervisor is None:
            if os.name == "nt":
                from wavedl_daemon.supervision.windows import WindowsSupervisor

                supervisor = WindowsSupervisor()
            else:
                from wavedl_daemon.supervision.posix import POSIXSupervisor

                supervisor = POSIXSupervisor()
        self.dispatch_lock = dispatch_lock
        self._pid = os.getpid()
        self._supervisor = supervisor
        self._max_processes = max_processes
        self._output_limit = output_limit
        self._probe_timeout = probe_timeout
        self._cancel_grace = cancel_grace
        self._lock = threading.Lock()
        self._records: dict[str, _Owned] = {}
        self._starting = 0
        self._waiters = ThreadPoolExecutor(
            max_workers=max_processes, thread_name_prefix="local-wait"
        )
        self._closed = False

    @property
    def max_processes(self) -> int:
        return self._max_processes

    @property
    def running(self) -> int:
        with self._lock:
            return self._starting + sum(
                not record.result.done() for record in self._records.values()
            )

    def _probe(self, plan: Plan, cwd: Path, environment: dict[str, str]) -> Probed:
        python = plan.spec.runtime.python
        process = self._supervisor.start(
            ProcessSpec(
                (python, "-c", _probe_code(plan)),
                cwd,
                environment,
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
            detail = result.stderr.decode("utf-8", "replace").strip()[-500:]
            raise UnsupportedPlan(
                "selected interpreter/device probe refused execution; no fallback"
                + (f": {detail}" if detail else "")
            )
        try:
            return _probed(python, result.stdout)
        except (ValueError, TypeError) as error:
            raise UnsupportedPlan(
                "selected interpreter probe identity is invalid; no fallback"
            ) from error

    def spawn(self, plan: Plan, attempt: int) -> tuple[Probed, LocalHandle]:
        """Claim ``attempts/N``, probe the interpreter, then start the group.

        Blocking: call from a worker. Any failure after the claim leaves the
        attempt claimed, so it is reported once and never retried.
        """
        require_dispatch(self.dispatch_lock)
        if self._pid != os.getpid():
            raise SupervisionLost("backend process ownership cannot survive fork")
        argv = attempt_argv(plan, attempt)
        if sum(len(arg.encode("utf-8")) + 1 for arg in argv) > 65536:
            raise UnsupportedPlan("Local command exceeds byte budget")
        environment = local_environment(plan)
        with self._lock:
            if self._closed:
                raise RuntimeError("Local backend is closed")
            running = sum(not record.result.done() for record in self._records.values())
            if running + self._starting >= self._max_processes:
                raise BudgetExhausted("Local process admission exhausted")
            self._starting += 1
        try:
            cwd = _claim(plan, attempt)
            probed = self._probe(plan, cwd, environment)
            # Probe latency is not permission to act with relinquished dispatch.
            require_dispatch(self.dispatch_lock)
            with self._lock:
                if self._closed:
                    raise RuntimeError("Local backend closed during the probe")
                process = self._supervisor.start(
                    ProcessSpec(argv, cwd, environment, inherit_environment=False),
                    output_limit=self._output_limit,
                )
                handle = LocalHandle(process.pid, secrets.token_urlsafe(32))
                try:
                    future = self._waiters.submit(process.wait)
                except BaseException:
                    process.cancel(grace_seconds=0)
                    process.wait()
                    raise
                self._records[handle.supervisor_id] = _Owned(
                    handle, process, future, (plan.run_id, attempt)
                )
        finally:
            with self._lock:
                self._starting -= 1
        return probed, handle

    def _owned(self, handle: LocalHandle) -> _Owned | None:
        if self._pid != os.getpid():
            return None
        with self._lock:
            record = self._records.get(handle.supervisor_id)
        return record if record is not None and record.handle == handle else None

    def started(self, run_id: str, attempt: int) -> LocalHandle | None:
        """The handle of the process this daemon process started for the attempt."""
        if self._pid != os.getpid():
            return None
        with self._lock:
            return next(
                (
                    record.handle
                    for record in self._records.values()
                    if record.attempt == (run_id, attempt)
                ),
                None,
            )

    def poll(self, handle: LocalHandle) -> LocalExit | NotOwned | None:
        """Non-blocking. None while the process group still runs."""
        record = self._owned(handle)
        if record is None:
            return NotOwned()
        if not record.result.done():
            return None
        try:
            code = record.result.result().returncode
        except Exception:
            return LocalExit(None, None)
        return LocalExit(code, None) if code >= 0 else LocalExit(None, -code)

    def cancel(self, handle: LocalHandle) -> None:
        """SIGINT the group now and SIGKILL it after the grace."""
        require_dispatch(self.dispatch_lock)
        record = self._owned(handle)
        if record is None:
            raise SupervisionLost("refusing to signal a persisted/reused PID")
        record.process.cancel(grace_seconds=self._cancel_grace, interrupt=True)

    def release(self, handle: LocalHandle) -> None:
        """Forget an exited process once its exit is durably recorded."""
        with self._lock:
            record = self._records.get(handle.supervisor_id)
            if record is not None and record.handle == handle and record.result.done():
                del self._records[handle.supervisor_id]

    def close(self) -> None:
        require_dispatch(self.dispatch_lock)
        with self._lock:
            self._closed = True
            for record in self._records.values():
                if not record.result.done():
                    record.process.cancel(grace_seconds=self._cancel_grace, interrupt=True)
        self._waiters.shutdown(wait=True)
