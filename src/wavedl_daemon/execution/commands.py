# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Shell-free command execution with real process-tree deadlines.

run_command is synchronous. Event-loop callers use submit_command and await its
handle. Output is diagnostic and explicitly truncated, never durable telemetry.
Each pipe receives half the aggregate output budget (rounding down).
"""

from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING

from .budget import BudgetExhausted, StorageWorkError, WorkBudget


if TYPE_CHECKING:
    from wavedl_daemon.supervision.base import ProcessResult, ProcessSpec, Supervisor

    from .pool import BoundedThreadPool, CancellationToken, WorkHandle


def run_command(
    spec: ProcessSpec,
    budget: WorkBudget,
    *,
    token: CancellationToken | None = None,
    supervisor: Supervisor | None = None,
) -> ProcessResult:
    """Run one command; timeout returns only after real work is terminated."""
    command_bytes = sum(len(arg.encode("utf-8")) + 1 for arg in spec.argv)
    command_bytes += sum(
        len(key.encode("utf-8")) + len(value.encode("utf-8")) + 2
        for key, value in (spec.environment or {}).items()
    )
    if command_bytes > budget.max_command_bytes:
        raise BudgetExhausted("command exceeds byte budget")
    if token is not None:
        token.check()
    if supervisor is None:
        if os.name == "nt":
            from wavedl_daemon.supervision.windows import WindowsSupervisor

            supervisor = WindowsSupervisor()
        else:
            from wavedl_daemon.supervision.posix import POSIXSupervisor

            supervisor = POSIXSupervisor()
    try:
        process = supervisor.start(spec, output_limit=budget.max_output_bytes // 2)
    except OSError as error:
        raise StorageWorkError("command could not start; operation failed") from error
    finished = threading.Event()

    def watch_cancellation() -> None:
        while not finished.wait(0.01):
            # Deadline belongs to wait(), whose result must report timed_out.
            if token is not None and token.cancellation_requested:
                process.cancel(grace_seconds=0.2)
                return

    watcher = None
    if token is not None:
        watcher = threading.Thread(target=watch_cancellation, daemon=True)
        watcher.start()
    timeout = budget.timeout_seconds
    if token is not None:
        timeout = min(timeout, max(0.0, token.deadline - time.monotonic()))
    try:
        try:
            return process.wait(timeout=timeout)
        except BaseException as error:
            # Never abandon live work when supervision or storage reports failure.
            process.cancel(grace_seconds=0.2)
            process.wait()
            if isinstance(error, OSError):
                raise StorageWorkError(
                    "command supervision failed; operation failed"
                ) from error
            raise
    finally:
        finished.set()
        if watcher is not None:
            watcher.join()


def submit_command(
    pool: BoundedThreadPool, spec: ProcessSpec, *, supervisor: Supervisor | None = None
) -> WorkHandle[ProcessResult]:
    """Admit a command without blocking the event loop.

    Like all pool work, cancellation/deadline raises WorkCancelled only after
    termination. Direct run_command returns ProcessResult with timed_out instead.
    """
    return pool.submit(
        lambda token: run_command(spec, pool.budget, token=token, supervisor=supervisor)
    )
