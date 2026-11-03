# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Turn active run heads into backend work, one bounded page per tick.

Ticks run on the thread that owns the database and never wait for I/O.
Blocking work (spawning, reading attempt files) runs in the pool and returns
the events to append with the head tokens it read, so a result that raced
another append is dropped as stale and recomputed on a later tick.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from wavedl_daemon.backends.contracts import UnsupportedPlan
from wavedl_daemon.backends.local_execution import SupervisionLost
from wavedl_daemon.events.metrics_tail import read_attempt_files, read_tail
from wavedl_daemon.execution.budget import BudgetExhausted
from wavedl_daemon.runs.layout import RunPaths
from wavedl_daemon.runs.lifecycle import (
    Append,
    Ended,
    Live,
    Work,
    WorkKind,
    next_work,
    settle,
)
from wavedl_daemon.runs.model import (
    CancelDelivered,
    Exited,
    LocalExit,
    LocalHandle,
    Observed,
    Outcome,
    Phase,
    Submitted,
)
from wavedl_daemon.storage.run_log import Expect, Stale


if TYPE_CHECKING:
    from collections.abc import Callable

    from wavedl_daemon.backends.contracts import Plan
    from wavedl_daemon.backends.local import LocalBackend
    from wavedl_daemon.execution.pool import BoundedThreadPool, WorkHandle
    from wavedl_daemon.runs.lifecycle import RunHead
    from wavedl_daemon.runs.model import NotOwned, RunEvent, RunId
    from wavedl_daemon.storage.plans import PlanStore
    from wavedl_daemon.storage.run_log import RunLog


PAGE_SIZE = 256
TAIL_INTERVAL_SECONDS = 1.0
MAX_ERRORS = 128


@dataclass(frozen=True)
class Done:
    """A worker's result: events to append under the tokens it read."""

    events: tuple[RunEvent, ...]
    expect: Expect
    release: LocalHandle | None = None  # forget the process once its exit commits


def _spawn(backend: LocalBackend, plan: Plan, head: RunHead) -> Done:
    try:
        probed, handle = backend.spawn(plan, head.attempt)
        # A started Local process is running; nothing queues it.
        running = Observed(Phase.RUNNING, None, None)
        events: tuple[RunEvent, ...] = (probed, Submitted(handle), running)
    except (UnsupportedPlan, SupervisionLost, BudgetExhausted, OSError) as error:
        events = (Exited(Outcome.FAILED, True, detail=str(error)[:1000]),)
    # The process may exist now, so a cancel requested meanwhile must not make
    # this result stale: only a different attempt could.
    return Done(events, Expect(offset=(head.attempt, head.metrics_offset)))


def _tail(head: RunHead) -> Done | None:
    chunk = read_tail(
        RunPaths(head.ref.run_dir).metrics(head.attempt), start=head.metrics_offset
    )
    if not chunk.rows:
        return None
    return Done(chunk.rows, Expect(offset=(head.attempt, head.metrics_offset)))


def _settle(head: RunHead, cause: LocalExit | NotOwned) -> Done:
    events = settle(head, cause, read_attempt_files(head))
    release = None
    if isinstance(head.stage, Live) and isinstance(head.stage.handle, LocalHandle):
        if events and isinstance(events[-1], Exited):
            release = head.stage.handle
    expect = Expect(revision=head.revision, offset=(head.attempt, head.metrics_offset))
    return Done(events, expect, release)


class RunController:
    """Owns no state that a restart needs: the run log is the work list."""

    def __init__(
        self,
        log: RunLog,
        plans: PlanStore,
        pool: BoundedThreadPool,
        *,
        local: LocalBackend | None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._log = log
        self._plans = plans
        self._local = local
        self._pool = pool
        self._clock = clock
        self._inflight: dict[RunId, tuple[WorkKind, WorkHandle[Done | None]]] = {}
        self._next_tail: dict[RunId, float] = {}
        self._offset = 0
        self._thread = threading.get_ident()
        self.errors: dict[str, str] = {}

    def _guard(self) -> None:
        if threading.get_ident() != self._thread:
            raise RuntimeError("The controller must stay on its database owner thread")

    def tick(self) -> None:
        self._guard()
        self._harvest()
        heads = self._log.heads(active=True, limit=PAGE_SIZE, offset=self._offset)
        self._offset = self._offset + len(heads) if len(heads) == PAGE_SIZE else 0
        for head in heads:
            if head.ref.run_id in self._inflight:
                continue
            try:
                self._advance(head)
            except (BudgetExhausted, Stale):
                continue  # retried on a later tick from the committed head
            except Exception as error:
                self._error(head, error)
            else:
                self.errors.pop(head.ref.plan_id, None)

    def _advance(self, head: RunHead) -> None:
        step = next_work(head, local_poll=self._poll(head))
        match step:
            case Append(events=events, then=then):
                if then is WorkKind.SPAWN and not self._admits():
                    return
                head = self._log.append(
                    head.ref.run_id, events, expect=Expect(revision=head.revision)
                )
                if then is not None:
                    self._run(then, head)
            case Work(kind=kind):
                self._run(kind, head)

    def _poll(self, head: RunHead) -> LocalExit | NotOwned | None:
        stage = head.stage
        if not isinstance(stage, Live) or not isinstance(stage.handle, LocalHandle):
            return None
        if self._local is None:
            raise UnsupportedPlan("The Local backend is not registered")
        return self._local.poll(stage.handle)

    def _admits(self) -> bool:
        if self._local is None:
            return False
        spawning = sum(kind is WorkKind.SPAWN for kind, _ in self._inflight.values())
        return self._local.running + spawning < self._local.max_processes

    def _run(self, kind: WorkKind, head: RunHead) -> None:
        run_id = head.ref.run_id
        match kind:
            case WorkKind.SPAWN:
                record = self._plans.get(head.ref.plan_id)
                local = self._local
                if record is None or local is None:
                    raise UnsupportedPlan("The run has no plan or no Local backend")
                self._submit(kind, run_id, lambda: _spawn(local, record.plan, head))
            case WorkKind.RECOVER:
                # The claim committed but its spawn result did not. The process
                # may exist, so it is never started again.
                ending = Exited(
                    Outcome.LOST,
                    False,
                    detail="The daemon stopped before the launch was recorded",
                )
                self._log.append(run_id, [ending], expect=Expect(revision=head.revision))
            case WorkKind.CANCEL:
                stage = head.stage
                if self._local is None or not isinstance(stage, Live):
                    raise UnsupportedPlan("Only Local runs can be cancelled for now")
                if isinstance(stage.handle, LocalHandle):
                    self._local.cancel(stage.handle)
                self._log.append(
                    run_id, [CancelDelivered()], expect=Expect(revision=head.revision)
                )
            case WorkKind.TAIL:
                now = self._clock()
                if now < self._next_tail.get(run_id, 0):
                    return
                self._next_tail[run_id] = now + TAIL_INTERVAL_SECONDS
                self._submit(kind, run_id, lambda: _tail(head))
            case WorkKind.SETTLE:
                cause = self._poll(head)
                if cause is None:
                    raise UnsupportedPlan("Slurm settling is being rebuilt")
                self._submit(kind, run_id, lambda: _settle(head, cause))
            case WorkKind.SBATCH:
                raise UnsupportedPlan("Slurm submission is being rebuilt")

    def _submit(
        self, kind: WorkKind, run_id: RunId, work: Callable[[], Done | None]
    ) -> None:
        self._inflight[run_id] = (kind, self._pool.control_lane.submit(lambda token: work()))

    def _harvest(self) -> None:
        for run_id, (kind, handle) in list(self._inflight.items()):
            if not handle.done():
                continue
            del self._inflight[run_id]
            try:
                done = handle.result()
                if done is None:
                    continue
                head = self._log.append(run_id, done.events, expect=done.expect)
            except Stale:
                continue
            except Exception as error:
                current = self._log.head(run_id)
                if current is not None:
                    self._error(current, error)
                continue
            if isinstance(head.stage, Ended):
                self._next_tail.pop(run_id, None)
                if done.release is not None and self._local is not None:
                    self._local.release(done.release)

    def _error(self, head: RunHead, error: BaseException) -> None:
        if head.ref.plan_id not in self.errors and len(self.errors) >= MAX_ERRORS:
            self.errors.pop(next(iter(self.errors)))
        self.errors[head.ref.plan_id] = type(error).__name__
