# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Turn active run heads into backend work, one bounded page per tick.

Ticks run on the thread that owns the database and never wait for I/O.
Blocking work (spawning, sbatch, scheduler queries, reading attempt files)
runs in the pool and returns the events to append with the head tokens it
read, so a result that raced another append is dropped as stale and
recomputed on a later tick.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from wavedl_daemon.backends.contracts import UnsupportedPlan
from wavedl_daemon.backends.local_execution import (
    CANCEL_GRACE_SECONDS,
    SupervisionLost,
)
from wavedl_daemon.backends.slurm import SubmissionUnknown
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
    react,
    settle,
)
from wavedl_daemon.runs.model import (
    BackendKind,
    CancelDelivered,
    Exited,
    LocalHandle,
    ObservationFailed,
    Observed,
    Outcome,
    Phase,
    SlurmHandle,
    SlurmObservation,
    Submitted,
)
from wavedl_daemon.storage.run_log import Expect, Stale


if TYPE_CHECKING:
    from collections.abc import Callable

    from wavedl_daemon.backends.contracts import Plan
    from wavedl_daemon.backends.local import LocalBackend
    from wavedl_daemon.backends.slurm import SlurmBackend
    from wavedl_daemon.execution.pool import BoundedThreadPool, WorkHandle
    from wavedl_daemon.runs.lifecycle import RunHead
    from wavedl_daemon.runs.model import LocalPoll, PlanId, RunEvent, RunId
    from wavedl_daemon.storage.plans import PlanRecord, PlanStore
    from wavedl_daemon.storage.run_log import RunLog


PAGE_SIZE = 256
TAIL_INTERVAL_SECONDS = 1.0
RETRY_SECONDS = 5.0
MAX_ERRORS = 128
OBSERVE_BATCH = 128
OBSERVE_INTERVAL_SECONDS = 30.0
OBSERVE_BACKOFF_SECONDS = (5.0, 120.0)


@dataclass(frozen=True)
class Done:
    """A worker's result: events to append under the tokens it read."""

    events: tuple[RunEvent, ...]
    expect: Expect
    release: LocalHandle | None = None  # forget the process once its exit commits
    # A Slurm settle may leave the attempt open (an undrained file, a second
    # requeue); its observation is folded again on the next tick.
    observation: SlurmObservation | None = None


@dataclass
class _Watch:
    """One Live Slurm run's polling state. A daemon restart starts it afresh."""

    plan_id: PlanId
    created_at: datetime  # bounds the accounting a reused job ID could match
    due: float = 0.0
    failures: int = 0
    error: str | None = None
    pending: SlurmObservation | ObservationFailed | None = None  # not yet reacted to
    kill_at: float | None = None


def _require_checkpoint(plan: Plan) -> None:
    """A test reads its source checkpoint in place, so it must still exist."""
    source = plan.spec.checkpoint_source
    if source is not None and not Path(source.path).is_dir():
        raise FileNotFoundError(
            f"The checkpoint {source.checkpoint} of run {source.run_id} is gone"
        )


def _spawn(backend: LocalBackend, plan: Plan, head: RunHead) -> Done:
    try:
        _require_checkpoint(plan)
        probed, handle = backend.spawn(plan, head.attempt)
        # A started Local process is running; nothing queues it.
        running = Observed(Phase.RUNNING, None, None)
        events: tuple[RunEvent, ...] = (probed, Submitted(handle), running)
    except (UnsupportedPlan, SupervisionLost, BudgetExhausted, OSError) as error:
        events = (Exited(Outcome.FAILED, True, detail=str(error)[:1000]),)
    # The process may exist now, so a cancel requested meanwhile must not make
    # this result stale: only a different attempt could.
    return Done(events, Expect(offset=(head.attempt, head.metrics_offset)))


def _sbatch(backend: SlurmBackend, plan: Plan, head: RunHead) -> Done:
    """Submit once. An unknown sbatch outcome raises and is recovered later."""
    try:
        _require_checkpoint(plan)
        events: tuple[RunEvent, ...] = (Submitted(backend.submit(plan)),)
    except (UnsupportedPlan, OSError) as error:
        events = (Exited(Outcome.FAILED, True, detail=str(error)[:1000]),)
    return Done(events, Expect(offset=(head.attempt, head.metrics_offset)))


def _recover(
    backend: SlurmBackend, record: PlanRecord, head: RunHead, unknown: str | None
) -> Done:
    """Find the job an unknown sbatch outcome, or a crash after sbatch, left.

    ``unknown`` is why this daemon's sbatch outcome is unknown, if it knows.
    """
    handle = backend.find(record.plan, created_at=datetime.fromisoformat(record.created_at))
    reason = unknown or "The daemon stopped before the submission was recorded"
    lost = Exited(Outcome.UNKNOWN, False, detail=f"{reason}, and no job was found"[:1000])
    events: tuple[RunEvent, ...] = (lost,) if handle is None else (Submitted(handle),)
    return Done(events, Expect(offset=(head.attempt, head.metrics_offset)))


def _scancel(
    backend: SlurmBackend, head: RunHead, handle: SlurmHandle, *, interrupt: bool
) -> Done:
    backend.cancel(head.ref.run_id, handle, interrupt=interrupt)
    return Done((CancelDelivered(),), Expect(revision=head.revision))


def _kill(backend: SlurmBackend, head: RunHead, handle: SlurmHandle) -> None:
    backend.cancel(head.ref.run_id, handle, interrupt=False)


def _tail(head: RunHead) -> Done | None:
    chunk = read_tail(
        RunPaths(head.ref.run_dir).metrics(head.attempt), start=head.metrics_offset
    )
    if not chunk.rows:
        return None
    return Done(chunk.rows, Expect(offset=(head.attempt, head.metrics_offset)))


def _settle(head: RunHead, cause: LocalPoll | SlurmObservation) -> Done:
    events = settle(head, cause, read_attempt_files(head))
    release = None
    if isinstance(head.stage, Live) and isinstance(head.stage.handle, LocalHandle):
        if events and isinstance(events[-1], Exited):
            release = head.stage.handle
    expect = Expect(revision=head.revision, offset=(head.attempt, head.metrics_offset))
    observation = cause if isinstance(cause, SlurmObservation) else None
    return Done(events, expect, release, observation)


class RunController:
    """Owns no state that a restart needs: the run log is the work list."""

    def __init__(
        self,
        log: RunLog,
        plans: PlanStore,
        pool: BoundedThreadPool,
        *,
        local: LocalBackend | None,
        slurm: SlurmBackend | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._log = log
        self._plans = plans
        self._local = local
        self._slurm = slurm
        self._pool = pool
        self._clock = clock
        self._inflight: dict[RunId, tuple[WorkKind, WorkHandle[Done | None]]] = {}
        self._next_tail: dict[RunId, float] = {}
        self._retry_at: dict[RunId, float] = {}
        self._watches: dict[RunId, _Watch] = {}
        self._unknown_submissions: dict[RunId, str] = {}
        self._observing: tuple[tuple[RunId, ...], WorkHandle[dict]] | None = None
        self._offset = 0
        self._thread = threading.get_ident()
        self.errors: dict[str, str] = {}
        self.observation_refusals = 0

    @property
    def observation_errors(self) -> dict[str, str]:
        """The last scheduler query failure of each Slurm run, by plan."""
        return {
            watch.plan_id: watch.error
            for watch in self._watches.values()
            if watch.error is not None
        }

    def _guard(self) -> None:
        if threading.get_ident() != self._thread:
            raise RuntimeError("The controller must stay on its database owner thread")

    def tick(self) -> None:
        self._guard()
        self._harvest()
        self._harvest_observations()
        heads = self._log.heads(active=True, limit=PAGE_SIZE, offset=self._offset)
        self._offset = self._offset + len(heads) if len(heads) == PAGE_SIZE else 0
        now = self._clock()
        for head in heads:
            run_id = head.ref.run_id
            if run_id in self._inflight or now < self._retry_at.get(run_id, 0):
                continue
            try:
                self._advance(head)
            except (BudgetExhausted, Stale):
                continue  # retried on a later tick from the committed head
            except Exception as error:
                self._error(head, error)
            else:
                self.errors.pop(head.ref.plan_id, None)
        self._observe(heads)

    def _advance(self, head: RunHead) -> None:
        poll = self._poll(head)
        watch = self._watches.get(head.ref.run_id)
        observation = None if watch is None else watch.pending
        step = None
        if watch is not None and observation is not None:
            watch.pending = None
            step = react(head, observation)
        if step is None and self._escalate(head):
            return
        if step is None:
            step = next_work(head, local_poll=poll)
        match step:
            case Append(events=events, then=then):
                if then is not None and not self._admits(then):
                    return
                head = self._log.append(
                    head.ref.run_id, events, expect=Expect(revision=head.revision)
                )
                if then is not None:
                    self._run(then, head)
            case Work(kind=WorkKind.SETTLE):
                cause = observation if isinstance(observation, SlurmObservation) else poll
                if cause is None:
                    raise UnsupportedPlan("There is no exit to settle")
                self._submit(WorkKind.SETTLE, head.ref.run_id, lambda: _settle(head, cause))
            case Work(kind=kind):
                self._run(kind, head)

    def _poll(self, head: RunHead) -> LocalPoll | None:
        stage = head.stage
        if not isinstance(stage, Live) or not isinstance(stage.handle, LocalHandle):
            return None
        if self._local is None:
            raise UnsupportedPlan("The Local backend is not registered")
        return self._local.poll(stage.handle)

    def _admits(self, kind: WorkKind) -> bool:
        """Whether the work a claim fences would start now.

        A claim commits only then, so a claim with nothing in flight is always
        one a daemon crash left behind.
        """
        if kind is WorkKind.SBATCH and self._slurm is None:
            raise UnsupportedPlan("The Slurm backend is not registered")
        if kind is WorkKind.SPAWN:
            if self._local is None:
                return False
            spawning = sum(kind is WorkKind.SPAWN for kind, _ in self._inflight.values())
            if self._local.running + spawning >= self._local.max_processes:
                return False
        return self._pool.control_lane.admits

    def _backend(self) -> SlurmBackend:
        if self._slurm is None:
            raise UnsupportedPlan("The Slurm backend is not registered")
        return self._slurm

    def _record(self, head: RunHead) -> PlanRecord:
        record = self._plans.get(head.ref.plan_id)
        if record is None:
            raise UnsupportedPlan("The run has no plan")
        return record

    def _run(self, kind: WorkKind, head: RunHead) -> None:
        run_id = head.ref.run_id
        slurm = head.ref.backend is BackendKind.SLURM
        match kind:
            case WorkKind.SPAWN:
                record, local = self._record(head), self._local
                if local is None:
                    raise UnsupportedPlan("The Local backend is not registered")
                self._submit(kind, run_id, lambda: _spawn(local, record.plan, head))
            case WorkKind.SBATCH:
                record, backend = self._record(head), self._backend()
                self._submit(kind, run_id, lambda: _sbatch(backend, record.plan, head))
            case WorkKind.RECOVER if slurm:
                record, backend = self._record(head), self._backend()
                unknown = self._unknown_submissions.get(run_id)
                self._submit(kind, run_id, lambda: _recover(backend, record, head, unknown))
            case WorkKind.RECOVER:
                # The claim committed but its spawn result did not. A process
                # another daemon started may exist, so it is never started again.
                local = self._local
                handle = None if local is None else local.started(run_id, head.attempt)
                if handle is None:
                    detail = "The daemon stopped before the launch was recorded"
                    events: tuple[RunEvent, ...] = (Exited(Outcome.LOST, False, detail=detail),)
                else:
                    events = (Submitted(handle), Observed(Phase.RUNNING, None, None))
                self._log.append(run_id, events, expect=Expect(revision=head.revision))
            case WorkKind.CANCEL:
                self._cancel(head)
            case WorkKind.TAIL:
                now = self._clock()
                if now < self._next_tail.get(run_id, 0):
                    return
                self._next_tail[run_id] = now + TAIL_INTERVAL_SECONDS
                self._submit(kind, run_id, lambda: _tail(head))

    def _cancel(self, head: RunHead) -> None:
        stage = head.stage
        if not isinstance(stage, Live):
            raise UnsupportedPlan("Only a live run can be cancelled")
        match stage.handle:
            case LocalHandle() as handle:
                if self._local is None:
                    raise UnsupportedPlan("The Local backend is not registered")
                self._local.cancel(handle)
                self._log.append(
                    head.ref.run_id, [CancelDelivered()], expect=Expect(revision=head.revision)
                )
            case SlurmHandle() as handle:
                backend = self._backend()
                # A queued job has no step to interrupt and nothing to save.
                interrupt = stage.phase is Phase.RUNNING
                self._submit(
                    WorkKind.CANCEL,
                    head.ref.run_id,
                    lambda: _scancel(backend, head, handle, interrupt=interrupt),
                )

    def _escalate(self, head: RunHead) -> bool:
        """End a Slurm job outright each time a cancel grace passes."""
        stage = head.stage
        if not (
            isinstance(stage, Live)
            and stage.cancel_delivered
            and isinstance(stage.handle, SlurmHandle)
        ):
            return False
        handle = stage.handle
        watch = self._watch(head)
        now = self._clock()
        if watch.kill_at is None:
            watch.kill_at = now + CANCEL_GRACE_SECONDS
        if now < watch.kill_at:
            return False
        backend = self._backend()
        watch.kill_at = now + CANCEL_GRACE_SECONDS
        self._submit(WorkKind.CANCEL, head.ref.run_id, lambda: _kill(backend, head, handle))
        return True

    def _watch(self, head: RunHead) -> _Watch:
        watch = self._watches.get(head.ref.run_id)
        if watch is None:
            created_at = datetime.fromisoformat(self._record(head).created_at)
            watch = self._watches[head.ref.run_id] = _Watch(head.ref.plan_id, created_at)
        return watch

    def _submit(
        self, kind: WorkKind, run_id: RunId, work: Callable[[], Done | None]
    ) -> None:
        handle = self._pool.control_lane.submit(lambda token: work(), deadline=False)
        self._inflight[run_id] = (kind, handle)

    def _harvest(self) -> None:
        for run_id, (kind, handle) in list(self._inflight.items()):
            if not handle.done():
                continue
            del self._inflight[run_id]
            done = None
            try:
                done = handle.result()
                if done is None:
                    continue
                head = self._log.append(run_id, done.events, expect=done.expect)
            except Stale:
                head = None
            except Exception as error:
                if isinstance(error, SubmissionUnknown):
                    self._unknown_submissions[run_id] = str(error)
                current = self._log.head(run_id)
                if current is not None:
                    self._error(current, error)
                continue
            if head is not None and isinstance(head.stage, Ended):
                self._forget(run_id)
                if done.release is not None and self._local is not None:
                    self._local.release(done.release)
            elif done.observation is not None and run_id in self._watches:
                watch = self._watches[run_id]
                watch.pending = watch.pending or done.observation

    def record_local_exits(self) -> None:
        """Settle every Live Local run whose process has exited, without the pool.

        Call once ticks have stopped and LocalExecution.close() has stopped and
        reaped its processes, before the run log closes. A restart could only
        call these runs lost.
        """
        self._guard()
        self._harvest()
        if self._local is None:
            return
        heads: list[RunHead] = []
        while page := self._log.heads(active=True, limit=PAGE_SIZE, offset=len(heads)):
            heads += page
        for head in heads:
            try:
                while (cause := self._poll(head)) is not None:
                    done = _settle(head, cause)
                    head = self._log.append(head.ref.run_id, done.events, expect=done.expect)
                    if done.release is not None:
                        self._local.release(done.release)
            except Exception as error:
                self._error(head, error)

    def _forget(self, run_id: RunId) -> None:
        self._next_tail.pop(run_id, None)
        self._retry_at.pop(run_id, None)
        self._watches.pop(run_id, None)
        self._unknown_submissions.pop(run_id, None)

    def _observe(self, heads: list[RunHead]) -> None:
        """Start one batched scheduler query for the Slurm runs that are due."""
        if self._slurm is None or self._observing is not None:
            return
        now = self._clock()
        batch: list[tuple[RunId, SlurmHandle]] = []
        since: datetime | None = None
        for head in heads:
            stage = head.stage
            if not isinstance(stage, Live) or not isinstance(stage.handle, SlurmHandle):
                continue
            watch = self._watch(head)
            if watch.due <= now and len(batch) < OBSERVE_BATCH:
                batch.append((head.ref.run_id, stage.handle))
                since = watch.created_at if since is None else min(since, watch.created_at)
        if since is None:
            return
        backend, start = self._slurm, since
        try:
            # Each scheduler command has its own timeout; a batch may take longer.
            work = self._pool.control_lane.submit(
                lambda token: backend.observe_many(batch, since=start), deadline=False
            )
        except BudgetExhausted:
            self.observation_refusals += 1
            return
        self._observing = (tuple(run_id for run_id, _ in batch), work)

    def _harvest_observations(self) -> None:
        if self._observing is None or not self._observing[1].done():
            return
        run_ids, work = self._observing
        self._observing = None
        try:
            results = work.result()
        except Exception as error:
            failed = ObservationFailed(f"{type(error).__name__}: {error}"[:512])
            results = dict.fromkeys(run_ids, failed)
        now = self._clock()
        initial, limit = OBSERVE_BACKOFF_SECONDS
        for run_id in run_ids:
            watch = self._watches.get(run_id)
            if watch is None:
                continue
            observation = results.get(run_id, ObservationFailed("No observation returned"))
            watch.pending = observation
            if isinstance(observation, ObservationFailed):
                watch.failures = min(watch.failures + 1, 16)
                watch.error = observation.detail
                watch.due = now + min(limit, initial * 2 ** (watch.failures - 1))
            else:
                watch.failures, watch.error = 0, None
                watch.due = now + OBSERVE_INTERVAL_SECONDS

    def _error(self, head: RunHead, error: BaseException) -> None:
        if head.ref.plan_id not in self.errors and len(self.errors) >= MAX_ERRORS:
            self.errors.pop(next(iter(self.errors)))
        self.errors[head.ref.plan_id] = type(error).__name__
        self._retry_at[head.ref.run_id] = self._clock() + RETRY_SECONDS
