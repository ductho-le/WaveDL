# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Bounded fleet polling with retained evidence and explicit retention uncertainty.

``tick`` is foreground/blocking: event-loop callers must use ``submit_tick`` with
the shared bounded pool. Scheduler subprocess deadlines/output limits belong to
the injected backend runner. Updates must be persisted by the lifecycle owner
before removing terminal handles; this component never scans runtime journals.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

from wavedl.core.scheduler_attempt import SchedulerIdentity
from wavedl_daemon.backends.contracts import (
    BackendObservation,
    Handle,
    ObservationStatus,
    SubmissionHandle,
    SubmissionObservation,
)
from wavedl_daemon.backends.slurm_observation import SlurmCapabilities
from wavedl_daemon.execution.budget import BudgetExhausted


if TYPE_CHECKING:
    from collections.abc import Callable

    from wavedl_daemon.execution.pool import CancellationToken, WorkHandle


T = TypeVar("T")


class WorkSubmitter(Protocol):
    def submit(self, callback: Callable[[CancellationToken], T]) -> WorkHandle[T]: ...


class FleetBackend(Protocol):
    def observe_many(
        self, handles: tuple[Handle | SubmissionHandle, ...]
    ) -> dict[str, BackendObservation | SubmissionObservation]: ...


@dataclass(frozen=True)
class PollPolicy:
    max_active: int = 1024
    max_batch: int = 128
    max_batches_per_tick: int = 8
    interval: float = 30.0
    backoff_initial: float = 5.0
    backoff_max: float = 120.0

    def __post_init__(self) -> None:
        for name in ("max_active", "max_batch", "max_batches_per_tick"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("interval", "backoff_initial", "backoff_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.backoff_initial > self.backoff_max:
            raise ValueError("initial backoff exceeds maximum")


@dataclass(frozen=True)
class PollUpdate:
    handle: Handle | SubmissionHandle
    observation: BackendObservation | SubmissionObservation | None
    last_known: BackendObservation | SubmissionObservation | None
    next_due: float
    retention_risk: bool = False
    detail: str = ""


@dataclass
class _Entry:
    update: PollUpdate
    last_seen: float
    errors: int = 0
    native_identity: SchedulerIdentity | None = None


class SchedulerReconciler:
    def __init__(
        self,
        backend: FleetBackend,
        *,
        policy: PollPolicy = PollPolicy(),
        capabilities: SlurmCapabilities = SlurmCapabilities(),
        clock: Callable[[], float] = time.monotonic,
        jitter: Callable[[], float] = lambda: 0.0,
    ) -> None:
        self.backend = backend
        self.policy = policy
        self.capabilities = capabilities
        self.clock = clock
        self.jitter = jitter
        self._entries: dict[Handle | SubmissionHandle, _Entry] = {}
        self._terminal_watch: frozenset[SubmissionHandle] = frozenset()
        self._lock = threading.Lock()

    def set_capabilities(self, capabilities: SlurmCapabilities) -> None:
        """Install a startup/refreshed probe without doing scheduler I/O here."""
        with self._lock:
            self.capabilities = capabilities
            due = self.clock() + self._interval()
            for entry in self._entries.values():
                entry.update = self._update(
                    entry, next_due=min(entry.update.next_due, due)
                )

    def set_handles(
        self,
        handles: tuple[Handle | SubmissionHandle, ...],
        *,
        native_generations: Mapping[SubmissionHandle, SchedulerIdentity] | None = None,
        terminal_watch: tuple[SubmissionHandle, ...] = (),
    ) -> None:
        """Replace the bounded active set atomically; changed identity resets history.

        Caller retains terminal handles until returned evidence is durably saved.
        Capacity refusal leaves all previous handles/evidence untouched.
        """
        if not isinstance(handles, tuple) or not all(
            isinstance(handle, (Handle, SubmissionHandle))
            and handle.backend == "slurm"
            and handle.job_id
            for handle in handles
        ):
            raise ValueError("Expected immutable Slurm handles with job IDs")
        if len(handles) > self.policy.max_active:
            raise BudgetExhausted("scheduler active-handle capacity exhausted")
        if len(set(handles)) != len(handles):
            raise ValueError("Duplicate handle identity")
        if (
            not isinstance(terminal_watch, tuple)
            or len(terminal_watch) > self.policy.max_active
            or len(set(terminal_watch)) != len(terminal_watch)
            or any(
                not isinstance(handle, SubmissionHandle) or handle not in handles
                for handle in terminal_watch
            )
        ):
            raise ValueError("Terminal watch requires a bounded active receipt subset")
        if native_generations is not None:
            if not isinstance(native_generations, Mapping):
                raise ValueError("Native generation scope must be a mapping")
            if len(native_generations) > self.policy.max_active:
                raise BudgetExhausted("scheduler native-generation scope exhausted")
            for handle, identity in native_generations.items():
                if (
                    not isinstance(handle, SubmissionHandle)
                    or handle not in handles
                    or not isinstance(identity, SchedulerIdentity)
                ):
                    raise ValueError("Native generation scope requires active receipts")
                if not self._matches_native_identity(handle, identity):
                    raise ValueError("Native generation identity differs from receipt")
        with self._lock:
            now = self.clock()
            entries = {}
            for handle in handles:
                entry = self._entries.get(handle)
                if (
                    entry is not None
                    and isinstance(handle, SubmissionHandle)
                    and native_generations is not None
                ):
                    expected = native_generations.get(handle)
                    if entry.native_identity != expected:
                        entry = None
                entries[handle] = entry or _Entry(
                    PollUpdate(handle, None, None, now),
                    now,
                    native_identity=(
                        native_generations.get(handle)
                        if isinstance(handle, SubmissionHandle)
                        and native_generations is not None
                        else None
                    ),
                )
            self._entries = entries
            self._terminal_watch = frozenset(terminal_watch)

    @property
    def snapshots(self) -> tuple[PollUpdate, ...]:
        with self._lock:
            return tuple(entry.update for entry in self._entries.values())

    def submit_tick(self, pool: WorkSubmitter) -> WorkHandle[tuple[PollUpdate, ...]]:
        """Admit polling through the shared bounded executor, never the event loop."""
        return pool.submit(lambda token: self.tick(check=token.check))

    @staticmethod
    def _update(entry: _Entry, **changes: object) -> PollUpdate:
        old = entry.update
        return PollUpdate(
            **{
                "handle": old.handle,
                "observation": old.observation,
                "last_known": old.last_known,
                "next_due": old.next_due,
                "retention_risk": old.retention_risk,
                "detail": old.detail,
                **changes,
            }
        )

    def _age(self) -> int | None:
        age = self.capabilities.min_job_age
        return (
            age
            if self.capabilities.accounting is not True and age is not None
            else None
        )

    @staticmethod
    def _error(
        handle: Handle | SubmissionHandle, detail: str
    ) -> BackendObservation | SubmissionObservation:
        if isinstance(handle, SubmissionHandle):
            return SubmissionObservation(ObservationStatus.ERROR, detail=detail)
        return BackendObservation(
            handle.attempt, ObservationStatus.ERROR, detail=detail
        )

    @staticmethod
    def _matches_native_identity(
        handle: SubmissionHandle, identity: SchedulerIdentity
    ) -> bool:
        raw = json.loads(handle.raw_json)
        job, _, job_cluster = handle.job_id.partition(";")
        return (
            isinstance(identity, SchedulerIdentity)
            and identity.run_id == handle.run_id
            and identity.request_id == handle.request_id
            and identity.owner == handle.owner
            and identity.job_id == job
            and identity.cluster == raw.get("cluster", job_cluster)
        )

    @staticmethod
    def _matches(
        handle: Handle | SubmissionHandle,
        observation: BackendObservation | SubmissionObservation | None,
    ) -> bool:
        """Keep native receipt evidence separate from proved attempt evidence."""
        if isinstance(handle, SubmissionHandle):
            return isinstance(observation, SubmissionObservation)
        return (
            isinstance(observation, BackendObservation)
            and observation.attempt == handle.attempt
        )

    def _interval(self) -> float:
        age = self._age()
        return (
            min(self.policy.interval, age / 2)
            if age and age > 0
            else self.policy.interval
        )

    def tick(
        self, *, check: Callable[[], None] = lambda: None
    ) -> tuple[PollUpdate, ...]:
        """Poll due entries in bounded batches; query errors are not execution failure.

        Due-first ordering prevents later pages starving behind early failures.
        Backoff is capped below known retention, but deadlines, outages and caller
        scheduling can still miss it; such risks remain explicit in each update.
        Concurrent ticks are refused, rather than queued without a bound.
        """
        if not self._lock.acquire(blocking=False):
            raise BudgetExhausted("scheduler reconciliation already running")
        try:
            return self._tick(check)
        finally:
            self._lock.release()

    def _tick(self, check: Callable[[], None]) -> tuple[PollUpdate, ...]:
        now = self.clock()
        due = sorted(
            (
                entry
                for entry in self._entries.values()
                if entry.update.next_due <= now
                and not (
                    entry.update.observation
                    and entry.update.observation.status == ObservationStatus.TERMINAL
                    and entry.update.handle not in self._terminal_watch
                )
            ),
            key=lambda entry: entry.update.next_due,
        )
        groups: dict[tuple[str, str], list[list[_Entry]]] = {}
        for entry in due:
            handle = entry.update.handle
            raw = json.loads(handle.raw_json)
            cluster = raw.get("cluster", (handle.job_id or "").partition(";")[2])
            lanes = groups.setdefault((handle.owner, str(cluster)), [])
            for lane in lanes:
                if len(lane) < self.policy.max_batch and all(
                    other.update.handle.job_id != handle.job_id for other in lane
                ):
                    lane.append(entry)
                    break
            else:
                lanes.append([entry])
        batches = sorted(
            (lane for lanes in groups.values() for lane in lanes),
            key=lambda lane: lane[0].update.next_due,
        )
        admitted = batches[: self.policy.max_batches_per_tick]
        polling = {id(entry) for batch in admitted for entry in batch}
        # Re-deliver cached terminal evidence until the owner removes the handle
        # after durable persistence; a failed store write must be retryable even
        # after the scheduler purges its terminal record.
        updates = [
            entry.update
            for entry in self._entries.values()
            if entry.update.observation
            and entry.update.observation.status == ObservationStatus.TERMINAL
            and id(entry) not in polling
        ]
        for batch in admitted:
            check()
            handles = tuple(entry.update.handle for entry in batch)
            try:
                observations = self.backend.observe_many(handles)
            except (OSError, ValueError, RuntimeError) as error:
                observations = {
                    handle.job_id: self._error(handle, str(error)[:512])
                    for handle in handles
                }
            finished = self.clock()
            for entry in batch:
                handle = entry.update.handle
                observation = observations.get(handle.job_id)
                if not self._matches(handle, observation):
                    observation = self._error(
                        handle,
                        "Batch lacks matching submission evidence"
                        if isinstance(handle, SubmissionHandle)
                        else "Batch lacks matching attempt evidence",
                    )
                if (
                    isinstance(handle, SubmissionHandle)
                    and isinstance(observation, SubmissionObservation)
                    and observation.native_identity is not None
                ):
                    identity = observation.native_identity
                    if not self._matches_native_identity(handle, identity) or (
                        entry.native_identity is not None
                        and identity.restart_count < entry.native_identity.restart_count
                    ):
                        observation = self._error(
                            handle, "Batch lacks matching native generation evidence"
                        )
                    else:
                        # Scope is a high-water mark, independent of last good
                        # state: typed failures can prove a newer generation.
                        entry.native_identity = identity
                good = observation.status not in {
                    ObservationStatus.ERROR,
                    ObservationStatus.ABSENT,
                    ObservationStatus.SUPERVISION_LOST,
                }
                age = self._age()
                risk = not good and (
                    entry.update.retention_risk
                    or (age is None and self.capabilities.accounting is not True)
                    or (
                        age is not None
                        and (age <= 0 or finished - entry.last_seen >= age)
                    )
                )
                if observation.status == ObservationStatus.ERROR:
                    entry.errors = min(entry.errors + 1, 64)
                    delay = min(
                        self.policy.backoff_max,
                        self.policy.backoff_initial * 2 ** (entry.errors - 1),
                    )
                    jitter = self.jitter()
                    if not math.isfinite(jitter) or not 0 <= jitter <= 1:
                        raise ValueError("jitter must be finite in [0, 1]")
                    delay = min(self.policy.backoff_max, delay * (1 + jitter))
                    if age is not None and age > 0:
                        delay = min(delay, age / 2)
                else:
                    entry.errors = 0
                    delay = self._interval()
                if not good and age is not None:
                    risk = risk or finished + delay - entry.last_seen >= age
                if good:
                    entry.last_seen = finished
                entry.update = PollUpdate(
                    handle,
                    observation,
                    observation if good else entry.update.last_known,
                    finished + delay,
                    risk,
                    "Retention window cannot be guaranteed; terminal outcome unknown"
                    if risk
                    else "",
                )
                updates.append(entry.update)
        return tuple(updates)
