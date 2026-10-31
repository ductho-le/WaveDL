# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Attempt-fenced negotiation; journal workers never commit runtime evidence."""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from wavedl.core.layout import RunLayout
from wavedl.core.protocol import ProtocolError, decode
from wavedl.core.safe_files import read_first_line
from wavedl_daemon.execution.budget import BudgetExhausted
from wavedl_daemon.runs.negotiation import (
    NegotiationStatus,
    RuntimeNegotiation,
    StaleNegotiation,
)

from .records import native_pending


if TYPE_CHECKING:
    from collections.abc import Iterable

    from wavedl.core.protocol import Envelope
    from wavedl_daemon.storage.executions import ExecutionRecord

    from .context import ControlContext


def _first_runtime_event(path: Path, token):
    """Read only the bounded first record; never open a FIFO or follow a link."""
    token.check()
    try:
        data = read_first_line(path, limit=65536)
    except FileNotFoundError:
        return None
    except ValueError as error:
        if "exceeds byte bound" in str(error):
            raise ProtocolError(str(error)) from error
        raise OSError("Runtime journal is not a safe regular file") from error
    token.check()
    return decode(data[:-1]) if data.endswith(b"\n") else None


class RuntimeHandshake:
    def __init__(
        self,
        context: ControlContext,
        *,
        handshake_timeout_seconds: float,
        capacity: int,
    ):
        self.context = context
        self._handshake_timeout = handshake_timeout_seconds
        self.capacity = capacity
        self._runtime = {}
        self._runtime_errors = {}
        self._runtime_work = {}
        self._runtime_skip = set()

    def tracker(self, record: ExecutionRecord) -> RuntimeNegotiation:
        if record.attempt is None:
            raise ProtocolError("Unbound submission has no accepted runtime attempt")
        tracker = self._runtime.get(record.plan_id)
        if tracker is not None and tracker.attempt == record.attempt:
            return tracker
        if tracker is None and len(self._runtime) >= self.capacity:
            # Accepted trackers are safely reconstructible from immutable SQLite
            # provenance. Waiting/failure clocks must never be silently reset.
            victim = next(
                (
                    key
                    for key, value in self._runtime.items()
                    if value.result.status == NegotiationStatus.ACCEPTED
                    and key not in self._runtime_work
                ),
                None,
            )
            if victim is None:
                raise BudgetExhausted("Runtime negotiation retention exhausted")
            del self._runtime[victim]
            self._runtime_errors.pop(victim, None)
            self._runtime_skip.discard(victim)
        tracker = RuntimeNegotiation(
            record.run_id,
            record.attempt,
            timeout_seconds=self._handshake_timeout,
        )
        if self.context.provenance_store is not None:
            persisted = self.context.provenance_store.get(
                record.plan_id, record.attempt
            )
            if persisted is not None:
                tracker.accept_event(persisted.event)
        self._runtime[record.plan_id] = tracker
        self._runtime_errors.pop(record.plan_id, None)
        return tracker

    def accept_events(self, plan_id: str, events: Iterable[Envelope]):
        """Validate actual attempt evidence and commit hello before exposing it.

        Projection and general journal replay belong to the future event ingestor.
        This acceptance path retains only negotiation evidence, not an event list.
        """
        self.context.guard()
        if self.context.provenance_store is None:
            raise RuntimeError("Runtime acceptance requires a provenance store")
        record = self.context.execution_store.get(plan_id)
        if record is None:
            raise KeyError(plan_id)
        if native_pending(record):
            raise ProtocolError(
                "Prior runtime telemetry is not current native execution"
            )
        tracker = self.tracker(record)
        for event in events:
            candidate = deepcopy(tracker)
            try:
                result = candidate.accept_event(event, now=self.context.clock())
            except StaleNegotiation:
                raise
            if (
                result.status == NegotiationStatus.ACCEPTED
                and tracker.result.status != NegotiationStatus.ACCEPTED
            ):
                try:
                    self.context.provenance_store.record(plan_id, event)
                except ProtocolError as error:
                    # The store also validates actual environment against the
                    # immutable approved request. Reject the original snapshot,
                    # not the candidate which briefly held uncommitted hello.
                    candidate = deepcopy(tracker)
                    result = candidate.reject(str(error))
            self._runtime[plan_id] = tracker = candidate
            if result.status == NegotiationStatus.FAILED:
                break
        return tracker.result

    def observe(self, plan_id: str, observation) -> None:
        if self.context.provenance_store is not None:
            record = self.context.execution_store.get(plan_id)
            if record.attempt is not None and not native_pending(record):
                self.tracker(record).observe_backend(
                    observation, now=self.context.clock()
                )

    def reject(self, plan_id: str, reason: str):
        """Latch invalid actual evidence without fabricating a backend exit."""
        self.context.guard()
        if self.context.provenance_store is None:
            raise RuntimeError("Runtime rejection requires a provenance store")
        record = self.context.execution_store.get(plan_id)
        if record is None:
            raise KeyError(plan_id)
        return self.tracker(record).reject(reason)

    def harvest(self) -> None:
        self.context.guard()
        for plan_id, (attempt, work) in tuple(self._runtime_work.items()):
            if not work.done():
                continue
            record = self.context.execution_store.get(plan_id)
            if record is None or record.attempt != attempt or native_pending(record):
                # Retrieving a finished old result must not latch errors or
                # skip clocks against the newly activated execution generation.
                try:
                    work.result()
                except Exception:
                    pass
                del self._runtime_work[plan_id]
                continue
            try:
                event = work.result()
            except ProtocolError as error:
                self._runtime_errors[plan_id] = str(error)[:8192]
            else:
                record = self.context.execution_store.get(plan_id)
                if (
                    record is not None
                    and record.attempt == attempt
                    and event is not None
                ):
                    try:
                        self.accept_events(plan_id, (event,))
                    except StaleNegotiation as error:
                        # The mismatch remains explicit protocol-failure evidence.
                        self._runtime_errors[plan_id] = str(error)
            # Storage/deadline failures intentionally propagate, never become a
            # fabricated runtime or backend exit.
            del self._runtime_work[plan_id]
            self._runtime_skip.add(plan_id)

    def probe(self, record: ExecutionRecord) -> None:
        if (
            self.context.provenance_store is None
            or record.handle is None
            or native_pending(record)
        ):
            return
        # A missing journal must not monopolize a single-worker budget and
        # starve backend observations (or cancellation dispatch).
        if record.plan_id in self._runtime_skip:
            self._runtime_skip.discard(record.plan_id)
            return
        tracker = self.tracker(record)
        if (
            tracker.result.status != NegotiationStatus.WAITING
            or record.plan_id in self._runtime_errors
            or record.plan_id in self._runtime_work
        ):
            return
        path = RunLayout(Path(record.handle.run_dir), record.run_id).journal(
            record.attempt
        )
        try:
            work = self.context.lane.submit(partial(_first_runtime_event, path))
        except BudgetExhausted:
            return
        self._runtime_work[record.plan_id] = (record.attempt, work)

    def negotiation_error(self, plan_id):
        return self._runtime_errors.get(plan_id)
