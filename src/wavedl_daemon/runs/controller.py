# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Durable, single-threaded control commits with bounded external work.

Call ticks on the thread that opened the stores. A tick never waits for backend
I/O. A lost submission reply is searched, never automatically submitted again.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from wavedl_daemon.execution.budget import BudgetExhausted, WorkBudget
from wavedl_daemon.execution.pool import BoundedThreadPool
from wavedl_daemon.storage.ownership import DispatchRequired, require_dispatch

from .control.context import ControlContext
from .control.derivation import derive_state
from .control.dispatch import SubmissionDispatch
from .control.fleet import FleetObserver
from .control.handshake import RuntimeHandshake
from .control.observations import ObservationRecorder
from .control.restarts import RestartWatcher


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from wavedl.core.protocol import Envelope
    from wavedl_daemon.backends.contracts import Backend
    from wavedl_daemon.runs.reducer import DerivedState, PublicationEvidence
    from wavedl_daemon.storage.executions import ExecutionRecord, ExecutionStore
    from wavedl_daemon.storage.ownership import DispatchLock
    from wavedl_daemon.storage.plans import PlanStore
    from wavedl_daemon.storage.runtime_provenance import RuntimeProvenanceStore


class SubmissionController:
    """Drive one bounded page per tick; no SQLite connection enters a worker."""

    def __init__(
        self,
        plan_store: PlanStore,
        execution_store: ExecutionStore,
        backends: Mapping[str, Backend],
        dispatch_lock: DispatchLock,
        work_budget: WorkBudget | None = None,
        *,
        provenance_store: RuntimeProvenanceStore | None = None,
        collection_store=None,
        clock: Callable[[], float] = time.monotonic,
        handshake_timeout_seconds: float = 30,
        max_runtime_active: int = 1000,
        pool: BoundedThreadPool | None = None,
    ) -> None:
        self.plan_store = plan_store
        self.execution_store = execution_store
        self.backends = dict(backends)
        self.dispatch_lock = dispatch_lock
        self.budget = work_budget or WorkBudget()
        if type(max_runtime_active) is not int or max_runtime_active < 1:
            raise ValueError("Runtime tracker capacity must be positive")
        self.provenance_store = provenance_store
        self.collection_store = collection_store
        self._owns_pool = pool is None
        self._pool = pool if pool is not None else BoundedThreadPool(self.budget)
        self.dispatch_errors = {}
        self._offset = 0
        self._thread = threading.get_ident()
        self._closed = False
        self._draining = False
        self.context = ControlContext(
            self.plan_store,
            self.execution_store,
            self.provenance_store,
            self.collection_store,
            self.backends,
            self.budget,
            self._pool.control_lane,
            clock,
            self._guard,
            self._guard_external,
            self.dispatch_errors,
            max_runtime_active,
        )
        self.handshake = RuntimeHandshake(
            self.context,
            handshake_timeout_seconds=handshake_timeout_seconds,
            capacity=max_runtime_active,
        )
        self.observations = ObservationRecorder(self.context, self.handshake)
        self.fleet = FleetObserver(self.context, self.observations, self.handshake)
        self.dispatch = SubmissionDispatch(
            self.context, self.observations, self.handshake, self.fleet
        )
        self.restarts = RestartWatcher(
            self.context, self.handshake, self.observations, self.fleet, self.dispatch
        )

    @property
    def restart_errors(self):
        return self.restarts.restart_errors

    @property
    def restart_watch_refusals(self):
        return self.restarts.restart_watch_refusals

    @property
    def capability_errors(self):
        return self.fleet.capability_errors

    def accept_runtime_events(self, plan_id: str, events: Iterable[Envelope]):
        self._guard()
        return self.handshake.accept_events(plan_id, events)

    def reject_runtime_events(self, plan_id: str, reason: str):
        self._guard()
        return self.handshake.reject(plan_id, reason)

    def _guard(self) -> None:
        if threading.get_ident() != self._thread:
            raise RuntimeError(
                "Controller/store commits must remain on their owner thread"
            )
        if self._closed and not self._draining:
            raise RuntimeError("Controller is closed")
        require_dispatch(self.dispatch_lock)
        stores = [self.plan_store, self.execution_store]
        if self.context.provenance_store is not None:
            stores.append(self.context.provenance_store)
        if self.collection_store is not None:
            stores.append(self.collection_store)
        for store in stores:
            if store.read_only:
                raise PermissionError("Read-only recovery cannot dispatch")
            if (
                store.path != self.plan_store.path
                or (
                    self.dispatch_lock.path.resolve()
                    != store.path.parent / "dispatch.lock"
                )
                or store.dispatch_lock is not self.dispatch_lock
            ):
                raise DispatchRequired(
                    "Controller requires matching store dispatch ownership"
                )

    def request_submit(self, plan_id: str) -> ExecutionRecord:
        self._guard()
        plan = self.plan_store.request_submission(plan_id)
        record = self.dispatch.ensure(plan)
        self.dispatch.schedule(plan, record)
        return self.execution_store.get(plan_id)

    def request_cancel(self, plan_id: str) -> ExecutionRecord:
        self._guard()
        plan = self.plan_store.request_cancellation(plan_id)
        record = self.dispatch.ensure(plan)
        self.dispatch.schedule(plan, record)
        return self.execution_store.get(plan_id)

    def _guard_external(self) -> None:
        if self._closed:
            raise RuntimeError("Controller closed before external dispatch")
        require_dispatch(self.dispatch_lock)
        if (
            self.dispatch_lock.path.resolve()
            != self.plan_store.path.parent / "dispatch.lock"
        ):
            raise DispatchRequired(
                "External operation has mismatched dispatch ownership"
            )

    def reconcile_once(self) -> list[ExecutionRecord]:
        self._guard()
        self.handshake.harvest()
        self.dispatch.harvest()
        self.restarts.harvest()
        self.fleet.harvest()
        self.fleet.probe_capabilities()
        limit = min(1000, self.budget.max_workers + self.budget.max_pending)
        plans = self.plan_store.list_pending(limit=limit, offset=self._offset)
        self._offset = self._offset + len(plans) if len(plans) == limit else 0
        records = []
        for plan in plans:
            record = self.dispatch.ensure(plan)
            try:
                self.restarts.schedule(plan, record)
                self.handshake.probe(record)
                self.dispatch.schedule(plan, record)
            except BudgetExhausted:
                if (
                    plan.plan_id not in self.dispatch_errors
                    and len(self.dispatch_errors) >= self.context.runtime_capacity
                ):
                    self.dispatch_errors.pop(next(iter(self.dispatch_errors)))
                self.dispatch_errors[plan.plan_id] = "BudgetExhausted"
            else:
                self.dispatch_errors.pop(plan.plan_id, None)
            records.append(self.execution_store.get(plan.plan_id))
        self.fleet.publish(records)
        return records

    def startup(self) -> list[ExecutionRecord]:
        return self.reconcile_once()

    def close(self) -> None:
        """Do not wait for uncooperative I/O or discard durable intent."""
        self._closed = True
        self.context.lane.shutdown(wait=False)
        if self._owns_pool:
            self._pool.shutdown(wait=False)

    def drain(self) -> None:
        """Finish admitted backend work before releasing dispatch/store ownership."""
        self.context.lane.shutdown(wait=True)
        if self._owns_pool:
            self._pool.shutdown(wait=True)
        self._draining = True
        try:
            self.dispatch.harvest()
        finally:
            self._draining = False

    def derive(
        self,
        plan_id: str,
        events: Iterable[Envelope] = (),
        *,
        publication: PublicationEvidence | None = None,
    ) -> DerivedState:
        return derive_state(
            self.context,
            self.handshake,
            self.dispatch,
            self.restarts,
            plan_id,
            events,
            publication=publication,
        )
