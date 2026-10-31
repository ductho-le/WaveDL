# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Bounded native fleet snapshots remain fenced to their admitted generation."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from wavedl_daemon.backends.contracts import ObservationStatus
from wavedl_daemon.execution.budget import BudgetExhausted
from wavedl_daemon.runs.reconciliation import SchedulerReconciler

from .records import (
    native_identity,
    observation_error,
    same_generation,
    watch_submission,
)


if TYPE_CHECKING:
    from .context import ControlContext
    from .handshake import RuntimeHandshake
    from .observations import ObservationRecorder


class FleetObserver:
    def __init__(
        self,
        context: ControlContext,
        observations: ObservationRecorder,
        handshake: RuntimeHandshake,
    ):
        self.context = context
        self.observations = observations
        self.handshake = handshake
        self._fleets = {
            name: SchedulerReconciler(backend)
            for name, backend in context.backends.items()
            if hasattr(backend, "observe_many")
        }
        self._fleet_work = {}
        self._fleet_active = {name: {} for name in self._fleets}
        self._cancel_observe = set()
        self._capability_work = {}
        self._capability_due = {}
        self.capability_errors = {}

    def harvest(self):
        self.context.guard()
        for name, (work, identities) in tuple(self._fleet_work.items()):
            if not work.done():
                continue
            try:
                updates = work.result()
            except Exception as error:
                for source in identities.values():
                    plan_id = source.plan_id
                    record = self.context.execution_store.get(plan_id)
                    if not same_generation(record, source):
                        continue
                    self.observations.record(
                        plan_id, observation_error(record, str(error)[:8192])
                    )
                    self.handshake.observe(
                        plan_id,
                        self.context.execution_store.get(plan_id).latest_observation,
                    )
            else:
                for update in updates:
                    if update.observation is not None:
                        source = identities[update.handle.request_id]
                        if not same_generation(
                            self.context.execution_store.get(source.plan_id), source
                        ):
                            continue
                        self.observations.record(source.plan_id, update.observation)
                        self.handshake.observe(
                            source.plan_id,
                            self.context.execution_store.get(
                                source.plan_id
                            ).latest_observation,
                        )
                        self._cancel_observe.discard(source.plan_id)
                        if update.observation.status == ObservationStatus.TERMINAL:
                            if not watch_submission(
                                self.context,
                                self.context.execution_store.get(source.plan_id),
                            ):
                                self._fleet_active[name].pop(
                                    update.handle.request_id, None
                                )
            del self._fleet_work[name]

    def probe_capabilities(self):
        self.context.guard()
        for name, fleet in self._fleets.items():
            # Scheduler probes use the same bounded I/O pool. Never take the
            # reconciler's mutex while its scheduler tick is still running.
            if name in self._fleet_work:
                continue
            pending = self._capability_work.get(name)
            if pending is not None and pending.done():
                try:
                    fleet.set_capabilities(pending.result())
                    self.capability_errors.pop(name, None)
                except Exception as error:
                    self.capability_errors[name] = str(error)[:8192]
                del self._capability_work[name]
                self._capability_due[name] = time.monotonic() + 300
            backend = self.context.backends[name]
            if (
                hasattr(backend, "capabilities")
                and name not in self._capability_work
                and time.monotonic() >= self._capability_due.get(name, 0)
            ):

                def probe(token, backend=backend):
                    token.check()
                    self.context.guard_external()
                    return backend.capabilities()

                try:
                    self._capability_work[name] = self.context.lane.submit(probe)
                except BudgetExhausted:
                    pass

    def publish(self, records):
        self.context.guard()
        for name, fleet in self._fleets.items():
            if name in self._fleet_work:
                continue
            known = self._fleet_active.setdefault(name, {})
            for record in records:
                native_handle = (
                    record.submission_handle
                    if watch_submission(self.context, record)
                    else record.handle or record.submission_handle
                )
                if record.backend != name or native_handle is None:
                    continue
                observation = (
                    record.observation
                    if record.attempt is not None
                    else record.submission_observation
                )
                if observation and observation.status == ObservationStatus.TERMINAL:
                    if not watch_submission(self.context, record):
                        known.pop(record.request_id, None)
                        continue
                if (
                    record.request_id not in known
                    and len(known) >= fleet.policy.max_active
                ):
                    victim = next(
                        (
                            key
                            for key, old in known.items()
                            if old.attempt is None
                            and old.submission_observation is not None
                            and old.submission_observation.status
                            == ObservationStatus.TERMINAL
                        ),
                        None,
                    )
                    if victim is not None:
                        known.pop(victim)
                        known[record.request_id] = record
                        continue
                    if (
                        len(self.context.dispatch_errors)
                        >= self.context.runtime_capacity
                        and record.plan_id not in self.context.dispatch_errors
                    ):
                        self.context.dispatch_errors.pop(
                            next(iter(self.context.dispatch_errors))
                        )
                    self.context.dispatch_errors[record.plan_id] = "BudgetExhausted"
                    continue
                known[record.request_id] = record
            active = tuple(known.values())
            if not active:
                fleet.set_handles(())
                continue
            fleet.set_handles(
                tuple(
                    record.submission_handle
                    if watch_submission(self.context, record)
                    else record.handle or record.submission_handle
                    for record in active
                ),
                native_generations={
                    record.submission_handle: native_identity(record)
                    for record in active
                    if (
                        record.attempt is None or watch_submission(self.context, record)
                    )
                    and native_identity(record) is not None
                },
                terminal_watch=tuple(
                    record.submission_handle
                    for record in active
                    if watch_submission(self.context, record)
                ),
            )
            try:
                work = fleet.submit_tick(self.context.lane)
            except BudgetExhausted:
                continue
            self._fleet_work[name] = (
                work,
                {record.request_id: record for record in active},
            )

    def request_cancel_observation(self, plan_id):
        self._cancel_observe.add(plan_id)
