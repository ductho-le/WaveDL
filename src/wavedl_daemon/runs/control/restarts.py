# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Native-backed restart CAS precedes publication of new runtime tracker state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wavedl.core.scheduler_attempt import SchedulerIdentity
from wavedl_daemon.backends.contracts import BackendObservation, ObservationStatus
from wavedl_daemon.backends.slurm_observation import RestartBindingUnavailable
from wavedl_daemon.execution.budget import BudgetExhausted
from wavedl_daemon.runs.negotiation import RuntimeNegotiation
from wavedl_daemon.storage.executions import ExecutionState, StaleExecution

from .records import native_identity, restart_hint, same_generation


if TYPE_CHECKING:
    from .context import ControlContext
    from .dispatch import SubmissionDispatch
    from .fleet import FleetObserver
    from .handshake import RuntimeHandshake
    from .observations import ObservationRecorder


class RestartWatcher:
    def __init__(
        self,
        context: ControlContext,
        handshake: RuntimeHandshake,
        observations: ObservationRecorder,
        fleet: FleetObserver,
        dispatch: SubmissionDispatch,
    ):
        self.context = context
        self.handshake = handshake
        self.observations = observations
        self.fleet = fleet
        self.dispatch = dispatch
        self._restart_work = {}
        self._restart_due = {}
        self.restart_errors = {}
        self.restart_watch_refusals = 0

    def schedule(self, plan, record):
        """Read native restart proposals only for an approved requeue plan."""
        if (
            dict(plan.plan.resources).get("requeue") is not True
            or record.backend != "slurm"
            or record.handle is None
            or record.state != ExecutionState.SUBMITTED
            or record.plan_id in self._restart_work
            or self.context.clock() < self._restart_due.get(record.plan_id, 0)
        ):
            return
        backend = self.context.backends[record.backend]
        if not hasattr(backend, "discover_restart"):
            return
        terminal = (
            record.observation is not None
            and record.observation.status == ObservationStatus.TERMINAL
        )
        if (
            record.plan_id not in self._restart_due
            and len(self._restart_due) >= self.context.runtime_capacity
        ):
            # Active executions take precedence over optional terminal watches.
            # Historical jobs cannot permanently occupy active admission.
            victim = None
            if not terminal:
                for key in self._restart_due:
                    if key in self._restart_work:
                        continue
                    old = self.context.execution_store.get(key)
                    if old is None or (
                        old.observation is not None
                        and old.observation.status == ObservationStatus.TERMINAL
                    ):
                        victim = key
                        break
            if victim is None:
                self.restart_watch_refusals = min(
                    2**63 - 1, self.restart_watch_refusals + 1
                )
                if terminal:
                    return
                raise BudgetExhausted("Scheduler restart retention exhausted")
            self._restart_due.pop(victim)
            self.restart_errors.pop(victim, None)
            self.restart_watch_refusals = min(
                2**63 - 1, self.restart_watch_refusals + 1
            )

        def discover(token):
            token.check()
            self.context.guard_external()
            mismatch = record.latest_observation is not None and (
                "Scheduler requeue changed execution identity"
                in record.latest_observation.detail
                or "Accounting contains a newer scheduler execution"
                in record.latest_observation.detail
            )
            if not terminal and not mismatch and not restart_hint(record.handle, token):
                return None
            return backend.discover_restart(record.handle)

        try:
            work = self.context.lane.submit(discover)
        except BudgetExhausted:
            return
        self._restart_work[record.plan_id] = (record, work)
        self._restart_due[record.plan_id] = self.context.clock() + 5

    def harvest(self):
        self.context.guard()
        for plan_id, (source, work) in tuple(self._restart_work.items()):
            if not work.done():
                continue
            record = self.context.execution_store.get(plan_id)
            if not same_generation(record, source) or (
                native_identity(record) != native_identity(source)
            ):
                try:
                    work.result()
                except Exception:
                    pass
                del self._restart_work[plan_id]
                continue
            try:
                work.token.check()
                proposal = work.result()
            except Exception as error:
                self.restart_errors[plan_id] = type(error).__name__
                terminal = (
                    record.observation is not None
                    and record.observation.status == ObservationStatus.TERMINAL
                )
                native_restart = getattr(error, "native_restart_count", None)
                if (
                    isinstance(error, RestartBindingUnavailable)
                    and record.submission_handle is not None
                ):
                    receipt = record.submission_handle
                    job, cluster = receipt.job_id.split(";")
                    identity = SchedulerIdentity(
                        receipt.run_id,
                        receipt.request_id,
                        cluster,
                        job,
                        receipt.owner,
                        native_restart,
                    )
                    try:
                        self.context.execution_store.record_native_generation(
                            plan_id, identity, expected_revision=record.revision
                        )
                    except StaleExecution:
                        # A delayed native reply cannot regress the durable head.
                        pass
                if not terminal or type(native_restart) is int:
                    self.context.execution_store.record_observation(
                        plan_id,
                        BackendObservation(
                            record.attempt,
                            ObservationStatus.ERROR,
                            detail=f"Restart observation unavailable: {type(error).__name__}",
                        ),
                    )
            else:
                self.restart_errors.pop(plan_id, None)
                if proposal is not None:
                    binding, candidate = proposal
                    # Prepare before committing, but never publish new tracker
                    # state until native-backed CAS succeeds. Old workers remain
                    # admitted until their real completion and are harvest-fenced.
                    tracker = RuntimeNegotiation(
                        record.run_id,
                        candidate.attempt,
                        timeout_seconds=self.handshake._handshake_timeout,
                    )
                    if (
                        plan_id not in self.handshake._runtime
                        and len(self.handshake._runtime)
                        >= self.context.runtime_capacity
                    ):
                        self.restart_errors[plan_id] = "BudgetExhausted"
                        del self._restart_work[plan_id]
                        continue
                    fleet = self.fleet._fleets.get(record.backend)
                    known = self.fleet._fleet_active.setdefault(record.backend, {})
                    if (
                        fleet is not None
                        and record.request_id not in known
                        and len(known) >= fleet.policy.max_active
                    ):
                        self.restart_errors[plan_id] = "BudgetExhausted"
                        del self._restart_work[plan_id]
                        continue
                    activated = self.context.execution_store.activate_requeue(
                        plan_id,
                        binding,
                        candidate,
                        expected_revision=record.revision,
                    )
                    if not same_generation(activated, record):
                        self.handshake._runtime[plan_id] = tracker
                        self.handshake._runtime_errors.pop(plan_id, None)
                        self.handshake._runtime_skip.discard(plan_id)
                        self.fleet._cancel_observe.discard(plan_id)
                        self._restart_due[plan_id] = self.context.clock() + 5
                        self.dispatch._collection_due.pop(plan_id, None)
                        known[record.request_id] = activated
            del self._restart_work[plan_id]
