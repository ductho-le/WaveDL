# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Durable intent dispatch; lost submission replies never trigger resubmission."""

from __future__ import annotations

import getpass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from wavedl.core.layout import RunLayout, allocate_attempt
from wavedl.core.manifest import read_manifest, write_manifest
from wavedl_daemon.backends.contracts import ObservationStatus, SubmissionObservation
from wavedl_daemon.execution.budget import BudgetExhausted
from wavedl_daemon.storage.executions import ExecutionState

from .dispatch_results import harvest_results
from .records import initial_hint, native_identity, native_pending


if TYPE_CHECKING:
    from wavedl_daemon.storage.executions import ExecutionRecord
    from wavedl_daemon.storage.plans import PlanRecord

    from .context import ControlContext
    from .fleet import FleetObserver
    from .handshake import RuntimeHandshake
    from .observations import ObservationRecorder


class SubmissionDispatch:
    def __init__(
        self,
        context: ControlContext,
        observations: ObservationRecorder,
        handshake: RuntimeHandshake,
        fleet: FleetObserver,
    ):
        self.context = context
        self.observations = observations
        self.handshake = handshake
        self.fleet = fleet
        self._inflight = {}
        self._inflight_sources = {}
        self._receipts = {}
        self._initial_due = {}
        self._collection_due = {}

    def ensure(self, plan_record: PlanRecord) -> ExecutionRecord:
        self.context.guard()
        existing = self.context.execution_store.get(plan_record.plan_id)
        if existing is not None:
            return existing
        plan = plan_record.plan
        backend = self.context.backends[plan.target]
        layout = RunLayout(Path(plan.run_dir), plan.run_id)
        # Existing layouts are legitimate immutable previews, not proof of execution.
        if not layout.run_dir.exists():
            layout.run_dir.parent.mkdir(parents=True, exist_ok=True)
            layout.run_dir.mkdir(mode=0o700)
            for directory in (
                layout.attempts,
                layout.checkpoints,
                layout.run_dir / "plots",
            ):
                directory.mkdir(mode=0o700)
        if not layout.run_dir.is_dir():
            raise ValueError("Persisted run directory is not a directory")
        if layout.manifest.exists():
            if read_manifest(layout.manifest) != plan.manifest:
                raise ValueError("Published manifest differs from persisted plan")
        else:
            write_manifest(layout, plan.manifest)
        # Writer-mode plans deliberately leave the runtime generation unbound.
        # The owned backend supervisor/scheduler allocates the first positive
        # attempt only after its native submission has been accepted.
        attempt = None if plan.allocation_mode == "writer" else allocate_attempt(layout)
        try:
            owner = backend.owner
        except AttributeError:
            owner = getpass.getuser()
        return self.context.execution_store.ensure(
            plan_record.plan_id,
            attempt,
            owner=owner,
        )

    def schedule(self, plan_record: PlanRecord, record: ExecutionRecord) -> None:
        self.context.guard()
        if record.plan_id in self._inflight:
            return
        if (
            self.context.lane.outstanding
            >= self.context.budget.max_workers + self.context.budget.max_pending
        ):
            return
        backend = self.context.backends[record.backend]
        plan = plan_record.plan
        if record.state == ExecutionState.PENDING:
            if not record.submission_requested or record.cancellation_requested:
                return
            record, claimed = self.context.execution_store.claim_submission(
                record.plan_id
            )
            if not claimed:
                return
            operation = "submit"
            callback = partial(
                backend.submit,
                plan,
                request_id=record.request_id,
                attempt=record.attempt,
            )
        elif (
            record.handle is None
            and record.submission_handle is None
            and record.state
            in {
                ExecutionState.SUBMITTING,
                ExecutionState.UNKNOWN,
            }
        ):
            if record.backend == "local":
                self.context.execution_store.mark_local_lost(
                    record.plan_id,
                    "Local submission supervision was lost; never adopt a PID",
                )
                return
            operation = "lookup"
            callback = partial(
                backend.find_by_request_id,
                plan,
                request_id=record.request_id,
                attempt=record.attempt,
            )
        elif (
            record.submission_handle is not None
            and record.attempt is None
            and record.state == ExecutionState.SUBMITTED
        ):
            terminal = (
                record.submission_observation is not None
                and record.submission_observation.status == ObservationStatus.TERMINAL
            )
            collected = self.submission_collection(record)
            if terminal and record.backend == "local":
                return
            if (
                terminal
                and self.context.collection_store is not None
                and collected is not None
                and dict(plan.resources).get("requeue") is not True
            ):
                self._initial_due.pop(record.plan_id, None)
                return
            if (
                record.cancellation_requested
                and not record.cancel_delivered
                and record.plan_id not in self.fleet._cancel_observe
            ):
                operation = "cancel"
                callback = partial(backend.request_cancel, record.submission_handle)
            elif (
                terminal
                and record.backend == "slurm"
                and self.context.collection_store is not None
                and collected is None
                and self.context.clock() >= self._collection_due.get(record.plan_id, 0)
            ):
                if hasattr(backend, "collect_submission"):
                    operation = "collect_generation"
                    callback = partial(
                        backend.collect_submission, record.submission_handle
                    )
                else:
                    operation = "collect_submission"
                    callback = partial(backend.collect, record.submission_handle)
                if (
                    record.plan_id not in self._collection_due
                    and len(self._collection_due) >= self.context.runtime_capacity
                ):
                    raise BudgetExhausted("Collected outcome retry retention exhausted")
                self._collection_due[record.plan_id] = self.context.clock() + 5
            else:
                if record.backend in self.fleet._fleets:
                    if self.context.clock() < self._initial_due.get(record.plan_id, 0):
                        return
                    if (
                        record.plan_id not in self._initial_due
                        and len(self._initial_due) >= self.context.runtime_capacity
                    ):
                        victim = next(
                            (
                                key
                                for key in self._initial_due
                                if key not in self._inflight
                                and (
                                    (old := self.context.execution_store.get(key))
                                    is None
                                    or old.attempt is not None
                                    or (
                                        old.submission_observation is not None
                                        and old.submission_observation.status
                                        == ObservationStatus.TERMINAL
                                    )
                                )
                            ),
                            None,
                        )
                        if victim is None:
                            if terminal:
                                return
                            raise BudgetExhausted(
                                "Initial binding retry retention exhausted"
                            )
                        self._initial_due.pop(victim)
                    self._initial_due[record.plan_id] = self.context.clock() + 5
                operation = "startup"
                callback = partial(
                    self.discover_initial,
                    backend,
                    record.submission_handle,
                    batched=record.backend in self.fleet._fleets,
                )
        elif native_pending(record) and record.state == ExecutionState.SUBMITTED:
            native = record.native_observation
            terminal = (
                native is not None and native.status == ObservationStatus.TERMINAL
            )
            if (
                record.cancellation_requested
                and not record.cancel_delivered
                and not terminal
            ):
                operation = "cancel"
                callback = partial(backend.request_cancel, record.submission_handle)
            elif (
                self.context.collection_store is not None
                and self.submission_collection(record) is None
                and hasattr(backend, "collect_submission")
                and self.context.clock() >= self._collection_due.get(record.plan_id, 0)
            ):
                operation = "collect_generation"
                callback = partial(backend.collect_submission, record.submission_handle)
                if (
                    record.plan_id not in self._collection_due
                    and len(self._collection_due) >= self.context.runtime_capacity
                ):
                    raise BudgetExhausted("Collected outcome retry retention exhausted")
                self._collection_due[record.plan_id] = self.context.clock() + 5
            else:
                return
        elif record.handle is not None and record.state == ExecutionState.SUBMITTED:
            terminal = (
                record.observation
                and record.observation.status == ObservationStatus.TERMINAL
            )
            if terminal:
                if (
                    record.backend != "slurm"
                    or self.context.collection_store is None
                    or not hasattr(backend, "collect")
                    or self.context.collection_store.get(record.plan_id, record.attempt)
                    is not None
                    or self.context.clock()
                    < self._collection_due.get(record.plan_id, 0)
                ):
                    return
                operation = "collect"
                callback = partial(backend.collect, record.handle)
                if (
                    record.plan_id not in self._collection_due
                    and len(self._collection_due) >= self.context.runtime_capacity
                ):
                    raise BudgetExhausted("Collected outcome retry retention exhausted")
                self._collection_due[record.plan_id] = self.context.clock() + 5
            elif (
                record.cancellation_requested
                and not record.cancel_delivered
                and record.plan_id not in self.fleet._cancel_observe
            ):
                operation = "cancel"
                callback = partial(backend.request_cancel, record.handle)
            elif (
                record.observation
                and record.observation.status == ObservationStatus.TERMINAL
            ):
                return
            else:
                if record.backend in self.fleet._fleets:
                    return
                operation = "observe"
                callback = partial(backend.observe, record.handle)
        else:
            return

        receipt = {}

        def work(token):
            token.check()
            self.context.guard_external()
            try:
                result = callback()
            except Exception as error:
                receipt["error"] = error
                raise
            # Preserve actual responses even when the pool's post-call deadline
            # check fires. Timeout is not proof that external acceptance failed.
            receipt["result"] = result
            return result

        try:
            handle = self.context.lane.submit(work)
        except BudgetExhausted:
            # A claimed but unstarted submit is still conservatively unknown.
            if operation == "submit":
                self.context.execution_store.mark_unknown(
                    record.plan_id, "Submission admission lost"
                )
            return
        self._inflight[record.plan_id] = (operation, handle)
        self._inflight_sources[record.plan_id] = record
        self._receipts[record.plan_id] = receipt

    def submission_collection(self, record):
        """Resolve exact native generation; never relabel older ACK collections."""
        if self.context.collection_store is None or record.backend != "slurm":
            return None
        identity = native_identity(record)
        if identity is not None:
            collected = self.context.collection_store.get_submission_generation(
                record.plan_id, identity
            )
            if collected is not None:
                return collected
            if identity.restart_count != 0:
                return None
        return self.context.collection_store.get_submission(record.plan_id)

    @staticmethod
    def discover_initial(backend, receipt, *, batched=False):
        """Return native observation and proof; a journal is never activation proof."""
        if batched:
            if not initial_hint(receipt):
                return None, None, None
            return None, backend.discover_initial(receipt), None
        observation = backend.observe(receipt)
        if not isinstance(observation, SubmissionObservation):
            raise ValueError("Unbound backend must return a submission observation")
        if observation.status == ObservationStatus.SUPERVISION_LOST:
            return observation, None, None
        try:
            binding = backend.discover_initial(receipt)
        except (OSError, ValueError, RuntimeError) as error:
            # Native lifecycle evidence remains independent of startup proof.
            # A torn/foreign binding does not erase a verified observation.
            return observation, None, error
        return observation, binding, None

    def harvest(self):
        self.context.guard()
        harvest_results(self)
