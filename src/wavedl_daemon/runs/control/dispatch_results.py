# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Harvest actual backend receipts on the owner thread, fenced by generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wavedl_daemon.backends.contracts import (
    ObservationStatus,
    SubmissionHandle,
    UnsupportedPlan,
)
from wavedl_daemon.backends.slurm_submission import SubmissionRejected
from wavedl_daemon.storage.executions import StaleExecution

from .records import native_identity, observation_error, same_generation


if TYPE_CHECKING:
    from .dispatch import SubmissionDispatch


def harvest_results(self: SubmissionDispatch) -> None:
    for plan_id, (operation, work) in tuple(self._inflight.items()):
        if not work.done():
            continue
        record = self.context.execution_store.get(plan_id)
        source = self._inflight_sources[plan_id]
        receipt = self._receipts[plan_id]
        same_source = same_generation(record, source)
        if operation == "collect" and record is not None:
            # A completed positive runtime receipt remains historical proof
            # when the native job requeues during the collection call.
            same_source = (
                record.attempt,
                record.handle,
                record.submission_handle,
            ) == (source.attempt, source.handle, source.submission_handle)
        if not same_source:
            try:
                work.result()
            except Exception:
                pass
            del self._inflight[plan_id]
            del self._inflight_sources[plan_id]
            del self._receipts[plan_id]
            continue
        try:
            if "error" in receipt:
                raise receipt["error"]
            result = receipt["result"] if "result" in receipt else work.result()
        except Exception as error:
            detail = f"{operation} failed: {error}"[:8192]
            if operation == "submit" and isinstance(
                error, (SubmissionRejected, UnsupportedPlan)
            ):
                self.context.execution_store.mark_rejected(plan_id, detail)
            elif operation in {"submit", "lookup"}:
                self.context.execution_store.mark_unknown(plan_id, detail)
            else:
                if operation == "cancel":
                    self.fleet._cancel_observe.add(plan_id)
                self.observations.record(plan_id, observation_error(record, detail))
                self.handshake.observe(
                    plan_id,
                    self.context.execution_store.get(plan_id).latest_observation,
                )
            del self._inflight[plan_id]
            del self._inflight_sources[plan_id]
            del self._receipts[plan_id]
            continue
        # Persistence failures are deliberately not caught as backend errors.
        if operation in {"submit", "lookup"}:
            if result is None:
                self.context.execution_store.mark_unknown(
                    plan_id, "No unique backend receipt; acceptance remains unknown"
                )
            else:
                if isinstance(result, SubmissionHandle):
                    self.context.execution_store.attach_submission(
                        plan_id, result, expected_revision=record.revision
                    )
                else:
                    self.context.execution_store.record_handle(plan_id, result)
        elif operation == "cancel":
            self.context.execution_store.mark_cancel_delivered(plan_id)
        elif operation == "collect":
            collected = self.context.collection_store.record_collection(
                plan_id, source.handle, result
            )
            if collected is not None:
                self._collection_due.pop(plan_id, None)
        elif operation == "collect_submission":
            collected = self.context.collection_store.record_submission_collection(
                plan_id, source.submission_handle, result
            )
            if collected is not None:
                self._collection_due.pop(plan_id, None)
        elif operation == "collect_generation":
            if result is not None:
                expected = native_identity(record)
                if (
                    expected is None
                    or result.identity.restart_count >= expected.restart_count
                ):
                    try:
                        self.context.execution_store.observe_native_generation(
                            plan_id,
                            result.observation,
                            expected_revision=record.revision,
                        )
                        current = self.context.execution_store.get(plan_id)
                        collected = (
                            self.context.collection_store.record_submission_generation(
                                plan_id,
                                source.submission_handle,
                                result,
                                expected_revision=current.revision,
                            )
                        )
                    except StaleExecution:
                        self.observations.record(
                            plan_id,
                            observation_error(
                                record,
                                "Native collection proof differs from current generation",
                            ),
                        )
                    else:
                        if collected is not None:
                            self._collection_due.pop(plan_id, None)
        elif operation == "startup":
            observation, binding, discovery_error = result
            if observation is not None:
                self.observations.record(plan_id, observation)
            current = self.context.execution_store.get(plan_id)
            if discovery_error is not None:
                self.context.execution_store.record_observation(
                    plan_id,
                    observation_error(
                        current,
                        f"Startup proof unavailable: {discovery_error}"[:8192],
                    ),
                )
            if binding is not None:
                try:
                    self.context.execution_store.activate_initial(
                        plan_id, binding, expected_revision=current.revision
                    )
                except StaleExecution:
                    self.observations.record(
                        plan_id,
                        observation_error(
                            current,
                            "Startup binding differs from current native generation",
                        ),
                    )
                else:
                    self._initial_due.pop(plan_id, None)
            elif (
                observation is not None
                and observation.status == ObservationStatus.SUPERVISION_LOST
            ):
                self.context.execution_store.mark_local_lost(
                    plan_id, observation.detail
                )
                self._initial_due.pop(plan_id, None)
            elif (
                observation is not None
                and observation.status == ObservationStatus.TERMINAL
                and record.backend == "local"
            ):
                release = getattr(
                    self.context.backends[record.backend], "release", None
                )
                if release is not None:
                    release(record.submission_handle)
            self.fleet._cancel_observe.discard(plan_id)
        else:
            self.observations.record(plan_id, result)
            self.handshake.observe(
                plan_id, self.context.execution_store.get(plan_id).latest_observation
            )
            if (
                result.status == ObservationStatus.TERMINAL
                and record.backend == "local"
            ):
                release = getattr(
                    self.context.backends[record.backend], "release", None
                )
                if release is not None:
                    release(record.handle)
            if result.status == ObservationStatus.SUPERVISION_LOST:
                self.context.execution_store.mark_local_lost(plan_id, result.detail)
            self.fleet._cancel_observe.discard(plan_id)
        del self._inflight[plan_id]
        del self._inflight_sources[plan_id]
        del self._receipts[plan_id]
