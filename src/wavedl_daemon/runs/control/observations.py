# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Persist observations on the owner thread without erasing retained evidence."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from wavedl_daemon.backends.contracts import ObservationStatus, SubmissionObservation
from wavedl_daemon.storage.executions import StaleExecution

from .records import native_identity, observation_error


if TYPE_CHECKING:
    from .context import ControlContext
    from .handshake import RuntimeHandshake


class ObservationRecorder:
    def __init__(self, context: ControlContext, handshake: RuntimeHandshake):
        self.context = context
        self.handshake = handshake

    def record(self, plan_id, observation):
        """Contradictory receipts remain uncertainty, not infinite retry loops."""
        self.context.guard()
        try:
            current = self.context.execution_store.get(plan_id)
            if (
                isinstance(observation, SubmissionObservation)
                and current.attempt is not None
            ):
                if observation != current.latest_native_observation:
                    self.context.execution_store.observe_native_generation(
                        plan_id, observation, expected_revision=current.revision
                    )
                updated = self.context.execution_store.get(plan_id)
                if (
                    observation.native_identity is not None
                    and observation.native_identity.restart_count
                    == json.loads(updated.handle.raw_json).get("restart_count")
                ):
                    bound = observation.bind_attempt(updated.attempt)
                    if bound != updated.latest_observation:
                        self.context.execution_store.record_observation(plan_id, bound)
            else:
                latest = (
                    current.latest_submission_observation
                    if isinstance(observation, SubmissionObservation)
                    else current.latest_observation
                )
                # Fleet terminal snapshots are re-delivered until their owner
                # persists them. Already durable evidence is not a new source
                # revision and must not churn revisions on every cached tick.
                if (
                    not isinstance(observation, SubmissionObservation)
                    or observation != latest
                ):
                    self.context.execution_store.record_observation(
                        plan_id, observation
                    )
        except StaleExecution:
            current = self.context.execution_store.get(plan_id)
            attempt = (
                current.attempt
                if isinstance(observation, SubmissionObservation)
                else observation.attempt
            )
            if current is None or current.attempt != attempt:
                raise
            expected = native_identity(current)
            identity = getattr(observation, "native_identity", None)
            if (
                expected is not None
                and identity is not None
                and identity.restart_count < expected.restart_count
            ):
                # Delayed native generation evidence is neither failure nor
                # uncertainty about the newer allocation already verified.
                return
            error = (
                SubmissionObservation(
                    ObservationStatus.ERROR,
                    detail="Contradictory native observation; retained prior evidence",
                )
                if isinstance(observation, SubmissionObservation)
                else observation_error(
                    current,
                    "Contradictory backend observation; retained prior evidence",
                )
            )
            if isinstance(error, SubmissionObservation) and current.attempt is not None:
                self.context.execution_store.observe_native_generation(
                    plan_id, error, expected_revision=current.revision
                )
                return
            self.context.execution_store.record_observation(
                plan_id,
                error,
            )

    def record_error(self, record, detail):
        self.record(record.plan_id, observation_error(record, detail))
