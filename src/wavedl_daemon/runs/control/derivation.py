# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Read-only state derivation; wrapper success never fabricates a runtime exit."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING

from wavedl_daemon.backends.contracts import (
    BackendObservation,
    ObservationStatus,
    OutcomeStatus,
    SubmissionObservation,
)
from wavedl_daemon.runs.negotiation import NegotiationStatus, RuntimeNegotiation
from wavedl_daemon.runs.reducer import RunIntent, RunState, SubmissionState, reduce
from wavedl_daemon.storage.executions import ExecutionState, native_runtime_pending


if TYPE_CHECKING:
    from collections.abc import Iterable

    from wavedl.core.protocol import Envelope
    from wavedl_daemon.runs.reducer import DerivedState, PublicationEvidence

    from .context import ControlContext
    from .dispatch import SubmissionDispatch
    from .handshake import RuntimeHandshake
    from .restarts import RestartWatcher


def derive_state(
    context: ControlContext,
    handshake: RuntimeHandshake,
    dispatch: SubmissionDispatch,
    restarts: RestartWatcher,
    plan_id: str,
    events: Iterable[Envelope] = (),
    *,
    publication: PublicationEvidence | None = None,
) -> DerivedState:
    """Read-only binding: wrapper success never fabricates a runtime exit."""
    plan = context.plan_store.get(plan_id)
    record = context.execution_store.get(plan_id)
    if plan is None or record is None:
        raise KeyError(plan_id)
    state = SubmissionState.__members__.get(record.state.name, SubmissionState.PENDING)
    native_pending = native_runtime_pending(record)
    native_lifecycle = native_pending or (
        record.attempt is None
        and record.native_identity is not None
        and record.latest_native_observation is not None
    )
    last_observation = (
        record.native_observation
        if native_lifecycle
        else (
            record.observation
            if record.attempt is not None
            else record.submission_observation
        )
    )
    observation = (
        record.latest_native_observation
        if native_lifecycle
        else (
            record.latest_observation
            if record.attempt is not None
            else record.latest_submission_observation
        )
    )
    if record.state == ExecutionState.LOCAL_LOST:
        observation = (
            BackendObservation(
                record.attempt,
                ObservationStatus.SUPERVISION_LOST,
                detail=record.detail,
            )
            if record.attempt is not None
            else SubmissionObservation(
                ObservationStatus.SUPERVISION_LOST, detail=record.detail
            )
        )
    tracker = handshake._runtime.get(plan_id)
    provenance = None
    if (
        context.provenance_store is not None
        and record.attempt is not None
        and not native_pending
    ):
        provenance = context.provenance_store.get(plan_id, record.attempt)
    validation_error = []
    validator = None
    if (
        context.provenance_store is not None
        and record.attempt is not None
        and not native_pending
    ):
        validator = (
            deepcopy(tracker)
            if tracker is not None and tracker.attempt == record.attempt
            else RuntimeNegotiation(
                record.run_id,
                record.attempt,
                timeout_seconds=handshake._handshake_timeout,
            )
        )
        if provenance is not None:
            validator.accept_event(provenance.event)

    def runtime_events():
        if record.attempt is None or native_pending:
            return
        if provenance is not None:
            yield provenance.event
        for event in events:
            if validator is not None:
                checked = validator.accept_event(event)
                if checked.status == NegotiationStatus.FAILED:
                    validation_error.append(checked.reason)
                    break
                if provenance is None:
                    validation_error.append(
                        "Runtime hello must be durably accepted before derivation"
                    )
                    break
            yield event

    collected = None
    requires_collection = (
        context.collection_store is not None
        and record.backend == "slurm"
        and record.attempt is not None
        and not native_pending
    )
    if requires_collection:
        collected = context.collection_store.get(plan_id, record.attempt)
    elif context.collection_store is not None and record.backend == "slurm":
        collected = dispatch.submission_collection(record)
    if collected is not None:
        collection_identity = None
        if not requires_collection:
            collection_identity = getattr(collected, "identity", None)
            if collection_identity is None:
                from wavedl_daemon.storage.submission_generations import (
                    receipt_identity,
                )

                collection_identity = receipt_identity(collected.handle, plan.plan, 0)

        def matches_collection(observed):
            return requires_collection or (
                isinstance(observed, SubmissionObservation)
                and observed.native_identity == collection_identity
            )

        if (
            last_observation is not None
            and last_observation.status == ObservationStatus.TERMINAL
            and matches_collection(last_observation)
        ):
            last_observation = replace(
                last_observation,
                batch=collected.result.batch,
                stage_out=collected.result.stage_out,
            )
        if (
            observation is not None
            and observation.status == ObservationStatus.TERMINAL
            and matches_collection(observation)
        ):
            observation = replace(
                observation,
                batch=collected.result.batch,
                stage_out=collected.result.stage_out,
            )
    derived = reduce(
        RunIntent(
            record.run_id,
            plan.plan.operation,
            None if native_pending else record.attempt,
            record.backend,
            state,
            record.submission_handle
            if native_pending
            else record.handle or record.submission_handle,
            record.cancellation_requested,
        ),
        runtime_events(),
        observation,
        publication=publication,
        last_observation=last_observation,
    )
    if native_pending:
        return derived
    if requires_collection and derived.state == RunState.COMPLETED:
        if (
            collected is None
            or collected.result.runtime.status != OutcomeStatus.SUCCESS
        ):
            derived = replace(
                derived,
                state=RunState.TERMINAL_UNKNOWN,
                confidence="unconfirmed",
                reason="Independent runtime, batch and stage-out receipts are not fully successful",
            )
    if restarts.restart_errors.get(plan_id) == "RestartBindingUnavailable":
        return replace(
            derived,
            state=RunState.OBSERVATION_UNAVAILABLE,
            confidence="unconfirmed",
            reason="Native scheduler restart observed; validated startup binding unavailable",
        )
    if record.handle is not None and record.native_identity is not None:
        native_restart = json.loads(record.handle.raw_json).get("restart_count")
        if (
            type(native_restart) is int
            and record.native_identity.restart_count > native_restart
        ):
            return replace(
                derived,
                state=RunState.OBSERVATION_UNAVAILABLE,
                confidence="unconfirmed",
                reason="Verified newer native generation awaits a runtime binding",
            )
    if record.state == ExecutionState.REJECTED:
        return replace(
            derived,
            state=RunState.FAILED,
            confidence="confirmed",
            reason=record.detail,
        )
    error = handshake.negotiation_error(plan_id)
    if validation_error:
        error = validation_error[0]
    if (
        tracker is not None
        and tracker.attempt == record.attempt
        and tracker.result.status == NegotiationStatus.FAILED
    ):
        error = tracker.result.reason
    if error is not None:
        return replace(
            derived,
            state=RunState.FAILED,
            confidence="confirmed",
            reason=f"Runtime negotiation failed: {error}",
        )
    return derived
