# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Native identity and bounded hints; filesystem hints never authorize activation."""

from __future__ import annotations

from pathlib import Path

from wavedl.core.safe_files import open_directory
from wavedl_daemon.backends.contracts import (
    BackendObservation,
    ObservationStatus,
    SubmissionObservation,
)
from wavedl_daemon.execution.budget import BudgetExhausted
from wavedl_daemon.storage.executions import native_runtime_pending


def same_generation(record, source):
    same = record is not None and (
        record.attempt,
        record.handle,
        record.submission_handle,
    ) == (
        source.attempt,
        source.handle,
        source.submission_handle,
    )
    if same:
        return native_identity(record) == native_identity(source)
    return same


def native_pending(record):
    return native_runtime_pending(record)


def native_identity(record):
    # Native identity is monotonic authority, not a known-good lifecycle
    # snapshot or the latest (possibly identity-free) diagnostic.
    if record.native_identity is not None:
        return record.native_identity
    observation = record.submission_observation
    return observation.native_identity if observation is not None else None


def watch_submission(context, record):
    """Retain a bounded native watch for approved, still-unbound requeues."""
    if record.submission_handle is None:
        return False
    plan = context.plan_store.get(record.plan_id)
    return plan is not None and dict(plan.plan.resources).get("requeue") is True


def observation_error(record, detail):
    if record.attempt is None or native_runtime_pending(record):
        return SubmissionObservation(ObservationStatus.ERROR, detail=detail)
    return BackendObservation(record.attempt, ObservationStatus.ERROR, detail=detail)


def initial_hint(receipt):
    """Bounded visibility hint; native verification remains authoritative."""
    path = Path(receipt.run_dir) / "submissions" / receipt.request_id / "launchers"
    try:
        with open_directory(path) as launchers:
            launchers.verify_private_owner()
            for count, entry in enumerate(launchers.entries(), 1):
                if count > 4096:
                    raise BudgetExhausted("Startup hint inventory exhausted")
                if not entry.is_directory:
                    continue
                with launchers.open_directory(entry.name) as claim:
                    try:
                        with claim.open_file("binding.json") as proof:
                            proof.verify_private_owner()
                            proof.guard()
                    except FileNotFoundError:
                        continue
                    launchers.guard()
                    return True
            launchers.guard()
    except FileNotFoundError:
        pass
    return False


def restart_hint(handle, token):
    """Filesystem hints only admit native discovery; never authorize CAS.

    Unchanged jobs do not add a per-run scheduler query to batched fleet
    polling. Abandoned reservations remain harmless hints, not executions.
    """
    with open_directory(Path(handle.run_dir) / "attempts") as attempts:
        attempts.verify_private_owner()
        for count, entry in enumerate(attempts.entries(), 1):
            token.check()
            if count > 4096:
                raise BudgetExhausted("Scheduler attempt hint inventory exhausted")
            if entry.is_directory and entry.name.isdecimal():
                if len(entry.name) > 19:
                    raise ValueError("Scheduler attempt hint integer exceeds bound")
                if int(entry.name) > handle.attempt:
                    attempts.guard()
                    return True
        attempts.guard()
    return False
