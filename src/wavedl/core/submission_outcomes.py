"""Immutable independent Slurm outcomes, including failure before runtime starts."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from .layout import OperationKind
from .outcomes import MAX_RECEIPT_BYTES, command_outcome
from .protocol import strict_loads
from .safe_files import open_directory
from .scheduler_attempt import SchedulerIdentity, read_scheduler_binding_in_directory
from .submission import (
    MAX_MANIFEST_BYTES,
    _open_writable_root,
    identity_for_submission_directory,
    open_submission_in_directory,
    publish_submission_member,
)


def submission_outcome_name(identity: SchedulerIdentity) -> str:
    """Full native execution identity distinguishes initial and restarted receipts."""
    if not isinstance(identity, SchedulerIdentity):
        raise ValueError("Typed scheduler identity required")
    return f"slurm-outcome-{identity.claim_name}.json"


def _object(raw):
    value = strict_loads(raw)

    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("Duplicate submission outcome JSON field")
            result[key] = item
        return result

    json.loads(raw, object_pairs_hook=unique)
    if type(value) is not dict:
        raise ValueError("Submission outcome must be an object")
    return value


def _read(directory, name, limit):
    with directory.open_file(name) as file:
        file.verify_private_owner()
        if file.stat().st_size > limit:
            raise ValueError("Submission outcome file exceeds byte bound")
        raw = file.read(limit)
        file.guard()
    directory.guard()
    return raw


@contextmanager
def _authority(run_directory, *, request_id, owner, plan_sha256, writable=False):
    if plan_sha256 is None:
        raise ValueError("Submission outcome requires a bound plan digest")
    path = Path(run_directory)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Submission outcome requires an absolute ordinary run path")
    admission = (
        _open_writable_root(path, deadline=time.monotonic() + 5, opener=open_directory)
        if writable
        else open_directory(path)
    )
    with admission as root:
        root.verify_private_owner()
        request = identity_for_submission_directory(
            root,
            request_id=request_id,
            backend="slurm",
            owner=owner,
            plan_sha256=plan_sha256,
        )
        manifest = _object(_read(root, "manifest.json", MAX_MANIFEST_BYTES))
        operation = OperationKind(manifest["operation"]).value
        with open_submission_in_directory(root, request, writable=writable) as selected:
            yield root, selected, request, operation
            root.guard()


def _validate_attempt(root, identity, attempt):
    if attempt is None:
        return
    if type(attempt) is not int or not 1 <= attempt <= 2**63 - 1:
        raise ValueError("Outcome attempt must be a positive runtime identity")
    binding = read_scheduler_binding_in_directory(root, identity, initial_attempt=None)
    if binding.attempt != attempt:
        raise ValueError("Outcome differs from immutable scheduler binding")


def _outcome(value):
    if type(value) is not dict or set(value) != {
        "status",
        "exit_code",
        "signal",
        "detail",
    }:
        raise ValueError("Invalid command outcome fields")
    code = value["exit_code"]
    if code is None:
        code = 125
    expected = command_outcome(code)
    if value != expected:
        raise ValueError("Command outcome status differs from exit code")
    return expected


def _validate(value, request, identity, operation, attempt):
    expected = {
        "schema_version": 2,
        **asdict(request),
        "cluster": identity.cluster,
        "job_id": identity.job_id,
        "restart_count": identity.restart_count,
        "operation": operation,
        "attempt": attempt,
    }
    if type(value) is not dict or set(value) != {
        *expected,
        "runtime",
        "batch",
        "stage_out",
    }:
        raise ValueError("Invalid submission outcome receipt fields")
    if (
        type(value["schema_version"]) is not int
        or type(value["restart_count"]) is not int
        or (attempt is not None and type(value["attempt"]) is not int)
        or any(value[key] != item for key, item in expected.items())
        or request.run_id != identity.run_id
        or request.request_id != identity.request_id
        or request.owner != identity.owner
    ):
        raise ValueError("Submission outcome identity mismatch")
    for name in ("runtime", "batch", "stage_out"):
        _outcome(value[name])
    if attempt is None and value["runtime"] != command_outcome(125):
        raise ValueError("Unbound submission cannot claim runtime execution")
    return value


def write_submission_outcome_receipt(
    run_directory: Path,
    *,
    request_id: str,
    owner: str,
    plan_sha256: str,
    job_id: str,
    cluster: str,
    restart_count: int,
    operation: str,
    runtime_exit: int,
    batch_exit: int,
    stage_out_exit: int,
    attempt: int | None = None,
) -> Path:
    """Publish once; pre-runtime command failure neither allocates nor journals."""
    runtime = command_outcome(runtime_exit)
    batch, stage_out = command_outcome(batch_exit), command_outcome(stage_out_exit)
    if attempt is None and runtime_exit != 125:
        raise ValueError("Unbound submission cannot claim runtime execution")
    with _authority(
        run_directory,
        request_id=request_id,
        owner=owner,
        plan_sha256=plan_sha256,
        writable=True,
    ) as (root, selected, request, expected_operation):
        identity = SchedulerIdentity(
            request.run_id, request_id, cluster, job_id, owner, restart_count
        )
        if OperationKind(operation).value != expected_operation:
            raise ValueError("Outcome operation differs from immutable manifest")
        _validate_attempt(root, identity, attempt)
        value = {
            "schema_version": 2,
            **asdict(request),
            "cluster": cluster,
            "job_id": job_id,
            "restart_count": restart_count,
            "operation": expected_operation,
            "attempt": attempt,
            "runtime": runtime,
            "batch": batch,
            "stage_out": stage_out,
        }
        _validate(value, request, identity, expected_operation, attempt)
        content = (json.dumps(value, allow_nan=False, sort_keys=True) + "\n").encode()
        if len(content) > MAX_RECEIPT_BYTES:
            raise ValueError("Submission outcome receipt exceeds byte bound")
        root.guard()
        return publish_submission_member(
            selected,
            submission_outcome_name(identity),
            content,
            limit=MAX_RECEIPT_BYTES,
        )


def read_submission_outcome_receipt(
    run_directory: Path,
    identity: SchedulerIdentity,
    *,
    plan_sha256: str,
    attempt: int | None = None,
) -> dict:
    """Read exact provenance and native identity; no completion inferred from paths."""
    if not isinstance(identity, SchedulerIdentity):
        raise ValueError("Typed scheduler identity required")
    with _authority(
        run_directory,
        request_id=identity.request_id,
        owner=identity.owner,
        plan_sha256=plan_sha256,
    ) as (root, selected, request, operation):
        _validate_attempt(root, identity, attempt)
        value = _object(
            _read(selected, submission_outcome_name(identity), MAX_RECEIPT_BYTES)
        )
        return _validate(value, request, identity, operation, attempt)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    for name in (
        "run-id",
        "request-id",
        "owner",
        "plan-sha256",
        "job-id",
        "cluster",
        "operation",
    ):
        parser.add_argument(f"--{name}", required=True)
    for name in ("restart-count", "runtime-exit", "batch-exit", "stage-out-exit"):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--attempt", type=int)
    values = vars(parser.parse_args(argv))
    run_id = values.pop("run_id")
    try:
        with _authority(
            values["run_directory"],
            request_id=values["request_id"],
            owner=values["owner"],
            plan_sha256=values["plan_sha256"],
        ) as (_, _, request, _):
            if request.run_id != run_id:
                raise ValueError("CLI run identity differs from submission")
        write_submission_outcome_receipt(**values)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Submission outcome refused: {type(error).__name__}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
