"""Dependency-light immutable Slurm command outcome receipts.

These receipts describe commands, not runtime journal negotiation or published
checkpoints. Every receipt is bound to a complete scheduler execution identity.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from uuid import UUID

from .layout import OperationKind
from .persistence import write_once


MAX_RECEIPT_BYTES = 16384


def receipt_name(restart_count: int) -> str:
    if type(restart_count) is not int or restart_count < 0:
        raise ValueError("restart_count must be nonnegative")
    return (
        "slurm-outcome.json"
        if restart_count == 0
        else f"slurm-outcome-restart-{restart_count}.json"
    )


def command_outcome(code: int) -> dict[str, object]:
    """125 is the wrapper's not-executed sentinel, never successful work."""
    if type(code) is not int or not 0 <= code <= 255:
        raise ValueError("Command exit must be in 0..255")
    return {
        "status": "unknown" if code == 125 else "success" if code == 0 else "failed",
        "exit_code": None if code == 125 else code,
        "signal": None,
        "detail": "Command return status; not runtime publication evidence",
    }


def write_outcome_receipt(
    directory: Path,
    *,
    run_id: str,
    request_id: str,
    attempt: int,
    job_id: str,
    cluster: str,
    restart_count: int,
    operation: str,
    runtime_exit: int,
    batch_exit: int,
    stage_out_exit: int,
) -> Path:
    """Publish complete bytes once; existing receipts cannot be overwritten."""
    for value in (run_id, request_id):
        if str(UUID(value)) != value:
            raise ValueError("Receipt IDs must be canonical UUIDs")
    if type(attempt) is not int or attempt < 1:
        raise ValueError("attempt must be positive")
    if re.fullmatch(r"[1-9][0-9]*", job_id) is None:
        raise ValueError("Invalid scheduler job ID")
    if re.fullmatch(r"[A-Za-z0-9_.-]+", cluster) is None:
        raise ValueError("Receipt requires an exact cluster")
    name = receipt_name(restart_count)
    data = {
        "schema_version": 1,
        "run_id": run_id,
        "request_id": request_id,
        "attempt": attempt,
        "job_id": job_id,
        "cluster": cluster,
        "restart_count": restart_count,
        "operation": OperationKind(operation).value,
        "runtime": command_outcome(runtime_exit),
        "batch": command_outcome(batch_exit),
        "stage_out": command_outcome(stage_out_exit),
    }
    content = (json.dumps(data, allow_nan=False, sort_keys=True) + "\n").encode()
    if len(content) > MAX_RECEIPT_BYTES:
        raise ValueError("Receipt exceeds size bound")
    path = Path(directory) / name
    write_once(path, content)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    for name in ("run-id", "request-id", "job-id", "cluster", "operation"):
        parser.add_argument(f"--{name}", required=True)
    for name in (
        "attempt",
        "restart-count",
        "runtime-exit",
        "batch-exit",
        "stage-out-exit",
    ):
        parser.add_argument(f"--{name}", type=int, required=True)
    write_outcome_receipt(**vars(parser.parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
