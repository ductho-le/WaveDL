"""
Bounded scheduler evidence about a cluster's jobs, never a training verdict.

Each row is matched to a job by ID, then checked against the job's owner, name
and cluster. A live row of another job fails the observation; an accounting row
of another job is dropped. A failed or truncated query cannot prove absence.

Field semantics: https://slurm.schedmd.com/{sacct,squeue,scontrol}.html

Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from wavedl_daemon.runs.model import ObservationFailed, Phase, SlurmObservation

from .slurm_plan import job_name


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from datetime import datetime

    from wavedl_daemon.runs.model import RunId


_TERMINAL = frozenset(
    {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "BOOT_FAIL",
        "DEADLINE",
        "PREEMPTED",
        "REVOKED",
    }
)
_RUNNING = frozenset({"RUNNING", "COMPLETING", "CONFIGURING", "STAGE_OUT"})
_QUEUED = frozenset(
    {"PENDING", "REQUEUED", "REQUEUE_FED", "RESIZING", "SUSPENDED", "STOPPED"}
)
_UNAVAILABLE = (
    "accounting storage is disabled",
    "accounting is not enabled",
    "accountingstorage/none",
    "accounting storage type is none",
)


def corroborates(comment: str, run_id: str) -> bool:
    """Whether a job's comment agrees with its run: the run ID, or none stored."""
    return comment in (run_id, "", "(null)")


def parse_controller(output: str) -> list[dict[str, str]]:
    """Parse ``scontrol --oneliner`` records without last-value-wins aliases.

    Only unquoted whitespace-delimited scalar identity tokens are supported.
    Human-readable fields are diagnostic, never native authority.
    """
    rows = []
    for line in output.splitlines():
        if not line.strip() or line.strip() == "No jobs in the system":
            continue
        row = {}
        for key, value in re.findall(r"(?:^|\s)([^\s=]+)=(\S*)", line):
            if key in row:
                raise ValueError("Duplicate native controller field")
            row[key] = value
        if "JobId" not in row:
            raise ValueError("Malformed native controller record")
        rows.append(row)
    return rows


def _phase(value: str, reason: str) -> Phase | None:
    """The live phase of a native state; None when it is terminal."""
    if not value.strip():
        raise ValueError("Missing native scheduler state")
    state = value.split()[0].rstrip("+").upper()
    if state in _TERMINAL:
        return None
    if (
        "HELD" in state
        or any(word in reason.lower() for word in ("hold", "held"))
        or state == "SPECIAL_EXIT"
    ):
        return Phase.HELD
    if state in _RUNNING:
        return Phase.RUNNING
    if state in _QUEUED:
        return Phase.QUEUED
    raise ValueError(f"Unsupported native scheduler state: {state[:64]}")


def _exit(value: str) -> tuple[int | None, int | None]:
    if not value:
        return None, None
    if not re.fullmatch(r"\d+:\d+", value):
        raise ValueError("Malformed scheduler exit status")
    code, signal = map(int, value.split(":"))
    return code, signal or None


def _restart_count(row: dict[str, str]) -> int:
    """Read Slurm's restart aliases without allowing conflicting evidence."""
    values = [row[key] for key in ("Restarts", "RestartCnt") if key in row]
    if not values or any(not value.isdigit() for value in values):
        raise ValueError("Native scheduler restart count unavailable")
    if len(set(values)) != 1:
        raise ValueError("Conflicting native scheduler restart counts")
    return int(values[0])


def observe_cluster(
    query: Callable[[tuple[str, ...]], str],
    *,
    owner: str,
    cluster: str,
    jobs: Mapping[str, RunId],
    since: datetime,
) -> dict[str, SlurmObservation | ObservationFailed]:
    """Observe ``jobs`` (job ID to run ID) on one cluster.

    ``query`` returns stdout or raises ValueError. A squeue failure fails the
    batch. The controller is asked about each job squeue lists, since only it
    knows a live job's restart count. The controller and accounting may each
    be missing, as long as the other still proves the job's state and restart
    count. ``since`` (UTC) bounds accounting, so a reused job ID's old records
    are never read.
    """
    queue = _rows(
        query(
            (
                "squeue",
                "--noheader",
                "--local",
                "--clusters",
                cluster,
                f"--user={owner}",
                "--states=all",
                "--format=%i|%u|%j|%T|%r|%k",
            )
        ),
        jobs,
    )
    failures = []
    try:
        accounting = query(
            (
                "sacct",
                "--noheader",
                "--parsable2",
                "--duplicates",
                "--clusters",
                cluster,
                "--jobs",
                ",".join(jobs),
                f"--starttime={since.strftime('%Y-%m-%dT%H:%M:%S')}",
                "--format=JobIDRaw,Cluster,User,JobName%256,State%64,ExitCode,Restarts,Submit,Comment%256",
            )
        )
    except ValueError as error:
        accounting = ""
        if not any(message in str(error).lower() for message in _UNAVAILABLE):
            failures.append(str(error)[:512])
    account = _rows(accounting, jobs)
    answer: dict[str, SlurmObservation | ObservationFailed] = {}
    for job, run_id in jobs.items():
        control, controller_failures = "", []
        if job in queue:
            try:
                control = query(
                    (
                        "scontrol",
                        "--clusters",
                        cluster,
                        "--oneliner",
                        "show",
                        "job",
                        job,
                    )
                )
            except ValueError as error:
                controller_failures.append(str(error)[:512])
        try:
            answer[job] = _observe_job(
                job,
                run_id,
                owner=owner,
                cluster=cluster,
                queue=queue.get(job, []),
                control=[
                    row for row in parse_controller(control) if row["JobId"] == job
                ],
                accounting=account.get(job, []),
                failures=controller_failures + failures,
            )
        except ValueError as error:
            answer[job] = ObservationFailed(str(error)[:512])
    return answer


def _rows(output: str, jobs: Mapping[str, RunId]) -> dict[str, list[list[str]]]:
    """``|`` rows by job ID, keeping only ``jobs``; IDs never contain ``|``."""
    rows: dict[str, list[list[str]]] = {}
    for line in output.splitlines():
        row = line.split("|")
        if row[0] in jobs:
            rows.setdefault(row[0], []).append(row)
    return rows


def _observe_job(
    job: str,
    run_id: RunId,
    *,
    owner: str,
    cluster: str,
    queue: list[list[str]],
    control: list[dict[str, str]],
    accounting: list[list[str]],
    failures: list[str],
) -> SlurmObservation | ObservationFailed:
    name = job_name(run_id)
    q, c = queue, control
    # This job's name and comment cannot contain "|", so a wider row is another job's.
    a = [
        row
        for row in accounting
        if len(row) == 9
        and row[1:4] == [cluster, owner, name]
        and corroborates(row[8], run_id)
    ]
    if any(
        len(row) != 6 or row[1:3] != [owner, name] or not corroborates(row[5], run_id)
        for row in q
    ) or any(
        row.get("UserId", "").split("(")[0] != owner
        or row.get("JobName") != name
        or not corroborates(row.get("Comment", ""), run_id)
        or (row.get("Cluster") is not None and row["Cluster"] != cluster)
        for row in c
    ):
        raise ValueError("Scheduler native identity mismatch")
    if len(q) > 1 or len(c) > 1:
        raise ValueError("Ambiguous live scheduler execution")
    if any(not row[6].isdigit() for row in a):
        raise ValueError("Accounting restart count unavailable")
    if c:
        restart = _restart_count(c[0])
    elif q:
        # squeue has no restart count, and accounting can lag a requeue.
        return ObservationFailed(
            "; ".join(failures) or "Scheduler restart count unavailable"
        )
    elif a:
        restart = max(int(row[6]) for row in a)
    elif failures:
        return ObservationFailed("; ".join(failures))
    else:
        # Every source answered and none knows the job: it ended and was purged.
        return SlurmObservation(0, None, True, None, None, None, None, {})
    if any(int(row[6]) > restart for row in a):
        raise ValueError("Accounting is ahead of the controller's restart count")
    current = [row for row in a if int(row[6]) == restart]
    if len(current) > 1:
        raise ValueError("Ambiguous accounting executions after requeue")
    earlier = {int(row[6]): row[4] for row in a if int(row[6]) < restart}
    # Live controller evidence takes precedence over stale accounting.
    state = c[0].get("JobState", "") if c else current[0][4] if current else ""
    reason = c[0].get("Reason", "") if c else ""
    if not state:
        raise ValueError("Missing native scheduler state")
    phase = _phase(state, reason)
    code = signal = None
    if phase is None:
        code, signal = _exit(c[0].get("ExitCode", "") if c else current[0][5])
    return SlurmObservation(
        restart, phase, phase is None, state, reason or None, code, signal, earlier
    )
