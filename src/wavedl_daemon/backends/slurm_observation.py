# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Bounded scheduler evidence, never a training verdict.

Native field semantics follow https://slurm.schedmd.com/{sacct,squeue,scontrol}.html.
Queries are batched by cluster; truncated or failed queries cannot prove absence.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from wavedl.core.layout import OperationKind
from wavedl.core.outcomes import MAX_RECEIPT_BYTES, receipt_name
from wavedl.core.protocol import strict_loads
from wavedl.core.safe_files import open_directory
from wavedl.core.scheduler_attempt import SchedulerBinding, SchedulerIdentity
from wavedl.core.submission import identity_for_submission, read_submission_file

from .contracts import (
    BackendObservation,
    Handle,
    ObservationStatus,
    OperationOutcome,
    OperationResult,
    OutcomeStatus,
    Plan,
    SubmissionCollectionResult,
    SubmissionHandle,
    SubmissionObservation,
)


if TYPE_CHECKING:
    from .scheduler import SchedulerRunner


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
_UNAVAILABLE = (
    "accounting storage is disabled",
    "accounting is not enabled",
    "accountingstorage/none",
    "accounting storage type is none",
)


@dataclass(frozen=True)
class SlurmCapabilities:
    accounting: bool | None = None
    min_job_age: int | None = None
    diagnostics: tuple[str, ...] = ()


class RestartBindingUnavailable(ValueError):
    """Native restart is proven, but its private attempt binding is unavailable."""

    def __init__(self, native_restart_count: int):
        if (
            type(native_restart_count) is not int
            or not 1 <= native_restart_count <= 2**63 - 1
        ):
            raise ValueError("Invalid confirmed native restart generation")
        self.native_restart_count = native_restart_count
        super().__init__("Validated scheduler restart binding unavailable")


def _observation(handle, *args, **kwargs):
    if isinstance(handle, SubmissionHandle):
        return SubmissionObservation(*args, **kwargs)
    return BackendObservation(handle.attempt, *args, **kwargs)


def _parse_native_control(output: str) -> list[dict[str, str]]:
    """Parse the flat oneliner fields without last-value-wins identity aliases.

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


def _state(value: str, reason: str = "") -> ObservationStatus:
    if not value.strip():
        raise ValueError("Missing native scheduler state")
    value = value.split()[0].rstrip("+").upper()
    if value in _TERMINAL:
        return ObservationStatus.TERMINAL
    if (
        "HELD" in value
        or any(word in reason.lower() for word in ("hold", "held"))
        or value == "SPECIAL_EXIT"
    ):
        return ObservationStatus.HELD
    if value in _RUNNING:
        return ObservationStatus.RUNNING
    if value in {
        "PENDING",
        "REQUEUED",
        "REQUEUE_FED",
        "RESIZING",
        "SUSPENDED",
        "STOPPED",
    }:
        return ObservationStatus.QUEUED
    return ObservationStatus.ERROR


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
        raise ValueError("Native scheduler restart generation unavailable")
    if len(set(values)) != 1:
        raise ValueError("Conflicting native scheduler restart generations")
    return int(values[0])


def _outcome(state: str, code: int | None, signal: int | None) -> OperationOutcome:
    status = OutcomeStatus.FAILED
    if state.startswith("CANCELLED"):
        status = OutcomeStatus.CANCELLED
    elif state == "COMPLETED" and code == 0 and signal is None:
        status = OutcomeStatus.SUCCESS
    return OperationOutcome(status, code, signal, "Scheduler batch outcome")


class SlurmObservation:
    def __init__(self, runner: SchedulerRunner, *, owner: str):
        if not isinstance(owner, str) or not re.fullmatch(
            r"[A-Za-z0-9_][A-Za-z0-9_.-]*", owner
        ):
            raise ValueError("Invalid scheduler owner")
        self.runner = runner
        self.owner = owner
        self._capabilities = SlurmCapabilities()

    def _query(self, argv: tuple[str, ...]) -> str:
        try:
            result = self.runner.run(argv)
        except RuntimeError as error:
            raise ValueError(f"{argv[0]} unavailable: {error}") from error
        if (
            result.returncode
            or result.timed_out
            or result.cancelled
            or result.stdout_truncated
            or result.stderr_truncated
        ):
            diagnostic = result.stderr.decode("utf-8", errors="replace")[:512]
            raise ValueError(f"{argv[0]} unavailable: {diagnostic}")
        return result.stdout.decode("utf-8", errors="strict")

    def capabilities(self) -> SlurmCapabilities:
        diagnostics = []
        accounting = None
        min_job_age = None
        try:
            config = self._query(("scontrol", "show", "config"))
            age = re.search(r"\bMinJobAge\s*=\s*(\d+)", config)
            if age:
                min_job_age = int(age[1])
            else:
                diagnostics.append("MinJobAge unavailable")
            storage = re.search(r"\bAccountingStorageType\s*=\s*(\S+)", config)
            if storage and storage[1].lower() in {
                "accounting_storage/none",
                "none",
                "(null)",
            }:
                accounting = False
            elif storage:
                # Empty output on a successful probe is not proof that accounting
                # is disabled: this owner may simply have no jobs.
                self._query(
                    (
                        "sacct",
                        "--noheader",
                        "--parsable2",
                        "--user",
                        self.owner,
                        "--format=JobIDRaw",
                        "--starttime=now-1minute",
                    )
                )
                accounting = True
            flags = re.search(r"\bAccountingStoreFlags\s*=\s*(\S+)", config)
            if accounting is not False and (
                not flags or "job_comment" not in flags[1].lower()
            ):
                diagnostics.append(
                    "Accounting job comments unavailable; accounting-only provenance cannot be verified"
                )
        except (ValueError, OSError, UnicodeError) as error:
            diagnostics.append(str(error)[:512])
        self._capabilities = SlurmCapabilities(
            accounting, min_job_age, tuple(diagnostics)
        )
        return self._capabilities

    def _identity(self, handle: Handle | SubmissionHandle) -> tuple[str, str, str, int]:
        try:
            raw = json.loads(handle.raw_json)
        except (TypeError, ValueError) as exc:
            raise ValueError("Handle scheduler identity is not valid JSON") from exc
        if not isinstance(raw, dict):
            raise ValueError("Handle scheduler identity must be an object")
        if handle.backend != "slurm" or handle.owner != self.owner:
            raise ValueError("Slurm handle owner/backend mismatch")
        job, separator, suffix = (handle.job_id or "").partition(";")
        cluster = raw.get("cluster", suffix)
        name = raw.get("job_name")
        restart = raw.get("restart_count", 0)
        request = str(UUID(handle.request_id))
        if (
            not re.fullmatch(r"[1-9]\d*", job)
            or not isinstance(cluster, str)
            or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", cluster)
            or not separator
            or cluster != suffix
            or not isinstance(name, str)
            or handle.request_id != request
            or name != f"wavedl-{request}"
            or raw.get("comment") != request
            or type(restart) is not int
            or restart < 0
        ):
            raise ValueError("Handle lacks durable scheduler identity")
        return job, cluster, name, restart

    def observe(
        self, handle: Handle | SubmissionHandle
    ) -> BackendObservation | SubmissionObservation:
        return self.observe_many((handle,))[handle.job_id or ""]

    def observe_submission(self, handle: SubmissionHandle) -> SubmissionObservation:
        """Observe a writer allocation before an attempt binding exists."""
        if not isinstance(handle, SubmissionHandle):
            raise ValueError("Submission observation requires a native receipt")
        return self.observe_many((handle,))[handle.job_id or ""]

    def discover_initial(self, handle: SubmissionHandle) -> SchedulerBinding | None:
        """Adopt a writer allocation only after native identity and binding checks."""
        if not isinstance(handle, SubmissionHandle):
            raise ValueError("Writer discovery requires a SubmissionHandle")
        job, cluster, name, _ = self._identity(handle)
        try:
            output = self._query(
                ("scontrol", "--clusters", cluster, "--oneliner", "show", "job", job)
            )
        except (OSError, ValueError, UnicodeError):
            output = ""
        rows = _parse_native_control(output)
        if not rows:
            # Native retained accounting can prove a completed writer even when
            # its live controller row was purged before daemon activation.
            observation = self.observe_submission(handle)
            if observation.status != ObservationStatus.TERMINAL:
                raise ValueError("Native submission execution identity unavailable")
            if observation.native_identity is None:
                raise ValueError("Native submission execution identity unavailable")
            return self._read_initial(handle, observation.native_identity)
        if len(rows) != 1:
            raise ValueError("Native submission identity absent or ambiguous")
        row = rows[0]
        if (
            row.get("JobId") != job
            or row.get("UserId", "").split("(")[0] != self.owner
            or row.get("JobName") != name
            or row.get("Comment") != handle.request_id
        ):
            raise ValueError("Native submission identity mismatch")
        status = _state(row.get("JobState", ""), row.get("Reason", ""))
        if status == ObservationStatus.ERROR:
            raise ValueError("Native submission state unavailable")

        if row.get("Cluster") is not None and row["Cluster"] != cluster:
            raise ValueError("Native submission restart identity unavailable")
        restart = _restart_count(row)
        identity = SchedulerIdentity(
            handle.run_id, handle.request_id, cluster, job, self.owner, restart
        )
        return self._read_initial(handle, identity)

    def _read_initial(self, handle, identity):
        from wavedl.core.scheduler_attempt import read_scheduler_binding

        try:
            self._verify_writer_plan(handle)
            binding = read_scheduler_binding(
                Path(handle.run_dir), identity, initial_attempt=None
            )
        except (OSError, ValueError):
            return None
        return binding

    def _verify_writer_plan(self, handle):
        """Tie native acceptance to the immutable approved submission namespace."""
        raw = json.loads(handle.raw_json)
        if raw.get("allocation_mode") != "writer":
            raise ValueError("Writer acceptance lacks allocation mode")
        request = identity_for_submission(
            Path(handle.run_dir),
            request_id=handle.request_id,
            backend="slurm",
            owner=self.owner,
            plan_sha256=raw.get("plan_sha256"),
        )
        plan = read_submission_file(Path(handle.run_dir), request, "plan.json")
        if hashlib.sha256(plan).hexdigest() != raw.get("plan_sha256"):
            raise ValueError("Writer acceptance differs from approved plan")
        approved = Plan.from_dict(strict_loads(plan))
        approved_cluster = dict(approved.resources).get("cluster")
        if (
            request.run_id != handle.run_id
            or approved.run_id != handle.run_id
            or approved.run_dir != handle.run_dir
            or approved.target != "slurm"
            or approved.allocation_mode != "writer"
            or approved.operation.value != raw.get("operation", "train")
            or (
                approved_cluster is not None
                and approved_cluster != self._identity(handle)[1]
            )
        ):
            raise ValueError("Writer acceptance differs from immutable authority")
        return request, approved

    def discover_restart(
        self, handle: Handle
    ) -> tuple[SchedulerBinding, Handle] | None:
        """Propose a native-proven execution binding; never activate a handle.

        Callers admit this bounded command/storage work to supervised workers.
        Missing or malformed native evidence/claims are unavailable, not adoption.
        The core reader refuses unverified host/execution mappings.
        """
        from wavedl.core.scheduler_attempt import read_scheduler_binding

        job, cluster, name, saved_restart = self._identity(handle)
        output = self._query(
            ("scontrol", "--clusters", cluster, "--oneliner", "show", "job", job)
        )
        rows = _parse_native_control(output)
        if len(rows) != 1:
            raise ValueError("Native restart evidence absent or ambiguous")
        row = rows[0]
        if (
            row.get("JobId") != job
            or row.get("UserId", "").split("(")[0] != self.owner
            or row.get("JobName") != name
            or row.get("Comment") != handle.request_id
            or (row.get("Cluster") is not None and row["Cluster"] != cluster)
        ):
            raise ValueError("Native restart identity mismatch")
        status = _state(row.get("JobState", ""), row.get("Reason", ""))
        restart = _restart_count(row)
        if restart < saved_restart:
            raise ValueError("Native restart generation regressed")
        identity = SchedulerIdentity(
            handle.run_id, handle.request_id, cluster, job, self.owner, restart
        )
        raw = json.loads(handle.raw_json)
        writer = raw.get("allocation_mode") == "writer"
        if status == ObservationStatus.ERROR:
            if writer and restart > saved_restart and row.get("JobState"):
                # Full native identity survives an unsupported lifecycle state.
                # The caller may retain the head, but must not activate a runtime.
                raise RestartBindingUnavailable(restart)
            raise ValueError("Native restart state unavailable")
        if restart == saved_restart:
            return None
        initial_attempt = None if writer else raw.get("initial_attempt", handle.attempt)
        if not writer and (
            type(initial_attempt) is not int
            or not 1 <= initial_attempt <= handle.attempt
        ):
            raise RestartBindingUnavailable(restart)
        try:
            if writer:
                self._verify_writer_plan(handle)
            binding = read_scheduler_binding(
                Path(handle.run_dir), identity, initial_attempt=initial_attempt
            )
        except (OSError, ValueError) as error:
            raise RestartBindingUnavailable(restart) from error
        if (
            binding.run_id != identity.run_id
            or binding.request_id != identity.request_id
            or binding.owner != identity.owner
            or binding.cluster != identity.cluster
            or binding.job_id != identity.job_id
            or binding.restart_count != identity.restart_count
        ):
            raise RestartBindingUnavailable(restart)
        if binding.attempt <= handle.attempt:
            raise RestartBindingUnavailable(restart)
        raw.update(restart_count=restart)
        if not writer:
            raw.update(initial_attempt=initial_attempt)
        candidate = replace(handle, attempt=binding.attempt, raw_json=json.dumps(raw))
        return binding, candidate

    def observe_many(
        self, handles: tuple[Handle | SubmissionHandle, ...]
    ) -> dict[str, BackendObservation | SubmissionObservation]:
        observations = {}
        groups = {}
        for handle in handles:
            key = handle.job_id or ""
            if key in observations or any(
                key == item.job_id for group in groups.values() for item in group
            ):
                raise ValueError("Duplicate full scheduler identity")
            try:
                _, cluster, _, _ = self._identity(handle)
                groups.setdefault(cluster, []).append(handle)
            except ValueError as error:
                observations[key] = _observation(
                    handle, ObservationStatus.ERROR, detail=str(error)
                )
        for cluster, group in groups.items():
            try:
                observations.update(self._observe_cluster(cluster, group))
            except (ValueError, OSError, UnicodeError) as error:
                for handle in group:
                    observations[handle.job_id] = _observation(
                        handle, ObservationStatus.ERROR, detail=str(error)[:512]
                    )
        return observations

    def _observe_cluster(
        self, cluster: str, handles: list[Handle | SubmissionHandle]
    ) -> dict[str, BackendObservation | SubmissionObservation]:
        jobs = ",".join(self._identity(handle)[0] for handle in handles)
        queue = self._query(
            (
                "squeue",
                "--noheader",
                "--local",
                "--clusters",
                cluster,
                "--jobs",
                jobs,
                "--states=all",
                "--format=%i|%u|%j|%T|%r|%k",
            )
        )
        control_error = None
        try:
            control = self._query(
                ("scontrol", "--clusters", cluster, "--oneliner", "show", "jobs")
            )
        except (ValueError, OSError, UnicodeError) as error:
            control = ""
            control_error = str(error)[:512]
            self._capabilities = SlurmCapabilities(
                self._capabilities.accounting,
                self._capabilities.min_job_age,
                (
                    f"Bounded whole-cluster controller fallback unavailable: {control_error}",
                ),
            )
        accounting_error = None
        try:
            accounting = self._query(
                (
                    "sacct",
                    "--noheader",
                    "--parsable2",
                    "--duplicates",
                    "--clusters",
                    cluster,
                    "--jobs",
                    jobs,
                    "--format=JobIDRaw,Cluster,User,JobName%256,State%64,ExitCode,Restarts,Submit,Comment%256",
                )
            )
        except ValueError as error:
            accounting_error = str(error)
            accounting = ""
            if any(message in accounting_error.lower() for message in _UNAVAILABLE):
                self._capabilities = SlurmCapabilities(
                    False, self._capabilities.min_job_age, (accounting_error,)
                )
                accounting_error = None
        queue_rows = [
            line.split("|")
            for line in queue.splitlines()
            if line and not line.startswith("CLUSTER:")
        ]
        account_rows = [line.split("|") for line in accounting.splitlines() if line]
        if any(len(row) != 6 for row in queue_rows) or any(
            len(row) != 9 for row in account_rows
        ):
            raise ValueError("Malformed native scheduler rows")
        controls = _parse_native_control(control)
        answer = {}
        for handle in handles:
            native = None
            try:
                job, _, name, restart = self._identity(handle)
                q = [row for row in queue_rows if row[0] == job]
                c = [row for row in controls if row.get("JobId") == job]
                a = [row for row in account_rows if row[0] == job]
                if (
                    any(
                        row[1:3] != [self.owner, name] or row[5] != handle.request_id
                        for row in q
                    )
                    or any(
                        row.get("UserId", "").split("(")[0] != self.owner
                        or row.get("JobName") != name
                        or row.get("Comment") != handle.request_id
                        or (
                            row.get("Cluster") is not None and row["Cluster"] != cluster
                        )
                        for row in c
                    )
                    or any(
                        row[1:4] != [cluster, self.owner, name]
                        or (
                            row[8] != handle.request_id
                            and not (row[8] in {"", "(null)"} and (q or c))
                        )
                        for row in a
                    )
                ):
                    raise ValueError("Scheduler native identity mismatch")
                if len(q) > 1 or len(c) > 1:
                    raise ValueError("Ambiguous live scheduler execution")
                if any(not row[6].isdigit() for row in a):
                    raise ValueError("Accounting execution identity unavailable")
                if isinstance(handle, SubmissionHandle):
                    if c:
                        restart = _restart_count(c[0])
                    elif a:
                        restart = max(int(row[6]) for row in a)
                if c and _restart_count(c[0]) != restart:
                    raise ValueError("Scheduler requeue changed execution identity")
                if any(not row[6].isdigit() for row in a):
                    raise ValueError("Accounting execution identity unavailable")
                if any(int(row[6]) > restart for row in a):
                    raise ValueError("Accounting requeue changed execution identity")
                a = [row for row in a if int(row[6]) == restart]
                if len(a) > 1:
                    raise ValueError("Ambiguous accounting executions after requeue")
                # Live controller evidence takes precedence over stale accounting.
                state = (
                    c[0].get("JobState", "")
                    if c
                    else q[0][3]
                    if q
                    else a[0][4]
                    if a
                    else ""
                )
                reason = c[0].get("Reason", "") if c else q[0][4] if q else ""
                if not state:
                    if c or q or a:
                        raise ValueError("Missing native scheduler state")
                    status = (
                        ObservationStatus.ERROR
                        if accounting_error or control_error
                        else ObservationStatus.ABSENT
                    )
                    answer[handle.job_id] = _observation(
                        handle,
                        status,
                        detail=accounting_error
                        or control_error
                        or "No native records; accounting/retention evidence absent",
                    )
                    continue
                status = _state(state, reason)
                # squeue alone cannot prove which restart produced a terminal row.
                if status == ObservationStatus.TERMINAL and not c and not a:
                    raise ValueError("Terminal execution restart identity unavailable")
                # Native generation proof is independent of lifecycle support.
                # An unknown state cannot erase a validated writer allocation
                # head; squeue alone still cannot supply its restart identity.
                if isinstance(handle, SubmissionHandle) and (c or a):
                    native = SchedulerIdentity(
                        handle.run_id,
                        handle.request_id,
                        cluster,
                        job,
                        self.owner,
                        restart,
                    )
                code = signal = None
                batch = None
                if status == ObservationStatus.TERMINAL:
                    native_exit = (
                        c[0].get("ExitCode", "") if c else a[0][5] if a else ""
                    )
                    code, signal = _exit(native_exit)
                    if code is not None:
                        batch = _outcome(state.split()[0].rstrip("+"), code, signal)
                answer[handle.job_id] = _observation(
                    handle,
                    status,
                    state,
                    code,
                    signal,
                    f"job={job};cluster={cluster};owner={self.owner};restart={restart}",
                    batch=batch,
                    **(
                        {"native_identity": native}
                        if isinstance(handle, SubmissionHandle)
                        else {}
                    ),
                )
            except ValueError as error:
                answer[handle.job_id] = _observation(
                    handle,
                    ObservationStatus.ERROR,
                    detail=str(error)[:512],
                    **(
                        {"native_identity": native}
                        if isinstance(handle, SubmissionHandle)
                        else {}
                    ),
                )
        return answer

    def collect_submission(
        self, handle: SubmissionHandle
    ) -> SubmissionCollectionResult | None:
        """Propose exact terminal native-generation evidence without allocating."""
        from wavedl.core.submission_outcomes import read_submission_outcome_receipt

        if not isinstance(handle, SubmissionHandle):
            raise ValueError("Submission collection requires an unbound native receipt")
        try:
            job, cluster, _, saved_restart = self._identity(handle)
            observation = self.observe_submission(handle)
            identity = observation.native_identity
            if observation.status != ObservationStatus.TERMINAL or identity is None:
                return None
            if (
                identity.run_id != handle.run_id
                or identity.request_id != handle.request_id
                or identity.job_id != job
                or identity.cluster != cluster
                or identity.owner != self.owner
                or identity.restart_count < saved_restart
            ):
                return None
            request, approved = self._verify_writer_plan(handle)
            if (
                identity.restart_count
                and dict(approved.resources).get("requeue") is not True
            ):
                return None
            receipt = read_submission_outcome_receipt(
                Path(handle.run_dir),
                identity,
                plan_sha256=request.plan_sha256,
                attempt=None,
            )
            outcomes = [
                OperationOutcome(
                    **(
                        receipt[name]
                        | {"status": OutcomeStatus(receipt[name]["status"])}
                    )
                )
                for name in ("runtime", "batch", "stage_out")
            ]
            return SubmissionCollectionResult(
                identity,
                request.plan_sha256,
                request.manifest_sha256,
                OperationResult(approved.operation, *outcomes),
                observation,
            )
        except (OSError, ValueError, TypeError, KeyError, RecursionError):
            return None

    def collect(self, handle: Handle | SubmissionHandle) -> OperationResult:
        job, cluster, _, restart = self._identity(handle)
        raw = json.loads(handle.raw_json)
        operation = OperationKind(raw.get("operation", "train"))
        unknown = OperationOutcome(
            OutcomeStatus.UNKNOWN, detail="No validated attempt outcome receipt"
        )
        if raw.get("allocation_mode") == "writer":
            return self._collect_writer(handle, operation, unknown)
        observation = self.observe(handle)
        batch = observation.batch or unknown
        if observation.status != ObservationStatus.TERMINAL:
            unavailable = OperationOutcome(
                OutcomeStatus.UNKNOWN,
                detail="Current terminal scheduler execution is not verified",
            )
            return OperationResult(operation, unavailable, batch, unavailable)
        filename = receipt_name(restart)
        try:
            # A fixed-size receipt avoids rescanning a potentially enormous journal.
            with (
                open_directory(handle.run_dir) as root,
                root.open_directory("attempts") as attempts,
                attempts.open_directory(str(handle.attempt)) as attempt,
                attempt.open_file(filename) as stream,
            ):
                for anchor in (root, attempts, attempt, stream):
                    anchor.verify_private_owner()
                    anchor.guard()
                if stream.stat().st_size > MAX_RECEIPT_BYTES:
                    raise ValueError("Outcome receipt must be a bounded regular file")
                content = stream.read(MAX_RECEIPT_BYTES + 1)
                for anchor in (stream, attempt, attempts, root):
                    anchor.verify_private_owner()
                    anchor.guard()
            if len(content) > MAX_RECEIPT_BYTES:
                raise ValueError("Outcome receipt exceeds byte bound")

            def unique(pairs):
                value = {}
                for key, item in pairs:
                    if key in value:
                        raise ValueError("Duplicate outcome receipt field")
                    value[key] = item
                return value

            # Shared finite-number decoding plus duplicate rejection at every
            # object depth. Neither ambiguous identity nor NaN is evidence.
            receipt = strict_loads(content)
            json.loads(content, object_pairs_hook=unique)
            expected = {
                "schema_version": 1,
                "run_id": handle.run_id,
                "request_id": handle.request_id,
                "attempt": handle.attempt,
                "job_id": job,
                "cluster": cluster,
                "restart_count": restart,
                "operation": operation.value,
            }
            if not isinstance(receipt, dict) or any(
                type(receipt.get(key)) is not type(value) or receipt.get(key) != value
                for key, value in expected.items()
            ):
                raise ValueError("Outcome receipt identity mismatch")
            outcomes = []
            for name in ("runtime", "batch", "stage_out"):
                value = receipt[name]
                if not isinstance(value, dict) or set(value) != {
                    "status",
                    "exit_code",
                    "signal",
                    "detail",
                }:
                    raise ValueError("Invalid outcome receipt fields")
                outcomes.append(
                    OperationOutcome(
                        **(value | {"status": OutcomeStatus(value["status"])})
                    )
                )
            # Native overall/epilogue failure remains separate observation
            # evidence; never overwrite the wrapper's independent command exits.
            return OperationResult(operation, *outcomes)
        except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
            diagnostic = OperationOutcome(
                OutcomeStatus.UNKNOWN,
                detail=f"Receipt unavailable: {type(error).__name__}",
            )
            return OperationResult(operation, diagnostic, batch, diagnostic)

    def _collect_writer(self, handle, operation, unknown):
        from wavedl.core.submission_outcomes import read_submission_outcome_receipt

        submission = (
            handle
            if isinstance(handle, SubmissionHandle)
            else SubmissionHandle(
                **{
                    key: value
                    for key, value in handle.to_dict().items()
                    if key != "attempt"
                }
            )
        )
        batch = unknown
        try:
            observation = self.observe(submission)
            batch = observation.batch or unknown
            if observation.status != ObservationStatus.TERMINAL:
                return OperationResult(
                    operation, unknown, observation.batch or unknown, unknown
                )
            if observation.native_identity is None:
                raise ValueError("Collected writer native identity unavailable")
            restart = observation.native_identity.restart_count
            job, cluster, _, saved_restart = self._identity(handle)
            # OperationResult has no native-generation field. Until collection
            # has a typed generation-scoped receipt, even an unbound ACK must
            # not attribute a later setup failure to its saved generation.
            if saved_restart != restart:
                raise ValueError(
                    "Collected writer execution differs from saved receipt generation"
                )
            identity = SchedulerIdentity(
                handle.run_id, handle.request_id, cluster, job, self.owner, restart
            )
            self._verify_writer_plan(handle)
            receipt = read_submission_outcome_receipt(
                Path(handle.run_dir),
                identity,
                plan_sha256=json.loads(handle.raw_json)["plan_sha256"],
                attempt=handle.attempt if isinstance(handle, Handle) else None,
            )
            outcomes = [
                OperationOutcome(
                    **(
                        receipt[name]
                        | {"status": OutcomeStatus(receipt[name]["status"])}
                    )
                )
                for name in ("runtime", "batch", "stage_out")
            ]
            return OperationResult(operation, *outcomes)
        except (OSError, ValueError, TypeError, KeyError, RecursionError):
            return OperationResult(operation, unknown, batch, unknown)
