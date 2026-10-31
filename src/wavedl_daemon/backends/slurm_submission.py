# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Fail-closed Slurm submission, identity recovery and cancellation requests."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from wavedl.core.layout import RunLayout
from wavedl.core.manifest import write_manifest
from wavedl.core.persistence import write_once
from wavedl.core.safe_files import open_directory
from wavedl.core.submission import (
    create_submission,
    identity_for_submission,
    publish_submission_file,
    read_submission_file,
)
from wavedl_daemon.backends.contracts import (
    Handle,
    Plan,
    SubmissionHandle,
    manifest_from_dict,
)
from wavedl_daemon.backends.slurm_plan import bind_submission, job_name
from wavedl_daemon.execution.budget import BudgetExhausted
from wavedl_daemon.storage.ownership import require_dispatch


if TYPE_CHECKING:
    from wavedl_daemon.backends.scheduler import CommandRunner
    from wavedl_daemon.storage.ownership import DispatchLock
    from wavedl_daemon.supervision.base import ProcessResult


class SubmissionUnknown(RuntimeError):
    """Acceptance cannot be established; automatic resubmission is prohibited."""


class SubmissionAmbiguous(SubmissionUnknown):
    """Multiple jobs match the same complete submission identity."""


class SubmissionRejected(RuntimeError):
    """The scheduler returned a definite submission failure."""


class SchedulerUnavailable(RuntimeError):
    """A recovery query failed; this is not evidence that a job is absent."""


class CancellationError(RuntimeError):
    """The scheduler did not acknowledge the cancellation request."""


_JOB = re.compile(r"[1-9][0-9]*(?:;[A-Za-z0-9_.-]+)?\Z")


def _read_bounded(path: Path, limit: int) -> bytes:
    """Refuse links, FIFOs and oversized files before reading persisted input."""
    if type(limit) is not int or limit < 0:
        raise ValueError("Submission byte bound must be nonnegative")
    path = Path(path)
    with open_directory(path.parent) as parent:
        parent.verify_private_owner()
        parent.guard()
        with parent.open_file(path.name) as stream:
            stream.verify_private_owner()
            stream.guard()
            if stream.stat().st_size > limit:
                raise BudgetExhausted("Submission input exceeds byte budget")
            payload = bytearray()
            while len(payload) <= limit:
                chunk = stream.read(min(65536, limit + 1 - len(payload)))
                if not chunk:
                    break
                payload.extend(chunk)
            if len(payload) > limit:
                raise BudgetExhausted("Submission input exceeds byte budget")
            stream.verify_private_owner()
            parent.verify_private_owner()
            stream.guard()
            parent.guard()
            return bytes(payload)


def _identity(plan: Plan, request_id: str, attempt: int | None) -> None:
    if plan.target != "slurm" or plan.script is None:
        raise ValueError("Submission requires a Slurm script plan")
    if str(UUID(request_id)) != request_id:
        raise ValueError("request_id must be a canonical full UUID")
    if plan.allocation_mode == "writer":
        if attempt is not None:
            raise ValueError("Writer submission cannot reserve an attempt")
    elif type(attempt) is not int or attempt < 1:
        raise ValueError("attempt must be positive")


def _incomplete(result: ProcessResult) -> bool:
    return (
        result.timed_out
        or result.cancelled
        or result.stdout_truncated
        or result.stderr_truncated
    )


def _cluster(plan: Plan) -> str:
    cluster = dict(plan.resources).get("cluster")
    if (
        not isinstance(cluster, str)
        or re.fullmatch(r"[A-Za-z0-9_.-]+", cluster) is None
    ):
        raise ValueError("Plan cluster must identify a single cluster")
    return cluster


class SlurmSubmission:
    """Mutations require dispatch ownership; read-only recovery does not."""

    def __init__(
        self,
        runner: CommandRunner,
        *,
        owner: str,
        dispatch_lock: DispatchLock | None = None,
        max_lookback_days: int = 7,
    ):
        if (
            not isinstance(owner, str)
            or not owner
            or any(character in owner for character in "\0\n\r|,")
        ):
            raise ValueError("Invalid scheduler owner")
        self.runner = runner
        self.owner = owner
        self.dispatch_lock = dispatch_lock
        if type(max_lookback_days) is not int or max_lookback_days < 1:
            raise ValueError("max_lookback_days must be positive")
        self.max_lookback_days = max_lookback_days

    def _handle(
        self, plan: Plan, request_id: str, attempt: int | None, job_id: str, **raw
    ) -> Handle | SubmissionHandle:
        fields = {} if attempt is None else {"attempt": attempt}
        if attempt is None:
            raw.update(
                allocation_mode="writer",
                plan_sha256=hashlib.sha256(plan.to_json().encode()).hexdigest(),
            )
        return (SubmissionHandle if attempt is None else Handle)(
            backend="slurm",
            run_id=plan.run_id,
            request_id=request_id,
            **fields,
            owner=self.owner,
            run_dir=plan.run_dir,
            job_id=job_id,
            raw_json=json.dumps(
                raw
                | {
                    "cluster": job_id.partition(";")[2] or None,
                    "job_name": job_name(request_id),
                    "comment": request_id,
                    "restart_count": 0,
                    "operation": plan.operation.value,
                }
            ),
        )

    def submit(
        self, plan: Plan, *, request_id: str, attempt: int | None
    ) -> Handle | SubmissionHandle:
        require_dispatch(self.dispatch_lock)
        _identity(plan, request_id, attempt)
        cluster = _cluster(plan)
        if (
            len(plan.manifest_json.encode()) > 1048576
            or len(plan.script.encode()) > 1048576
        ):
            raise BudgetExhausted("Slurm plan exceeds byte budget")
        script = bind_submission(plan, request_id, attempt)
        script_bytes = script.encode()
        if len(script_bytes) > 1048576:
            raise BudgetExhausted("Slurm script exceeds byte budget")
        layout = RunLayout(Path(plan.run_dir), plan.run_id)
        layout.run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if layout.manifest.exists():
            if (
                manifest_from_dict(json.loads(_read_bounded(layout.manifest, 1048576)))
                != plan.manifest
            ):
                raise ValueError("Existing manifest differs from submission plan")
        else:
            write_manifest(layout, plan.manifest)
        (layout.run_dir / "logs").mkdir(exist_ok=True)
        if attempt is None:
            identity = identity_for_submission(
                layout.run_dir,
                request_id=request_id,
                backend="slurm",
                owner=self.owner,
                plan_sha256=hashlib.sha256(plan.to_json().encode()).hexdigest(),
            )
            try:
                create_submission(layout.run_dir, identity)
            except FileExistsError as exc:
                raise SubmissionUnknown(
                    "Submission request is already reserved"
                ) from exc
            publish_submission_file(
                layout.run_dir, identity, "plan.json", plan.to_json().encode()
            )
            publish_submission_file(
                layout.run_dir,
                identity,
                "intent.json",
                json.dumps({"created_at": datetime.now(UTC).isoformat()}).encode(),
            )
            script_path = publish_submission_file(
                layout.run_dir, identity, "submission.sh", script_bytes
            )
        else:
            script_path = self._reserve_legacy(
                layout, request_id, attempt, script_bytes
            )
        require_dispatch(self.dispatch_lock)
        argv = ("sbatch", "--parsable", f"--comment={request_id}")
        if cluster:
            argv += (f"--clusters={cluster}",)
        try:
            result = self.runner.run((*argv, str(script_path)))
        except Exception as exc:
            raise SubmissionUnknown("sbatch outcome is unknown") from exc
        if _incomplete(result):
            raise SubmissionUnknown("Incomplete sbatch acknowledgement")
        if result.returncode != 0:
            raise SubmissionRejected(result.stderr.decode("utf-8", errors="replace"))
        try:
            acknowledgement = result.stdout.decode("utf-8").strip()
        except UnicodeError as exc:
            raise SubmissionUnknown("Invalid sbatch acknowledgement") from exc
        if _JOB.fullmatch(acknowledgement) is None:
            raise SubmissionUnknown("Unparseable sbatch acknowledgement")
        job_id, _, acknowledged_cluster = acknowledgement.partition(";")
        if cluster and acknowledged_cluster and cluster != acknowledged_cluster:
            raise SubmissionUnknown("sbatch acknowledgement has a different cluster")
        identity = (
            f"{job_id};{acknowledged_cluster or cluster}"
            if (acknowledged_cluster or cluster)
            else job_id
        )
        return self._handle(
            plan, request_id, attempt, identity, acknowledgement=acknowledgement
        )

    def _reserve_legacy(self, layout, request_id, attempt, script_bytes):
        attempt_dir = layout.attempt_dir(attempt)
        attempt_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        (attempt_dir / "rank").mkdir(exist_ok=True)
        script_path = attempt_dir / "submission.sh"
        # Exclusive publication reserves this invocation. An abandoned file is
        # never rewritten and can only be reconciled, not automatically retried.
        try:
            write_once(
                attempt_dir / "submission.json",
                json.dumps(
                    {
                        "request_id": request_id,
                        "attempt": attempt,
                        "owner": self.owner,
                        "created_at": datetime.now(UTC).isoformat(),
                    }
                ).encode(),
            )
        except FileExistsError as exc:
            raise SubmissionUnknown("Submission attempt is already reserved") from exc
        write_once(script_path, script_bytes)
        return script_path

    def find_by_request_id(
        self, plan: Plan, *, request_id: str, attempt: int | None
    ) -> Handle | SubmissionHandle | None:
        """Search live and retained accounting records, never authorize a retry."""
        _identity(plan, request_id, attempt)
        cluster = _cluster(plan)
        selection = (f"--clusters={cluster}",)
        start = datetime.now(UTC) - timedelta(days=self.max_lookback_days)
        intent = Path(plan.run_dir) / "attempts" / str(attempt) / "submission.json"
        if attempt is None:
            try:
                identity = identity_for_submission(
                    Path(plan.run_dir),
                    request_id=request_id,
                    backend="slurm",
                    owner=self.owner,
                    plan_sha256=hashlib.sha256(plan.to_json().encode()).hexdigest(),
                )
                if (
                    read_submission_file(Path(plan.run_dir), identity, "plan.json")
                    != plan.to_json().encode()
                ):
                    raise ValueError("Recovery plan differs from submission")
                saved = json.loads(
                    read_submission_file(
                        Path(plan.run_dir), identity, "intent.json", limit=16384
                    )
                )
                created = datetime.fromisoformat(saved["created_at"])
                if created.tzinfo is None:
                    raise ValueError("Submission timestamp must have a timezone")
                start = max(start, created - timedelta(minutes=5))
            except (ValueError, KeyError, OSError, TypeError) as exc:
                raise SchedulerUnavailable(
                    "Cannot read submission recovery intent"
                ) from exc
        elif intent.exists():
            try:
                saved = json.loads(_read_bounded(intent, 16384))
                if (saved["request_id"], saved["attempt"], saved["owner"]) != (
                    request_id,
                    attempt,
                    self.owner,
                ):
                    raise ValueError("Submission identity differs")
                created = datetime.fromisoformat(saved["created_at"])
                if created.tzinfo is None:
                    raise ValueError("Submission timestamp must have a timezone")
                start = max(start, created - timedelta(minutes=5))
            except (ValueError, KeyError, OSError, TypeError, BudgetExhausted) as exc:
                raise SchedulerUnavailable(
                    "Cannot read submission recovery intent"
                ) from exc
        commands = (
            (
                "squeue",
                "--noheader",
                f"--user={self.owner}",
                "--Format=JobID:100|,UserName:256|,Name:256|,Comment:256|,Cluster:256",
                *selection,
            ),
            (
                "sacct",
                "--noheader",
                "--parsable2",
                "--allocations",
                "--duplicates",
                f"--user={self.owner}",
                f"--starttime={start.astimezone(UTC).strftime('%Y-%m-%dT%H:%M:%S')}",
                "--format=JobIDRaw,User%256,JobName%256,Comment%256,Cluster%256",
                *((f"--clusters={cluster}",) if cluster else ()),
            ),
        )
        matches: dict[str, str] = {}
        for argv in commands:
            try:
                result = self.runner.run(argv)
                if _incomplete(result) or result.returncode != 0:
                    raise SchedulerUnavailable(f"{argv[0]} query unavailable")
                output = result.stdout.decode("utf-8")
            except SchedulerUnavailable:
                raise
            except Exception as exc:
                raise SchedulerUnavailable(f"{argv[0]} query unavailable") from exc
            for line in output.splitlines():
                if not line.strip():
                    continue
                fields = [field.strip() for field in line.split("|")]
                if len(fields) != 5:
                    raise SchedulerUnavailable(f"Malformed {argv[0]} recovery row")
                job, owner, name, comment, found_cluster = fields
                if (owner, name, comment) != (
                    self.owner,
                    job_name(request_id),
                    request_id,
                ):
                    continue
                if cluster is not None and found_cluster != cluster:
                    continue
                identity = f"{job};{found_cluster}" if found_cluster else job
                if _JOB.fullmatch(identity) is None:
                    raise SchedulerUnavailable("Invalid recovered job identity")
                matches[identity] = line
        if len(matches) > 1:
            raise SubmissionAmbiguous("Multiple jobs match the full request identity")
        if not matches:
            return None
        identity, row = next(iter(matches.items()))
        return self._handle(plan, request_id, attempt, identity, recovery_row=row)

    def request_cancel(self, handle: Handle | SubmissionHandle) -> None:
        require_dispatch(self.dispatch_lock)
        if handle.backend != "slurm" or handle.owner != self.owner:
            raise ValueError("Cannot cancel a foreign backend or owner")
        if handle.job_id is None or _JOB.fullmatch(handle.job_id) is None:
            raise ValueError("Invalid Slurm job identity")
        UUID(handle.request_id)
        try:
            raw = json.loads(handle.raw_json)
        except (TypeError, ValueError) as exc:
            raise ValueError("Incomplete Slurm handle identity") from exc
        if not isinstance(raw, dict):
            raise ValueError("Incomplete Slurm handle identity")
        job, _, cluster = handle.job_id.partition(";")
        if (
            raw.get("comment") != handle.request_id
            or raw.get("job_name") != job_name(handle.request_id)
            or raw.get("cluster") != (cluster or None)
        ):
            raise ValueError("Incomplete Slurm handle identity")
        # A purged/reused numeric ID must not cancel somebody else's new job.
        try:
            result = self.runner.run(
                (
                    "squeue",
                    "--noheader",
                    f"--jobs={job}",
                    f"--clusters={cluster}",
                    "--Format=JobID:100|,UserName:256|,Name:256|,Comment:256|,Cluster:256",
                )
            )
            if _incomplete(result) or result.returncode != 0:
                raise CancellationError("Cancellation identity query unavailable")
            rows = [
                tuple(field.strip() for field in line.split("|"))
                for line in result.stdout.decode("utf-8").splitlines()
                if line.strip()
            ]
            if rows != [
                (
                    job,
                    self.owner,
                    job_name(handle.request_id),
                    handle.request_id,
                    cluster,
                )
            ]:
                raise CancellationError(
                    "Live cancellation identity is absent or differs"
                )
        except CancellationError:
            raise
        except Exception as exc:
            raise CancellationError("Cancellation identity query unavailable") from exc
        require_dispatch(self.dispatch_lock)
        argv = (
            "scancel",
            f"--user={self.owner}",
            f"--name={job_name(handle.request_id)}",
        )
        if cluster:
            argv += (f"--clusters={cluster}",)
        try:
            result = self.runner.run((*argv, job))
        except Exception as exc:
            raise CancellationError("scancel outcome is unknown") from exc
        if _incomplete(result) or result.returncode != 0:
            raise CancellationError("scancel request was not acknowledged")
