# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""The Slurm backend: submit a planned batch script, find, observe and cancel it.

A job is named ``wavedl-<run_id>`` with the run ID as its comment, so the
owner, name and comment identify it in every scheduler source. Submission is
never retried: an unknown sbatch outcome is recovered with ``find``.
"""

from __future__ import annotations

import re
from datetime import UTC, timedelta
from typing import TYPE_CHECKING

from wavedl_daemon.runs.model import ObservationFailed, SlurmHandle
from wavedl_daemon.storage.ownership import require_dispatch

from .contracts import UnsupportedPlan
from .slurm_observation import observe_cluster
from .slurm_plan import SlurmPlanner, SlurmSite, job_name


if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from wavedl_daemon.runs.model import RunId, SlurmObservation
    from wavedl_daemon.storage.ownership import DispatchLock

    from .contracts import Plan, RunSpec, Topology
    from .scheduler import CommandRunner


_ACK = re.compile(r"([1-9][0-9]*)(?:;([A-Za-z0-9_.-]+))?\Z")
# Accounting rows may be stamped by a clock slightly behind the daemon's.
_CLOCK_SKEW = timedelta(minutes=5)


class SubmissionRejected(RuntimeError):
    """sbatch definitely refused the job."""


class SubmissionUnknown(RuntimeError):
    """sbatch may have queued the job; only ``find`` may settle it."""


class SchedulerUnavailable(RuntimeError):
    """A query failed; this is not evidence that a job is absent."""


class SlurmBackend:
    name = "slurm"

    def __init__(
        self,
        site: SlurmSite,
        runner: CommandRunner,
        *,
        owner: str,
        dispatch_lock: DispatchLock | None = None,
    ):
        if not isinstance(owner, str) or not re.fullmatch(
            r"[A-Za-z0-9_][A-Za-z0-9_.-]*", owner
        ):
            raise ValueError("Invalid scheduler owner")
        self._planner = SlurmPlanner(site)
        self.runner = runner
        self.owner = owner
        self.dispatch_lock = dispatch_lock

    def plan(self, spec: RunSpec, *, topology: Topology) -> Plan:
        return self._planner.plan(spec, topology=topology)

    def _query(self, argv: tuple[str, ...]) -> str:
        """Stdout of a complete, successful command, else ValueError."""
        try:
            result = self.runner.run(argv)
        except (RuntimeError, OSError) as error:
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
        try:
            return result.stdout.decode("utf-8")
        except UnicodeError as error:
            raise ValueError(f"{argv[0]} output is not UTF-8") from error

    def submit(self, plan: Plan) -> SlurmHandle:
        """Write ``submission.sh`` once and sbatch it. Blocking: call from a worker."""
        require_dispatch(self.dispatch_lock)
        cluster = dict(plan.resources).get("cluster")
        if plan.target != self.name or plan.script is None or not isinstance(cluster, str):
            raise UnsupportedPlan("A Slurm script plan with a cluster is required")
        paths = plan.spec.paths
        for directory in (paths.run_dir, paths.output, paths.attempt(1).parent, paths.logs):
            directory.mkdir(mode=0o700, exist_ok=True)
        with paths.submission_script.open("x", encoding="utf-8") as stream:
            stream.write(plan.script)
        require_dispatch(self.dispatch_lock)
        argv = ("sbatch", "--parsable", f"--clusters={cluster}", str(paths.submission_script))
        try:
            result = self.runner.run(argv)
        except Exception as error:
            raise SubmissionUnknown("sbatch outcome is unknown") from error
        if (
            result.timed_out
            or result.cancelled
            or result.stdout_truncated
            or result.stderr_truncated
        ):
            raise SubmissionUnknown("Incomplete sbatch acknowledgement")
        if result.returncode != 0:
            raise SubmissionRejected(
                result.stderr.decode("utf-8", errors="replace").strip()[:1000]
            )
        match = _ACK.fullmatch(result.stdout.decode("utf-8", errors="replace").strip())
        if match is None:
            raise SubmissionUnknown("Unparseable sbatch acknowledgement")
        job, acknowledged = match.groups()
        if acknowledged not in (None, cluster):
            raise SubmissionUnknown("sbatch acknowledged a different cluster")
        return SlurmHandle(job, cluster, self.owner)

    def find(self, plan: Plan, *, created_at: datetime) -> SlurmHandle | None:
        """The job a lost submission created, if any. Never authorizes a retry.

        Searches the live queue and accounting since the plan was created, by
        owner, job name and comment.
        """
        cluster = dict(plan.resources).get("cluster")
        if not isinstance(cluster, str):
            raise UnsupportedPlan("A Slurm plan with a cluster is required")
        name, comment = job_name(plan.run_id), plan.run_id
        since = (created_at - _CLOCK_SKEW).astimezone(UTC)
        commands = (
            (
                "squeue",
                "--noheader",
                f"--user={self.owner}",
                "--Format=JobID:100|,UserName:256|,Name:256|,Comment:256|,Cluster:256",
                f"--clusters={cluster}",
            ),
            (
                "sacct",
                "--noheader",
                "--parsable2",
                "--allocations",
                "--duplicates",
                f"--user={self.owner}",
                f"--starttime={since.strftime('%Y-%m-%dT%H:%M:%S')}",
                "--format=JobIDRaw,User%256,JobName%256,Comment%256,Cluster%256",
                f"--clusters={cluster}",
            ),
        )
        jobs = set()
        for argv in commands:
            try:
                output = self._query(argv)
            except ValueError as error:
                raise SchedulerUnavailable(str(error)) from error
            for line in output.splitlines():
                if not line.strip():
                    continue
                fields = [field.strip() for field in line.split("|")]
                if len(fields) != 5:
                    raise SchedulerUnavailable(f"Malformed {argv[0]} recovery row")
                job, owner, found_name, found_comment, found_cluster = fields
                if (owner, found_name, found_comment, found_cluster) != (
                    self.owner,
                    name,
                    comment,
                    cluster,
                ):
                    continue
                if not _ACK.fullmatch(job):
                    raise SchedulerUnavailable("Invalid recovered job ID")
                jobs.add(job)
        if len(jobs) > 1:
            raise SchedulerUnavailable("Several jobs carry this run's identity")
        return SlurmHandle(jobs.pop(), cluster, self.owner) if jobs else None

    def observe_many(
        self, targets: Sequence[tuple[RunId, SlurmHandle]]
    ) -> dict[RunId, SlurmObservation | ObservationFailed]:
        """Observe jobs in one batch of queries per cluster. Blocking."""
        answer: dict[RunId, SlurmObservation | ObservationFailed] = {}
        clusters: dict[str, dict[str, RunId]] = {}
        for run_id, handle in targets:
            if handle.owner != self.owner:
                answer[run_id] = ObservationFailed("The job belongs to another owner")
            else:
                clusters.setdefault(handle.cluster, {})[handle.job_id] = run_id
        for cluster, jobs in clusters.items():
            try:
                observed = observe_cluster(
                    self._query, owner=self.owner, cluster=cluster, jobs=jobs
                )
            except (ValueError, UnicodeError) as error:
                failed = ObservationFailed(str(error)[:512])
                observed = dict.fromkeys(jobs, failed)
            answer.update((jobs[job], result) for job, result in observed.items())
        return answer

    def cancel(self, run_id: RunId, handle: SlurmHandle, *, interrupt: bool) -> None:
        """Interrupt the job's steps, or end the job. Blocking.

        The name and user filters keep a reused job ID from cancelling another
        job.
        """
        require_dispatch(self.dispatch_lock)
        if handle.owner != self.owner:
            raise ValueError("Cannot cancel another owner's job")
        argv = (
            "scancel",
            f"--clusters={handle.cluster}",
            f"--user={self.owner}",
            f"--name={job_name(run_id)}",
        )
        # No --full/--batch: SIGINT reaches the srun steps, not the batch shell that writes the exit record.
        if interrupt:
            argv += ("--signal=INT",)
        try:
            self._query((*argv, handle.job_id))
        except ValueError as error:
            raise SchedulerUnavailable(str(error)) from error
