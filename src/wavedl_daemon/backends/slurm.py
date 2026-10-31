# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Slurm backend composition: preview, identity recovery and native evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .contracts import Topology
from .slurm_observation import SlurmObservation
from .slurm_plan import SlurmPlanner, SlurmSite
from .slurm_submission import SlurmSubmission


if TYPE_CHECKING:
    from wavedl.core.manifest import Manifest
    from wavedl.core.scheduler_attempt import SchedulerBinding
    from wavedl_daemon.storage.ownership import DispatchLock

    from .contracts import (
        BackendObservation,
        Handle,
        OperationResult,
        Plan,
        SubmissionCollectionResult,
        SubmissionHandle,
        SubmissionObservation,
    )
    from .scheduler import CommandRunner
    from .slurm_observation import SlurmCapabilities


class SlurmBackend:
    """Submit only approved immutable plans under matching dispatch ownership."""

    name = "slurm"

    def __init__(
        self,
        site: SlurmSite,
        runner: CommandRunner,
        *,
        owner: str,
        dispatch_lock: DispatchLock | None = None,
    ):
        self.owner = owner
        self.dispatch_lock = dispatch_lock
        self._planner = SlurmPlanner(site)
        self._submission = SlurmSubmission(
            runner, owner=owner, dispatch_lock=dispatch_lock
        )
        self._observation = SlurmObservation(runner, owner=owner)

    def plan(
        self,
        manifest: Manifest,
        *,
        topology: Topology = Topology(),
        arguments: tuple[str, ...] = (),
        allocation_mode: str = "reserved",
    ) -> Plan:
        return self._planner.plan(
            manifest,
            topology=topology,
            arguments=arguments,
            allocation_mode=allocation_mode,
        )

    def submit(
        self, plan: Plan, *, request_id: str, attempt: int | None
    ) -> Handle | SubmissionHandle:
        return self._submission.submit(plan, request_id=request_id, attempt=attempt)

    def find_by_request_id(
        self, plan: Plan, *, request_id: str, attempt: int | None
    ) -> Handle | SubmissionHandle | None:
        return self._submission.find_by_request_id(
            plan, request_id=request_id, attempt=attempt
        )

    def observe(
        self, handle: Handle | SubmissionHandle
    ) -> BackendObservation | SubmissionObservation:
        return self._observation.observe(handle)

    def discover_initial(self, handle: SubmissionHandle) -> SchedulerBinding | None:
        return self._observation.discover_initial(handle)

    def discover_restart(
        self, handle: Handle
    ) -> tuple[SchedulerBinding, Handle] | None:
        """Return a native-proven proposal; controller CAS activation is separate."""
        return self._observation.discover_restart(handle)

    def observe_many(
        self, handles: tuple[Handle | SubmissionHandle, ...]
    ) -> dict[str, BackendObservation | SubmissionObservation]:
        return self._observation.observe_many(handles)

    def request_cancel(self, handle: Handle | SubmissionHandle) -> None:
        self._submission.request_cancel(handle)

    def collect(self, handle: Handle | SubmissionHandle) -> OperationResult:
        return self._observation.collect(handle)

    def collect_submission(
        self, handle: SubmissionHandle
    ) -> SubmissionCollectionResult | None:
        return self._observation.collect_submission(handle)

    def capabilities(self) -> SlurmCapabilities:
        return self._observation.capabilities()
