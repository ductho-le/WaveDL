# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""The Slurm backend, being rebuilt on the run log; it refuses every plan."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .contracts import UnsupportedPlan
from .slurm_plan import SlurmSite


if TYPE_CHECKING:
    from wavedl_daemon.storage.ownership import DispatchLock

    from .contracts import Plan, RunSpec, Topology
    from .scheduler import CommandRunner


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
        if not isinstance(site, SlurmSite):
            raise UnsupportedPlan("Explicit SlurmSite configuration is required")
        self.site = site
        self.runner = runner
        self.owner = owner
        self.dispatch_lock = dispatch_lock

    def plan(self, spec: RunSpec, *, topology: Topology) -> Plan:
        raise UnsupportedPlan("Slurm is being rebuilt; use the Local backend for now")
