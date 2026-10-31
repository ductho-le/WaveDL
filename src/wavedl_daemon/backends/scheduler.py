# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Shell-free scheduler commands with bounded output and real deadlines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from wavedl_daemon.execution.budget import BudgetExhausted, WorkBudget
from wavedl_daemon.execution.commands import run_command
from wavedl_daemon.supervision.base import ProcessResult, ProcessSpec


if TYPE_CHECKING:
    from collections.abc import Mapping

    from wavedl_daemon.supervision.base import Supervisor


class CommandRunner(Protocol):
    def run(self, argv: tuple[str, ...]) -> ProcessResult:
        """Return bounded command evidence, including truncation and timeout."""
        ...


SchedulerRunner = CommandRunner
_COMMANDS = frozenset({"sbatch", "scancel", "squeue", "sacct", "scontrol"})
_OPTION_PREFIXES = ("SBATCH_", "SCANCEL_", "SQUEUE_", "SACCT_", "SCONTROL_")
_SITE_VARIABLES = frozenset({"SLURM_CONF", "SLURM_CONF_SERVER", "SLURM_JWT"})


class SchedulerCommands:
    """Synchronous adapter; event-loop owners must admit work through a pool.

    Executables and environment are administrator configuration, not HTTP input.
    Missing programs and unavailable schedulers are failures, never absence.
    """

    def __init__(
        self,
        *,
        cwd: Path,
        budget: WorkBudget | None = None,
        executables: Mapping[str, str] | None = None,
        environment: Mapping[str, str] | None = None,
        supervisor: Supervisor | None = None,
    ):
        self.cwd = Path(cwd).resolve()
        self.budget = budget or WorkBudget()
        self.executables = dict(executables or {})
        if set(self.executables) - _COMMANDS or any(
            not isinstance(value, str) or not value or "\0" in value
            for value in self.executables.values()
        ):
            raise ValueError("Invalid scheduler executable configuration")
        # Slurm option variables override script directives. Site configuration
        # such as SLURM_CONF remains available, but invocation options cannot
        # silently change an approved plan or cancellation/query selection.
        self.environment = {
            key: value
            for key, value in (
                os.environ if environment is None else environment
            ).items()
            if not key.upper().startswith(_OPTION_PREFIXES)
            and (not key.upper().startswith("SLURM_") or key.upper() in _SITE_VARIABLES)
        }
        self.environment["LC_ALL"] = "C"
        self.environment["TZ"] = "UTC"
        self.supervisor = supervisor

    def run(self, argv: tuple[str, ...]) -> ProcessResult:
        if not isinstance(argv, tuple) or not argv or argv[0] not in _COMMANDS:
            raise ValueError("Unsupported scheduler command")
        if any(not isinstance(arg, str) or "\0" in arg for arg in argv):
            raise ValueError("Scheduler arguments must be NUL-free strings")
        if sum(len(arg.encode()) + 1 for arg in argv) > self.budget.max_command_bytes:
            raise BudgetExhausted("scheduler command exceeds byte budget")
        command = (self.executables.get(argv[0], argv[0]), *argv[1:])
        return run_command(
            ProcessSpec(command, self.cwd, self.environment, inherit_environment=False),
            self.budget,
            supervisor=self.supervisor,
        )
