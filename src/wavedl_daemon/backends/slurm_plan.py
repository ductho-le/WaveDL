# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Site configuration for Slurm plans: literal scheduler tokens and trusted hooks."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .contracts import UnsupportedPlan


_TOKEN = re.compile(r"[A-Za-z0-9_.:+,=/\-]+\Z")


def _text(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or any(char in value for char in ("\0", "\r", "\n"))
    ):
        raise UnsupportedPlan(f"Invalid {name}")


@dataclass(frozen=True)
class SlurmSite:
    """Explicit site resources and trusted administrator shell hooks.

    Hooks are shell programs, not user-provided training arguments. Container
    and bundled runtimes are deliberately refused until their contracts exist.
    Memory is per node; CPUs are per launcher task (one task per machine).
    """

    partition: str
    account: str | None = None
    time: str = "01:00:00"
    cpus_per_task: int = 1
    memory: str = "4G"
    gpus_per_node: int = 0
    constraint: str | None = None
    cluster: str | None = None
    rendezvous_port: int = 29500
    setup_commands: tuple[str, ...] = ()
    stage_out_command: str | None = None
    requeue: bool = False

    def __post_init__(self):
        if type(self.requeue) is not bool:
            raise UnsupportedPlan("requeue must be explicitly boolean")
        for name in ("partition", "account", "time", "memory", "constraint", "cluster"):
            value = getattr(self, name)
            if value is None and name in {"account", "constraint", "cluster"}:
                continue
            _text(value, name)
            if not _TOKEN.fullmatch(value):
                raise UnsupportedPlan(f"Unsupported scheduler token: {name}")
        if not re.fullmatch(r"(?:\d+-)?\d+:[0-5]\d:[0-5]\d", self.time):
            raise UnsupportedPlan("time must be [days-]hours:minutes:seconds")
        if not re.fullmatch(r"[1-9]\d*[KMGT]?", self.memory):
            raise UnsupportedPlan("memory must be a positive Slurm memory quantity")
        if type(self.cpus_per_task) is not int or self.cpus_per_task < 1:
            raise UnsupportedPlan("cpus_per_task must be positive")
        if type(self.gpus_per_node) is not int or self.gpus_per_node < 0:
            raise UnsupportedPlan("gpus_per_node must be nonnegative")
        if (
            type(self.rendezvous_port) is not int
            or not 1 <= self.rendezvous_port <= 65535
        ):
            raise UnsupportedPlan("Invalid rendezvous port")
        if not isinstance(self.setup_commands, tuple):
            raise UnsupportedPlan("setup_commands must be an immutable tuple")
        for command in self.setup_commands:
            _text(command, "trusted setup command")
        if self.stage_out_command is not None:
            _text(self.stage_out_command, "trusted stage-out command")
