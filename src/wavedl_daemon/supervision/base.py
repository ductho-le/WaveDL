# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Portable, stdlib-only process supervision contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ProcessSpec:
    argv: tuple[str, ...]
    cwd: Path
    environment: dict[str, str] | None = None
    inherit_environment: bool = True

    def __post_init__(self) -> None:
        if type(self.inherit_environment) is not bool:
            raise ValueError("inherit_environment must be a boolean")
        if not isinstance(self.argv, tuple) or not self.argv:
            raise ValueError("argv must be a nonempty tuple")
        if any(not isinstance(arg, str) or "\0" in arg for arg in self.argv):
            raise ValueError("argv entries must be strings without NUL")
        if not self.argv[0]:
            raise ValueError("argv executable must not be empty")
        if not isinstance(self.cwd, Path) or not self.cwd.is_dir():
            raise ValueError("cwd must be an existing directory Path")
        if self.environment is not None:
            if not isinstance(self.environment, dict) or any(
                not isinstance(key, str)
                or not key
                or "=" in key
                or "\0" in key
                or not isinstance(value, str)
                or "\0" in value
                for key, value in self.environment.items()
            ):
                raise ValueError("environment must contain valid string keys/values")
            object.__setattr__(self, "environment", dict(self.environment))


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    cancelled: bool = False
    timed_out: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False


def validate_seconds(value: float, name: str) -> None:
    """Reject non-finite and negative deadlines before changing process state."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")


def validate_output_limit(output_limit: int) -> None:
    if isinstance(output_limit, bool) or not isinstance(output_limit, int):
        raise ValueError("output_limit must be a nonnegative integer")
    if output_limit < 0:
        raise ValueError("output_limit must be a nonnegative integer")


class SupervisedProcess(Protocol):
    @property
    def pid(self) -> int: ...

    def cancel(self, grace_seconds: float = 5) -> None: ...

    def wait(self, timeout: float | None = None) -> ProcessResult: ...


class Supervisor(Protocol):
    def start(
        self, spec: ProcessSpec, *, output_limit: int = 1048576
    ) -> SupervisedProcess: ...
