# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Admission limits and fail-closed storage policy for blocking daemon work."""

from __future__ import annotations

import math
from dataclasses import dataclass


class BudgetExhausted(RuntimeError):
    """Work was refused before submission or process creation."""


class StorageWorkError(RuntimeError):
    """Storage failed; callers must fail the operation, never report durable success.

    There is no emergency spool or silent degraded-success mode. The original
    OSError is retained as the cause, allowing lifecycle owners to report failure.
    """


@dataclass(frozen=True)
class WorkBudget:
    max_workers: int = 4
    max_pending: int = 8
    max_command_bytes: int = 65536
    max_output_bytes: int = 1048576
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        for name in (
            "max_workers",
            "max_pending",
            "max_command_bytes",
            "max_output_bytes",
        ):
            value = getattr(self, name)
            minimum = 0 if name in {"max_pending", "max_output_bytes"} else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        value = self.timeout_seconds
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
