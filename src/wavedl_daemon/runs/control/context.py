# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Shared stores and ownership guards; workers never commit through this context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

    from wavedl_daemon.backends.contracts import Backend
    from wavedl_daemon.execution.budget import WorkBudget
    from wavedl_daemon.execution.pool import WorkLane
    from wavedl_daemon.storage.collections import CollectionStore
    from wavedl_daemon.storage.executions import ExecutionStore
    from wavedl_daemon.storage.plans import PlanStore
    from wavedl_daemon.storage.runtime_provenance import RuntimeProvenanceStore


@dataclass
class ControlContext:
    plan_store: PlanStore
    execution_store: ExecutionStore
    provenance_store: RuntimeProvenanceStore | None
    collection_store: CollectionStore | None
    backends: dict[str, Backend]
    budget: WorkBudget
    lane: WorkLane
    clock: Callable[[], float]
    guard: Callable[[], None]
    guard_external: Callable[[], None]
    dispatch_errors: dict[str, str]
    runtime_capacity: int
