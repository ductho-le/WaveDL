# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""The complete owned service boundary required by the hosted HTTP app."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from wavedl_daemon.events.broker import EventBroker
    from wavedl_daemon.execution.pool import BoundedThreadPool
    from wavedl_daemon.execution.service_host import AsyncRunService
    from wavedl_daemon.runs.adoption_service import AdoptionResponse
    from wavedl_daemon.runs.artifacts import ArtifactStore
    from wavedl_daemon.runs.catalog import AsyncCatalogService
    from wavedl_daemon.runs.runtime_service import _AsyncEventLog, _AsyncIngestor


class DaemonServices(Protocol):
    """An inert facade; ownership starts and ends with the app lifespan."""

    daemon_id: str
    run_service: AsyncRunService
    catalog: AsyncCatalogService
    ingestor: _AsyncIngestor
    event_log: _AsyncEventLog
    broker: EventBroker
    artifacts: ArtifactStore
    pool: BoundedThreadPool

    async def resolve_plan(self, run_id: str) -> str: ...

    async def health(self) -> dict: ...

    async def adopt(
        self, path: str, *, attempt: int | None = None, idempotency_key: str
    ) -> AdoptionResponse: ...

    async def inspect(self, root: str, path: str) -> dict: ...

    async def preview(self, request: dict) -> dict: ...

    async def host_profile(self) -> dict: ...

    async def start(self) -> None: ...

    async def close(self) -> None: ...


def require_services(services: DaemonServices) -> None:
    """Fail at assembly, not on the first request to an incomplete facade."""
    attributes = (
        "daemon_id",
        "run_service",
        "catalog",
        "ingestor",
        "event_log",
        "broker",
        "artifacts",
        "pool",
    )
    methods = (
        "resolve_plan",
        "health",
        "adopt",
        "inspect",
        "preview",
        "host_profile",
        "start",
        "close",
    )
    missing = [name for name in attributes if not hasattr(services, name)]
    missing.extend(
        name for name in methods if not callable(getattr(services, name, None))
    )
    if missing:
        raise TypeError("Incomplete daemon services: " + ", ".join(missing))
