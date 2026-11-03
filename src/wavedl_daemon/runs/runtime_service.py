# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Owned service sessions and async facades for the HTTP control plane."""

from __future__ import annotations

import asyncio
import getpass
import hashlib
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

from wavedl_daemon import paths
from wavedl_daemon.backends.local import LocalBackend
from wavedl_daemon.events.broker import EventBroker
from wavedl_daemon.execution.budget import BudgetExhausted, WorkBudget
from wavedl_daemon.execution.pool import BoundedThreadPool
from wavedl_daemon.execution.service_host import AsyncRunService, ServiceHost
from wavedl_daemon.http.services import DaemonServices
from wavedl_daemon.runs.catalog import AsyncCatalogService, CatalogService
from wavedl_daemon.runs.controller import RunController
from wavedl_daemon.runs.host_profile import HostProfiler
from wavedl_daemon.runs.inspection import DatasetInspector
from wavedl_daemon.runs.preview_service import DatasetPreviewService
from wavedl_daemon.runs.service import RunService
from wavedl_daemon.storage.database import open_database
from wavedl_daemon.storage.ownership import acquire_dispatch_lock
from wavedl_daemon.storage.plans import PlanStore
from wavedl_daemon.storage.run_log import RunLog


if TYPE_CHECKING:
    from wavedl_daemon.storage.settings import Settings


DATABASE_FILE = "registry.db"
PUMP_INTERVAL_SECONDS = 0.05


class ServiceSession:
    """Construct inside ServiceHost, never on an ASGI or filesystem worker thread.

    Nothing opens until entering the context. SQLite calls all stay on this
    context's owner thread; bounded workers only perform filesystem and
    backend work. The run log is the work list, so startup needs no recovery.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        broker=None,
        budget=None,
        backend_factory=None,
        worker_pool=None,
        browse_roots=None,
    ):
        self.settings = settings
        self.broker = broker or EventBroker()
        self.budget = budget or WorkBudget()
        self.backend_factory = backend_factory
        self._worker_pool = worker_pool
        self.browse_roots = browse_roots
        self._stack = None
        self._pump_task = None
        self.errors = {}

    def __enter__(self):
        self.settings.validate()
        # Another user who can write the registry could plan commands as us.
        if not paths.is_private(self.settings.state_root.stat()):
            raise PermissionError("State root is accessible to other users")
        stack = ExitStack()
        self._stack = stack
        try:
            lock = stack.enter_context(
                acquire_dispatch_lock(self.settings.state_root / "dispatch.lock")
            )
            path = self.settings.state_root / DATABASE_FILE
            connection = open_database(path, dispatch_lock=lock)
            stack.callback(connection.close)
            info = path.stat()
            principal = (
                f"uid:{os.getuid()}"
                if hasattr(os, "getuid")
                else f"user:{getpass.getuser()}"
            )
            identity = json.dumps(
                {
                    "state_root": str(self.settings.state_root.resolve()),
                    "database_device": info.st_dev,
                    "database_inode": info.st_ino,
                    "principal": principal,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            self.daemon_id = hashlib.sha256(identity).hexdigest()
            self.log = RunLog(connection, notify=self.broker.notify)
            plans = PlanStore(connection)
            self.pool = self._worker_pool or BoundedThreadPool(self.budget)
            stack.callback(self.pool.shutdown, wait=True)
            self.backends = (
                self.backend_factory(lock)
                if self.backend_factory
                else {"local": LocalBackend(dispatch_lock=lock)}
            )
            for backend in self.backends.values():
                if hasattr(backend, "close"):
                    stack.callback(backend.close)
            self.controller = RunController(
                self.log,
                plans,
                self.pool,
                local=self.backends.get("local"),
                slurm=self.backends.get("slurm"),
            )
            self.runs = RunService(self.settings, plans, self.log, self.backends)
            self.catalog = CatalogService(
                self.browse_roots
                if self.browse_roots is not None
                else {"runs": self.settings.run_root}
            )
            self.inspector = DatasetInspector(
                sys.executable,
                Path.cwd(),
                self.pool,
                budget=self.budget,
                catalog=self.catalog,
            )
            self.previewer = DatasetPreviewService(
                sys.executable, Path.cwd(), self.pool, self.catalog
            )
            self.host_profiler = HostProfiler(
                sys.executable, Path.cwd(), self.pool, self.backends, budget=self.budget
            )
            return self
        except BaseException:
            stack.close()
            self._stack = None
            raise

    async def start(self):
        if self._pump_task is None:
            self._pump_task = asyncio.create_task(self._pump())

    async def _pump(self):
        while True:
            try:
                self.controller.tick()
                self.errors.pop("control", None)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._monitor_error("control", type(error).__name__)
            await asyncio.sleep(PUMP_INTERVAL_SECONDS)

    def _monitor_error(self, source, failure):
        if source not in self.errors and len(self.errors) >= 128:
            self.errors.pop(next(iter(self.errors)))
        self.errors[source] = failure

    async def stop(self):
        if self._pump_task is not None:
            self._pump_task.cancel()
            await asyncio.gather(self._pump_task, return_exceptions=True)
            self._pump_task = None

    def __exit__(self, *exc_info):
        # ServiceHost drains RPCs and stops its owner loop before context exit.
        # Complete background cancellation on that same loop before closing DBs.
        if self._pump_task is not None:
            asyncio.get_event_loop().run_until_complete(self.stop())
        if self._stack is not None:
            self._stack.close()
            self._stack = None


class _AsyncEventLog:
    def __init__(self, host):
        self.host = host

    async def read_after(self, run_id, cursor=(0, -1), *, limit=256):
        return await self.host.call(
            lambda session: session.log.read_after(run_id, cursor, limit=limit)
        )


class HostedServices(DaemonServices):
    """One inert shared facade, started/closed by the app lifespan."""

    def __init__(
        self,
        settings: Settings,
        *,
        budget=None,
        backend_factory=None,
        browse_roots=None,
    ):
        self.daemon_id = ""
        self.broker = EventBroker()
        self.pool = BoundedThreadPool(budget or WorkBudget())
        self.host = ServiceHost(
            lambda: ServiceSession(
                settings,
                broker=self.broker,
                budget=budget,
                backend_factory=backend_factory,
                worker_pool=self.pool,
                browse_roots=browse_roots,
            ),
            budget=budget,
        )
        self.run_service = AsyncRunService(self.host)
        self.catalog = AsyncCatalogService(self.host)
        self.event_log = _AsyncEventLog(self.host)
        self._start_task = None
        self._close_task = None

    async def start(self):
        if self._close_task is not None:
            raise BudgetExhausted("Hosted services are closed")
        if self._start_task is None:
            self._start_task = asyncio.create_task(self._finish_start())
        await self._start_task

    async def _finish_start(self):
        try:
            await self.host.start()
            self.daemon_id = await self.host.call(lambda session: session.daemon_id)
            await self.host.call("start")
        except BaseException:
            try:
                await self.close()
            except BaseException:
                pass
            raise

    async def close(self):
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._finish_close())
        await asyncio.shield(self._close_task)

    async def _finish_close(self):
        try:
            # Context exit stops the pump on the owner loop, even when startup
            # failed or RPC admission is saturated. Cleanup needs no new RPC.
            await self.host.close()
        finally:
            await asyncio.to_thread(self.pool.shutdown, wait=True)

    async def resolve_plan(self, run_id):
        return (await self.run_service.get(run_id)).plan_id

    async def health(self):
        """Expose bounded secret-free monitoring failures, not runtime proof."""
        return await self.host.call(
            lambda session: {
                "registered_backends": sorted(session.backends),
                "monitoring_errors": [
                    {"source": source, "error": failure}
                    for source, failure in sorted(session.errors.items())
                ],
                "filesystem_work_outstanding": session.pool.outstanding,
                # Requeues are found by the same Slurm observation as status.
                "restart_watch_refusals": session.controller.observation_refusals,
                "restart_monitoring_errors": [
                    {"plan_id": key, "error": value}
                    for key, value in sorted(
                        session.controller.observation_errors.items()
                    )
                ],
                "dispatch_errors": [
                    {"plan_id": key, "error": value}
                    for key, value in sorted(session.controller.errors.items())
                ],
            }
        )

    async def inspect(self, root, path):
        return await self.host.call(
            lambda session: session.inspector.inspect(root, path), concurrent_async=True
        )

    async def preview(self, request):
        return await self.host.call(
            lambda session: session.previewer.preview(request), concurrent_async=True
        )

    async def host_profile(self):
        return await self.host.call(
            lambda session: session.host_profiler.profile(), concurrent_async=True
        )
