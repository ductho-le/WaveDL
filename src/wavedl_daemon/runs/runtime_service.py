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

from wavedl.core.layout import RunLayout
from wavedl.core.safe_files import open_directory
from wavedl_daemon.backends.local import LocalBackend
from wavedl_daemon.events.broker import EventBroker
from wavedl_daemon.events.ingestion import (
    Ingestor,
    archived_execution_matches,
    registered_handle_matches,
    standalone_adoption,
)
from wavedl_daemon.execution.budget import BudgetExhausted, WorkBudget
from wavedl_daemon.execution.pool import BoundedThreadPool
from wavedl_daemon.execution.service_host import AsyncRunService, ServiceHost
from wavedl_daemon.http.services import DaemonServices
from wavedl_daemon.runs.adoption_service import VerifiedAdoptionService
from wavedl_daemon.runs.artifacts import ArtifactStore
from wavedl_daemon.runs.catalog import AsyncCatalogService, CatalogService
from wavedl_daemon.runs.controller import SubmissionController
from wavedl_daemon.runs.host_profile import HostProfiler
from wavedl_daemon.runs.inspection import DatasetInspector
from wavedl_daemon.runs.preview_service import DatasetPreviewService
from wavedl_daemon.runs.service import RunService
from wavedl_daemon.storage.collections import CollectionStore
from wavedl_daemon.storage.event_log import EventLogStore
from wavedl_daemon.storage.executions import ExecutionStore, native_runtime_pending
from wavedl_daemon.storage.ownership import acquire_dispatch_lock
from wavedl_daemon.storage.plans import PlanStore
from wavedl_daemon.storage.registry import Registry
from wavedl_daemon.storage.runtime_provenance import RuntimeProvenanceStore


if TYPE_CHECKING:
    from wavedl_daemon.storage.settings import Settings


class ServiceSession:
    """Construct inside ServiceHost, never on an ASGI or filesystem worker thread.

    Nothing opens until entering the context. SQLite and intent/control calls all
    stay on this context's owner thread; bounded workers only perform filesystem
    and backend work. Startup only dispatches previously persisted user intent.
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
        self._offset = 0
        self._history_epoch = 0

    @staticmethod
    def _attempts(layout, current, selected=None):
        result = []
        with open_directory(layout.attempts) as directory:
            if selected is not None:
                # Managed ancestry comes from SQLite, not directory names.
                if len(selected) > 4096 or any(
                    type(number) is not int or not 1 <= number <= current
                    for number in selected
                ):
                    raise ValueError("Invalid bounded registered attempt selection")
                for number in selected:
                    try:
                        with directory.open_directory(str(number)) as attempt:
                            attempt.verify_private_owner()
                            attempt.guard()
                    except FileNotFoundError:
                        continue
                    result.append(number)
                return sorted(result)
            # Explicit standalone adoption retains its separate verification
            # path: each historical candidate must pass full standalone proof.
            for count, entry in enumerate(directory.entries(), 1):
                if count > 4096:
                    raise ValueError("Attempt inventory exceeds monitoring bound")
                if (
                    entry.is_directory
                    and entry.name.isdecimal()
                    and str(int(entry.name)) == entry.name
                ):
                    number = int(entry.name)
                    if 1 <= number <= current:
                        result.append(number)
        return sorted(result)

    def _lineage_attempts(self, record, epoch):
        """Read a fair bounded archive window on the SQLite owner thread."""
        store = self.runs.execution_store
        if not registered_handle_matches(record, record.handle):
            raise ValueError("Current execution lacks registered journal lineage")
        count = store.history_count(record.plan_id)
        if type(count) is not int or not 0 <= count <= 2**63 - 1:
            raise ValueError("Invalid execution history count")
        selected = [record.attempt]
        if count:
            start = (epoch * 2) % count
            page = store.history(
                record.plan_id, limit=min(2, count - start), offset=start
            )
            if len(page) < min(2, count):
                page += store.history(record.plan_id, limit=2 - len(page), offset=0)
            for archived in page:
                if not archived_execution_matches(record, archived):
                    raise ValueError("Archived execution has foreign journal lineage")
                selected.append(archived.attempt)
        return sorted(set(selected))

    def __enter__(self):
        self.settings.validate()
        with open_directory(self.settings.state_root) as state:
            state.verify_private_owner(confidential=True)
            state.guard()
        stack = ExitStack()
        self._stack = stack
        try:
            lock = stack.enter_context(
                acquire_dispatch_lock(self.settings.state_root / "dispatch.lock")
            )
            path = self.settings.state_root / "registry.db"
            self.registry = stack.enter_context(Registry(path, dispatch_lock=lock))
            info = self.registry.path.stat()
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
            plans = stack.enter_context(PlanStore(path, dispatch_lock=lock))
            executions = stack.enter_context(ExecutionStore(path, dispatch_lock=lock))
            provenance = stack.enter_context(
                RuntimeProvenanceStore(path, dispatch_lock=lock)
            )
            collections = stack.enter_context(CollectionStore(path, dispatch_lock=lock))
            self.event_log = EventLogStore(path)
            stack.callback(self.event_log.close)
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
            self.controller = SubmissionController(
                plans,
                executions,
                self.backends,
                lock,
                self.budget,
                provenance_store=provenance,
                collection_store=collections,
                pool=self.pool,
            )
            stack.callback(self.controller.drain)
            stack.callback(self.controller.close)
            self.runs = RunService(
                self.settings,
                plans,
                executions,
                self.controller,
                self.backends,
                self.registry,
                provenance,
                collections=collections,
            )
            self.ingestor = Ingestor(
                self.registry,
                self.controller,
                self.broker,
                self.pool,
                event_log=self.event_log,
            )
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
            self.adoptions = VerifiedAdoptionService(
                self.settings,
                plans,
                executions,
                provenance,
                self.registry,
                self.controller,
                self.pool,
                self.backends,
            )
            return self
        except BaseException:
            stack.close()
            self._stack = None
            raise

    async def start(self):
        self.controller.startup()
        if self._pump_task is None:
            self._pump_task = asyncio.create_task(self._pump())

    async def _pump(self):
        while True:
            try:
                self.controller.reconcile_once()
                self.errors.pop("control", None)
                epoch = self._history_epoch
                plans = self.runs.plan_store.list_pending(limit=4, offset=self._offset)
                if len(plans) == 4:
                    self._offset += 4
                else:
                    self._offset = 0
                    self._history_epoch += 1
                for plan in plans:
                    record = self.runs.execution_store.get(plan.plan_id)
                    native_pending = record is not None and native_runtime_pending(
                        record
                    )
                    if (
                        record is None
                        or record.attempt is None
                        or record.handle is None
                    ):
                        continue
                    # A native requeue keeps the prior runtime binding as the
                    # last-started attempt while the replacement is pending.
                    # Continue tailing that journal strictly as historical
                    # diagnostics; no active-runtime cursor/provenance writes.
                    layout = RunLayout(Path(record.handle.run_dir), record.run_id)
                    try:
                        selected = (
                            None
                            if standalone_adoption(record)
                            else self._lineage_attempts(record, epoch)
                        )
                        attempts = await self.pool.submit(
                            lambda token,
                            layout=layout,
                            current=record.attempt,
                            selected=selected: (
                                self._attempts(layout, current, selected)
                            )
                        ).wait()
                        failure = None
                        for attempt in self._attempt_batch(
                            attempts, record.attempt, epoch if selected is None else 0
                        ):
                            try:
                                await self.ingestor.poll_once(
                                    plan.plan_id,
                                    record.run_id,
                                    attempt,
                                    layout.journal(attempt),
                                    historical=native_pending
                                    or attempt < record.attempt,
                                )
                            except Exception as error:
                                # A corrupt old attempt cannot suppress active
                                # delivery or the next fair historical window.
                                failure = failure or type(error).__name__
                        if failure is None:
                            self.errors.pop(plan.plan_id, None)
                        else:
                            self._monitor_error(plan.plan_id, failure)
                    except Exception as error:
                        self._monitor_error(plan.plan_id, type(error).__name__)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._monitor_error("control", type(error).__name__)
            await asyncio.sleep(0.05)

    @staticmethod
    def _attempt_batch(attempts, current, epoch):
        """Always tail the active attempt, then two fair historical sources.

        Epoch advances once per complete plan sweep, not once per plan/page:
        fleet size cannot alias with an attempt count and starve a history range.
        A pump lap admits at most four plans and twelve journal polls.
        """
        historical = [number for number in attempts if number < current]
        result = [current] if current in attempts else []
        if historical:
            start = (epoch * 2) % len(historical)
            chosen = {
                historical[(start + offset) % len(historical)]
                for offset in range(min(2, len(historical)))
            }
            result.extend(sorted(chosen))
        return result

    def _monitor_error(self, plan_id, failure):
        if plan_id not in self.errors and len(self.errors) >= 128:
            self.errors.pop(next(iter(self.errors)))
        self.errors[plan_id] = failure

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
            lambda session: session.event_log.read_after(run_id, cursor, limit=limit)
        )


class _AsyncIngestor:
    def __init__(self, host):
        self.host = host

    async def push(self, plan_id, run_id, attempt, events):
        return await self.host.call(
            lambda session: session.ingestor.push(plan_id, run_id, attempt, events)
        )

    async def reject_runtime_events(self, plan_id, run_id, attempt, reason):
        return await self.host.call(
            lambda session: session.ingestor.reject_runtime_events(
                plan_id, run_id, attempt, reason
            )
        )


def _unresolved_layout(run_id):
    raise RuntimeError("Artifacts require owner-resolved layout")


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
        self.ingestor = _AsyncIngestor(self.host)
        self.artifacts = ArtifactStore(_unresolved_layout)
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
                "restart_watch_refusals": session.controller.restart_watch_refusals,
                "restart_monitoring_errors": [
                    {"plan_id": key, "error": value}
                    for key, value in sorted(session.controller.restart_errors.items())
                ],
                "dispatch_errors": [
                    {"plan_id": key, "error": value}
                    for key, value in sorted(session.controller.dispatch_errors.items())
                ],
            }
        )

    async def adopt(self, path, *, attempt=None, idempotency_key):
        return await self.host.call(
            lambda session: session.adoptions.adopt(
                path, attempt=attempt, idempotency_key=idempotency_key
            ),
            concurrent_async=True,
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
