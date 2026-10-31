# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""An inert, bounded bridge to one SQLite-owning thread and asyncio loop.

A caller deadline requests cooperative asynchronous cancellation and limits
waiting, never forcibly terminating a thread. Noncooperative synchronous work
retains admission until it actually finishes; shutdown drains it honestly.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from concurrent.futures import Future
from contextlib import ExitStack
from copy import deepcopy
from typing import TYPE_CHECKING

from .budget import BudgetExhausted, StorageWorkError, WorkBudget
from .pool import CancellationToken, WorkDeadlineExceeded


if TYPE_CHECKING:
    from collections.abc import Callable

    from wavedl.core.layout import RunLayout
    from wavedl_daemon.runs.service import PlanResponse, RunResponse


class ServiceHost:
    """Construct and close all service resources on a single dedicated thread."""

    def __init__(self, factory: Callable, *, budget: WorkBudget | None = None):
        self.factory = factory
        self.budget = budget or WorkBudget()
        self._lock = threading.Lock()
        self._tokens: set[CancellationToken] = set()
        self._ready: Future = Future()
        self._stopped: Future = Future()
        self._thread: threading.Thread | None = None
        self._loop = None
        self._closed = False
        self._running = False
        self._abandoned_start = False
        self._close_task = None

    @property
    def outstanding(self) -> int:
        with self._lock:
            return len(self._tokens)

    async def start(self) -> None:
        with self._lock:
            if self._closed:
                raise BudgetExhausted("Service host is closed")
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="wavedl-service-owner", daemon=True
                )
                self._thread.start()
        ready = asyncio.wrap_future(self._ready)
        ready.add_done_callback(
            lambda done: None if done.cancelled() else done.exception()
        )
        try:
            await asyncio.wait_for(
                asyncio.shield(ready),
                self.budget.timeout_seconds,
            )
        except (TimeoutError, asyncio.CancelledError):
            # Late initialization must dispose resources itself. The thread is
            # not killed and a blocked factory is not falsely declared stopped.
            with self._lock:
                self._closed = True
                self._abandoned_start = True
                if self._running:
                    self._loop.call_soon_threadsafe(self._loop.stop)
            raise WorkDeadlineExceeded("Service startup deadline exceeded") from None

    def _run(self) -> None:
        stack = ExitStack()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._serial = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()
        self._async_tasks = {}
        try:
            candidate = self.factory()
            if hasattr(candidate, "__enter__") and hasattr(candidate, "__exit__"):
                self._service = stack.enter_context(candidate)
            else:
                self._service = candidate
                if hasattr(candidate, "close"):
                    stack.callback(candidate.close)
            with self._lock:
                closed = self._closed
                self._running = not closed
            self._ready.set_result(None)
            if not closed:
                loop.run_forever()
        except BaseException as error:
            if not self._ready.done():
                failure = StorageWorkError("Service initialization failed")
                failure.__cause__ = error
                self._ready.set_exception(failure)
        finally:
            with self._lock:
                self._running = False
            try:
                stack.close()
            except BaseException as error:
                if not self._stopped.done():
                    self._stopped.set_exception(error)
            finally:
                loop.close()
                if not self._stopped.done():
                    self._stopped.set_result(None)

    async def call(
        self, method: str | Callable, *args, concurrent_async: bool = False, **kwargs
    ):
        """Optionally release RPC serialization during audited asynchronous work.

        Opt-in callbacks must keep any SQLite transaction synchronous and may
        not require exclusive shared state across their await points.
        """
        if type(concurrent_async) is not bool:
            raise ValueError("concurrent_async must be a boolean")
        token = CancellationToken(time.monotonic() + self.budget.timeout_seconds)
        with self._lock:
            if (
                self._closed
                or not self._ready.done()
                or self._ready.exception() is not None
            ):
                raise BudgetExhausted("Service host is unavailable")
            if len(self._tokens) >= self.budget.max_workers + self.budget.max_pending:
                raise BudgetExhausted("Service owner capacity exhausted")
            self._tokens.add(token)
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._invoke(method, args, kwargs, token, concurrent_async),
                    self._loop,
                )
            except BaseException:
                self._tokens.remove(token)
                raise
        wrapped = asyncio.wrap_future(future)
        # Retrieve late failures even if the caller has already timed out.
        wrapped.add_done_callback(
            lambda done: None if done.cancelled() else done.exception()
        )
        try:
            return await asyncio.wait_for(
                asyncio.shield(wrapped), self.budget.timeout_seconds
            )
        except (TimeoutError, asyncio.CancelledError):
            self._request_cancel(token)
            raise WorkDeadlineExceeded(
                "Service call wait ended; execution may continue"
            ) from None

    async def _invoke(self, method, args, kwargs, token, concurrent_async):
        task = asyncio.current_task()
        self._tasks.add(task)
        try:
            async with self._serial:
                token.check()
                callback = (
                    getattr(self._service, method)
                    if isinstance(method, str)
                    else method
                )
                result = (
                    callback(*args, **kwargs)
                    if isinstance(method, str)
                    else callback(self._service, *args, **kwargs)
                )
                if not concurrent_async:
                    result = await self._await_result(result, token, task)
                    token.check()
            if concurrent_async:
                result = await self._await_result(result, token, task)
                token.check()
            return result
        finally:
            self._tasks.discard(task)
            with self._lock:
                self._tokens.remove(token)

    async def _await_result(self, result, token, task):
        if not inspect.isawaitable(result):
            return result
        self._async_tasks[token] = task
        try:
            if token.cancelled:
                task.cancel()
            return await result
        finally:
            self._async_tasks.pop(token, None)

    def _cancel_async(self, token):
        task = self._async_tasks.get(token)
        if task is not None and not task.cancelling():
            task.cancel()

    def _request_cancel(self, token):
        token.cancel()
        # A synchronous callback blocks this owner loop and cannot be killed.
        # Only an invocation currently awaiting cooperative asynchronous work
        # is cancelled; its finally blocks retain admission through cleanup.
        try:
            self._loop.call_soon_threadsafe(self._cancel_async, token)
        except RuntimeError:
            pass

    async def close(self) -> None:
        """Stop admission, drain real callbacks, then close/join off the main loop."""
        with self._lock:
            self._closed = True
            for token in self._tokens:
                self._request_cancel(token)
            if self._thread is None:
                return
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._finish_close())
        await asyncio.shield(self._close_task)

    async def _finish_close(self) -> None:
        try:
            await asyncio.shield(asyncio.wrap_future(self._ready))
        except BaseException:
            pass
        with self._lock:
            running = self._running and not self._abandoned_start
        if running and not self._stopped.done():
            drain = asyncio.run_coroutine_threadsafe(self._drain(), self._loop)
            await asyncio.shield(asyncio.wrap_future(drain))
            self._loop.call_soon_threadsafe(self._loop.stop)
        try:
            await asyncio.shield(asyncio.wrap_future(self._stopped))
        finally:
            await asyncio.to_thread(self._thread.join)

    async def _drain(self) -> None:
        # run_coroutine_threadsafe enqueue ordering ensures all already-admitted
        # invocations register before this drain coroutine runs.
        while self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)


class AsyncRunService:
    """Explicit lifecycle adapter; Session containers may expose a runs facade."""

    def __init__(self, host: ServiceHost):
        self.host = host

    async def _call(self, method, *args, **kwargs):
        def invoke(session):
            runs = getattr(session, "runs", session)
            return getattr(runs, method)(*args, **kwargs)

        return await self.host.call(invoke)

    async def plan(self, payload: dict, idempotency_key: str) -> PlanResponse:
        payload = deepcopy(payload)
        if type(payload) is not dict or "checkpoint_source" not in payload:
            return await self._call("plan", payload, idempotency_key)

        def capture(session):
            runs = getattr(session, "runs", session)
            source = runs.checkpoint_selection(payload, idempotency_key)
            if source is None:
                return None
            from wavedl.core.checkpoint_source import capture_checkpoint_source

            layout, generation, digest = source
            return session.pool.submit(
                lambda token: capture_checkpoint_source(
                    layout, generation, manifest_sha256=digest, check=token.check
                )
            )

        work = await self.host.call(capture)
        selection = await work.wait() if work is not None else None
        if selection is None:
            return await self._call("plan", payload, idempotency_key)
        return await self._call(
            "plan", payload, idempotency_key, checkpoint_selection=selection
        )

    async def submit(self, plan_id: str) -> RunResponse:
        return await self._call("submit", plan_id)

    async def cancel(self, run_id: str) -> RunResponse:
        return await self._call("cancel", run_id)

    async def get(self, run_id: str) -> RunResponse:
        return await self._call("get", run_id)

    async def list(
        self, limit: int = 50, offset: int = 0, state: str | None = None
    ) -> list[RunResponse]:
        return await self._call("list", limit, offset, state)

    async def layout(self, run_id: str) -> RunLayout:
        return await self._call("layout", run_id)

    async def attempts(self, run_id: str, *, limit=50, offset=0):
        return await self._call("attempts", run_id, limit=limit, offset=offset)
