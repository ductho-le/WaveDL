# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Bounded cooperative threads; hard deadlines belong in supervised processes.

A deadline requests cancellation. A callback must check its token during work.
No Python thread can be forcibly stopped: admission remains occupied until the
callback actually returns, including when it ignores cancellation. A timeout on
result() only bounds that caller's wait, and does not cancel any work.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Generic, TypeVar

from .budget import BudgetExhausted, StorageWorkError, WorkBudget


if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


class WorkCancelled(RuntimeError):
    """Cooperative work acknowledged a cancellation request."""


class WorkDeadlineExceeded(WorkCancelled):
    """Cooperative work acknowledged its deadline."""


class CancellationToken:
    def __init__(self, deadline: float) -> None:
        self.deadline = deadline
        self._event = threading.Event()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set() or time.monotonic() >= self.deadline

    @property
    def cancellation_requested(self) -> bool:
        """Explicit cancellation, distinct from expiry of the deadline."""
        return self._event.is_set()

    def cancel(self) -> None:
        """Request cancellation; this is not an acknowledgement of termination."""
        self._event.set()

    def check(self) -> None:
        if time.monotonic() >= self.deadline:
            raise WorkDeadlineExceeded("work deadline exceeded")
        if self._event.is_set():
            raise WorkCancelled("work cancellation requested")

    def wait(self, timeout: float | None = None) -> bool:
        """Wait interruptibly, waking at cancellation or the deadline."""
        end = self.deadline
        if timeout is not None:
            end = min(end, time.monotonic() + max(0.0, timeout))
        while not self.cancelled:
            remaining = end - time.monotonic()
            if remaining <= 0:
                break
            # Native timed waits may wake early (notably on Windows). Re-wait
            # for the remaining interval, never report an unexpired deadline.
            self._event.wait(remaining)
        return self.cancelled


class WorkHandle(Generic[T]):
    def __init__(self, future: Future[T], token: CancellationToken) -> None:
        self._future = future
        self.token = token

    def cancel(self) -> None:
        self.token.cancel()

    def done(self) -> bool:
        return self._future.done()

    def result(self, timeout: float | None = None) -> T:
        return self._future.result(timeout)

    async def wait(self) -> T:
        """Await without blocking an event loop; caller cancellation signals work."""
        wrapped = asyncio.wrap_future(self._future)
        wrapped.add_done_callback(
            lambda done: None if done.cancelled() else done.exception()
        )
        try:
            return await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            self.cancel()
            raise


class WorkLane:
    def __init__(
        self,
        budget: WorkBudget,
        executor: ThreadPoolExecutor,
        threads: int,
        permits: threading.BoundedSemaphore,
    ) -> None:
        self.budget = budget
        self._executor = executor
        self._capacity = threads + budget.max_pending
        # Shared by every lane of one pool: caps running callbacks, not admission.
        self._permits = permits
        self._lock = threading.Condition()
        self._tokens: set[CancellationToken] = set()
        self._closed = False

    @property
    def outstanding(self) -> int:
        with self._lock:
            return len(self._tokens)

    def submit(self, callback: Callable[[CancellationToken], T]) -> WorkHandle[T]:
        """Admit at most lane threads + max_pending uncompleted operations."""
        token = CancellationToken(time.monotonic() + self.budget.timeout_seconds)
        with self._lock:
            if self._closed:
                raise RuntimeError("pool is closed")
            if len(self._tokens) >= self._capacity:
                raise BudgetExhausted("blocking worker capacity exhausted")
            self._tokens.add(token)
            try:
                future = self._executor.submit(self._run, callback, token)
            except BaseException:
                self._tokens.remove(token)
                self._lock.notify_all()
                raise
        future.add_done_callback(lambda done: self._complete(token))
        return WorkHandle(future, token)

    def _complete(self, token: CancellationToken) -> None:
        with self._lock:
            self._tokens.remove(token)
            self._lock.notify_all()

    def _run(
        self, callback: Callable[[CancellationToken], T], token: CancellationToken
    ) -> T:
        token.check()
        while not self._permits.acquire(timeout=0.05):
            token.check()
        try:
            token.check()
            try:
                result = callback(token)
            except OSError as error:
                raise StorageWorkError(
                    "blocking operation failed; durability unknown"
                ) from error
            token.check()
            return result
        finally:
            self._permits.release()

    def shutdown(self, *, wait: bool = True) -> None:
        """Request cooperative shutdown; wait=True waits for real termination."""
        with self._lock:
            self._closed = True
            for token in self._tokens:
                token.cancel()
            if wait:
                self._lock.wait_for(lambda: not self._tokens)


class BoundedThreadPool:
    """Shared callbacks leave reserved running permits available to control.

    Control may borrow all idle capacity; running callbacks never exceed max_workers.
    """

    def __init__(self, budget: WorkBudget) -> None:
        self.budget = budget
        self._closed = False
        reserved = max(1, budget.max_workers // 4)
        shared_threads = max(1, budget.max_workers - reserved)
        shared = ThreadPoolExecutor(max_workers=shared_threads)
        control = (
            shared
            if budget.max_workers == 1
            else ThreadPoolExecutor(max_workers=budget.max_workers)
        )
        self._executors = (shared,) if shared is control else (shared, control)
        permits = threading.BoundedSemaphore(budget.max_workers)
        self._shared_lane = WorkLane(budget, shared, shared_threads, permits)
        self.control_lane = WorkLane(budget, control, budget.max_workers, permits)

    @property
    def outstanding(self) -> int:
        return self._shared_lane.outstanding + self.control_lane.outstanding

    def submit(self, callback: Callable[[CancellationToken], T]) -> WorkHandle[T]:
        return self._shared_lane.submit(callback)

    def shutdown(self, *, wait: bool = True) -> None:
        self._closed = True
        self._shared_lane.shutdown(wait=False)
        self.control_lane.shutdown(wait=False)
        for executor in self._executors:
            executor.shutdown(wait=wait)

    def __enter__(self) -> BoundedThreadPool:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown()
