# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Bounded wakeups only: SQLite remains the event delivery authority."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field

from wavedl_daemon.execution.budget import BudgetExhausted


@dataclass(eq=False)
class Subscription:
    run_id: str
    queue: asyncio.Queue = field(repr=False)
    loop: asyncio.AbstractEventLoop = field(repr=False)
    thread_id: int
    overflowed: bool = False
    closed: bool = False
    pending: int = 0
    scheduled: bool = False

    async def wait(self, timeout: float) -> None:
        if self.closed:
            return
        try:
            await asyncio.wait_for(self.queue.get(), timeout)
        except TimeoutError:
            pass


class EventBroker:
    """Cross-thread notifications schedule at most one callback per subscriber."""

    def __init__(self, *, queue_size: int = 8, max_subscribers: int = 256):
        if any(
            type(value) is not int or not 1 <= value <= 4096
            for value in (queue_size, max_subscribers)
        ):
            raise ValueError("Broker bounds must be positive")
        self.queue_size, self.max_subscribers = queue_size, max_subscribers
        self._subscribers: set[Subscription] = set()
        self._lock = threading.RLock()

    def subscribe(self, run_id: str) -> Subscription:
        with self._lock:
            if len(self._subscribers) >= self.max_subscribers:
                raise BudgetExhausted("Event subscriber capacity exhausted")
            subscription = Subscription(
                run_id,
                asyncio.Queue(self.queue_size),
                asyncio.get_running_loop(),
                threading.get_ident(),
            )
            self._subscribers.add(subscription)
            return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        with self._lock:
            if subscription.closed:
                return
            subscription.closed = True
            self._subscribers.discard(subscription)
            if subscription.thread_id == threading.get_ident():
                self._wake_closed(subscription)
            else:
                try:
                    subscription.loop.call_soon_threadsafe(
                        self._wake_closed, subscription
                    )
                except RuntimeError:
                    pass

    def _wake_closed(self, subscription):
        if subscription.queue.empty():
            subscription.queue.put_nowait(None)

    def _enqueue(self, subscription):
        try:
            subscription.queue.put_nowait(None)
        except asyncio.QueueFull:
            subscription.overflowed = True
            self.unsubscribe(subscription)

    def _deliver(self, subscription):
        # Runs only on the subscriber's event loop, never the SQLite owner loop.
        with self._lock:
            count, subscription.pending = subscription.pending, 0
            subscription.scheduled = False
            if subscription.closed:
                return
            for _ in range(count):
                self._enqueue(subscription)
                if subscription.closed:
                    break

    def notify(self, run_id: str) -> None:
        with self._lock:
            for subscription in tuple(self._subscribers):
                if subscription.run_id != run_id:
                    continue
                if subscription.thread_id == threading.get_ident():
                    self._enqueue(subscription)
                    continue
                subscription.pending = min(
                    self.queue_size + 1, subscription.pending + 1
                )
                if subscription.scheduled:
                    continue
                subscription.scheduled = True
                try:
                    subscription.loop.call_soon_threadsafe(self._deliver, subscription)
                except RuntimeError:
                    subscription.closed = True
                    self._subscribers.discard(subscription)
