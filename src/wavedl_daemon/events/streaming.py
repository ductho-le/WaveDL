# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Gap-free replay/live handoff with periodic durable fallback."""

from __future__ import annotations

import inspect
import json
import math
import sqlite3

from wavedl.core.protocol import encode
from wavedl_daemon.execution.budget import BudgetExhausted, StorageWorkError
from wavedl_daemon.execution.pool import WorkCancelled
from wavedl_daemon.storage.event_log import validate_cursor
from wavedl_daemon.storage.ownership import DispatchRequired
from wavedl_daemon.storage.plans import PlanRecoveryRequired
from wavedl_daemon.storage.settings import UnsupportedStorage


EVENT_SERVICE_ERRORS = (
    BudgetExhausted,
    StorageWorkError,
    WorkCancelled,
    DispatchRequired,
    PermissionError,
    PlanRecoveryRequired,
    UnsupportedStorage,
    sqlite3.DatabaseError,
    OSError,
)


async def stream_events(
    store,
    broker,
    run_id: str,
    cursor=(0, -1),
    *,
    page_size=256,
    poll_interval=1.0,
    subscription=None,
    initial_page=None,
):
    validate_cursor(cursor)
    if not math.isfinite(poll_interval) or not 0.01 <= poll_interval <= 60:
        raise ValueError("Invalid replay polling interval")
    # Register BEFORE the first read, including before replay watermarks exist.
    subscription = subscription or broker.subscribe(run_id)
    try:
        while True:
            if subscription.closed and not subscription.overflowed:
                return
            if subscription.overflowed:
                data = json.dumps(
                    {"resumePosition": {"attempt": cursor[0], "seq": cursor[1]}},
                    separators=(",", ":"),
                )
                yield f"event: overflow\ndata: {data}\n\n".encode()
                return
            try:
                if initial_page is not None:
                    events, initial_page = initial_page, None
                else:
                    events = store.read_after(run_id, cursor, limit=page_size)
                    if inspect.isawaitable(events):
                        events = await events
            except EVENT_SERVICE_ERRORS:
                data = json.dumps(
                    {
                        "error": "Event service unavailable",
                        "resumePosition": {"attempt": cursor[0], "seq": cursor[1]},
                    },
                    separators=(",", ":"),
                )
                yield f"event: error\ndata: {data}\n\n".encode()
                return
            if subscription.closed:
                continue
            if events:
                for event in events:
                    position = (event.attempt, event.seq)
                    yield (
                        f"id: {event.attempt}:{event.seq}\ndata: ".encode()
                        + encode(event)
                        + b"\n\n"
                    )
                    cursor = position
                continue
            await subscription.wait(poll_interval)
            if not subscription.overflowed:
                # Comments are not events and never advance the resume cursor.
                yield b": heartbeat\n\n"
    finally:
        broker.unsubscribe(subscription)
