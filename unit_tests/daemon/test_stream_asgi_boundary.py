import asyncio

import pytest

from tools.stream_asgi_boundary import (
    StreamBoundary,
    query_cursor,
    verify_boundary_record,
)
from unit_tests.daemon.test_event_stream_live_boundaries import (
    RUN,
    TOKEN,
    authenticated_app,
    event,
    loopback_app,
)
from wavedl.core.client_transport import HttpTransport
from wavedl_daemon.events.broker import EventBroker
from wavedl_daemon.storage.event_log import EventLogStore
from wavedl_daemon.storage.registry import Registry


def test_query_cursor_requires_exact_numeric_pair():
    assert query_cursor(b"attempt=2&seq=7") == {"attempt": 2, "seq": 7}
    assert query_cursor(b"seq=-1&attempt=0") == {"attempt": 0, "seq": -1}
    assert query_cursor(b"attempt=2") is None
    assert query_cursor(b"attempt=2&seq=7&seq=8") is None
    assert query_cursor(b"attempt=true&seq=7") is None
    assert query_cursor(b"attempt=0&seq=7") is None
    assert query_cursor(b"attempt=9223372036854775808&seq=7") is None


def test_verifier_rejects_claim_of_local_cursor_or_uncommitted_event():
    base = {
        "kind": "asgi-event-boundary-v1",
        "status": 200,
        "handlerReturned": True,
        "browserObserved": False,
        "localCursor": None,
        "forwardedControls": [],
        "forwardedEventIds": [[1, 3]],
        "registry": {"eventCount": 4, "cursors": {"1": 3}},
    }
    assert verify_boundary_record(base) == {"scope": "asgi-only", "forwardedEvents": 1}
    claimed = {**base, "localCursor": [1, 3]}
    with pytest.raises(ValueError, match="ASGI-only"):
        verify_boundary_record(claimed)
    uncommitted = {**base, "registry": {"eventCount": 3, "cursors": {"1": 2}}}
    with pytest.raises(ValueError, match="authority"):
        verify_boundary_record(uncommitted)


def test_real_authenticated_handler_snapshot_and_subsequent_request_cursor(tmp_path):
    """Real registry and routes; client is native HTTP, explicitly not a browser."""

    async def scenario():
        with Registry(tmp_path / "registry.db") as registry:
            store = EventLogStore(registry.path)
            broker = EventBroker()
            try:
                registry.apply_events(RUN, 1, [event(0)])
                real_app = authenticated_app(None, store, broker)
                handler_bytes, forwarded_bytes = [], []

                async def fragmented(scope, receive, send):
                    async def fragment_send(message):
                        if message["type"] == "http.response.body" and message.get(
                            "body"
                        ):
                            handler_bytes.append(message["body"])
                            for value in message["body"]:
                                await send({**message, "body": bytes((value,))})
                        else:
                            await send(message)

                    await real_app(scope, receive, fragment_send)

                boundary = StreamBoundary(fragmented, registry)
                pending = bytearray()

                def disconnect_after_event(payload, disconnected):
                    forwarded_bytes.append(payload)
                    pending.extend(payload)
                    if pending.endswith(b"\n\n") and b"id: 1:" in pending:
                        pending.clear()
                        disconnected.set()

                async with loopback_app(boundary, on_body=disconnect_after_event) as (
                    url,
                    _,
                ):
                    first = [
                        item async for item in HttpTransport(url, TOKEN).events(RUN)
                    ]
                    assert first == [event(0)]
                    registry.apply_events(RUN, 1, [event(1)])
                    second = [
                        item
                        async for item in HttpTransport(url, TOKEN).events(
                            RUN, cursor=(1, 0)
                        )
                    ]
                    assert second == [event(1)]
                assert b"".join(handler_bytes) == b"".join(forwarded_bytes)
                assert [record["queryCursor"] for record in boundary.records] == [
                    {"attempt": 0, "seq": -1},
                    {"attempt": 1, "seq": 0},
                ]
                assert boundary.records[0]["registry"] == {
                    "eventCount": 1,
                    "cursors": {"0": -1, "1": 0},
                }
                assert boundary.records[1]["registry"] == {
                    "eventCount": 2,
                    "cursors": {"1": 1},
                }
                for record in boundary.records:
                    assert verify_boundary_record(record)["scope"] == "asgi-only"
            finally:
                store.close()

    asyncio.run(asyncio.wait_for(scenario(), timeout=5))
