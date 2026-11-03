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
    frames,
    loopback_app,
    metrics,
    open_log,
    read_events,
    start,
)
from wavedl_daemon.events.broker import EventBroker


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
        "runLog": {"eventCount": 4, "cursors": {"1": 3}},
    }
    assert verify_boundary_record(base) == {"scope": "asgi-only", "forwardedEvents": 1}
    claimed = {**base, "localCursor": [1, 3]}
    with pytest.raises(ValueError, match="ASGI-only"):
        verify_boundary_record(claimed)
    uncommitted = {**base, "runLog": {"eventCount": 3, "cursors": {"1": 2}}}
    with pytest.raises(ValueError, match="run log"):
        verify_boundary_record(uncommitted)


def test_real_authenticated_handler_snapshot_and_subsequent_request_cursor(tmp_path):
    """Real run log and routes; client is native HTTP, explicitly not a browser."""

    async def scenario():
        with open_log(tmp_path) as log:
            start(log)
            broker = EventBroker()
            real_app = authenticated_app(log, broker)
            handler_bytes, forwarded_bytes = [], []

            async def fragmented(scope, receive, send):
                async def fragment_send(message):
                    if message["type"] == "http.response.body" and message.get("body"):
                        handler_bytes.append(message["body"])
                        for value in message["body"]:
                            await send({**message, "body": bytes((value,))})
                    else:
                        await send(message)

                await real_app(scope, receive, fragment_send)

            boundary = StreamBoundary(fragmented, log)
            pending = bytearray()

            def disconnect_after_head(payload, disconnected):
                forwarded_bytes.append(payload)
                pending.extend(payload)
                # Each request ends once it forwarded the last committed row.
                if pending.endswith(b"\n\n") and (
                    b"id: 1:2" in pending or b"id: 1:3" in pending
                ):
                    pending.clear()
                    disconnected.set()

            async with loopback_app(boundary, on_body=disconnect_after_head) as (url, _):
                first = [item async for item in read_events(url, TOKEN, RUN)]
                assert first == frames(log)
                metrics(log, 1)
                second = [item async for item in read_events(url, TOKEN, RUN, cursor=(1, 2))]
                assert second == frames(log, (1, 2))
            assert b"".join(handler_bytes) == b"".join(forwarded_bytes)
            assert [record["queryCursor"] for record in boundary.records] == [
                {"attempt": 0, "seq": -1},
                {"attempt": 1, "seq": 2},
            ]
            assert boundary.records[0]["forwardedEventIds"] == [[1, 0], [1, 1], [1, 2]]
            assert boundary.records[0]["runLog"] == {
                "eventCount": 3,
                "cursors": {"0": -1, "1": 2},
            }
            assert boundary.records[1]["forwardedEventIds"] == [[1, 3]]
            assert boundary.records[1]["runLog"] == {
                "eventCount": 4,
                "cursors": {"1": 3},
            }
            for record in boundary.records:
                assert verify_boundary_record(record)["scope"] == "asgi-only"

    asyncio.run(asyncio.wait_for(scenario(), timeout=5))
