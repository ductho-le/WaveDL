import asyncio
import hashlib
import os
from contextlib import asynccontextmanager, suppress
from types import SimpleNamespace

import pytest

from wavedl.core.client_transport import (
    ArtifactRef,
    ClientTransportError,
    HttpTransport,
    StreamControl,
)
from wavedl.core.protocol import Envelope, encode


RUN = "00000000-0000-4000-8000-000000000001"
TOKEN = "private-transport-test-token"


def transport_failure_detail(error):
    cause = error.__context__
    # Native failures are deliberately hidden at the public transport boundary.
    # Identify the failing internal operation without printing private paths,
    # traceback source lines, locals, or the OS error's potentially private text.
    frames = []
    traceback = cause.__traceback__ if cause is not None else None
    while traceback is not None:
        frame = traceback.tb_frame
        if frame.f_globals.get("__name__") in {
            "wavedl.core.client_transport",
            "wavedl.core.safe_files",
            "wavedl.core.windows_ownership",
        }:
            frames.append(f"{frame.f_code.co_name}:{traceback.tb_lineno}")
        traceback = traceback.tb_next
    return (
        f"{type(cause).__name__}, errno={getattr(cause, 'errno', None)}, "
        f"winerror={getattr(cause, 'winerror', None)}, "
        f"operations={'/'.join(frames) or 'unavailable'}"
    )


async def ready_or_task_failure(ready, task):
    """A pre-request failure must fail the test, not strand its readiness wait."""
    readiness = asyncio.create_task(ready.wait())
    observed = False
    try:
        done, _ = await asyncio.wait(
            {readiness, task}, timeout=5, return_when=asyncio.FIRST_COMPLETED
        )
        if task in done:
            try:
                await task
            except ClientTransportError as error:
                pytest.fail(
                    "Transport failed before readiness: "
                    + transport_failure_detail(error)
                )
            pytest.fail("Transport finished before readiness")
        assert readiness in done, "Loopback readiness deadline exceeded"
        observed = True
    finally:
        readiness.cancel()
        await asyncio.gather(readiness, return_exceptions=True)
        if not observed:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@asynccontextmanager
async def server(handler):
    tasks = set()

    async def accepted(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        try:
            await handler(reader, writer)
        finally:
            writer.close()
            with suppress(ConnectionResetError):
                await writer.wait_closed()
            tasks.discard(task)

    listener = await asyncio.start_server(accepted, "127.0.0.1", 0)
    try:
        yield f"http://127.0.0.1:{listener.sockets[0].getsockname()[1]}"
    finally:
        listener.close()
        await listener.wait_closed()
        for task in tuple(tasks):
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def envelope(seq, attempt=1):
    return Envelope(
        RUN, attempt, "log", seq, "2026-10-04T00:00:00Z", {"message": "Tiếng Việt 🌊"}
    )


def frame(event):
    return f"id: {event.attempt}:{event.seq}\ndata: ".encode() + encode(event) + b"\n"


def test_real_loopback_json_query_auth_and_last_event_id():
    async def run():
        async def handler(reader, writer):
            request = await reader.readuntil(b"\r\n\r\n")
            assert b"GET /api/runs?limit=2&state=running HTTP/1.1" in request
            assert f"Authorization: Bearer {TOKEN}".encode() in request
            assert b"Last-Event-ID: 1:2" in request
            body = b'{"runs":[]}'
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                + str(len(body)).encode()
                + b"\r\n\r\n"
                + body
            )
            await writer.drain()

        async with server(handler) as url:
            transport = HttpTransport(url, TOKEN)
            assert TOKEN not in repr(transport)
            assert await transport.request(
                "GET",
                "/api/runs",
                query={"limit": 2, "state": "running"},
                last_event_id="1:2",
            ) == {"runs": []}

    asyncio.run(run())


def test_bound_daemon_identity_is_sent_before_request_and_cannot_be_rebound():
    async def run():
        requests = []

        async def handler(reader, writer):
            requests.append(await reader.readuntil(b"\r\n\r\n"))
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}"
            )
            await writer.drain()

        async with server(handler) as url:
            transport = HttpTransport(url, TOKEN)
            transport.bind_daemon_id("approved-daemon")
            transport.bind_daemon_id("approved-daemon")
            with pytest.raises(ClientTransportError, match="changed"):
                transport.bind_daemon_id("foreign-daemon")
            assert await transport.request("POST", "/api/runs", body={}) == {}
        assert len(requests) == 1
        assert b"X-WaveDL-Daemon-ID: approved-daemon\r\n" in requests[0]

    asyncio.run(run())


@pytest.mark.parametrize("status", [302, 401, 409, 503])
def test_http_refusal_preserves_only_status_for_retry_policy(status):
    async def run():
        async def handler(reader, writer):
            await reader.readuntil(b"\r\n\r\n")
            writer.write(
                f"HTTP/1.1 {status} raw-private-reason\r\nContent-Length: 20\r\n\r\nprivate-response-data".encode()
            )
            await writer.drain()

        async with server(handler) as url:
            with pytest.raises(ClientTransportError) as captured:
                await HttpTransport(url, TOKEN).request("GET", "/api/health")
            assert captured.value.status_code == status
            assert str(captured.value) == "Daemon request refused"
            assert "private" not in str(captured.value) + repr(captured.value)

    asyncio.run(run())


def test_chunked_utf8_stream_deduplicates_and_controls_never_advance():
    async def run():
        async def handler(reader, writer):
            await reader.readuntil(b"\r\n\r\n")
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
            )
            body = (
                frame(envelope(0))
                + frame(envelope(0))
                + frame(envelope(1))
                + b'event: overflow\ndata: {"resumePosition":{"attempt":1,"seq":1}}\n\n'
            )
            for byte in body:
                writer.write(b"1\r\n" + bytes((byte,)) + b"\r\n")
            writer.write(b"0\r\n\r\n")
            await writer.drain()

        async with server(handler) as url:
            events = [item async for item in HttpTransport(url, TOKEN).events(RUN)]
            assert events == [
                envelope(0),
                envelope(1),
                StreamControl("overflow", (1, 1)),
            ]

    asyncio.run(run())


def test_heartbeats_use_idle_timeout_not_total_stream_duration():
    async def run():
        async def handler(reader, writer):
            await reader.readuntil(b"\r\n\r\n")
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
            )
            for _ in range(8):
                writer.write(b"3\r\n:\n\n\r\n")
                await writer.drain()
                await asyncio.sleep(0.02)
            body = frame(envelope(0))
            writer.write(f"{len(body):X}\r\n".encode() + body + b"\r\n0\r\n\r\n")
            await writer.drain()

        async with server(handler) as url:
            assert [
                item
                async for item in HttpTransport(url, TOKEN, deadline=0.07).events(RUN)
            ] == [envelope(0)]

    asyncio.run(run())


@pytest.mark.parametrize(
    "response",
    [
        b"HTTP/1.1 302 Found\r\nLocation: http://foreign/\r\nContent-Length: 0\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nContent-Length: 2\r\n\r\n{}",
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nTransfer-Encoding: chunked\r\n\r\n{}",
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\nG\r\nsecret\r\n0\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 10\r\n\r\n{}",
        b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 13\r\n\r\n{"x":1,"x":2}',
    ],
)
def test_invalid_response_is_refused_without_leaking_token(response):
    async def run():
        async def handler(reader, writer):
            await reader.readuntil(b"\r\n\r\n")
            writer.write(response)
            await writer.drain()

        async with server(handler) as url:
            with pytest.raises(ClientTransportError) as error:
                await HttpTransport(url, TOKEN).request("GET", "/api/health")
            assert TOKEN not in str(error.value)

    asyncio.run(run())


def test_stream_cancellation_closes_actual_socket():
    async def run():
        ready, disconnected = asyncio.Event(), asyncio.Event()

        async def handler(reader, writer):
            await reader.readuntil(b"\r\n\r\n")
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
            )
            await writer.drain()
            ready.set()
            # Cancelling while the response header is still unread can close
            # TCP with RST rather than FIN. Both prove the socket was closed.
            try:
                assert await reader.read() == b""
            except ConnectionResetError:
                pass
            disconnected.set()

        async with server(handler) as url:
            stream = HttpTransport(url, TOKEN).events(RUN)
            task = asyncio.create_task(anext(stream))
            await ready_or_task_failure(ready, task)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(disconnected.wait(), 1)

    asyncio.run(run())


@pytest.mark.parametrize("windows_path_branch", [False, True])
def test_binary_download_verified_exclusive_and_partial_cleanup(
    tmp_path, monkeypatch, windows_path_branch
):
    from wavedl.core import client_transport

    if windows_path_branch:
        # Exercise Windows pathname publication on POSIX too, while retaining
        # genuine loopback bytes and an actually held writable directory.
        monkeypatch.setattr(
            client_transport, "os", SimpleNamespace(**(vars(os) | {"name": "nt"}))
        )
    original_open = client_transport.open_directory
    opened = []
    synced = []

    def held_parent(path, *, writable=False):
        assert writable, "Download publication requires a writable capability"
        parent = original_open(path, writable=writable)
        original_sync = parent.sync

        def sync():
            synced.append(parent)
            original_sync()

        parent.sync = sync
        opened.append(parent)
        return parent

    monkeypatch.setattr(client_transport, "open_directory", held_parent)
    data = b"binary-content" * 30000

    async def run():
        async def handler(reader, writer):
            request = await reader.readuntil(b"\r\n\r\n")
            assert b"/gen-000001/nested/report%20file.bin" in request
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: "
                + str(len(data)).encode()
                + b"\r\n\r\n"
                + data
            )
            await writer.drain()

        ref = ArtifactRef(
            RUN,
            "gen-000001",
            "nested/report file.bin",
            len(data),
            hashlib.sha256(data).hexdigest(),
        )
        async with server(handler) as url:
            transport = HttpTransport(url, TOKEN)
            destination = tmp_path / "download.bin"
            try:
                actual = await transport.download(ref, destination, max_bytes=len(data))
            except ClientTransportError as error:
                pytest.fail(
                    "Binary download failed: " + transport_failure_detail(error)
                )
            assert actual == destination
            assert destination.read_bytes() == data
            with pytest.raises(ClientTransportError):
                await transport.download(ref, destination, max_bytes=len(data))
            bad = ArtifactRef(RUN, "gen-000001", ref.path, len(data), "0" * 64)
            with pytest.raises(ClientTransportError):
                await transport.download(bad, tmp_path / "bad.bin", max_bytes=len(data))
            assert not (tmp_path / "bad.bin").exists()
            assert not list(tmp_path.glob(".wavedl-download-*"))

    asyncio.run(run())
    assert len(opened) == 3
    assert synced == [opened[0], opened[0]]
    assert all(parent.handle is None for parent in opened)


def test_plain_http_external_host_is_refused_before_connection():
    with pytest.raises(ValueError):
        HttpTransport("http://example.com", TOKEN)


def test_failed_download_before_readiness_surfaces_sanitized_context_without_hanging():
    async def run():
        async def failed():
            try:
                cause = PermissionError(13, "private path must not enter diagnostics")
                cause.winerror = 32
                raise cause
            except PermissionError:
                raise ClientTransportError("Artifact download unavailable") from None

        task = asyncio.create_task(failed())
        with pytest.raises(pytest.fail.Exception) as failure:
            await ready_or_task_failure(asyncio.Event(), task)
        assert str(failure.value) == (
            "Transport failed before readiness: PermissionError, errno=13, "
            "winerror=32, operations=unavailable"
        )
        assert task.done()

    asyncio.run(run())


def test_transport_failure_diagnostics_identify_internal_phase_without_private_data(
    monkeypatch, tmp_path
):
    from wavedl.core import client_transport

    private = "private-token-and-path-must-not-appear"

    def fail_open(*args, **kwargs):
        error = PermissionError(13, private, str(tmp_path / private))
        error.winerror = 32
        raise error

    monkeypatch.setattr(client_transport, "open_directory", fail_open)
    ref = ArtifactRef(RUN, "gen-000001", "file.bin", 0, "0" * 64)
    with pytest.raises(ClientTransportError) as caught:
        asyncio.run(
            HttpTransport("http://127.0.0.1:1", TOKEN).download(
                ref, tmp_path / "result.bin", max_bytes=1
            )
        )
    detail = transport_failure_detail(caught.value)
    assert "PermissionError, errno=13, winerror=32, operations=download:" in detail
    assert private not in detail and TOKEN not in detail
    assert str(tmp_path) not in detail and "fail_open" not in detail


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory-rename race")
def test_download_parent_replacement_cannot_publish_or_cleanup_in_foreign_directory(
    tmp_path,
):
    chosen, moved, foreign = (
        tmp_path / "chosen",
        tmp_path / "moved",
        tmp_path / "foreign",
    )
    chosen.mkdir(mode=0o700)
    foreign.mkdir(mode=0o700)
    sentinel = foreign / ".wavedl-download-sentinel"
    sentinel.write_bytes(b"keep")
    data = b"content"

    async def run():
        async def handler(reader, writer):
            await reader.readuntil(b"\r\n\r\n")
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: 7\r\n\r\ncon"
            )
            await writer.drain()
            deadline = asyncio.get_running_loop().time() + 1
            while (
                not list(chosen.glob(".wavedl-download-*"))
                and asyncio.get_running_loop().time() < deadline
            ):
                await asyncio.sleep(0.005)
            assert list(chosen.glob(".wavedl-download-*"))
            chosen.rename(moved)
            chosen.symlink_to(foreign, target_is_directory=True)
            writer.write(b"tent")
            await writer.drain()

        async with server(handler) as url:
            ref = ArtifactRef(
                RUN,
                "gen-000001",
                "file.bin",
                len(data),
                hashlib.sha256(data).hexdigest(),
            )
            with pytest.raises(ClientTransportError):
                await HttpTransport(url, TOKEN).download(
                    ref, chosen / "file.bin", max_bytes=10
                )
        assert not (foreign / "file.bin").exists()
        assert not (moved / "file.bin").exists()
        assert not list(moved.glob(".wavedl-download-*"))
        assert sentinel.read_bytes() == b"keep"

    asyncio.run(run())


def test_download_refuses_existing_destination_symlink(tmp_path):
    target = tmp_path / "existing"
    target.write_bytes(b"keep")
    destination = tmp_path / "link"
    destination.symlink_to(target)
    ref = ArtifactRef(RUN, "gen-000001", "file.bin", 0, hashlib.sha256(b"").hexdigest())
    with pytest.raises(ClientTransportError):
        asyncio.run(
            HttpTransport("http://127.0.0.1:1", TOKEN).download(
                ref, destination, max_bytes=0
            )
        )
    assert target.read_bytes() == b"keep"
    assert destination.is_symlink()


def test_resolved_plain_http_peer_is_checked_before_bearer_write(monkeypatch):
    class Writer:
        def __init__(self):
            self.writes = []
            self.closed = False

        def get_extra_info(self, name):
            return ("192.0.2.10", 80)

        def write(self, data):
            self.writes.append(data)

        def close(self):
            self.closed = True

        async def wait_closed(self):
            pass

    writer = Writer()

    async def connect(*args, **kwargs):
        return None, writer

    monkeypatch.setattr(asyncio, "open_connection", connect)

    async def run():
        with pytest.raises(ClientTransportError):
            await HttpTransport("http://localhost", TOKEN).request("GET", "/api/health")
        assert writer.writes == []
        assert writer.closed

    asyncio.run(run())


def test_idle_timeout_and_cancelled_download_close_socket_and_remove_temp(tmp_path):
    async def run():
        ready, disconnected = asyncio.Event(), asyncio.Event()

        async def handler(reader, writer):
            await reader.readuntil(b"\r\n\r\n")
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: 100\r\n\r\nfirst"
            )
            await writer.drain()
            ready.set()
            assert await reader.read() == b""
            disconnected.set()

        async with server(handler) as url:
            ref = ArtifactRef(RUN, "gen-000001", "file.bin", 100, "0" * 64)
            transport = HttpTransport(url, TOKEN)
            downloading = asyncio.create_task(
                transport.download(ref, tmp_path / "cancel.bin", max_bytes=100)
            )
            await ready_or_task_failure(ready, downloading)
            await asyncio.sleep(0.01)
            downloading.cancel()
            with pytest.raises(asyncio.CancelledError):
                await downloading
            await asyncio.wait_for(disconnected.wait(), 1)
            assert not (tmp_path / "cancel.bin").exists()
            assert not list(tmp_path.glob(".wavedl-download-*"))

        async def stalled(reader, writer):
            await reader.readuntil(b"\r\n\r\n")
            await reader.read()

        async with server(stalled) as url:
            with pytest.raises(ClientTransportError):
                await HttpTransport(url, TOKEN, deadline=0.03).request(
                    "GET", "/api/health"
                )

    asyncio.run(run())


@pytest.mark.parametrize(
    "body",
    [
        b"\xff\n\n",
        b"data: {}\n\n",
        b'event: overflow\ndata: {"resumePosition":{"attempt":1,"seq":999}}\n\n',
    ],
)
def test_invalid_stream_encoding_identity_or_control_cannot_advance(body):
    async def run():
        async def handler(reader, writer):
            await reader.readuntil(b"\r\n\r\n")
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: "
                + str(len(body)).encode()
                + b"\r\n\r\n"
                + body
            )
            await writer.drain()

        async with server(handler) as url:
            with pytest.raises(ClientTransportError):
                _ = [item async for item in HttpTransport(url, TOKEN).events(RUN)]

    asyncio.run(run())
