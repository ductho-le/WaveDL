"""Actual loopback delivery, bounded IO cancellation, and file-authority fallbacks."""

import json
import socket
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from wavedl.core.events import (
    ArtifactPayload,
    ErrorPayload,
    ExitPayload,
    HelloPayload,
    MetricPayload,
    StatePayload,
    SupersedePayload,
    WarningPayload,
)
from wavedl.core.journal_reader import JournalReader
from wavedl.core.live_push import LivePushClient, LivePushConfig
from wavedl.core.manifest import RuntimeSelection
from wavedl.core.protocol import decode
from wavedl.journal import RunJournal, bootstrap_run


def wait_until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


@contextmanager
def server(mode="normal"):
    requests = []
    closed = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            payload = json.loads(body)
            requests.append((self.path, dict(self.headers), payload))
            event = payload["events"][0]
            if mode in {"trickle_header", "trickle_body"}:
                if mode == "trickle_body":
                    self.wfile.write(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n")
                try:
                    for value in (
                        b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n"
                        if mode == "trickle_header"
                        else b"x" * 100
                    ):
                        self.wfile.write(bytes([value]))
                        self.wfile.flush()
                        time.sleep(0.03)
                except OSError:
                    closed.set()
                return
            if mode in {"slow_header", "slow_body"}:
                if mode == "slow_header":
                    self.wfile.write(b"HTTP/1.1 200 OK\r\n")
                else:
                    self.wfile.write(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{}"
                    )
                self.wfile.flush()
                self.connection.settimeout(1.5)
                try:
                    if self.rfile.read(1) == b"":
                        closed.set()
                except OSError:
                    pass
                return
            ack = {"attempt": event["attempt"], "seq": event["seq"]}
            if mode == "ahead":
                ack["seq"] += 10
            elif mode == "wrong_attempt":
                ack["attempt"] += 1
            elif mode == "empty":
                ack = {}
            elif mode == "bool":
                ack["seq"] = True
            elif mode == "nan":
                ack["seq"] = float("nan")
            elif mode == "behind":
                ack["seq"] -= 1
            raw = json.dumps(ack).encode()
            self.send_response(503 if mode == "http_error" else 200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    instance = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=instance.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{instance.server_port}", requests, closed
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(1)


def journal(tmp_path, client=None, *, rank=0, bootstrap=None):
    if bootstrap is None:
        bootstrap = bootstrap_run(
            tmp_path / "run",
            resolved_config={},
            runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
            environment={},
        )
    return RunJournal(
        bootstrap,
        hello=HelloPayload(1, 1, "test", "3.11", "cpu", ["train"], sys.executable),
        push_client=client,
        rank=rank,
        is_main_process=rank == 0,
    )


def read_events(run):
    bootstrap = run.bootstrap
    return JournalReader(
        bootstrap.layout.journal(bootstrap.attempt),
        run_id=bootstrap.layout.run_id,
        attempt=bootstrap.attempt,
    ).read()


@pytest.mark.parametrize("mode", ["normal", "ahead"])
def test_actual_server_full_canonical_envelopes_and_order(tmp_path, mode):
    with server(mode) as (url, requests, _):
        client = LivePushClient(LivePushConfig(url, "token%2Fsecret"))
        run = journal(tmp_path, client)
        run.emit("state", StatePayload("running"))
        run.log("hello log")
        run.emit("metric", MetricPayload("loss", 2.0, 1))
        run.emit(
            "artifact",
            ArtifactPayload(
                "checkpoint", "checkpoints/gen-000001/COMPLETE.json", "a" * 64, 2
            ),
        )
        run.emit("supersede", SupersedePayload(1, 1, "gen-000001", source_seq=0))
        run.emit("warning", WarningPayload("TEST", "warning"))
        run.emit("error", ErrorPayload("TEST", "recoverable", recoverable=True))
        run.emit("exit", ExitPayload(0, "completed"))
        wait_until(lambda: client.counters["acknowledged"] == 9)
        run.close()
        assert client.counters["pending_bytes"] == 0
        authoritative = read_events(run)
        assert len(requests) == len(authoritative) == 9
        for request, event in zip(requests, authoritative, strict=True):
            path, headers, payload = request
            assert path == f"/api/runs/{event.run_id}/events"
            assert headers["Authorization"] == "Bearer token%2Fsecret"
            assert decode(json.dumps(payload["events"][0])) == event
        assert [item[2]["events"][0]["seq"] for item in requests] == list(range(9))


@pytest.mark.parametrize(
    "mode", ["wrong_attempt", "empty", "bool", "nan", "behind", "http_error"]
)
def test_bad_ack_disables_push_without_harming_file(tmp_path, mode):
    with server(mode) as (url, requests, _):
        client = LivePushClient(LivePushConfig(url, "secret"))
        run = journal(tmp_path, client)
        wait_until(lambda: client.counters["disabled"])
        run.emit("state", StatePayload("running"))
        run.emit("exit", ExitPayload(0, "completed"))
        run.close()
        assert len(requests) == 1
        assert [event.seq for event in read_events(run)] == [0, 1, 2]
        assert "secret" not in repr(client.counters)


def test_outage_uses_authoritative_file(tmp_path):
    with socket.socket() as unused:
        unused.bind(("127.0.0.1", 0))
        port = unused.getsockname()[1]
    config = LivePushConfig(f"http://127.0.0.1:{port}", "secret")
    client = LivePushClient(config)
    run = journal(tmp_path, client)
    try:
        # The network attempt may consume its entire product deadline before
        # Windows reports refusal. Allow finite worker/OS scheduling overhead;
        # this test checks file fallback, not the stricter IO-deadline contract.
        wait_until(lambda: client.counters["disabled"], timeout=config.deadline + 3)
        run.emit("state", StatePayload("running"))
    finally:
        run.close()
    assert client.counters["pending_bytes"] == 0
    wait_until(lambda: not client._thread.is_alive(), timeout=3)
    events = read_events(run)
    assert [event.seq for event in events] == [0, 1]
    assert [event.type for event in events] == ["hello", "state"]
    assert events[1].payload == {"state": "running", "message": ""}


@pytest.mark.parametrize(
    "mode", ["slow_header", "slow_body", "trickle_header", "trickle_body"]
)
def test_absolute_deadline_closes_actual_socket(tmp_path, mode):
    with server(mode) as (url, _, closed):
        client = LivePushClient(LivePushConfig(url, "secret", deadline=0.15))
        started = time.monotonic()
        run = journal(tmp_path, client)
        wait_until(lambda: client.counters["disabled"], timeout=0.8)
        assert closed.wait(0.3)
        assert time.monotonic() - started < 0.8
        run.close()
        assert client.counters["pending_bytes"] == 0
        wait_until(lambda: not client._thread.is_alive())


def test_overflow_cancels_live_only_and_non_main_never_submits(tmp_path):
    with server("slow_body") as (url, requests, closed):
        client = LivePushClient(
            LivePushConfig(url, "secret", max_record_bytes=1024, max_pending_bytes=1024)
        )
        run = journal(tmp_path, client)
        wait_until(lambda: bool(requests))
        run.log("x" * 1500)
        assert client.counters["disabled"]
        assert closed.wait(0.5)
        run.emit("state", StatePayload("completed"))
        run.close()
        assert len(read_events(run)) == 3

        class NoTraffic:
            def submit(self, event):
                pytest.fail("Follower must not push")

        follower = journal(tmp_path, NoTraffic(), rank=1, bootstrap=run.bootstrap)
        follower.log("follower log")
        follower.close()
        assert len(requests) == 1


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "http://127.1",
        "https://user:secret@example.com",
        "http://localhost?token=secret",
        "https://example.com/#secret",
        "https://bad%20host",
        "http://127.0.0.1:0",
    ],
)
def test_unsafe_origins_rejected_without_secrets(url):
    with pytest.raises(ValueError) as error:
        LivePushConfig(url, "token%2Fsecret")
    assert "secret" not in str(error.value)
    assert "secret" not in repr(LivePushConfig("https://example.com", "token%2Fsecret"))


def test_opt_in_configuration_and_bounded_close(tmp_path):
    assert LivePushConfig.from_environment({}) is None
    with pytest.raises(ValueError, match="Incomplete"):
        LivePushConfig.from_environment({"WAVEDL_DAEMON_TOKEN": "secret"})
    with server("slow_header") as (url, requests, closed):
        client = LivePushClient(LivePushConfig(url, "secret", deadline=10))
        run = journal(tmp_path, client)
        wait_until(lambda: bool(requests))
        started = time.monotonic()
        run.close()
        assert time.monotonic() - started < 0.5
        assert closed.wait(0.5)
        wait_until(lambda: not client._thread.is_alive())


def test_runtime_opt_in_does_not_persist_control_credentials(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from wavedl.runtime import open_training_runtime

    token = "ephemeral%2Fsecret%2Bcontrol"
    with server() as (url, requests, _):
        monkeypatch.setenv("WAVEDL_DAEMON_URL", url)
        monkeypatch.setenv("WAVEDL_DAEMON_TOKEN", token)
        accelerator = SimpleNamespace(
            process_index=0,
            is_main_process=True,
            num_processes=1,
            device=SimpleNamespace(type="cpu"),
            wait_for_everyone=lambda: None,
        )
        run = open_training_runtime(
            SimpleNamespace(output_dir=str(tmp_path / "run"), cv=0), accelerator
        )
        run.emit("exit", ExitPayload(0, "completed"))
        run.close()
        wait_until(lambda: len(requests) == 4)
        for path in (
            run.bootstrap.layout.manifest,
            run.bootstrap.layout.attempt_dir(1) / "runtime_request.json",
            run.bootstrap.layout.journal(1),
        ):
            assert token not in path.read_text()
            assert "WAVEDL_DAEMON_TOKEN" not in path.read_text()


def test_durable_append_precedes_push_and_optional_delivery_errors_are_isolated(
    tmp_path,
):
    observed = []

    class BrokenDelivery:
        def submit(self, event):
            path = tmp_path / "run/attempts/1/events.jsonl"
            assert decode(path.read_bytes().splitlines()[-1]) == event
            observed.append(event)
            if event.seq == 1:
                raise RuntimeError("optional push failed")

        def close(self):
            raise RuntimeError("optional close failed")

    run = journal(tmp_path, BrokenDelivery())
    run.emit("state", StatePayload("running"))
    run.emit("exit", ExitPayload(0, "completed"))
    run.close()
    assert [item.seq for item in observed] == [0, 1]
    assert [item.seq for item in read_events(run)] == [0, 1, 2]
