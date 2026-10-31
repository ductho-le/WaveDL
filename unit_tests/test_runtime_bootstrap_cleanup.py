"""Runtime startup failures release journals before lifecycle entry."""

from types import SimpleNamespace

import pytest

from wavedl import journal as journal_module, runtime
from wavedl.core.journal_writer import JournalWriter
from wavedl.core.live_push import LivePushClient
from wavedl.core.training_cli import parse_training_args


@pytest.fixture
def startup(tmp_path, monkeypatch):
    for name in ("WAVEDL_RUN_ID", "WAVEDL_REQUEST_ID", "WAVEDL_ATTEMPT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("WAVEDL_DAEMON_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("WAVEDL_DAEMON_TOKEN", "startup-test-token")
    # Exercise a real idle push thread without starting network requests.
    monkeypatch.setattr(LivePushClient, "submit", lambda *args: False)
    writers, clients = [], []

    def writer(*args, **kwargs):
        result = JournalWriter(*args, **kwargs)
        writers.append(result)
        return result

    def client(config):
        result = LivePushClient(config)
        clients.append(result)
        return result

    monkeypatch.setattr(journal_module, "JournalWriter", writer)
    monkeypatch.setattr(runtime, "LivePushClient", client)
    monkeypatch.setattr(runtime, "wait_for_workers", lambda *args: None)
    args, _ = parse_training_args(["--output_dir", str(tmp_path)])
    accelerator = SimpleNamespace(
        process_index=0,
        is_main_process=True,
        num_processes=1,
        device=SimpleNamespace(type="cpu"),
    )
    yield args, accelerator, writers, clients
    # Failed assertions must not leak the resources this test observes.
    for resource in writers + clients:
        resource.close()


@pytest.mark.parametrize("event_type", ["hello", "log", "state"])
def test_startup_append_failure_closes_real_writer_and_push_thread(
    startup, monkeypatch, event_type
):
    args, accelerator, writers, clients = startup
    original = JournalWriter.append
    primary = OSError(f"{event_type} append failed")

    def append(self, kind, *args, **kwargs):
        if kind == event_type:
            raise primary
        return original(self, kind, *args, **kwargs)

    monkeypatch.setattr(JournalWriter, "append", append)
    with pytest.raises(OSError) as caught:
        runtime.open_training_runtime(args, accelerator)
    assert caught.value is primary
    assert len(writers) == len(clients) == 1
    assert writers[0]._stream.closed
    assert clients[0]._closing
    clients[0]._thread.join(1)
    assert not clients[0]._thread.is_alive()


@pytest.mark.parametrize("primary", [OSError("startup failed"), KeyboardInterrupt()])
def test_startup_and_close_failures_preserve_both_exceptions(
    startup, monkeypatch, primary
):
    args, accelerator, writers, clients = startup
    cleanup = OSError("journal close failed")
    original_close = runtime.RunJournal.close

    def log(*args, **kwargs):
        raise primary

    def close(self):
        original_close(self)
        raise cleanup

    monkeypatch.setattr(runtime.RunJournal, "log", log)
    monkeypatch.setattr(runtime.RunJournal, "close", close)
    with pytest.raises(BaseExceptionGroup) as caught:
        runtime.open_training_runtime(args, accelerator)
    assert caught.value.exceptions == (primary, cleanup)
    assert writers[0]._stream.closed
    assert clients[0]._closing
    clients[0]._thread.join(1)
    assert not clients[0]._thread.is_alive()
