"""Directory durability waits only for restrictive native reader admission."""

import ctypes
from pathlib import Path
from types import SimpleNamespace

import pytest

from wavedl.core import persistence


class NativeFunction:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        return self.function(*args)


def native_sync(monkeypatch, errors, *, flush_error=0):
    calls = []
    clock = [0.0]
    last_error = [0]
    invalid = ctypes.c_void_p(-1).value

    def create(*args):
        calls.append(("create", args))
        last_error[0] = next(errors)
        return invalid if last_error[0] else 123

    def flush(handle):
        calls.append(("flush", handle))
        last_error[0] = flush_error
        return not flush_error

    def sleep(seconds):
        calls.append(("sleep", seconds))
        clock[0] += seconds

    def error(code):
        result = PermissionError(f"native error {code}")
        result.winerror = code
        return result

    kernel = SimpleNamespace(
        CreateFileW=NativeFunction(create),
        FlushFileBuffers=NativeFunction(flush),
        CloseHandle=NativeFunction(lambda handle: calls.append(("close", handle))),
    )
    monkeypatch.setattr(persistence, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(
        persistence, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep)
    )
    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **kw: kernel, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: last_error[0], raising=False)
    monkeypatch.setattr(ctypes, "WinError", error, raising=False)
    return calls, clock


def test_native_directory_flush_waits_for_reader_without_weaker_sharing(monkeypatch):
    calls, _ = native_sync(monkeypatch, iter([32, 32, 0]))
    persistence.sync_directory(Path("attempt"))
    assert [call[0] for call in calls] == [
        "create",
        "sleep",
        "create",
        "sleep",
        "create",
        "flush",
        "close",
    ]
    for kind, arguments in calls:
        if kind == "create":
            assert arguments == ("attempt", 0xC0000000, 7, None, 3, 0x02000000, None)
    assert [value for kind, value in calls if kind == "sleep"] == [0.02, 0.02]


def test_native_directory_flush_sharing_deadline_is_not_reset(monkeypatch):
    calls, clock = native_sync(monkeypatch, iter(lambda: 32, None))
    with pytest.raises(PermissionError) as caught:
        persistence.sync_directory(Path("attempt"))
    assert caught.value.winerror == 32
    assert clock[0] == 5
    assert not any(kind in ("flush", "close") for kind, _ in calls)
    assert calls[-1][0] == "sleep"


@pytest.mark.parametrize("error", [5, 6, 33, 50])
def test_native_directory_flush_other_admission_errors_are_immediate(
    monkeypatch, error
):
    calls, clock = native_sync(monkeypatch, iter([error]))
    with pytest.raises(PermissionError) as caught:
        persistence.sync_directory(Path("attempt"))
    assert caught.value.winerror == error
    assert [kind for kind, _ in calls] == ["create"]
    assert clock[0] == 0


@pytest.mark.parametrize("error", [5, 32, 50])
def test_native_directory_flush_failure_is_not_retried(monkeypatch, error):
    calls, _ = native_sync(monkeypatch, iter([0]), flush_error=error)
    with pytest.raises(PermissionError) as caught:
        persistence.sync_directory(Path("attempt"))
    assert caught.value.winerror == error
    assert [kind for kind, _ in calls] == ["create", "flush", "close"]
