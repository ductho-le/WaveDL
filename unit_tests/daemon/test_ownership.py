from pathlib import Path

import pytest

from wavedl_daemon.storage.ownership import (
    DispatchHeld,
    DispatchLock,
    DispatchRequired,
    acquire_dispatch_lock,
    require_dispatch,
)


HOLDER = """
import sys, time
sys.path.insert(0, {src!r})
from wavedl_daemon.storage.ownership import DispatchLock
lock = DispatchLock({path!r})
lock.acquire()
print("held", flush=True)
time.sleep(300)
"""


def holder_script(path):
    return HOLDER.format(src=str(Path("src").resolve()), path=str(path))


def test_second_daemon_is_refused(tmp_path, paused_process):
    path = tmp_path / "dispatch.lock"
    with (
        paused_process(holder_script(path), "held"),
        pytest.raises(DispatchHeld, match="another process owns dispatch"),
    ):
        DispatchLock(path).acquire()


def test_killed_daemon_releases_the_lock(tmp_path, paused_process):
    """SIGKILL runs no cleanup, so the kernel must be the one releasing it."""
    path = tmp_path / "dispatch.lock"
    with paused_process(holder_script(path), "held") as holder:
        holder.kill()
        holder.wait(timeout=10)
        with acquire_dispatch_lock(path) as successor:
            assert successor.held


def test_a_stale_lock_file_does_not_strand_the_successor(tmp_path, paused_process):
    """The file surviving with a dead pid inside it must not block anyone."""
    path = tmp_path / "dispatch.lock"
    with paused_process(holder_script(path), "held") as holder:
        holder.kill()
        holder.wait(timeout=10)
        recorded = path.read_text().strip()
        assert recorded == str(holder.pid)  # the pid is recorded ...
    assert path.exists()  # ... and the file outlives its writer

    with acquire_dispatch_lock(path) as lock:  # ... and neither blocks ownership
        assert lock.held


def test_release_makes_ownership_available_again(tmp_path):
    path = tmp_path / "dispatch.lock"
    first = DispatchLock(path)
    first.acquire()
    first.release()

    with acquire_dispatch_lock(path) as second:
        assert second.held


def test_context_manager_releases_on_exception(tmp_path):
    path = tmp_path / "dispatch.lock"

    with pytest.raises(ZeroDivisionError), acquire_dispatch_lock(path):
        raise ZeroDivisionError

    with acquire_dispatch_lock(path) as lock:
        assert lock.held


def test_mutation_without_ownership_is_refused(tmp_path):
    """Every externally visible mutation is gated on this."""
    lock = DispatchLock(tmp_path / "dispatch.lock")

    with pytest.raises(DispatchRequired):
        require_dispatch(lock)
    with pytest.raises(DispatchRequired):
        require_dispatch(None)

    lock.acquire()
    require_dispatch(lock)
    lock.release()
    with pytest.raises(DispatchRequired):
        require_dispatch(lock)


def test_double_acquire_by_the_same_object_is_refused(tmp_path):
    lock = DispatchLock(tmp_path / "dispatch.lock")
    lock.acquire()
    try:
        with pytest.raises(DispatchHeld, match="already acquired"):
            lock.acquire()
    finally:
        lock.release()


def test_lock_directory_is_created_on_demand(tmp_path):
    path = tmp_path / "missing" / "dispatch.lock"

    with acquire_dispatch_lock(path) as lock:
        assert lock.held
        assert path.exists()


def test_same_process_separate_lock_objects_contend(tmp_path):
    path = tmp_path / "dispatch.lock"
    with acquire_dispatch_lock(path), pytest.raises(DispatchHeld):
        DispatchLock(path).acquire()


def test_forked_child_cannot_use_parent_ownership(tmp_path, monkeypatch):
    from wavedl_daemon.storage import ownership

    with acquire_dispatch_lock(tmp_path / "dispatch.lock") as lock:
        assert lock.held
        monkeypatch.setattr(ownership.os, "getpid", lambda: -1)
        with pytest.raises(DispatchRequired):
            require_dispatch(lock)


def test_windows_lock_starts_at_byte_zero_even_for_existing_file(tmp_path, monkeypatch):
    import importlib.util
    import os
    import sys
    import types

    import wavedl_daemon.storage.ownership as ownership

    offsets = []
    fake = types.ModuleType("msvcrt")
    fake.LK_NBLCK = 1
    fake.LK_UNLCK = 2
    fake.locking = lambda fd, mode, size: offsets.append(os.lseek(fd, 0, os.SEEK_CUR))
    spec = importlib.util.spec_from_file_location(
        "windows_ownership", ownership.__file__
    )
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as patches:
        patches.setattr(sys, "platform", "win32")
        patches.setitem(sys.modules, "msvcrt", fake)
        spec.loader.exec_module(module)
    path = tmp_path / "dispatch.lock"
    path.write_text("12345\n")
    with module.acquire_dispatch_lock(path):
        pass
    assert offsets == [0, 0]
