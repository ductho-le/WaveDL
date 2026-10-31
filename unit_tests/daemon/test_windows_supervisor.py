"""Portable lifecycle probes plus native Windows Job Object acceptance tests."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from wavedl_daemon.supervision.base import ProcessSpec
from wavedl_daemon.supervision.windows import (
    CREATE_NEW_PROCESS_GROUP,
    CREATE_SUSPENDED,
    WindowsSupervisor,
    _Win32,
)


native = pytest.mark.skipif(sys.platform != "win32", reason="native Win32 required")


class FakeAPI:
    """Tests call ordering/cleanup, not native Windows semantics."""

    def __init__(self, fail=None):
        self.events = []
        self.fail = fail
        self.members = 1

    def create_job(self):
        self.events.append("job")
        return 10

    def spawn(self, spec, flags):
        assert flags & CREATE_SUSPENDED
        assert flags & CREATE_NEW_PROCESS_GROUP
        self.events.append("suspended")
        stdout, out_write = os.pipe()
        stderr, err_write = os.pipe()
        os.write(out_write, b"stdout")
        os.write(err_write, b"stderr")
        os.close(out_write)
        os.close(err_write)
        self.reads = (stdout, stderr)
        return 11, 12, 123, stdout, stderr

    def assign(self, job, process):
        self.events.append("assign")
        if self.fail == "assign":
            raise OSError("assignment failed")

    def resume(self, thread):
        self.events.append("resume")
        if self.fail == "resume":
            raise OSError("resume failed")

    def close(self, handle):
        self.events.append(("close", handle))

    def terminate_process(self, process):
        self.events.append("terminate_process")
        self.members = 0

    def terminate_job(self, job):
        self.events.append("terminate_job")
        self.members = 0

    def active(self, job):
        return self.members

    def send_break(self, pid):
        self.events.append("break")

    def exit_code(self, process):
        return 1

    def leader_running(self, process):
        return True


def test_win32_spawn_forwards_isolated_environment_without_ambient_merge(
    tmp_path, monkeypatch
):
    captured = []
    monkeypatch.setenv("SBATCH_JOB_NAME", "ambient-must-not-return")
    monkeypatch.setattr(
        subprocess,
        "STARTUPINFO",
        lambda **kwargs: SimpleNamespace(**kwargs),
        raising=False,
    )
    monkeypatch.setattr(subprocess, "STARTF_USESTDHANDLES", 256, raising=False)
    monkeypatch.setattr(os, "set_handle_inheritable", lambda *args: None, raising=False)
    api = object.__new__(_Win32)
    api.msvcrt = SimpleNamespace(get_osfhandle=lambda fd: fd)

    def create(*args):
        captured.append(args[6])
        return 11, 12, 123, None

    api.winapi = SimpleNamespace(CreateProcess=create)
    _, _, _, stdout, stderr = api.spawn(
        ProcessSpec(
            ("unused",), tmp_path, {"EXPLICIT": "yes"}, inherit_environment=False
        ),
        CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP,
    )
    os.close(stdout)
    os.close(stderr)
    assert captured == [{"EXPLICIT": "yes", "WAVEDL_SUPERVISED": "1"}]


def test_portable_assign_before_resume_and_cancel_order(tmp_path):
    api = FakeAPI()
    process = WindowsSupervisor(_api=api).start(
        ProcessSpec(("unused",), tmp_path), output_limit=3
    )
    process.cancel(grace_seconds=0)
    process.cancel(grace_seconds=0)
    result = process.wait()
    assert process.wait() is result
    assert api.events[:4] == ["job", "suspended", "assign", "resume"]
    assert api.events.count("break") == 1
    assert api.events.count("terminate_job") == 1
    assert api.events.index("break") < api.events.index("terminate_job")
    assert result.stdout == b"std" and result.stderr == b"std"
    assert result.stdout_truncated and result.stderr_truncated
    assert result.cancelled and not result.timed_out
    assert api.events[-2:] == [("close", 10), ("close", 11)]


@pytest.mark.parametrize("failure", ["assign", "resume"])
def test_portable_failed_spawn_reaps_suspended_child(tmp_path, failure):
    api = FakeAPI(failure)
    with pytest.raises(OSError, match=f"{failure}.*failed"):
        WindowsSupervisor(_api=api).start(ProcessSpec(("unused",), tmp_path))
    assert "terminate_process" in api.events
    assert api.events[-3:] == [("close", 12), ("close", 11), ("close", 10)]
    for fd in api.reads:
        with pytest.raises(OSError):
            os.fstat(fd)
    if failure == "assign":
        assert "resume" not in api.events


def test_portable_timeout_kills_job(tmp_path):
    api = FakeAPI()
    process = WindowsSupervisor(_api=api).start(ProcessSpec(("unused",), tmp_path))
    result = process.wait(timeout=0)
    assert result.timed_out and result.cancelled
    assert "terminate_job" in api.events


def test_portable_timeout_accelerates_existing_long_grace(tmp_path):
    api = FakeAPI()
    process = WindowsSupervisor(_api=api).start(ProcessSpec(("unused",), tmp_path))
    process.cancel(grace_seconds=60)
    started = time.monotonic()
    result = process.wait(timeout=0.05)
    assert result.timed_out and time.monotonic() - started < 1
    assert api.events.count("break") == 1
    assert api.events.count("terminate_job") == 1


def test_portable_leader_exit_cleans_descendants_without_wait_timeout(tmp_path):
    api = FakeAPI()
    api.leader_running = lambda process: False
    api.exit_code = lambda process: 0
    process = WindowsSupervisor(_api=api).start(ProcessSpec(("unused",), tmp_path))
    result = process.wait()
    assert result.returncode == 0
    assert not result.timed_out and not result.cancelled
    assert "terminate_job" in api.events


@native
def test_native_large_independent_streams_and_closed_stdin(tmp_path):
    code = (
        "import os,sys; assert sys.stdin.buffer.read()==b''; "
        "assert os.environ['WAVEDL_SUPERVISED']=='1'; "
        "os.write(1,b'o'*2000000); os.write(2,b'e'*2000000)"
    )
    process = WindowsSupervisor().start(
        ProcessSpec((sys.executable, "-c", code), tmp_path, {"WAVEDL_SUPERVISED": "0"}),
        output_limit=1024,
    )
    result = process.wait(timeout=20)
    assert result.returncode == 0 and not result.timed_out
    assert result.stdout == b"o" * 1024 and result.stderr == b"e" * 1024
    assert result.stdout_truncated and result.stderr_truncated
    process.cancel(grace_seconds=0)
    assert process.wait() is result


@native
def test_native_assignment_failure_never_executes_child(tmp_path):
    marker = tmp_path / "executed"

    class FailingAssignment(_Win32):
        def assign(self, job, process):
            self.child = process
            raise OSError("deliberate assignment failure")

        def terminate_process(self, process):
            super().terminate_process(process)
            assert self.winapi.GetExitCodeProcess(process) != 259

    api = FailingAssignment()
    with pytest.raises(OSError, match="deliberate assignment"):
        WindowsSupervisor(_api=api).start(
            ProcessSpec(
                (
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).touch()",
                ),
                tmp_path,
            )
        )
    assert not marker.exists()


@native
def test_native_suspended_assignment_and_descendant_inherits_job(tmp_path):
    marker = tmp_path / "executed"

    class InspectAssignment(_Win32):
        def assign(self, job, process):
            assert not marker.exists()
            super().assign(job, process)
            assert self.active(job) == 1

        def resume(self, thread):
            assert not marker.exists()
            super().resume(thread)

    # The descendant checks its own job membership, not a mocked parent flag.
    child = (
        "import ctypes; from ctypes import wintypes; "
        "k=ctypes.WinDLL('kernel32',use_last_error=True); "
        "k.IsProcessInJob.argtypes=[wintypes.HANDLE,wintypes.HANDLE,"
        "ctypes.POINTER(wintypes.BOOL)]; "
        "k.GetCurrentProcess.restype=wintypes.HANDLE; b=wintypes.BOOL(); "
        "assert k.IsProcessInJob(k.GetCurrentProcess(),None,ctypes.byref(b)); "
        "assert b.value; print('in-job')"
    )
    code = (
        f"from pathlib import Path; Path({str(marker)!r}).touch(); "
        f"import subprocess,sys; subprocess.run([sys.executable,'-c',{child!r}],"
        "check=True)"
    )
    result = (
        WindowsSupervisor(_api=InspectAssignment())
        .start(ProcessSpec((sys.executable, "-c", code), tmp_path))
        .wait(timeout=20)
    )
    assert result.returncode == 0 and result.stdout.strip() == b"in-job"


@native
@pytest.mark.parametrize("leader_exits", [False, True])
def test_native_ignoring_descendant_killed_even_after_leader_exit(
    tmp_path, leader_exits
):
    import _winapi

    ready = tmp_path / "ready"
    pid_file = tmp_path / "descendant-pid"
    release = tmp_path / "release-leader"
    child = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGBREAK,signal.SIG_IGN); "
        f"Path({str(pid_file)!r}).write_text(str(os.getpid())); "
        f"Path({str(ready)!r}).touch(); time.sleep(120)"
    )
    # Hold the leader until the descendant has installed its signal handler
    # and the test owns a native handle. Otherwise automatic leader-exit
    # cleanup can correctly kill a slow-starting interpreter before readiness.
    code = (
        "import subprocess,sys,time,signal; "
        "from pathlib import Path; "
        "signal.signal(signal.SIGBREAK,signal.SIG_IGN); "
        f"subprocess.Popen([sys.executable,'-c',{child!r}]); "
        f"release=Path({str(release)!r}); "
        "deadline=time.monotonic()+30\n"
        "while not release.exists() and time.monotonic()<deadline: "
        "time.sleep(.01)\n"
        "assert release.exists(), 'leader release timed out'\n"
    )

    class InspectTermination(_Win32):
        def __init__(self):
            super().__init__()
            self.kills = 0

        def terminate_job(self, job):
            super().terminate_job(job)
            self.kills += 1

    api = InspectTermination()
    process = WindowsSupervisor(_api=api).start(
        ProcessSpec((sys.executable, "-c", code), tmp_path)
    )
    descendant = None
    try:
        deadline = time.monotonic() + 15
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        descendant = _winapi.OpenProcess(
            0x00100000,
            False,
            int(pid_file.read_text()),  # SYNCHRONIZE
        )
        assert _winapi.WaitForSingleObject(descendant, 0) == _winapi.WAIT_TIMEOUT
        started = time.monotonic()
        if leader_exits:
            release.touch()
        else:
            process.cancel(grace_seconds=0.1)
        result = process.wait(timeout=5)
        assert result.cancelled is not leader_exits
        assert not result.timed_out
        if leader_exits:
            # A successful leader exit remains truthful even though its
            # ignoring descendant required forced Job Object termination.
            assert result.returncode == 0
            assert time.monotonic() - started >= 0.19
        else:
            assert result.returncode != 0
            assert time.monotonic() - started >= 0.09
        assert api.kills == 1
        assert _winapi.WaitForSingleObject(descendant, 0) == _winapi.WAIT_OBJECT_0
        assert process.wait() is result
    finally:
        try:
            process.cancel(grace_seconds=0)
            process.wait()
        finally:
            if descendant is not None:
                _winapi.CloseHandle(descendant)


@native
def test_native_leader_exit_finalizes_without_caller_timeout(tmp_path):
    code = (
        "import subprocess,sys; subprocess.Popen([sys.executable,'-c',"
        "'import signal,time; signal.signal(signal.SIGBREAK,signal.SIG_IGN); "
        "time.sleep(120)'])"
    )
    process = WindowsSupervisor().start(
        ProcessSpec((sys.executable, "-c", code), tmp_path)
    )
    try:
        # The worker must independently notice leader exit and close the job;
        # no wait(timeout) call is allowed to trigger the cleanup under test.
        process._worker.join(timeout=20)
        assert not process._worker.is_alive()
        result = process.wait()
        assert result.returncode == 0
        assert not result.cancelled and not result.timed_out
    finally:
        process.cancel(grace_seconds=0)
        process.wait(timeout=0)


def test_portable_api_failure_propagates_after_handle_cleanup(tmp_path):
    class FailingBreak(FakeAPI):
        def send_break(self, pid):
            raise OSError("break failed")

    api = FailingBreak()
    process = WindowsSupervisor(_api=api).start(ProcessSpec(("unused",), tmp_path))
    process.cancel(grace_seconds=0)
    with pytest.raises(OSError, match="break failed"):
        process.wait()
    assert "terminate_job" in api.events
    assert api.events[-2:] == [("close", 10), ("close", 11)]


@native
def test_native_daemon_death_closes_job_and_kills_descendant(tmp_path):
    import _winapi

    pid_file = tmp_path / "descendant-pid"
    child = (
        "import os,time; from pathlib import Path; "
        f"Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(120)"
    )
    leader = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(120)"
    )
    daemon = (
        "import os,sys,time; from pathlib import Path; "
        "from wavedl_daemon.supervision.base import ProcessSpec; "
        "from wavedl_daemon.supervision.windows import WindowsSupervisor; "
        f"p=WindowsSupervisor().start(ProcessSpec((sys.executable,'-c',{leader!r}),"
        f"Path({str(tmp_path)!r}))); "
        f"marker=Path({str(pid_file)!r}); deadline=time.monotonic()+15\n"
        "while not marker.exists() and time.monotonic()<deadline: time.sleep(.01)\n"
        "os._exit(0 if marker.exists() else 1)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", daemon], timeout=25, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    pid = int(pid_file.read_text())
    try:
        handle = _winapi.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    except OSError as error:
        # A process which has already disappeared cannot be opened.
        assert error.winerror == 87
    else:
        try:
            assert _winapi.WaitForSingleObject(handle, 5000) == _winapi.WAIT_OBJECT_0
        finally:
            _winapi.CloseHandle(handle)
