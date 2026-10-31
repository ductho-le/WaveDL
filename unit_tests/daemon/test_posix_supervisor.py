"""Real OS process-tree tests independent of ML and Qt."""

import concurrent.futures
import errno
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from wavedl_daemon.supervision.base import ProcessSpec
from wavedl_daemon.supervision.posix import POSIXProcess, POSIXSupervisor


pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")


def start(tmp_path, code, output_limit=1048576):
    return POSIXSupervisor().start(
        ProcessSpec((sys.executable, "-c", code), tmp_path),
        output_limit=output_limit,
    )


def assert_not_live(pid):
    # macOS cannot waitpid an orphan grandchild. A zombie is dead, awaiting init.
    for _ in range(100):
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if not state or state.startswith("Z"):
            return
        time.sleep(0.01)
    pytest.fail(f"live orphan remains: {pid} ({state})")


def test_large_separate_output_is_drained_but_bounded(tmp_path):
    process = start(
        tmp_path,
        "import os; os.write(1,b'a'*2000000); os.write(2,b'b'*2000000)",
        8192,
    )
    result = process.wait(5)
    assert result.returncode == 0
    assert result.stdout == b"a" * 8192
    assert result.stderr == b"b" * 8192
    assert result.stdout_truncated and result.stderr_truncated
    assert not result.timed_out
    assert process.wait() is result
    with pytest.raises(ChildProcessError):
        os.waitpid(process.pid, os.WNOHANG)


def test_zero_capture_and_stdin_closed(tmp_path):
    result = start(
        tmp_path,
        "import sys; assert sys.stdin.read()==''; print('hello')",
        0,
    ).wait(5)
    assert result.returncode == 0
    assert result.stdout == b"" and result.stdout_truncated


def test_supervised_marker_cannot_be_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv("WAVEDL_PARENT_ENV", "inherited")
    process = POSIXSupervisor().start(
        ProcessSpec(
            (
                sys.executable,
                "-c",
                "import os; print(os.environ['WAVEDL_SUPERVISED']); "
                "print(os.environ['WAVEDL_PARENT_ENV']); "
                "print(os.environ['CALLER_ENV'])",
            ),
            tmp_path,
            {"WAVEDL_SUPERVISED": "0", "CALLER_ENV": "caller"},
        )
    )
    assert process.wait(5).stdout == b"1\ninherited\ncaller\n"


def tree_code(pid_file, leader_exits=False):
    child = (
        "import signal,time,os; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"open({str(pid_file)!r},'w').write(str(os.getpid())); "
        "time.sleep(60)"
    )
    return (
        "import subprocess,sys,time,pathlib; "
        f"subprocess.Popen([sys.executable,'-c',{child!r}]); "
        f"p=pathlib.Path({str(pid_file)!r}); "
        "\nwhile not p.exists(): time.sleep(.005)\n"
        + ("sys.exit(0)" if leader_exits else "time.sleep(60)")
    )


def wait_pid(path):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.exists() and path.read_text():
            return int(path.read_text())
        time.sleep(0.005)
    pytest.fail("child did not initialize")


def test_exited_leader_still_escalates_ignoring_grandchild(tmp_path):
    pid_file = tmp_path / "child.pid"
    process = start(tmp_path, tree_code(pid_file, True))
    result = process.wait(5)
    assert result.returncode == 0
    assert not result.cancelled and not result.timed_out
    assert_not_live(wait_pid(pid_file))


def test_exited_leader_closed_pipes_do_not_hide_descendant(tmp_path):
    pid_file = tmp_path / "closed.pid"
    child = (
        "import signal,time,os; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "os.close(1); os.close(2); "
        f"open({str(pid_file)!r},'w').write(str(os.getpid())); time.sleep(60)"
    )
    code = (
        "import subprocess,sys,time,pathlib; "
        f"subprocess.Popen([sys.executable,'-c',{child!r}]); "
        f"p=pathlib.Path({str(pid_file)!r}); "
        "\nwhile not p.exists(): time.sleep(.005)\n"
    )
    process = start(tmp_path, code)
    result = process.wait(5)
    assert result.returncode == 0 and not result.timed_out
    assert_not_live(wait_pid(pid_file))


def test_ignoring_leader_requires_forceful_phase(tmp_path):
    pid_file = tmp_path / "ready.pid"
    process = start(
        tmp_path,
        "import signal,os,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"open({str(pid_file)!r},'w').write(str(os.getpid())); time.sleep(60)",
    )
    assert wait_pid(pid_file) == process.pid
    process.cancel(0.02)
    result = process.wait(5)
    assert result.returncode == -signal.SIGKILL and result.cancelled
    assert_not_live(process.pid)


def test_repeated_cancel_and_concurrent_wait_share_result(tmp_path):
    pid_file = tmp_path / "child.pid"
    process = start(tmp_path, tree_code(pid_file))
    child_pid = wait_pid(pid_file)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        waits = [pool.submit(process.wait) for _ in range(3)]
        process.cancel(0.05)
        process.cancel(30)
        results = [future.result(timeout=5) for future in waits]
    assert all(result is results[0] for result in results)
    assert results[0].cancelled and not results[0].timed_out
    assert results[0].returncode == -signal.SIGTERM
    assert_not_live(child_pid)
    process.cancel()
    assert process.wait() is results[0]


def test_timeout_kills_actual_tree(tmp_path):
    pid_file = tmp_path / "child.pid"
    process = start(tmp_path, tree_code(pid_file))
    child_pid = wait_pid(pid_file)
    began = time.monotonic()
    result = process.wait(0.02)
    assert time.monotonic() - began < 3
    assert result.timed_out and result.cancelled
    assert_not_live(child_pid)


@pytest.mark.parametrize("argv", [(), [], ("",), ("a\0b",), (123,)])
def test_invalid_argv(tmp_path, argv):
    with pytest.raises(ValueError):
        ProcessSpec(argv, tmp_path)


def test_invalid_cwd(tmp_path):
    for cwd in (str(tmp_path), tmp_path / "missing", Path(__file__)):
        with pytest.raises(ValueError):
            ProcessSpec((sys.executable,), cwd)


@pytest.mark.parametrize(
    "environment", [{"": "x"}, {"a=b": "x"}, {"x": "\0"}, {"x": 1}, []]
)
def test_invalid_environment(tmp_path, environment):
    with pytest.raises(ValueError):
        ProcessSpec((sys.executable,), tmp_path, environment)


@pytest.mark.parametrize("limit", [-1, True, 1.5, "1"])
def test_invalid_limits(tmp_path, limit):
    with pytest.raises(ValueError):
        start(tmp_path, "pass", limit)


@pytest.mark.parametrize("seconds", [-1, float("nan"), float("inf"), True])
def test_invalid_deadlines_do_not_change_process(tmp_path, seconds):
    process = start(tmp_path, "import time; time.sleep(.1)")
    with pytest.raises(ValueError):
        process.wait(seconds)
    with pytest.raises(ValueError):
        process.cancel(seconds)
    assert process.wait(5).returncode == 0


def test_cleanup_error_cannot_strand_waiter(tmp_path, monkeypatch):
    """A real child's failing pipe close must still publish supervisor error."""
    native = subprocess.Popen(
        (sys.executable, "-c", "import time; time.sleep(60)"),
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    close = native.stdout.close

    def failing_close():
        close()
        raise OSError("simulated pipe teardown failure")

    def failing_selector():
        raise OSError("simulated selector storage failure")

    monkeypatch.setattr(native.stdout, "close", failing_close)
    monkeypatch.setattr(
        "wavedl_daemon.supervision.posix.selectors.DefaultSelector", failing_selector
    )
    process = POSIXProcess(native, 1024)
    process._worker.join(timeout=3)
    assert not process._worker.is_alive()
    began = time.monotonic()
    with pytest.raises(RuntimeError, match="supervision failed") as captured:
        process.wait(timeout=0.05)
    assert time.monotonic() - began < 1
    assert "selector storage failure" in str(captured.value.__cause__)
    assert native.returncode is not None


def test_dead_worker_without_evidence_is_not_an_infinite_wait(tmp_path, monkeypatch):
    """A catastrophically broken worker is reported, never treated as running."""

    def broken_worker(self):
        self._process.kill()
        self._process.wait()
        self._process.stdout.close()
        self._process.stderr.close()

    monkeypatch.setattr(POSIXProcess, "_run", broken_worker)
    process = start(tmp_path, "import time; time.sleep(60)")
    process._worker.join(timeout=3)
    assert not process._worker.is_alive()
    with pytest.raises(RuntimeError, match="supervision failed") as captured:
        process.wait()
    assert "without terminal evidence" in str(captured.value.__cause__)


@pytest.mark.parametrize("exit_code", [0, 7])
def test_fast_exit_retains_identity_until_last_group_signal(
    tmp_path, monkeypatch, exit_code
):
    """No poll/reap can release the leader ID while killpg still uses it."""
    native = subprocess.Popen(
        (sys.executable, "-c", f"raise SystemExit({exit_code})"),
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    real_wait = native.wait
    real_killpg = os.killpg
    events = []

    def forbidden_poll():
        raise AssertionError("poll would reap and permit foreign PID reuse")

    def tracked_wait(*args, **kwargs):
        events.append("reap")
        return real_wait(*args, **kwargs)

    def owned_killpg(pid, signum):
        assert pid == native.pid
        assert "reap" not in events
        assert native.returncode is None
        events.append(signum)
        return real_killpg(pid, signum)

    monkeypatch.setattr(native, "poll", forbidden_poll)
    monkeypatch.setattr(native, "wait", tracked_wait)
    monkeypatch.setattr(os, "killpg", owned_killpg)
    result = POSIXProcess(native, 1024).wait(5)
    assert result.returncode == exit_code
    assert events[-1] == "reap"
    assert signal.SIGTERM in events
    with pytest.raises(ChildProcessError):
        os.waitpid(native.pid, os.WNOHANG)


def test_fast_native_failure_exit_stress(tmp_path):
    """A fast failed probe is a process result, not a cleanup EPERM."""
    for _ in range(12):
        result = start(tmp_path, "raise SystemExit(7)").wait(5)
        assert result.returncode == 7
        assert not result.cancelled and not result.timed_out


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin kqueue registration")
@pytest.mark.parametrize("exit_code", [0, 7])
@pytest.mark.parametrize("descendant", [False, True])
def test_exited_before_kqueue_registration_drains_without_early_reaping(
    tmp_path, monkeypatch, exit_code, descendant
):
    # Python 3.13 exposes waitid on Darwin too. Exercise the genuine native
    # kqueue fallback explicitly, rather than depending on interpreter version.
    monkeypatch.delattr(os, "waitid", raising=False)
    pid_file = tmp_path / "registration-child.pid"
    code = f"print('retained output'); raise SystemExit({exit_code})"
    if descendant:
        code = "print('retained output'); " + tree_code(pid_file, True).replace(
            "sys.exit(0)", f"sys.exit({exit_code})"
        )
    native = subprocess.Popen(
        (
            sys.executable,
            "-c",
            code,
        ),
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    # Observe a zombie without wait/poll: it still reserves the owned PGID.
    deadline = time.monotonic() + 5
    while True:
        state = subprocess.run(
            ["/bin/ps", "-o", "stat=", "-p", str(native.pid)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if state.startswith("Z"):
            break
        if time.monotonic() >= deadline:
            native.kill()
            native.wait()
            pytest.fail("child did not exit before registration")
        time.sleep(0.005)
    real_kqueue = select.kqueue
    registrations = []

    class RegistrationRace:
        def __init__(self):
            self.queue = real_kqueue()

        def control(self, changes, maximum, timeout=None):
            if changes and any(
                event.filter == select.KQ_FILTER_PROC for event in changes
            ):
                registrations.append(native.pid)
                raise ProcessLookupError(errno.ESRCH, "owned child already exited")
            return self.queue.control(changes, maximum, timeout)

        def close(self):
            self.queue.close()

        def fileno(self):
            return self.queue.fileno()

    def forbidden_poll():
        raise AssertionError("cannot reap before draining and group cleanup")

    monkeypatch.setattr(select, "kqueue", RegistrationRace)
    monkeypatch.setattr(native, "poll", forbidden_poll)
    result = POSIXProcess(native, 1024).wait(5)
    assert registrations == [native.pid]
    assert result.returncode == exit_code
    assert result.stdout == b"retained output\n"
    assert not result.cancelled and not result.timed_out
    if descendant:
        assert_not_live(wait_pid(pid_file))
    with pytest.raises(ChildProcessError):
        os.waitpid(native.pid, os.WNOHANG)


def test_permission_error_with_live_group_is_not_absence(tmp_path, monkeypatch):
    """Inaccessible live groups must remain visible as supervision failures."""
    native = subprocess.Popen(
        (sys.executable, "-c", "import time; time.sleep(60)"),
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    def denied(pid, signum):
        assert pid == native.pid
        raise PermissionError("simulated inaccessible owned group")

    monkeypatch.setattr(os, "killpg", denied)
    process = POSIXProcess(native, 1024)
    process.cancel(0)
    with pytest.raises(RuntimeError, match="supervision failed") as captured:
        process.wait(5)
    assert isinstance(captured.value.__cause__, PermissionError)
    assert native.returncode == -signal.SIGKILL


@pytest.mark.parametrize(
    "platform,snapshot,dead",
    [
        ("darwin", "123 123 Z", True),
        ("darwin", "123 123 Z\n124 123 Z", True),
        ("darwin", "123 123 Z\n124 123 S", False),
        ("darwin", "123 999 Z", False),
        ("darwin", "124 123 Z", False),
        ("darwin", "", False),
        ("darwin", "unknown evidence", False),
        ("linux", "123 123 Z", False),
    ],
)
def test_signal_eperm_uses_strong_exit_proof_despite_stale_observation(
    monkeypatch, platform, snapshot, dead
):
    """Only an unreaped owned leader and exclusively zombie group prove exit."""
    process = POSIXProcess.__new__(POSIXProcess)
    process._process = type("OwnedProcess", (), {"pid": 123})()
    process._reaped = False
    process._leader_exited = False
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(process, "_group_snapshot", lambda: snapshot)
    denied = PermissionError(errno.EPERM, "owned group signal denied")

    def killpg(pid, signum):
        assert pid == 123 and signum == signal.SIGKILL
        raise denied

    monkeypatch.setattr(os, "killpg", killpg)
    if dead:
        assert process._signal_group(signal.SIGKILL) is False
        assert process._leader_exited
    else:
        with pytest.raises(PermissionError) as captured:
            process._signal_group(signal.SIGKILL)
        assert captured.value is denied
        assert not process._leader_exited
    assert not process._reaped  # Strong exit observation never releases PID/PGID.
    process._reaped = True
    with pytest.raises(RuntimeError, match="released process-group identity"):
        process._signal_group(signal.SIGKILL)


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin kqueue registration")
def test_registration_esrch_without_terminal_owned_leader_fails_closed(
    tmp_path, monkeypatch
):
    monkeypatch.delattr(os, "waitid", raising=False)
    real_kqueue = select.kqueue

    class RegistrationDenied:
        def __init__(self):
            self.queue = real_kqueue()

        def control(self, changes, maximum, timeout=None):
            if changes and any(
                event.filter == select.KQ_FILTER_PROC for event in changes
            ):
                raise ProcessLookupError(errno.ESRCH, "not terminal evidence")
            return self.queue.control(changes, maximum, timeout)

        def close(self):
            self.queue.close()

        def fileno(self):
            return self.queue.fileno()

    monkeypatch.setattr(select, "kqueue", RegistrationDenied)
    process = start(tmp_path, "import time; time.sleep(60)")
    with pytest.raises(RuntimeError, match="supervision failed") as captured:
        process.wait(5)
    assert isinstance(captured.value.__cause__, ProcessLookupError)
    assert process._process.returncode == -signal.SIGKILL
    with pytest.raises(ChildProcessError):
        os.waitpid(process.pid, os.WNOHANG)


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin zombie-group proof")
@pytest.mark.parametrize(
    "snapshot",
    [
        "",
        "broken row",
        "invalid invalid Z",
        "{pid} {pid} Z\n99 {pid} S",
        "99 {pid} Z",
    ],
)
def test_zombie_proof_requires_owned_leader_and_no_live_members(
    tmp_path, monkeypatch, snapshot
):
    process = start(tmp_path, "pass")
    process.wait(5)
    monkeypatch.setattr(
        process,
        "_group_snapshot",
        lambda: snapshot.format(pid=process.pid),
    )
    assert not process._darwin_group_is_dead()


@pytest.mark.parametrize(
    ("code", "error"),
    [
        ("import time; time.sleep(60)", TimeoutError),
        ("import os; os.write(1, b'x' * 1000000)", OSError),
    ],
)
def test_group_snapshot_is_bounded_and_reaps_probe(tmp_path, monkeypatch, code, error):
    process = start(tmp_path, "pass")
    process.wait(5)
    native_popen = subprocess.Popen
    probes = []

    def probe_popen(argv, **kwargs):
        assert argv[:2] == ("/bin/ps", "-g")
        assert argv[2] == str(process.pid)
        probe = native_popen((sys.executable, "-c", code), **kwargs)
        probes.append(probe)
        return probe

    monkeypatch.setattr(subprocess, "Popen", probe_popen)
    began = time.monotonic()
    with pytest.raises(error):
        process._group_snapshot()
    assert time.monotonic() - began < 3
    assert len(probes) == 1 and probes[0].returncode is not None
    with pytest.raises(ChildProcessError):
        os.waitpid(probes[0].pid, os.WNOHANG)


def test_cancellation_drains_continuous_output_with_bounded_capture(tmp_path):
    process = start(
        tmp_path,
        "import os\nwhile True: os.write(1, b'x' * 65536)",
        128,
    )
    time.sleep(0.05)
    process.cancel(0.02)
    result = process.wait(5)
    assert result.cancelled and not result.timed_out
    assert result.stdout == b"x" * 128 and result.stdout_truncated


@pytest.mark.parametrize("cancel", [False, True])
def test_escaped_session_retaining_pipes_fails_closed_and_reaps_leader(
    tmp_path, cancel
):
    """An escaped pipe holder cannot strand wait or prove tree termination."""
    pid_file = tmp_path / "escaped.pid"
    code = (
        "import os, pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(60)'], start_new_session=True)\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))\n"
        + ("time.sleep(60)\n" if cancel else "")
    )
    process = start(tmp_path, code)
    escaped_pid = None
    try:
        escaped_pid = wait_pid(pid_file)
        assert os.getpgid(escaped_pid) == escaped_pid
        began = time.monotonic()
        if cancel:
            process.cancel(0)
        with pytest.raises(RuntimeError, match="supervision failed") as captured:
            process.wait(5)
        assert time.monotonic() - began < 4
        assert isinstance(captured.value.__cause__, TimeoutError)
        assert "escaped descendant" in str(captured.value.__cause__)
        assert process._result is None
        process._worker.join(timeout=1)
        assert not process._worker.is_alive()
        assert process._process.stdout.closed and process._process.stderr.closed
        with pytest.raises(ChildProcessError):
            os.waitpid(process.pid, os.WNOHANG)
        # Fail-closed evidence must not claim this outside-group child was killed.
        assert os.getpgid(escaped_pid) == escaped_pid
        with pytest.raises(RuntimeError, match="supervision failed"):
            process.wait()
    finally:
        if escaped_pid is None and pid_file.exists():
            escaped_pid = int(pid_file.read_text())
        if escaped_pid is not None:
            try:
                os.kill(escaped_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            assert_not_live(escaped_pid)
        if process._worker.is_alive():
            process.cancel(0)
            process._worker.join(timeout=3)


def test_failing_group_evidence_cannot_skip_direct_child_reap(tmp_path, monkeypatch):
    native = subprocess.Popen(
        (sys.executable, "-c", "import time; time.sleep(60)"),
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    def failed_signal(self, signum):
        raise ValueError("simulated malformed group evidence")

    monkeypatch.setattr(POSIXProcess, "_signal_group", failed_signal)
    process = POSIXProcess(native, 1024)
    process.cancel(0)
    with pytest.raises(RuntimeError, match="supervision failed") as captured:
        process.wait(5)
    assert isinstance(captured.value.__cause__, ValueError)
    with pytest.raises(ChildProcessError):
        os.waitpid(native.pid, os.WNOHANG)
