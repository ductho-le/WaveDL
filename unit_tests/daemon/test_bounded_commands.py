import os
import sys
import threading
import time
from pathlib import Path

import pytest

from wavedl_daemon.execution import (
    BoundedThreadPool,
    BudgetExhausted,
    StorageWorkError,
    WorkBudget,
    WorkCancelled,
)
from wavedl_daemon.execution.commands import run_command, submit_command
from wavedl_daemon.supervision.base import ProcessResult, ProcessSpec


def spec(script, tmp_path):
    return ProcessSpec((sys.executable, "-c", script), tmp_path)


def test_command_budget_refuses_before_start(tmp_path):
    with pytest.raises(BudgetExhausted):
        run_command(spec("print('no')", tmp_path), WorkBudget(max_command_bytes=1))


def test_output_aggregate_bound(tmp_path):
    result = run_command(
        spec("import os; os.write(1,b'x'*100000); os.write(2,b'y'*100000)", tmp_path),
        WorkBudget(max_output_bytes=257),
    )
    assert result.returncode == 0
    assert len(result.stdout) + len(result.stderr) <= 257
    assert result.stdout_truncated and result.stderr_truncated


def test_shell_free_and_stdin_closed(tmp_path):
    command = ProcessSpec(
        (
            sys.executable,
            "-c",
            "import sys; print(repr(sys.stdin.read())); print(sys.argv[1])",
            "$(touch forbidden); echo injected",
        ),
        tmp_path,
    )
    result = run_command(command, WorkBudget())
    newline = os.linesep.encode()
    assert result.stdout == newline.join(
        [b"''", b"$(touch forbidden); echo injected", b""]
    )
    assert not (tmp_path / "forbidden").exists()


def test_start_failure_is_explicit(tmp_path):
    with pytest.raises(StorageWorkError) as captured:
        run_command(
            ProcessSpec(("/missing/wavedl-executable",), tmp_path), WorkBudget()
        )
    assert isinstance(captured.value.__cause__, OSError)


def test_storage_error_during_supervision_cleans_work_and_fails(tmp_path):
    class Process:
        pid = 123
        cancelled = False
        waits = 0

        def cancel(self, grace_seconds=5):
            self.cancelled = True

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise OSError("storage unavailable")
            return ProcessResult(-9, b"", b"", cancelled=True)

    process = Process()

    class Supervisor:
        def start(self, spec, *, output_limit):
            return process

    with pytest.raises(StorageWorkError) as captured:
        run_command(spec("pass", tmp_path), WorkBudget(), supervisor=Supervisor())
    assert isinstance(captured.value.__cause__, OSError)
    assert process.cancelled
    assert process.waits == 2


def test_bounded_command_cancel_terminates_before_releasing_slot(tmp_path):
    ready = tmp_path / "ready"
    marker = tmp_path / "after"
    script = (
        "import pathlib,time; "
        f"pathlib.Path({str(ready)!r}).touch(); time.sleep(1); "
        f"pathlib.Path({str(marker)!r}).touch()"
    )
    with BoundedThreadPool(WorkBudget(max_workers=1, max_pending=0)) as pool:
        handle = submit_command(pool, spec(script, tmp_path))
        deadline = time.monotonic() + 2
        while not ready.exists() and time.monotonic() < deadline:
            threading.Event().wait(0.01)
        assert ready.exists()
        handle.cancel()
        with pytest.raises(WorkCancelled):
            handle.result(2)
        assert pool.outstanding == 0
        time.sleep(1)
        assert not marker.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX PID liveness probe")
def test_timeout_kills_real_process_and_descendant(tmp_path):
    pid_file = tmp_path / "child.pid"
    marker = tmp_path / "survived"
    child = (
        "import time,pathlib; time.sleep(1); "
        f"pathlib.Path({str(marker)!r}).write_text('alive')"
    )
    script = (
        "import subprocess,sys,time,pathlib; "
        f"p=subprocess.Popen([sys.executable,'-c',{child!r}]); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid)); time.sleep(10)"
    )
    started = time.monotonic()
    result = run_command(spec(script, tmp_path), WorkBudget(timeout_seconds=0.3))
    assert result.timed_out
    assert time.monotonic() - started < 3
    assert pid_file.exists()
    time.sleep(1)
    assert not marker.exists()
    # A reparented zombie may briefly exist, but can no longer execute work.
    pid = int(pid_file.read_text())
    status = Path(f"/proc/{pid}/stat")
    if status.exists():
        assert status.read_text().split()[2] == "Z"
