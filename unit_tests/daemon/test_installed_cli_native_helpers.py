"""Portable fail-closed probes, not native Windows runtime acceptance."""

import asyncio
from types import SimpleNamespace

import pytest

from wavedl_daemon.supervision.base import ProcessResult

from . import test_installed_cli as acceptance


def result(**changes):
    values = {
        "returncode": 0,
        "stdout": b"output",
        "stderr": b"Application shutdown complete",
    }
    return ProcessResult(**(values | changes))


class OwnedProcess:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []
        self.pid = 123
        self._stdout = SimpleNamespace(data=bytearray(b"ready\nsecond\n"))

    def wait(self, timeout=None):
        self.calls.append(("wait", timeout))
        return self.outcome

    def cancel(self, grace_seconds):
        self.calls.append(("cancel", grace_seconds))


def daemon(outcome=None, **changes):
    instance = object.__new__(acceptance._WindowsInstalledDaemon)
    instance.process = OwnedProcess(outcome or result())
    instance.api = SimpleNamespace(
        kernel=SimpleNamespace(GenerateConsoleCtrlEvent=lambda event, pid: 1),
        observed_empty=True,
        forced_kills=0,
        reserved_process=456,
        leader_running=lambda handle: True,
        close_reservation=lambda: instance.process.calls.append(("close", 456)),
        **changes,
    )
    instance.position = 0
    return instance


@pytest.mark.parametrize(
    "changes",
    [
        {"returncode": 1},
        {"cancelled": True},
        {"timed_out": True},
        {"stdout_truncated": True},
        {"stderr_truncated": True},
    ],
)
def test_windows_checked_rejects_unsuccessful_or_incomplete_result(changes):
    with pytest.raises((AssertionError, acceptance.subprocess.TimeoutExpired)):
        acceptance._require_windows_result(result(**changes), ["owned"], 2)


def test_windows_checked_uses_owned_timeout_and_cleanup(monkeypatch, tmp_path):
    process = OwnedProcess(result())
    starts = []

    def start(argv, cwd, environment):
        starts.append((argv, cwd, environment))
        return process

    monkeypatch.setattr(acceptance, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(acceptance, "_start_windows_installed", start)
    assert (
        acceptance.checked(
            ["owned"], cwd=tmp_path, environment={"EXPLICIT": "yes"}, timeout=2
        )
        == "output"
    )
    assert starts == [(["owned"], tmp_path, {"EXPLICIT": "yes"})]
    assert process.calls == [("wait", 2), ("cancel", 0), ("wait", None)]


def test_windows_daemon_readline_snapshots_only_owned_drain():
    instance = daemon()
    assert asyncio.run(instance.readline()) == b"ready\n"
    assert asyncio.run(instance.readline()) == b"second\n"
    assert instance.process.calls == []


@pytest.mark.parametrize("returncode", [0, 3])
def test_windows_daemon_records_actual_exit_without_coercion(capsys, returncode):
    instance = daemon(result(returncode=returncode))
    signals = []
    properties = []
    instance.api.kernel.GenerateConsoleCtrlEvent = lambda event, pid: (
        signals.append((event, pid)) or 1
    )
    assert asyncio.run(instance.shutdown(lambda *args: properties.append(args))) == (
        b"output",
        b"Application shutdown complete",
    )
    assert f"Native installed daemon exit code: {returncode}" in capsys.readouterr().out
    assert properties == [("native_windows_daemon_exit_code", returncode)]
    assert signals == [(1, 123)]  # Never broadcast to process group zero.
    assert instance.process.calls == [
        ("wait", 15),
        ("cancel", 0),
        ("wait", None),
        ("close", 456),
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"returncode": 1},
        {"returncode": 259},
        {"returncode": 7},
        {"returncode": 0xC000013A},
        {"returncode": True},
        {"stderr": b"shutdown did not finish"},
        {"timed_out": True},
        {"cancelled": True},
        {"stdout_truncated": True},
        {"stderr_truncated": True},
    ],
)
def test_windows_daemon_rejects_incomplete_graceful_evidence(changes):
    instance = daemon(result(**changes))
    with pytest.raises(AssertionError):
        asyncio.run(instance.shutdown())
    assert instance.process.calls[-3:] == [
        ("cancel", 0),
        ("wait", None),
        ("close", 456),
    ]


@pytest.mark.parametrize(
    "field,value", [("forced_kills", 1), ("observed_empty", False)]
)
def test_windows_daemon_rejects_forced_or_unreaped_job(field, value):
    instance = daemon()
    setattr(instance.api, field, value)
    with pytest.raises(AssertionError):
        asyncio.run(instance.shutdown())
    assert instance.process.calls[-3:] == [
        ("cancel", 0),
        ("wait", None),
        ("close", 456),
    ]


def test_windows_daemon_ctrl_break_failure_is_not_graceful(monkeypatch):
    import ctypes

    instance = daemon()
    instance.api.kernel.GenerateConsoleCtrlEvent = lambda event, pid: 0
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 6, raising=False)
    with pytest.raises(OSError, match="CTRL_BREAK failed: winerror=6"):
        asyncio.run(instance.shutdown())
    assert instance.process.calls == [("cancel", 0), ("wait", None), ("close", 456)]


def test_windows_daemon_without_console_fails_before_spawn(monkeypatch, tmp_path):
    monkeypatch.setattr(
        acceptance,
        "_InstalledWindowsAPI",
        lambda: SimpleNamespace(kernel=SimpleNamespace(GetConsoleCP=lambda: 0)),
    )
    monkeypatch.setattr(
        acceptance,
        "_start_windows_installed",
        lambda *args, **kwargs: pytest.fail("No console must fail before spawn"),
    )
    with pytest.raises(OSError, match="requires a console"):
        acceptance._WindowsInstalledDaemon(["owned"], tmp_path, {})


def test_windows_owned_start_preserves_isolated_environment(monkeypatch, tmp_path):
    captures = []

    class Supervisor:
        def __init__(self, **kwargs):
            assert kwargs == {"_api": None}

        def start(self, spec, *, output_limit):
            captures.append((spec, output_limit))
            return "owned"

    monkeypatch.setattr(acceptance, "WindowsSupervisor", Supervisor)
    assert (
        acceptance._start_windows_installed(["owned"], tmp_path, {"EXPLICIT": "yes"})
        == "owned"
    )
    [(spec, limit)] = captures
    assert spec.argv == ("owned",)
    assert spec.cwd == tmp_path
    assert spec.environment == {"EXPLICIT": "yes"}
    assert spec.inherit_environment is False
    assert limit == 8 * 1024 * 1024


def test_windows_checked_never_exposes_token_diagnostics(monkeypatch, tmp_path):
    secret = "private-auth-material"
    process = OwnedProcess(result(returncode=1, stderr=secret.encode()))
    monkeypatch.setattr(acceptance, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(acceptance, "_start_windows_installed", lambda *args: process)
    with pytest.raises(AssertionError, match="leaked authentication material") as error:
        acceptance.checked(
            ["owned"],
            cwd=tmp_path,
            environment={"WAVEDL_DAEMON_TOKEN": secret},
            timeout=1,
        )
    assert secret not in str(error.value)
    assert process.calls[-2:] == [("cancel", 0), ("wait", None)]


def test_windows_daemon_already_exited_is_never_signalled():
    instance = daemon()
    checked_handles = []
    instance.api.leader_running = lambda handle: checked_handles.append(handle) or False
    instance.api.kernel.GenerateConsoleCtrlEvent = lambda *args: pytest.fail(
        "Exited leader must never receive a console event"
    )
    with pytest.raises(AssertionError, match="exited before CTRL_BREAK"):
        asyncio.run(instance.shutdown())
    assert checked_handles == [456]
    assert instance.process.calls == [("cancel", 0), ("wait", None), ("close", 456)]


def test_windows_daemon_failed_start_closes_reservation(monkeypatch, tmp_path):
    closed = []
    api = SimpleNamespace(
        kernel=SimpleNamespace(GetConsoleCP=lambda: 1),
        close_reservation=lambda: closed.append(456),
    )
    monkeypatch.setattr(acceptance, "_InstalledWindowsAPI", lambda: api)

    def failed(*args, **kwargs):
        assert kwargs["api"] is api
        raise OSError("assignment failed")

    monkeypatch.setattr(acceptance, "_start_windows_installed", failed)
    with pytest.raises(OSError, match="assignment failed"):
        acceptance._WindowsInstalledDaemon(["owned"], tmp_path, {})
    assert closed == [456]


def test_windows_reservation_precedes_assignment_and_closes_once(monkeypatch):
    api = object.__new__(acceptance._InstalledWindowsAPI)
    api.reserved_process = None
    calls = []

    def duplicate(*args):
        assert args == (99, 11, 99, 0, False, 2)
        calls.append("duplicate")
        return 456

    api.winapi = SimpleNamespace(
        GetCurrentProcess=lambda: 99, DuplicateHandle=duplicate
    )
    api.close = lambda handle: calls.append(("close", handle))

    def assign(self, job, process):
        assert self.reserved_process == 456
        assert (job, process) == (10, 11)
        calls.append("assign")

    monkeypatch.setattr(acceptance._Win32, "assign", assign)
    api.assign(10, 11)
    api.close_reservation()
    api.close_reservation()
    assert calls == ["duplicate", "assign", ("close", 456)]


@pytest.mark.parametrize(
    "output,phase",
    [
        (b"private https://secret.example/token", "unknown"),
        (b"Processing private.whl\r\nCollecting private", "resolve"),
        (b"Processing private.whl\nInstalling collected packages: private", "unpack"),
        (
            b"Installing collected packages: private\n  Attempting uninstall: private",
            "uninstall",
        ),
        (b"Uninstalling private\nSuccessfully installed private", "complete"),
        (
            b"Attempting uninstall: private\nSuccessfully uninstalled private",
            "uninstall-complete",
        ),
        (b"Attempting uninstall: private\nProcessing private.whl", "resolve"),
    ],
)
def test_pip_timeout_phase_is_finite_and_never_publishes_raw_output(output, phase):
    assert acceptance._pip_install_phase(output) == phase


@pytest.mark.parametrize("enabled", [False, True])
def test_pip_timeout_diagnostic_is_opt_in_sanitized_and_never_passes(
    monkeypatch, tmp_path, capsys, enabled
):
    private = b"private-auth https://private.example/token"
    process = OwnedProcess(
        result(
            returncode=1,
            stdout=b"Installing collected packages: " + private,
            stderr=private,
            timed_out=True,
            cancelled=True,
        )
    )
    monkeypatch.setattr(acceptance, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(acceptance, "_start_windows_installed", lambda *args: process)
    with pytest.raises(acceptance.subprocess.TimeoutExpired):
        acceptance.checked(
            ["owned"],
            cwd=tmp_path,
            environment={},
            timeout=300,
            pip_diagnostic=enabled,
        )
    assert capsys.readouterr().out == (
        "Installed dependency timeout: last_progress=unpack, timed_out=True, "
        "cancelled=True, stdout_truncated=False, stderr_truncated=False\n"
        if enabled
        else ""
    )
    assert process.calls == [("wait", 300), ("cancel", 0), ("wait", None)]
