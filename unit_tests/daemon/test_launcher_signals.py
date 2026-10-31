"""Real launcher subprocesses with a fake accelerate, never Torch training."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from wavedl import launcher


def test_handler_does_not_wait_and_restores():
    child = Mock()
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    forwarder = launcher._install_sigterm_handler(child)
    try:
        forwarder.forward(signal.SIGTERM, None)
        forwarder.forward(signal.SIGINT, None)
        assert forwarder.received_signal == signal.SIGTERM
        child.send_signal.assert_called_once_with(signal.SIGTERM)
        child.wait.assert_not_called()
        child.poll.assert_not_called()
    finally:
        forwarder.restore()
    assert all(signal.getsignal(sig) == handler for sig, handler in previous.items())


def test_main_restores_on_spawn_failure(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    monkeypatch.setattr(sys, "argv", ["wavedl-train", "--num_gpus", "1"])
    monkeypatch.setattr(launcher, "setup_environment", lambda: None)
    monkeypatch.setattr(
        launcher.subprocess, "Popen", Mock(side_effect=OSError("spawn"))
    )
    with pytest.raises(OSError, match="spawn"):
        launcher.main()
    assert all(signal.getsignal(sig) == handler for sig, handler in previous.items())


@pytest.mark.skipif(os.name != "posix", reason="owned POSIX group")
def test_signal_during_spawn_is_forwarded_before_reap(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WAVEDL_SUPERVISED", raising=False)
    monkeypatch.setattr(sys, "argv", ["wavedl-train", "--num_gpus", "1"])
    monkeypatch.setattr(launcher, "setup_environment", lambda: None)
    child = Mock(pid=4242, returncode=-signal.SIGKILL)
    observer = Mock()
    observer.observe.return_value = True
    observer.group_is_dead.return_value = True
    monkeypatch.setattr(launcher, "_ChildExitObserver", lambda proc: observer)
    monkeypatch.setattr(launcher, "_GRACE_SECONDS", 0)
    calls = []

    def spawn(*args, **kwargs):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return child

    def reap(*args, **kwargs):
        # Deliver another signal exactly where wait would release the PID.
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        calls.append("wait")
        return child.returncode

    child.wait.side_effect = reap
    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    monkeypatch.setattr(
        launcher.os, "killpg", lambda group, sig: calls.append((group, sig))
    )
    assert launcher.main() == 143
    assert calls == [(4242, signal.SIGTERM), (4242, signal.SIGKILL), "wait"]
    child.poll.assert_not_called()


@pytest.mark.skipif(os.name != "posix", reason="owned POSIX process group")
def test_group_identity_is_disarmed_before_reap(monkeypatch):
    child = Mock(pid=4242)
    observer = Mock()
    observer.observe.return_value = True
    observer.group_is_dead.return_value = True
    forwarder = launcher._SignalForwarder(child, child.pid)
    calls = []

    def reap():
        assert forwarder.process is None
        assert forwarder.process_group is None
        forwarder.forward(signal.SIGTERM, None)
        calls.append("wait")

    child.wait.side_effect = reap
    monkeypatch.setattr(
        launcher.os, "killpg", lambda group, sig: calls.append((group, sig))
    )
    launcher._terminate_child(
        child,
        timeout=0,
        process_group=child.pid,
        observer=observer,
        forwarder=forwarder,
    )
    assert calls == [(4242, signal.SIGTERM), (4242, signal.SIGKILL), "wait"]
    child.poll.assert_not_called()


def test_waitid_observation_does_not_release_child(monkeypatch):
    child = Mock(pid=4242)
    for name, value in (("P_PID", 1), ("WEXITED", 2), ("WNOHANG", 4), ("WNOWAIT", 8)):
        monkeypatch.setattr(launcher.os, name, value, raising=False)
    waitid = Mock(side_effect=[None, Mock(si_pid=child.pid)])
    monkeypatch.setattr(launcher.os, "waitid", waitid, raising=False)
    observer = launcher._ChildExitObserver(child)
    try:
        assert not observer.observe()
        assert observer.observe()
        assert observer.observe()
        assert waitid.call_count == 2
        waitid.assert_called_with(1, child.pid, 14)
        child.poll.assert_not_called()
        child.wait.assert_not_called()
    finally:
        observer.close()


@pytest.mark.skipif(os.name != "posix", reason="owned POSIX group")
def test_observer_failure_kills_and_reaps_reserved_group(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WAVEDL_SUPERVISED", raising=False)
    monkeypatch.setattr(sys, "argv", ["wavedl-train", "--num_gpus", "1"])
    monkeypatch.setattr(launcher, "setup_environment", lambda: None)
    real_popen = subprocess.Popen
    children = []

    def spawn(*args, **kwargs):
        child = real_popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )
        children.append(child)
        return child

    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    monkeypatch.setattr(
        launcher, "_ChildExitObserver", Mock(side_effect=OSError("observe failed"))
    )
    with pytest.raises(OSError, match="observe failed"):
        launcher.main()
    assert len(children) == 1
    assert children[0].returncode == -signal.SIGKILL
    with pytest.raises(ProcessLookupError):
        os.kill(children[0].pid, 0)


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin kqueue fallback")
def test_fast_exit_before_kqueue_registration_keeps_owned_zombie(monkeypatch):
    monkeypatch.delattr(launcher.os, "waitid", raising=False)
    child = Mock(pid=4242)
    queue = Mock()
    queue.control.side_effect = ProcessLookupError()
    monkeypatch.setattr(launcher.select, "kqueue", lambda: queue)
    snapshot = Mock(return_value=[(child.pid, "Z")])
    monkeypatch.setattr(launcher._ChildExitObserver, "snapshot", snapshot)
    observer = launcher._ChildExitObserver(child)
    try:
        assert observer.observe()
        assert observer.group_is_dead()
        child.poll.assert_not_called()
        child.wait.assert_not_called()
    finally:
        observer.close()
    queue.close.assert_called_once()


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin zombie-group EPERM")
def test_permission_error_waits_for_owned_zombie_evidence(monkeypatch):
    observer = launcher._ChildExitObserver.__new__(launcher._ChildExitObserver)
    observer.proc = Mock(pid=4242)
    observer.exited = False
    observer.observe = Mock(return_value=False)
    observer.snapshot = Mock(side_effect=[[(4242, "E")], [(4242, "Z"), (4243, "Z")]])
    assert observer.confirm_group_is_dead()
    assert observer.exited
    assert observer.snapshot.call_count == 2
    assert all(
        0 < call.kwargs["timeout"] <= 0.1 for call in observer.snapshot.call_args_list
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin zombie-group EPERM")
@pytest.mark.parametrize("members", [[(4242, "Z"), (4243, "S")], [], [(4243, "Z")]])
def test_permission_error_never_accepts_live_or_missing_leader(monkeypatch, members):
    child = Mock(pid=4242)
    observer = launcher._ChildExitObserver.__new__(launcher._ChildExitObserver)
    observer.proc = child
    observer.exited = False
    observer.observe = Mock(return_value=False)
    observer.snapshot = Mock(return_value=members)
    monkeypatch.setattr(launcher.os, "killpg", Mock(side_effect=PermissionError()))
    started = time.monotonic()
    with pytest.raises(PermissionError):
        launcher._terminate_child(
            child, timeout=0, process_group=child.pid, observer=observer
        )
    assert time.monotonic() - started < 0.3
    child.wait.assert_not_called()
    child.poll.assert_not_called()


@pytest.mark.parametrize(
    "args",
    [
        ["--num_gpus", "0"],
        ["--cpu", "--num_gpus", "65"],
        ["--cpu", "--num_gpus", "0"],
        ["--cpu", "--num_machines", "2"],
        ["--cpu", "--machine_rank", "1"],
    ],
)
def test_rejects_conflicting_cpu_and_nonpositive_counts(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["wavedl-train", *args])
    with pytest.raises(SystemExit) as error:
        launcher.parse_args()
    assert error.value.code == 2


def _await_file(path, proc):
    deadline = time.monotonic() + 5
    while not path.exists():
        assert proc.poll() is None, proc.communicate()
        assert time.monotonic() < deadline, "fake accelerate did not start"
        time.sleep(0.01)
    return json.loads(path.read_text())


@pytest.fixture
def launch_fake(tmp_path):
    if os.name != "posix":
        pytest.skip("POSIX native signals and executable stub")
    commands = tmp_path / "accelerate" / "commands"
    commands.mkdir(parents=True)
    (commands.parent / "__init__.py").write_text("")
    (commands / "__init__.py").write_text("")
    executable = commands / "launch.py"
    executable.write_text(
        "import json, os, signal, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "mode = os.environ['FAKE_MODE']\n"
        "if mode == 'ignore': signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "grandchild = None\n"
        "if mode in ('tree', 'tree-exit'):\n"
        "    grandchild = subprocess.Popen([sys.executable, '-c', "
        "'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); '"
        "'time.sleep(60)'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "    time.sleep(.1)\n"
        "Path(os.environ['FAKE_READY']).write_text(json.dumps({"
        "'pid': os.getpid(), 'pgid': os.getpgrp(), 'ppid': os.getppid(), "
        "'executable': sys.executable, "
        "'parent_pgid': os.getpgid(os.getppid()), 'argv': sys.argv, "
        "'grandchild': grandchild.pid if grandchild else None}))\n"
        "if mode in ('exit', 'tree-exit'): sys.exit(7)\n"
        "time.sleep(60)\n"
    )
    processes = []

    def start(mode="sleep", marker=None, cpu=False):
        ready = tmp_path / f"ready-{len(processes)}.json"
        env = os.environ.copy()
        env.update(
            PYTHONPATH=os.pathsep.join(
                [str(tmp_path), str(Path(__file__).resolve().parents[2] / "src")]
            ),
            FAKE_MODE=mode,
            FAKE_READY=str(ready),
        )
        env.pop("WAVEDL_SUPERVISED", None)
        if marker is not None:
            env["WAVEDL_SUPERVISED"] = marker
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from wavedl import launcher; launcher._GRACE_SECONDS=.25; "
                "raise SystemExit(launcher.main())",
                "--num_gpus",
                "1",
                "--model",
                "fake",
            ]
            + (["--cpu"] if cpu else []),
            env=env,
            cwd=tmp_path,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append((proc, ready))
        return proc, _await_file(ready, proc)

    yield start
    for proc, _ready in processes:
        # The launcher, not this fixture, owns its child group. Never send to a
        # saved PGID after the launcher could already have reaped its leader.
        proc.terminate()
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate(timeout=5)


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
@pytest.mark.parametrize("attempt", range(20))
def test_real_launcher_reports_shell_signal_status(launch_fake, signum, attempt):
    proc, info = launch_fake()
    assert info["pgid"] != info["parent_pgid"]
    proc.send_signal(signum)
    out, err = proc.communicate(timeout=5)
    assert proc.returncode == 128 + signum, (out, err)
    with pytest.raises(ProcessLookupError):
        os.kill(info["pid"], 0)


def test_ignoring_child_escalates_and_repeated_signals_are_harmless(launch_fake):
    proc, info = launch_fake("ignore")
    started = time.monotonic()
    proc.send_signal(signal.SIGTERM)
    time.sleep(0.05)
    proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=5)
    assert proc.returncode == 143, (out, err)
    assert time.monotonic() - started >= 0.2
    with pytest.raises(ProcessLookupError):
        os.kill(info["pid"], 0)


def test_standalone_kills_grandchild_after_direct_child_exits(launch_fake):
    proc, info = launch_fake("tree")
    proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=5)
    assert proc.returncode == 143, (out, err)
    # Reparented zombies may remain until the OS reaps them: assert they cannot
    # execute, rather than claiming this launcher can waitpid a non-child.
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(info["grandchild"])],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    assert not state or state.startswith("Z"), state


def test_natural_leader_exit_cleans_descendant_with_closed_pipes(launch_fake):
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        proc, info = launch_fake("tree-exit")
        out, err = proc.communicate(timeout=5)
        assert proc.returncode == 7, (out, err)
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(info["grandchild"])],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        assert not state or state.startswith("Z"), state
        assert unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait(timeout=5)


def test_supervised_child_inherits_group(launch_fake):
    proc, info = launch_fake(marker="1")
    assert info["pgid"] == info["parent_pgid"] == proc.pid
    os.killpg(proc.pid, signal.SIGTERM)
    out, err = proc.communicate(timeout=5)
    assert proc.returncode == 143, (out, err)


def test_malformed_marker_does_not_disable_standalone_ownership(launch_fake):
    proc, info = launch_fake(marker="true")
    assert info["pgid"] != info["parent_pgid"]
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=5) == 143


def test_early_child_exit_preserves_status_and_command(launch_fake):
    proc, info = launch_fake("exit")
    out, err = proc.communicate(timeout=5)
    assert proc.returncode == 7, (out, err)
    assert info["executable"] == sys.executable
    assert info["argv"][1:] == [
        "--num_processes=1",
        "--num_machines=1",
        "--machine_rank=0",
        "--mixed_precision=bf16",
        "--dynamo_backend=no",
        "-m",
        "wavedl.train",
        "--model",
        "fake",
    ]


def test_cpu_flag_belongs_to_accelerate(launch_fake):
    proc, info = launch_fake("exit", cpu=True)
    assert proc.wait(timeout=5) == 7
    args = info["argv"][1:]
    assert "--cpu" in args[: args.index("wavedl.train")]
    assert "--cpu" not in args[args.index("wavedl.train") + 1 :]
