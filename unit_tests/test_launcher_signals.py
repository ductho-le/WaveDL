"""
Launcher process ownership and signal forwarding tests.

Real-process tests run the launcher's supervision in a fresh interpreter against
a fake training command (no Torch), then check that no process survives.
"""

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from wavedl import launcher


LAUNCHER_PATH = Path(launcher.__file__).resolve()
FORWARDED = (signal.SIGTERM, signal.SIGINT)
posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")


def _handlers():
    return {signum: signal.getsignal(signum) for signum in FORWARDED}


def _gone(pid: int) -> bool:
    """A process cannot run any more (missing, or a zombie awaiting its reaper)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return not state or state.startswith("Z")


# ==============================================================================
# ARGUMENTS AND COMMANDS
# ==============================================================================
@pytest.mark.parametrize(
    "args",
    [
        ["--num_gpus", "0"],
        ["--cpu", "--num_gpus", "65"],
        ["--cpu", "--num_machines", "2", "--main_process_ip", "10.0.0.1"],
    ],
)
def test_rejects_invalid_process_counts(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["wavedl-train", *args])
    with pytest.raises(SystemExit) as error:
        launcher.parse_args()
    assert error.value.code == 2


def _launch_commands(monkeypatch, *args):
    calls = []
    monkeypatch.setattr(sys, "argv", ["wavedl-train", *args, "--model", "cnn"])
    monkeypatch.setattr(launcher, "setup_environment", lambda: None)
    monkeypatch.setattr(
        launcher,
        "_run_training_process",
        lambda cmd, env=None, **kwargs: calls.append((cmd, env, kwargs)) or 0,
    )
    assert launcher.main() == 0
    return calls[0]


def test_single_process_launch_uses_accelerate(monkeypatch):
    cmd, env, kwargs = _launch_commands(monkeypatch, "--num_gpus", "1")
    assert cmd[1:3] == ["-m", "accelerate.commands.launch"]
    assert env == dict(os.environ, PYTHONIOENCODING="utf-8:backslashreplace")
    # accelerate's single-process launcher does not forward signals itself.
    assert kwargs == {"signal_leader_only": False}


def test_multi_gpu_launch_keeps_ranks_in_launcher_group(monkeypatch):
    cmd, env, kwargs = _launch_commands(monkeypatch, "--num_gpus", "4")
    assert cmd[1:3] == ["-m", "wavedl.distributed_launch"]
    assert "--num_processes=4" in cmd and "--multi_gpu" in cmd
    assert cmd[cmd.index("-m", 3) :][:2] == ["-m", "wavedl.train"]
    assert env == dict(os.environ, PYTHONIOENCODING="utf-8:backslashreplace")
    assert kwargs == {"signal_leader_only": True}


def test_multi_process_cpu_uses_torchrun(monkeypatch):
    cmd, env, kwargs = _launch_commands(monkeypatch, "--cpu", "--num_gpus", "3")
    assert cmd[1:3] == ["-m", "torch.distributed.run"]
    assert "--nproc_per_node=3" in cmd and "--standalone" in cmd
    assert "--run-path" in cmd and "--start-method=spawn" in cmd
    script = cmd.index("--run-path") + 1
    assert Path(cmd[script]) == LAUNCHER_PATH.with_name("train.py")
    assert cmd[script + 1 :] == ["--model", "cnn", "--precision", "bf16"]
    assert env["ACCELERATE_USE_CPU"] == "true"
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert kwargs == {"signal_leader_only": True}


def test_cross_validation_keeps_one_cpu_process(monkeypatch):
    cmd, _env, _kwargs = _launch_commands(
        monkeypatch, "--cpu", "--num_gpus", "3", "--cv", "2"
    )
    assert cmd[1:3] == ["-m", "accelerate.commands.launch"]
    assert "--num_processes=1" in cmd


def test_distributed_launch_routes_ranks_through_run_path(monkeypatch):
    import accelerate.commands.launch as accelerate_launch

    from wavedl import distributed_launch

    launched = []
    monkeypatch.setattr(accelerate_launch, "launch_command", launched.append)
    distributed_launch.main(
        ["--multi_gpu", "--num_processes=2", "-m", "wavedl.train", "--epochs", "1"]
    )
    (args,) = launched
    assert Path(args.training_script) == LAUNCHER_PATH.with_name("train.py")
    assert args.run_path and not args.module
    assert args.start_method == "spawn"
    assert args.training_script_args == ["--epochs", "1"]


@pytest.mark.parametrize(
    "argv",
    [
        ["--num_processes=2", "-m", "wavedl.train"],
        ["--multi_gpu", "--cpu", "-m", "wavedl.train"],
    ],
)
def test_distributed_launch_rejects_other_launches(argv):
    from wavedl import distributed_launch

    with pytest.raises(ValueError):
        distributed_launch.main(argv)


# ==============================================================================
# HANDLER LIFECYCLE AND EXIT CODES (in-process)
# ==============================================================================
def test_spawn_failure_restores_handlers_and_is_reported(monkeypatch, capsys):
    previous = _handlers()
    monkeypatch.setattr(sys, "argv", ["wavedl-train", "--num_gpus", "1"])
    monkeypatch.setattr(launcher, "setup_environment", lambda: None)
    monkeypatch.setattr(
        launcher.subprocess, "Popen", Mock(side_effect=OSError("no python"))
    )
    assert launcher.main() == 1
    assert "Could not launch training: no python" in capsys.readouterr().err
    assert _handlers() == previous


def test_unexpected_spawn_failure_propagates_and_restores(monkeypatch):
    previous = _handlers()
    monkeypatch.setattr(
        launcher.subprocess, "Popen", Mock(side_effect=RuntimeError("spawn"))
    )
    with pytest.raises(RuntimeError, match="spawn"):
        launcher._run_training_process(["train"])
    assert _handlers() == previous


@posix_only
def test_ignored_signals_stay_ignored(monkeypatch):
    """nohup-style SIG_IGN must not be replaced by a forwarding handler."""
    seen = {}

    def spawn(cmd, **kwargs):
        seen["hup"] = signal.getsignal(signal.SIGHUP)
        seen["term"] = signal.getsignal(signal.SIGTERM)
        raise OSError("stop here")

    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    previous = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        with pytest.raises(OSError, match="stop here"):
            launcher._run_training_process(["train"])
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGHUP, previous)
    assert seen["hup"] is signal.SIG_IGN
    assert callable(seen["term"])


@posix_only
def test_leader_only_forwarding_still_kills_group(monkeypatch):
    child = Mock(pid=4242, returncode=None)
    observer = Mock()
    observer.observe.return_value = True
    calls = []

    def spawn(cmd, **kwargs):
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        return child

    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    monkeypatch.setattr(launcher, "_ChildExitObserver", lambda proc: observer)
    monkeypatch.setattr(launcher, "_group_alive", lambda pgid: False)
    monkeypatch.setattr(
        launcher.os, "kill", lambda pid, sig: calls.append(("kill", pid, sig))
    )
    monkeypatch.setattr(
        launcher.os, "killpg", lambda pgid, sig: calls.append(("killpg", pgid, sig))
    )
    result = launcher._run_training_process(["train"], signal_leader_only=True)
    assert result == 128 + signal.SIGINT
    assert calls == [("kill", 4242, signal.SIGINT), ("killpg", 4242, signal.SIGKILL)]


@posix_only
def test_observer_failure_kills_and_reaps_group(monkeypatch):
    children = []
    real_popen = subprocess.Popen

    def spawn(cmd, **kwargs):
        children.append(
            real_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        )
        return children[-1]

    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    monkeypatch.setattr(
        launcher, "_ChildExitObserver", Mock(side_effect=OSError("observe failed"))
    )
    with pytest.raises(OSError, match="observe failed"):
        launcher._run_training_process(["train"])
    assert children[0].returncode == -signal.SIGKILL


@posix_only
@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("raise SystemExit(7)", 7),
        ("import os, signal; os.kill(os.getpid(), signal.SIGKILL)", 137),
        ("import os, signal; os.kill(os.getpid(), signal.SIGTERM)", 143),
    ],
)
def test_child_status_maps_to_shell_exit_code(code, expected):
    previous = _handlers()
    assert launcher._run_training_process([sys.executable, "-c", code]) == expected
    assert _handlers() == previous


# ==============================================================================
# REAL SIGNALS (fresh launcher interpreter, fake training tree)
# ==============================================================================
FAKE_TRAINING = textwrap.dedent(
    """
    import json, os, signal, subprocess, sys, time

    mode, ready = sys.argv[1], sys.argv[2]
    grandchild = None
    if mode == "ignore":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if mode in ("tree", "tree-exit"):
        # A descendant (think DataLoader worker) that ignores SIGTERM.
        grandchild = subprocess.Popen([sys.executable, "-c",
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "time.sleep(60)"])
    if mode == "graceful":
        # A descendant that needs time after SIGTERM (emergency checkpoint)
        # while its parent exits immediately (like a launcher that does not wait).
        grandchild = subprocess.Popen([sys.executable, "-c",
            "import signal, sys, time\\n"
            "def stop(*_):\\n"
            "    time.sleep(0.5)\\n"
            "    open(sys.argv[1], 'w').write('saved')\\n"
            "    sys.exit(0)\\n"
            "signal.signal(signal.SIGTERM, stop)\\n"
            "print('armed', flush=True)\\n"
            "time.sleep(60)", ready + ".checkpoint"], stdout=subprocess.PIPE)
        grandchild.stdout.readline()
    time.sleep(0.1)
    with open(ready + ".tmp", "w") as handle:
        json.dump({"pid": os.getpid(), "pgid": os.getpgrp(),
                   "grandchild": grandchild.pid if grandchild else None}, handle)
    os.replace(ready + ".tmp", ready)
    if mode in ("exit", "tree-exit"):
        sys.exit(7)
    time.sleep(60)
    """
)

HARNESS = textwrap.dedent(
    """
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("launcher_under_test", sys.argv[1])
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    launcher._GRACE_SECONDS = float(sys.argv[2])
    leader_only = sys.argv[3] == "leader"
    sys.exit(launcher._run_training_process(
        [sys.executable, *sys.argv[4:]], signal_leader_only=leader_only
    ))
    """
)


@pytest.fixture
def launch_fake(tmp_path):
    if os.name != "posix":
        pytest.skip("POSIX signals")
    fake = tmp_path / "fake_training.py"
    fake.write_text(FAKE_TRAINING)
    started = []
    spawned = []

    def start(mode="sleep", grace=5.0, target="group"):
        ready = tmp_path / f"ready-{len(started)}.json"
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                HARNESS,
                str(LAUNCHER_PATH),
                str(grace),
                target,
                str(fake),
                mode,
                str(ready),
            ],
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        started.append(proc)
        deadline = time.monotonic() + 15
        while not ready.exists():
            assert proc.poll() is None, proc.communicate()
            assert time.monotonic() < deadline, "fake training did not start"
            time.sleep(0.01)
        info = json.loads(ready.read_text())
        spawned.extend(pid for pid in (info["pid"], info["grandchild"]) if pid)
        return proc, info, ready

    yield start
    for proc in started:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        proc.stdout.close()
        proc.stderr.close()
    # Safety net so a failing test cannot leak fake training processes.
    for pid in spawned:
        if not _gone(pid):
            os.kill(pid, signal.SIGKILL)


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP])
def test_forwarded_signal_stops_whole_tree(launch_fake, signum):
    proc, info, _ready = launch_fake("tree", grace=0.5)
    assert info["pgid"] == info["pid"] != os.getpgrp()
    proc.send_signal(signum)
    out, err = proc.communicate(timeout=10)
    assert proc.returncode == 128 + signum, (out, err)
    assert _gone(info["pid"])
    assert _gone(info["grandchild"])


def test_ignoring_child_is_killed_after_grace(launch_fake):
    proc, info, _ready = launch_fake("ignore", grace=0.5)
    started = time.monotonic()
    proc.send_signal(signal.SIGTERM)
    time.sleep(0.1)
    proc.send_signal(signal.SIGTERM)  # Repeats must not extend the grace period.
    out, err = proc.communicate(timeout=10)
    elapsed = time.monotonic() - started
    assert proc.returncode == 143, (out, err)
    assert 0.4 <= elapsed < 5
    assert _gone(info["pid"])


def test_descendants_get_grace_after_leader_exits(launch_fake):
    proc, info, ready = launch_fake("graceful", grace=10.0)
    proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=15)
    assert proc.returncode == 143, (out, err)
    # Killed only after its checkpoint, not when the leader died.
    assert Path(str(ready) + ".checkpoint").read_text() == "saved"
    assert _gone(info["grandchild"])


def test_natural_exit_preserves_status_and_cleans_descendants(launch_fake):
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        proc, info, _ready = launch_fake("tree-exit")
        out, err = proc.communicate(timeout=10)
        assert proc.returncode == 7, (out, err)
        assert _gone(info["grandchild"])
        assert unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait(timeout=5)


def test_leader_only_signal_kills_unsignalled_descendants_after_grace(launch_fake):
    """torchrun-style launches: only the leader is signalled, nothing escapes."""
    proc, info, ready = launch_fake("graceful", grace=1.0, target="leader")
    started = time.monotonic()
    proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=10)
    assert proc.returncode == 143, (out, err)
    assert time.monotonic() - started >= 0.9
    # The descendant never saw SIGTERM; it was killed once the grace expired.
    assert not Path(str(ready) + ".checkpoint").exists()
    assert _gone(info["grandchild"])
