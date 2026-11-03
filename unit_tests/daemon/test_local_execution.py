"""Disposable stub runtimes exercise real Local process ownership, not ML."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from wavedl_daemon.backends.contracts import Runtime, UnsupportedPlan
from wavedl_daemon.backends.local import LocalBackend
from wavedl_daemon.backends.local_execution import (
    CANCEL_GRACE_SECONDS,
    SupervisionLost,
    attempt_argv,
    reserved_local_environment,
)
from wavedl_daemon.execution.budget import BudgetExhausted, WorkBudget
from wavedl_daemon.execution.pool import BoundedThreadPool
from wavedl_daemon.runs.model import LocalExit, NotOwned, Operation
from wavedl_daemon.runs.service import build_spec
from wavedl_daemon.storage.ownership import DispatchLock, DispatchRequired


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="executable stubs require POSIX; native Windows supervisor tests cover Jobs",
)

# The child records what it was started with next to the attempt file it was
# given, because the supervisor keeps process output in memory only.
STUB = """
import json,os,signal,subprocess,sys,time
if sys.argv[1] == '-c':
    probe = os.environ.get('STUB_PROBE')
    if probe == 'hang': time.sleep(60)
    if probe == 'refuse': sys.exit(7)
    if probe == 'relative':
        print(json.dumps({'interpreter':'python','prefix':sys.prefix,'devices':['cpu']}))
        sys.exit(0)
    if probe == 'truncated':
        print('x' * 40000)
        sys.exit(0)
    if probe in ('no_cuda','no_devices','no_allocation','no_cpu'):
        import types
        fake=types.ModuleType('torch')
        fake.cuda=types.SimpleNamespace(
            is_available=lambda: probe!='no_cuda',
            device_count=lambda: 0 if probe=='no_devices' else 1,
            get_device_name=lambda index: 'stub')
        def empty(*args,**kwargs):
            if probe in ('no_cuda','no_devices'): return object()
            raise RuntimeError('allocation refused')
        fake.empty=empty
        sys.modules['torch']=fake
        exec(sys.argv[2])
        sys.exit(0)
    print(json.dumps({'interpreter':sys.executable,'prefix':sys.prefix,'devices':['cpu']}))
    sys.exit(0)
target = next(sys.argv[i + 1] for i, arg in enumerate(sys.argv)
              if arg in ('--metrics_jsonl', '--metrics_json'))
record = os.path.join(os.path.dirname(target), 'child.json')
def write(**extra):
    with open(record, 'w') as stream:
        json.dump({'argv': sys.argv, 'cwd': os.getcwd(), 'env': dict(os.environ), **extra}, stream)
mode=os.environ.get('STUB_MODE','exit')
if mode=='interrupt':
    signal.signal(signal.SIGINT, lambda *_: sys.exit(42))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(43))
if mode=='tree':
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
    child=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'])
    write(child=child.pid)
else:
    write()
if mode in ('sleep','interrupt','tree'): time.sleep(float(os.environ.get('STUB_SLEEP','60')))
sys.exit(int(os.environ.get('STUB_EXIT','0')))
"""


@pytest.fixture
def dispatch(tmp_path):
    with DispatchLock(tmp_path / "dispatch.lock") as lock:
        yield lock


def stub(tmp_path):
    executable = tmp_path / ("runtime-" + uuid4().hex)
    executable.write_text(f"#!{sys.executable}\n" + STUB)
    executable.chmod(0o700)
    return executable


def prepared(
    tmp_path, operation=Operation.TRAIN, device="cpu", executable=None, **environment
):
    (tmp_path / "runs").mkdir(mode=0o700, exist_ok=True)
    config = (
        {"epochs": 1}
        if operation is Operation.TRAIN
        else {"checkpoint": "checkpoint", "data_path": "data.npz"}
    )
    spec = build_spec(
        tmp_path / "runs" / str(uuid4()),
        operation=operation,
        resolved_config=config,
        runtime=Runtime(
            str(executable or stub(tmp_path)), "selected", device, environment
        ),
        submission_cwd=tmp_path,
    )
    return LocalBackend().plan(spec)


def exited(backend, handle, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = backend.poll(handle)
        if result is not None:
            return result
        time.sleep(0.01)
    raise AssertionError("process did not exit")


def child(plan, attempt=1):
    return json.loads((plan.spec.paths.attempt(attempt) / "child.json").read_text())


def test_spawn_claims_the_attempt_and_names_its_files(tmp_path, dispatch):
    plan = prepared(tmp_path)
    backend = LocalBackend(dispatch_lock=dispatch)
    try:
        probed, handle = backend.spawn(plan, 1)
        assert exited(backend, handle) == LocalExit(0, None)
        paths = plan.spec.paths
        assert paths.attempt(1).stat().st_mode & 0o777 == 0o700
        assert paths.output.is_dir()
        started = child(plan)
        assert started["argv"][-4:] == [
            "--output_dir",
            str(paths.output),
            "--metrics_jsonl",
            str(paths.metrics(1)),
        ]
        assert Path(started["cwd"]).resolve() == paths.run_dir.resolve()
        assert probed.python == plan.spec.runtime.python
        assert probed.devices == ("cpu",)
        assert Path(probed.interpreter).is_absolute()
        assert backend.running == 0
    finally:
        backend.close()


def test_attempt_files_follow_the_operation(tmp_path):
    train = prepared(tmp_path)
    test = prepared(tmp_path, Operation.TEST, device="auto")
    plot = prepared(tmp_path, Operation.PLOT, device="auto")
    assert attempt_argv(train, 2)[-2:] == (
        "--metrics_jsonl",
        str(train.spec.paths.metrics(2)),
    )
    assert attempt_argv(test, 1)[-2:] == (
        "--metrics_json",
        str(test.spec.paths.evaluation(1)),
    )
    assert attempt_argv(plot, 1) == plot.argv


def test_an_attempt_is_started_at_most_once(tmp_path, dispatch):
    plan = prepared(tmp_path)
    backend = LocalBackend(dispatch_lock=dispatch)
    restarted = LocalBackend(dispatch_lock=dispatch)
    try:
        _, handle = backend.spawn(plan, 1)
        exited(backend, handle)
        record = child(plan)
        backend.release(handle)
        for owner in (backend, restarted):
            with pytest.raises(SupervisionLost, match="already claimed"):
                owner.spawn(plan, 1)
        assert child(plan) == record
        _, second = backend.spawn(plan, 2)
        assert exited(backend, second) == LocalExit(0, None)
    finally:
        restarted.close()
        backend.close()


def test_ambient_launcher_identity_is_removed_from_probe_and_child(
    tmp_path, dispatch, monkeypatch
):
    polluted = {
        "WAVEDL_SCHEDULER_BINDING": "1",
        "WAVEDL_DAEMON_TOKEN": "ambient-secret",
        "SLURM_JOB_ID": "foreign",
        "SBATCH_JOB_NAME": "foreign",
        "RANK": "4",
        "WORLD_SIZE": "99",
        "MASTER_ADDR": "foreign",
        "MASTER_PORT": "9",
    }
    for key, value in polluted.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("ORDINARY_PATH_SETTING", "preserved")
    plan = prepared(tmp_path, CUDA_VISIBLE_DEVICES="declared-device")
    backend = LocalBackend(dispatch_lock=dispatch)
    specs = []
    original = backend._supervisor.start

    class RecordingSupervisor:
        def start(self, spec, **kwargs):
            specs.append(spec)
            return original(spec, **kwargs)

    backend._supervisor = RecordingSupervisor()
    try:
        _, handle = backend.spawn(plan, 1)
        exited(backend, handle)
        environment = child(plan)["env"]
        assert all(key not in environment for key in polluted)
        assert environment["ORDINARY_PATH_SETTING"] == "preserved"
        assert environment["CUDA_VISIBLE_DEVICES"] == "declared-device"
        assert environment["WAVEDL_SUPERVISED"] == "1"
        assert len(specs) == 2
        for spec in specs:
            assert not set(polluted) & set(spec.environment)
        assert reserved_local_environment("wavedl_scheduler_binding")
    finally:
        backend.close()


@pytest.mark.parametrize(
    "key",
    ["WAVEDL_SCHEDULER_BINDING", "wavedl_daemon_token", "SLURM_JOB_ID", "RANK"],
)
def test_explicit_reserved_local_environment_is_refused_at_planning(tmp_path, key):
    with pytest.raises(UnsupportedPlan):
        prepared(tmp_path, **{key: "unapproved"})


def test_admission_is_bounded_by_running_processes(tmp_path, dispatch):
    plan = prepared(tmp_path, STUB_MODE="sleep")
    other = prepared(tmp_path)
    backend = LocalBackend(dispatch_lock=dispatch, max_processes=1, cancel_grace=0.05)
    try:
        assert backend.max_processes == 1
        _, handle = backend.spawn(plan, 1)
        assert backend.running == 1
        assert backend.poll(handle) is None
        with pytest.raises(BudgetExhausted):
            backend.spawn(other, 1)
        assert not other.spec.paths.attempt(1).exists()
        backend.cancel(handle)
        exited(backend, handle)
        assert backend.running == 0
        _, second = backend.spawn(other, 1)
        exited(backend, second)
    finally:
        backend.close()


def test_local_runs_are_not_capped_by_the_worker_budget(tmp_path, dispatch):
    backend = LocalBackend(dispatch_lock=dispatch, max_processes=3, cancel_grace=0.05)
    try:
        with BoundedThreadPool(WorkBudget(max_workers=1)) as pool:
            handles = [
                pool.control_lane.submit(
                    lambda token, plan=prepared(
                        tmp_path, STUB_MODE="sleep"
                    ): backend.spawn(plan, 1)[1]
                ).result(10)
                for _ in range(3)
            ]
        assert backend.running == 3
        assert all(backend.poll(handle) is None for handle in handles)
    finally:
        backend.close()


def test_a_saved_or_altered_handle_is_never_signalled(tmp_path, dispatch):
    plan = prepared(tmp_path, STUB_MODE="sleep")
    original = LocalBackend(dispatch_lock=dispatch, cancel_grace=0.05)
    restarted = LocalBackend(dispatch_lock=dispatch)
    try:
        _, handle = original.spawn(plan, 1)
        for foreign in (
            handle,
            replace(handle, pid=1),
            replace(handle, supervisor_id="forged"),
        ):
            owner = restarted if foreign == handle else original
            assert owner.poll(foreign) == NotOwned()
            with pytest.raises(SupervisionLost):
                owner.cancel(foreign)
        assert original.poll(handle) is None
    finally:
        restarted.close()
        original.close()


def test_cancel_interrupts_and_waits_the_launcher_grace(tmp_path, dispatch):
    plan = prepared(tmp_path, STUB_MODE="interrupt")
    backend = LocalBackend(dispatch_lock=dispatch)
    try:
        _, handle = backend.spawn(plan, 1)
        deadline = time.monotonic() + 5
        while not (plan.spec.paths.attempt(1) / "child.json").exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        backend.cancel(handle)
        assert exited(backend, handle) == LocalExit(42, None)
    finally:
        backend.close()
    assert CANCEL_GRACE_SECONDS == 45


def test_cancel_kills_a_tree_that_ignores_the_interrupt(tmp_path, dispatch):
    plan = prepared(tmp_path, STUB_MODE="tree")
    backend = LocalBackend(dispatch_lock=dispatch, cancel_grace=0.05)
    try:
        _, handle = backend.spawn(plan, 1)
        deadline = time.monotonic() + 5
        while not (plan.spec.paths.attempt(1) / "child.json").exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        backend.cancel(handle)
        backend.cancel(handle)
        assert exited(backend, handle) == LocalExit(None, signal.SIGKILL)
    finally:
        backend.close()


def test_nonzero_exit_is_reported_as_the_return_code(tmp_path, dispatch):
    plan = prepared(tmp_path, STUB_EXIT="3")
    backend = LocalBackend(dispatch_lock=dispatch)
    try:
        _, handle = backend.spawn(plan, 1)
        assert exited(backend, handle) == LocalExit(3, None)
    finally:
        backend.close()


def test_mutations_require_live_dispatch(tmp_path, dispatch):
    plan = prepared(tmp_path, STUB_MODE="sleep")
    backend = LocalBackend(dispatch_lock=dispatch, cancel_grace=0.05)
    missing = LocalBackend()
    with pytest.raises(DispatchRequired):
        missing.spawn(plan, 1)
    assert not plan.spec.paths.run_dir.exists()
    _, handle = backend.spawn(plan, 1)
    dispatch.release()
    try:
        with pytest.raises(DispatchRequired):
            backend.cancel(handle)
        with pytest.raises(DispatchRequired):
            backend.spawn(plan, 2)
    finally:
        dispatch.acquire()
        backend.close()
        missing.dispatch_lock = dispatch
        missing.close()


@pytest.mark.parametrize("probe", ["refuse", "hang", "truncated", "relative"])
def test_selected_runtime_probe_fails_closed(tmp_path, dispatch, probe):
    plan = prepared(tmp_path, STUB_PROBE=probe)
    backend = LocalBackend(dispatch_lock=dispatch, probe_timeout=0.2)
    try:
        with pytest.raises(UnsupportedPlan, match="no fallback"):
            backend.spawn(plan, 1)
        assert not (plan.spec.paths.attempt(1) / "child.json").exists()
        with pytest.raises(SupervisionLost):
            backend.spawn(plan, 1)
        assert backend.running == 0
    finally:
        backend.close()


@pytest.mark.parametrize(
    "probe,device",
    [
        ("no_cuda", "cuda"),
        ("no_devices", "cuda"),
        ("no_allocation", "cuda"),
        ("no_cpu", "cpu"),
    ],
)
def test_probe_executes_failclosed_device_code(tmp_path, dispatch, probe, device):
    environment = {"STUB_PROBE": probe, "PYTHONOPTIMIZE": "1"}
    if device == "cuda":
        environment["CUDA_VISIBLE_DEVICES"] = "0"
    plan = prepared(tmp_path, device=device, **environment)
    backend = LocalBackend(dispatch_lock=dispatch)
    try:
        with pytest.raises(UnsupportedPlan, match="no fallback"):
            backend.spawn(plan, 1)
    finally:
        backend.close()


def test_a_missing_interpreter_fails_without_fallback(tmp_path, dispatch):
    plan = prepared(tmp_path, executable=tmp_path / "missing-python")
    backend = LocalBackend(dispatch_lock=dispatch)
    try:
        with pytest.raises(OSError):
            backend.spawn(plan, 1)
        with pytest.raises(SupervisionLost):
            backend.spawn(plan, 1)
    finally:
        backend.close()


def test_a_failed_start_keeps_the_attempt_claimed(tmp_path, dispatch):
    plan = prepared(tmp_path)
    backend = LocalBackend(dispatch_lock=dispatch)
    original_start = backend._supervisor.start
    starts = 0

    def fail_spawn(*args, **kwargs):
        nonlocal starts
        starts += 1
        if starts > 1:
            raise OSError("runtime spawn failed")
        return original_start(*args, **kwargs)

    backend._supervisor.start = fail_spawn
    try:
        with pytest.raises(OSError, match="runtime spawn failed"):
            backend.spawn(plan, 1)
        assert backend.running == 0
        with pytest.raises(SupervisionLost, match="already claimed"):
            backend.spawn(plan, 1)
        assert starts == 2
    finally:
        backend.close()


def test_command_byte_bound_is_checked_before_the_claim(tmp_path, dispatch):
    plan = prepared(tmp_path)
    huge = replace(
        plan, argv=(*plan.argv[:-2], "--run_name", "x" * 70000, *plan.argv[-2:])
    )
    backend = LocalBackend(dispatch_lock=dispatch)
    try:
        with pytest.raises(UnsupportedPlan, match="byte budget"):
            backend.spawn(huge, 1)
        assert not plan.spec.paths.run_dir.exists()
    finally:
        backend.close()


def test_release_forgets_only_an_exited_process(tmp_path, dispatch):
    plan = prepared(tmp_path, STUB_MODE="sleep")
    backend = LocalBackend(dispatch_lock=dispatch, cancel_grace=0.05)
    try:
        _, handle = backend.spawn(plan, 1)
        backend.release(handle)
        backend.release(replace(handle, supervisor_id="forged"))
        assert backend.poll(handle) is None
        backend.cancel(handle)
        exited(backend, handle)
        backend.release(handle)
        assert backend.poll(handle) == NotOwned()
        backend.release(handle)
    finally:
        backend.close()


def test_close_stops_live_processes_and_refuses_new_ones(tmp_path, dispatch):
    plan = prepared(tmp_path, STUB_MODE="interrupt")
    backend = LocalBackend(dispatch_lock=dispatch, cancel_grace=5)
    _, handle = backend.spawn(plan, 1)
    deadline = time.monotonic() + 5
    while not (plan.spec.paths.attempt(1) / "child.json").exists():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    backend.close()
    assert backend.poll(handle) == LocalExit(42, None)
    with pytest.raises(RuntimeError, match="closed"):
        backend.spawn(prepared(tmp_path), 1)
