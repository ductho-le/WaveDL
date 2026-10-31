"""Disposable stub runtimes exercise real Local process ownership, not ML."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from wavedl.core.layout import OperationKind, RunLayout, allocate_attempt
from wavedl.core.live_push import LivePushConfig
from wavedl.core.manifest import RuntimeSelection, build_manifest, write_manifest
from wavedl_daemon.backends.contracts import (
    ObservationStatus,
    OutcomeStatus,
    UnsupportedPlan,
)
from wavedl_daemon.backends.local import LocalBackend
from wavedl_daemon.backends.local_execution import SupervisionLost
from wavedl_daemon.execution.budget import BudgetExhausted, WorkBudget
from wavedl_daemon.execution.pool import BoundedThreadPool
from wavedl_daemon.storage.ownership import DispatchLock, DispatchRequired


STUB = """
import json,os,signal,subprocess,sys,time
if sys.argv[1] == '-c':
    if os.environ.get('STUB_PROBE') == 'hang': time.sleep(60)
    if os.environ.get('STUB_PROBE') == 'refuse': sys.exit(7)
    if os.environ.get('STUB_PROBE') == 'duplicate':
        print('{"interpreter":"/foreign","interpreter":"/actual","prefix":"/env","base_prefix":"/base"}')
        sys.exit(0)
    if os.environ.get('STUB_PROBE') == 'truncated':
        print('WAVEDL_LOCAL_PROBE_OK'+' ' * 10000)
        sys.exit(0)
    if os.environ.get('STUB_PROBE') in ('no_cuda','no_devices','no_allocation','no_cpu'):
        import types
        fake=types.ModuleType('torch')
        fake.cuda=types.SimpleNamespace(is_available=lambda:os.environ['STUB_PROBE']!='no_cuda', device_count=lambda:0 if os.environ['STUB_PROBE']=='no_devices' else 1)
        def empty(*args,**kwargs):
            if os.environ['STUB_PROBE'] in ('no_cuda','no_devices'): return object()
            raise RuntimeError('allocation refused')
        fake.empty=empty
        sys.modules['torch']=fake
        exec(sys.argv[2])
        sys.exit(0)
    print(json.dumps({'interpreter':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix}))
    sys.exit(0)
print(json.dumps({k:os.environ[k] for k in ('WAVEDL_RUN_ID','WAVEDL_REQUEST_ID','WAVEDL_ATTEMPT','WAVEDL_OWNER')}),flush=True)
mode=os.environ.get('STUB_MODE','exit')
if mode=='output':
    sys.stdout.write('x'*1000000)
    sys.stderr.write('y'*1000000)
if mode=='tree':
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
    child=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'])
    print('CHILD='+str(child.pid),flush=True)
    time.sleep(60)
if mode=='sleep': time.sleep(float(os.environ.get('STUB_SLEEP','60')))
sys.exit(int(os.environ.get('STUB_EXIT','0')))
"""


@pytest.fixture
def dispatch(tmp_path):
    with DispatchLock(tmp_path / "dispatch.lock") as lock:
        yield lock


def prepared(tmp_path, **environment):
    # Executable stubs exercise the selected-interpreter seam without torch.
    # Native Windows process invocation uses its own supervisor CI tests.
    executable = tmp_path / ("runtime-" + uuid4().hex)
    executable.write_text(f"#!{sys.executable}\n" + STUB)
    executable.chmod(0o700)
    layout = RunLayout.create(tmp_path / "runs", str(uuid4()))
    manifest = build_manifest(
        layout,
        operation=OperationKind.TRAIN,
        resolved_config={"epochs": 1},
        runtime=RuntimeSelection(str(executable), "selected", "cpu", environment),
        submission_cwd=tmp_path,
    )
    write_manifest(layout, manifest)
    attempt = allocate_attempt(layout)
    return LocalBackend().plan(manifest), attempt


def terminal(backend, handle):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        observation = backend.observe(handle)
        if observation.status == ObservationStatus.TERMINAL:
            return observation
        time.sleep(0.01)
    raise AssertionError("process did not become terminal")


def test_local_runs_not_capped_by_worker_budget(tmp_path, dispatch):
    budget = WorkBudget(max_workers=1)
    # The backend's own process cap is separate; lift it to isolate the worker budget.
    backend = LocalBackend(dispatch_lock=dispatch, max_processes=3)
    handles = []
    try:
        with BoundedThreadPool(budget) as pool:
            for index in range(3):
                plan, attempt = prepared(tmp_path, STUB_MODE="sleep", STUB_SLEEP="0.5")
                handle = pool.control_lane.submit(
                    lambda token,
                    plan=plan,
                    attempt=attempt,
                    index=index: backend.submit(
                        plan, request_id=f"request-{index}", attempt=attempt
                    )
                ).result(5)
                handles.append(handle)
            assert len(handles) == 3
            assert all(
                backend.observe(handle).status == ObservationStatus.RUNNING
                for handle in handles
            )
    finally:
        backend.close()


def test_ambient_launcher_identity_is_removed_from_probe_and_actual_child(
    tmp_path, dispatch, monkeypatch
):
    from wavedl_daemon.backends.local_execution import reserved_local_environment

    polluted = {
        "WAVEDL_SCHEDULER_BINDING": "1",
        "WAVEDL_DAEMON_TOKEN": "ambient-secret",
        "SLURM_JOB_ID": "foreign",
        "SLURM_RESTART_COUNT": "99",
        "SBATCH_JOB_NAME": "foreign",
        "RANK": "4",
        "LOCAL_RANK": "4",
        "WORLD_SIZE": "99",
        "MASTER_ADDR": "foreign",
        "MASTER_PORT": "9",
    }
    for key, value in polluted.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("ORDINARY_PATH_SETTING", "preserved")
    plan, attempt = prepared(tmp_path, CUDA_VISIBLE_DEVICES="declared-device")
    executable = Path(plan.argv[0])
    executable.write_text(
        f"#!{sys.executable}\n" + "import os,sys,json\n"
        "if sys.argv[1]=='-c':\n"
        " assert all(key not in os.environ for key in " + repr(tuple(polluted)) + ")\n"
        " print(json.dumps({'interpreter':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix})); sys.exit(0)\n"
        "print(json.dumps(dict(os.environ)))\n"
    )
    backend = LocalBackend(dispatch_lock=dispatch, owner="controller")
    try:
        handle = backend.submit(plan, request_id="request", attempt=attempt)
        terminal(backend, handle)
        environment = json.loads(backend.process_result(handle).stdout)
        assert all(key not in environment for key in polluted)
        assert environment["ORDINARY_PATH_SETTING"] == "preserved"
        assert environment["CUDA_VISIBLE_DEVICES"] == "declared-device"
        assert environment["WAVEDL_RUN_ID"] == plan.run_id
        assert environment["WAVEDL_REQUEST_ID"] == "request"
        assert environment["WAVEDL_SUPERVISED"] == "1"
        assert reserved_local_environment("wavedl_scheduler_binding")
    finally:
        backend.close()


@pytest.mark.parametrize(
    "key",
    [
        "WAVEDL_SCHEDULER_BINDING",
        "wavedl_daemon_token",
        "SLURM_JOB_ID",
        "RANK",
        "MASTER_ADDR",
    ],
)
def test_explicit_reserved_local_environment_is_refused_before_execution(tmp_path, key):
    with pytest.raises(UnsupportedPlan):
        prepared(tmp_path, **{key: "unapproved"})


def test_live_credentials_are_ephemeral_child_environment_only(tmp_path, dispatch):
    plan, attempt = prepared(tmp_path)
    token = "ephemeral-private-token"
    backend = LocalBackend(
        dispatch_lock=dispatch,
        live_push=LivePushConfig("http://127.0.0.1:8420", token),
    )
    specs = []
    original = backend._supervisor.start

    class RecordingSupervisor:
        def start(self, spec, **kwargs):
            specs.append(spec)
            return original(spec, **kwargs)

    backend._supervisor = RecordingSupervisor()
    try:
        handle = backend.submit(plan, request_id="request", attempt=attempt)
        terminal(backend, handle)
        assert "WAVEDL_DAEMON_TOKEN" not in specs[0].environment
        assert specs[1].environment["WAVEDL_DAEMON_TOKEN"] == token
        assert specs[1].environment["WAVEDL_DAEMON_URL"] == "http://127.0.0.1:8420"
        assert token not in plan.to_json()
        root = Path(plan.run_dir)
        assert token.encode() not in (root / "manifest.json").read_bytes()
        assert (
            token.encode()
            not in (root / "attempts/1/local-submission.json").read_bytes()
        )
    finally:
        backend.close()


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="executable stubs require POSIX; native Windows supervisor tests cover Jobs",
)


def test_fast_exit_identity_and_independent_runtime_outcome(tmp_path, dispatch):
    plan, attempt = prepared(tmp_path)
    backend = LocalBackend(dispatch_lock=dispatch, owner="controller")
    try:
        handle = backend.submit(plan, request_id="request", attempt=attempt)
        evidence = terminal(backend, handle)
        assert evidence.exit_code == 0
        result = backend.process_result(handle)
        assert json.loads(result.stdout) == {
            "WAVEDL_RUN_ID": plan.run_id,
            "WAVEDL_REQUEST_ID": "request",
            "WAVEDL_ATTEMPT": "1",
            "WAVEDL_OWNER": "controller",
        }
        assert backend.collect(handle).batch.status == OutcomeStatus.SUCCESS
        assert backend.collect(handle).runtime.status == OutcomeStatus.UNKNOWN
        assert backend.collect(handle).stage_out.status == OutcomeStatus.UNKNOWN
        assert backend.observe(handle) == evidence
        assert backend.submit(plan, request_id="request", attempt=attempt) == handle
        with pytest.raises(ValueError, match="another request"):
            backend.submit(plan, request_id="other-request", attempt=attempt)
        with pytest.raises(ValueError, match="another plan"):
            backend.submit(
                replace(plan, warnings=("different snapshot",)),
                request_id="request",
                attempt=attempt,
            )
    finally:
        backend.close()


def test_live_retry_does_not_spawn_and_admission_is_bounded(tmp_path, dispatch):
    plan, attempt = prepared(tmp_path, STUB_MODE="sleep")
    other, other_attempt = prepared(tmp_path)
    backend = LocalBackend(dispatch_lock=dispatch, max_processes=1, max_handles=1)
    try:
        handle = backend.submit(plan, request_id="request", attempt=attempt)
        assert backend.submit(plan, request_id="request", attempt=attempt) == handle
        with pytest.raises(BudgetExhausted):
            backend.submit(other, request_id="other", attempt=other_attempt)
        backend.request_cancel(handle)
        assert terminal(backend, handle).batch.status == OutcomeStatus.CANCELLED
        with pytest.raises(BudgetExhausted):
            backend.submit(other, request_id="other", attempt=other_attempt)
    finally:
        backend.close()


def test_restarted_backend_never_signals_saved_pid(tmp_path, dispatch):
    plan, attempt = prepared(tmp_path, STUB_MODE="sleep")
    original = LocalBackend(dispatch_lock=dispatch)
    restarted = LocalBackend(dispatch_lock=dispatch)
    try:
        handle = original.submit(plan, request_id="request", attempt=attempt)
        assert restarted.observe(handle).status == ObservationStatus.SUPERVISION_LOST
        with pytest.raises(SupervisionLost):
            restarted.request_cancel(handle)
        assert original.observe(handle).status == ObservationStatus.RUNNING
        with pytest.raises(SupervisionLost):
            restarted.collect(handle)
    finally:
        restarted.close()
        original.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("owner", "foreign"),
        ("run_id", "11111111-1111-4111-8111-111111111111"),
        ("run_dir", "/foreign"),
        ("pid", 1),
        ("supervisor_id", "forged"),
    ],
)
def test_altered_handle_refused(tmp_path, dispatch, field, value):
    plan, attempt = prepared(tmp_path, STUB_MODE="sleep")
    backend = LocalBackend(dispatch_lock=dispatch)
    try:
        handle = backend.submit(plan, request_id="request", attempt=attempt)
        foreign = replace(handle, **{field: value})
        with pytest.raises(ValueError, match="foreign"):
            backend.request_cancel(foreign)
        with pytest.raises(ValueError, match="foreign"):
            backend.observe(foreign)
    finally:
        backend.close()


def test_cancel_escalates_real_tree(tmp_path, dispatch):
    plan, attempt = prepared(tmp_path, STUB_MODE="tree")
    backend = LocalBackend(dispatch_lock=dispatch, cancel_grace=0.05)
    try:
        handle = backend.submit(plan, request_id="request", attempt=attempt)
        # Give the disposable child time to install its TERM handler.
        time.sleep(0.15)
        backend.request_cancel(handle)
        backend.request_cancel(handle)
        observation = terminal(backend, handle)
        assert observation.signal == 9
        assert observation.batch.status == OutcomeStatus.CANCELLED
        assert observation.native_state == "CANCELLED"
        assert b"CHILD=" in backend.process_result(handle).stdout
    finally:
        backend.close()


def test_output_drains_with_bounded_retention(tmp_path, dispatch):
    plan, attempt = prepared(tmp_path, STUB_MODE="output")
    backend = LocalBackend(dispatch_lock=dispatch, output_limit=512)
    try:
        handle = backend.submit(plan, request_id="request", attempt=attempt)
        assert terminal(backend, handle).exit_code == 0
        result = backend.process_result(handle)
        assert len(result.stdout) == len(result.stderr) == 512
        assert result.stdout_truncated and result.stderr_truncated
    finally:
        backend.close()


def test_mutations_require_live_dispatch(tmp_path, dispatch):
    plan, attempt = prepared(tmp_path, STUB_MODE="sleep")
    backend = LocalBackend(dispatch_lock=dispatch)
    missing = LocalBackend()
    with pytest.raises(DispatchRequired):
        missing.submit(plan, request_id="request", attempt=attempt)
    handle = backend.submit(plan, request_id="request", attempt=attempt)
    dispatch.release()
    try:
        with pytest.raises(DispatchRequired):
            backend.request_cancel(handle)
        with pytest.raises(DispatchRequired):
            backend.submit(plan, request_id="request", attempt=attempt)
    finally:
        dispatch.acquire()
        backend.close()
        missing.dispatch_lock = dispatch
        missing.close()


def test_snapshot_and_reservation_refusals(tmp_path, dispatch):
    plan, attempt = prepared(tmp_path)
    backend = LocalBackend(dispatch_lock=dispatch)
    try:
        with pytest.raises(UnsupportedPlan, match="snapshot"):
            backend.submit(
                replace(plan, argv=(sys.executable, "-c", "print('forged')")),
                request_id="bad",
                attempt=attempt,
            )
        with pytest.raises(ValueError, match="reserved"):
            backend.submit(plan, request_id="bad", attempt=2)
        manifest_path = Path(plan.run_dir) / "manifest.json"
        payload = json.loads(manifest_path.read_text())
        payload["resolved_config"]["epochs"] = 99
        manifest_path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="immutable"):
            backend.submit(plan, request_id="bad", attempt=attempt)
    finally:
        backend.close()


@pytest.mark.parametrize("probe", ["refuse", "hang", "truncated", "duplicate"])
def test_selected_runtime_probe_fails_closed(tmp_path, dispatch, probe):
    plan, attempt = prepared(tmp_path, STUB_PROBE=probe)
    backend = LocalBackend(dispatch_lock=dispatch, probe_timeout=0.05)
    try:
        with pytest.raises(UnsupportedPlan, match="no fallback"):
            backend.submit(plan, request_id="request", attempt=attempt)
    finally:
        backend.close()


def test_cuda_probe_checks_allocation_in_selected_runtime(tmp_path, dispatch):
    plan, attempt = prepared(tmp_path, STUB_PROBE="refuse")
    manifest = replace(
        plan.manifest,
        runtime=replace(
            plan.manifest.runtime,
            device="cuda",
            environment={"CUDA_VISIBLE_DEVICES": "0", "STUB_PROBE": "refuse"},
        ),
    )
    plan = LocalBackend().plan(manifest)
    # Replace test fixture manifest to match this new test request exactly.
    (Path(plan.run_dir) / "manifest.json").write_text(plan.manifest_json)
    backend = LocalBackend(dispatch_lock=dispatch)
    try:
        with pytest.raises(UnsupportedPlan, match="no fallback"):
            backend.submit(plan, request_id="request", attempt=attempt)
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
    plan, attempt = prepared(tmp_path, STUB_PROBE=probe)
    environment = {"STUB_PROBE": probe, "PYTHONOPTIMIZE": "1"}
    if device == "cuda":
        environment["CUDA_VISIBLE_DEVICES"] = "0"
    manifest = replace(
        plan.manifest,
        runtime=replace(plan.manifest.runtime, device=device, environment=environment),
    )
    plan = LocalBackend().plan(manifest)
    (Path(plan.run_dir) / "manifest.json").write_text(plan.manifest_json)
    backend = LocalBackend(dispatch_lock=dispatch)
    try:
        with pytest.raises(UnsupportedPlan, match="no fallback"):
            backend.submit(plan, request_id="request", attempt=attempt)
    finally:
        backend.close()


def test_interpreter_missing_does_not_fallback(tmp_path, dispatch):
    plan, attempt = prepared(tmp_path)
    Path(plan.argv[0]).unlink()
    backend = LocalBackend(dispatch_lock=dispatch)
    try:
        with pytest.raises(FileNotFoundError):
            backend.submit(plan, request_id="request", attempt=attempt)
    finally:
        backend.close()


def test_spawn_failure_retains_attempt_fence_and_never_retries(tmp_path, dispatch):
    """A failed Local spawn remains an uncertain, non-retryable attempt."""
    plan, attempt = prepared(tmp_path)
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
            backend.submit(plan, request_id="request", attempt=attempt)
        with pytest.raises(SupervisionLost, match="never resubmit"):
            backend.submit(plan, request_id="request", attempt=attempt)
    finally:
        backend.close()


def test_plan_byte_bound_before_runtime_spawn(tmp_path, dispatch):
    plan, attempt = prepared(tmp_path, STUB_PADDING="x" * 70000)
    backend = LocalBackend(dispatch_lock=dispatch)
    try:
        with pytest.raises(BudgetExhausted, match="command"):
            backend.submit(plan, request_id="request", attempt=attempt)
    finally:
        backend.close()


def test_release_terminal_reclaims_capacity_without_allowing_resubmit(
    tmp_path, dispatch
):
    first, attempt = prepared(tmp_path)
    second, second_attempt = prepared(tmp_path)
    backend = LocalBackend(dispatch_lock=dispatch, max_handles=1)
    restarted = LocalBackend(dispatch_lock=dispatch)
    try:
        handle = backend.submit(first, request_id="first", attempt=attempt)
        terminal(backend, handle)
        marker = (
            Path(first.run_dir) / "attempts" / str(attempt) / "local-submission.json"
        )
        reserved = marker.read_bytes()
        backend.release(handle)
        with pytest.raises(SupervisionLost, match="reservation"):
            backend.submit(first, request_id="first", attempt=attempt)
        with pytest.raises(SupervisionLost, match="reservation"):
            restarted.submit(first, request_id="first", attempt=attempt)
        with pytest.raises(SupervisionLost, match="reservation"):
            restarted.submit(first, request_id="different-request", attempt=attempt)
        assert marker.read_bytes() == reserved
        other = backend.submit(second, request_id="second", attempt=second_attempt)
        terminal(backend, other)
        backend.release(other)
        next_attempt = allocate_attempt(RunLayout(Path(first.run_dir), first.run_id))
        retried = backend.submit(first, request_id="first", attempt=next_attempt)
        assert retried.attempt == 2
        terminal(backend, retried)
        backend.release(retried)
    finally:
        restarted.close()
        backend.close()


def test_release_refuses_live_foreign_and_missing_dispatch(tmp_path, dispatch):
    plan, attempt = prepared(tmp_path, STUB_MODE="sleep")
    backend = LocalBackend(dispatch_lock=dispatch)
    try:
        handle = backend.submit(plan, request_id="request", attempt=attempt)
        with pytest.raises(RuntimeError, match="active"):
            backend.release(handle)
        with pytest.raises(ValueError, match="foreign"):
            backend.release(replace(handle, owner="foreign"))
        backend.dispatch_lock = None
        with pytest.raises(DispatchRequired):
            backend.release(handle)
        backend.dispatch_lock = dispatch
        backend.request_cancel(handle)
        terminal(backend, handle)
        backend.release(handle)
        with pytest.raises(SupervisionLost):
            backend.release(handle)
    finally:
        backend.dispatch_lock = dispatch
        backend.close()
