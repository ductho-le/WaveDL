"""Submission ambiguity must never silently create or cancel a different job."""

import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4

import pytest

from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.manifest import RuntimeSelection, build_manifest
from wavedl.core.persistence import write_once
from wavedl.core.safe_files import UnsafeOwnership
from wavedl_daemon.backends import slurm_submission
from wavedl_daemon.backends.contracts import Plan, UnsupportedPlan
from wavedl_daemon.backends.scheduler import SchedulerCommands
from wavedl_daemon.backends.slurm_plan import SlurmPlanner, SlurmSite, job_name
from wavedl_daemon.backends.slurm_submission import (
    CancellationError,
    SchedulerUnavailable,
    SlurmSubmission,
    SubmissionAmbiguous,
    SubmissionRejected,
    SubmissionUnknown,
)
from wavedl_daemon.execution.budget import BudgetExhausted, WorkBudget
from wavedl_daemon.storage.ownership import DispatchLock, DispatchRequired
from wavedl_daemon.supervision.base import ProcessResult


class Runner:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def run(self, argv):
        self.calls.append(argv)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def result(stdout=b"", **kwargs):
    return ProcessResult(kwargs.pop("returncode", 0), stdout, b"", **kwargs)


@pytest.fixture
def plan(tmp_path):
    manifest = build_manifest(
        RunLayout(tmp_path / "run", str(uuid4())),
        operation=OperationKind.TRAIN,
        resolved_config={"epochs": 1, "model": "mlp", "data_path": "train.npz"},
        runtime=RuntimeSelection(sys.executable, "selected", "cpu", {}),
        submission_cwd=tmp_path,
    )
    planner = SlurmPlanner(SlurmSite(partition="cpu", cluster="alpha"))
    if os.name == "nt":
        # The fake scheduler below tests submission state, not Windows Slurm
        # execution. Preserve the real native-path rendering refusal first.
        with pytest.raises(UnsupportedPlan, match="Slurm directive paths"):
            planner.plan(manifest)
        assert not Path(manifest.run_directory).exists()
        return Plan(
            "slurm",
            json.dumps(asdict(manifest)),
            script=(
                "# fixture-only fake scheduler input, never executed as a shell\n"
                "#SBATCH --job-name=@@WAVEDL_JOB_NAME@@\n"
                "#SBATCH --comment=@@WAVEDL_REQUEST_ID@@\n"
                "export WAVEDL_REQUEST_ID=@@WAVEDL_REQUEST_ID@@\n"
                "export WAVEDL_ATTEMPT=@@WAVEDL_ATTEMPT@@\n"
            ),
            resources=(("cluster", "alpha"),),
        )
    return planner.plan(manifest)


@pytest.fixture
def lock(tmp_path):
    with DispatchLock(tmp_path / "dispatch.lock") as held:
        yield held


def row(request, job="123", owner="alice", cluster="alpha"):
    return f"{job}|{owner}|{job_name(request)}|{request}|{cluster}\n".encode()


def test_submit_prepares_identity_bound_files_and_full_ack(plan, lock):
    request = str(uuid4())
    runner = Runner(result(b"123;alpha\n"))
    backend = SlurmSubmission(runner, owner="alice", dispatch_lock=lock)
    handle = backend.submit(plan, request_id=request, attempt=2)
    assert handle.job_id == "123;alpha"
    assert handle.request_id == request
    script = Path(runner.calls[0][-1])
    assert script.read_text().count(request) >= 2
    assert (Path(plan.run_dir) / "logs").is_dir()
    assert (Path(plan.run_dir) / "manifest.json").is_file()
    assert script.parent.name == "2"
    assert "--parsable" in runner.calls[0]
    with pytest.raises(SubmissionUnknown, match="reserved"):
        backend.submit(plan, request_id=request, attempt=2)
    assert len(runner.calls) == 1


@pytest.mark.parametrize(
    "ack",
    [
        result(b"accepted maybe"),
        result(b"12;alpha", timed_out=True),
        result(b"12;alpha", stdout_truncated=True),
        result(b"12;beta"),
        result(b"\xff"),
        OSError("lost acknowledgement"),
    ],
)
def test_uncertain_submission_never_retries(plan, lock, ack):
    runner = Runner(ack)
    backend = SlurmSubmission(runner, owner="alice", dispatch_lock=lock)
    request = str(uuid4())
    with pytest.raises(SubmissionUnknown):
        backend.submit(plan, request_id=request, attempt=1)
    with pytest.raises(SubmissionUnknown):
        backend.submit(plan, request_id=request, attempt=1)
    assert len(runner.calls) == 1


def test_scheduler_rejection_is_distinct(plan, lock):
    backend = SlurmSubmission(
        Runner(result(returncode=1)), owner="alice", dispatch_lock=lock
    )
    with pytest.raises(SubmissionRejected):
        backend.submit(plan, request_id=str(uuid4()), attempt=1)


def test_mutations_require_live_dispatch_before_files_or_commands(plan, lock):
    runner = Runner()
    backend = SlurmSubmission(runner, owner="alice", dispatch_lock=lock)
    lock.release()
    with pytest.raises(DispatchRequired):
        backend.submit(plan, request_id=str(uuid4()), attempt=1)
    assert not Path(plan.run_dir).exists()
    assert not runner.calls


def test_recovery_reads_accounting_and_deduplicates_live_job(plan):
    request = str(uuid4())
    runner = Runner(result(row(request)), result(row(request)))
    backend = SlurmSubmission(runner, owner="alice")
    handle = backend.find_by_request_id(plan, request_id=request, attempt=1)
    assert handle.job_id == "123;alpha"
    assert runner.calls[1][0] == "sacct"
    assert not any("1970" in arg for arg in runner.calls[1])
    assert not Path(plan.run_dir).exists()


def test_recovery_refuses_two_jobs_and_filters_owner_comment_name_cluster(plan):
    request = str(uuid4())
    backend = SlurmSubmission(
        Runner(result(row(request)), result(row(request, "456"))), owner="alice"
    )
    with pytest.raises(SubmissionAmbiguous):
        backend.find_by_request_id(plan, request_id=request, attempt=1)
    wrong = row(request, owner="bob") + row(request, cluster="beta")
    wrong += row(str(uuid4()))
    backend = SlurmSubmission(Runner(result(wrong), result()), owner="alice")
    assert backend.find_by_request_id(plan, request_id=request, attempt=1) is None


@pytest.mark.parametrize(
    "failure",
    [result(returncode=1), result(timed_out=True), result(b"bad"), OSError("offline")],
)
def test_failed_recovery_is_not_absence(plan, failure):
    backend = SlurmSubmission(Runner(failure), owner="alice")
    with pytest.raises(SchedulerUnavailable):
        backend.find_by_request_id(plan, request_id=str(uuid4()), attempt=1)


def test_cancel_confirms_live_identity_and_ack_is_not_terminal(plan, lock):
    request = str(uuid4())
    runner = Runner(result(b"123;alpha"), result(row(request)), result())
    backend = SlurmSubmission(runner, owner="alice", dispatch_lock=lock)
    handle = backend.submit(plan, request_id=request, attempt=1)
    assert backend.request_cancel(handle) is None
    assert runner.calls[-1] == (
        "scancel",
        "--user=alice",
        f"--name={job_name(request)}",
        "--clusters=alpha",
        "123",
    )
    with pytest.raises(ValueError, match="foreign"):
        backend.request_cancel(replace(handle, owner="bob"))


def test_cancel_reused_job_or_missing_identity_never_invokes_scancel(plan, lock):
    request = str(uuid4())
    runner = Runner(result(b"123;alpha"), result(row(str(uuid4()))))
    backend = SlurmSubmission(runner, owner="alice", dispatch_lock=lock)
    handle = backend.submit(plan, request_id=request, attempt=1)
    with pytest.raises(CancellationError, match="identity"):
        backend.request_cancel(handle)
    assert all(call[0] != "scancel" for call in runner.calls)
    with pytest.raises(ValueError, match="identity"):
        backend.request_cancel(replace(handle, raw_json="{}"))


@pytest.mark.parametrize("raw_json", ["not-json", "[]", "null"])
def test_cancel_corrupted_handle_payload_never_queries_scheduler(plan, lock, raw_json):
    request = str(uuid4())
    runner = Runner(result(b"123;alpha"))
    backend = SlurmSubmission(runner, owner="alice", dispatch_lock=lock)
    handle = backend.submit(plan, request_id=request, attempt=1)
    # Simulate corruption after the frozen constructor's persistence validation.
    object.__setattr__(handle, "raw_json", raw_json)
    runner.calls.clear()
    with pytest.raises(ValueError, match="identity"):
        backend.request_cancel(handle)
    assert runner.calls == []


def test_cancel_timeout_is_explicit(plan, lock):
    request = str(uuid4())
    runner = Runner(result(b"123;alpha"), result(row(request)), result(timed_out=True))
    backend = SlurmSubmission(runner, owner="alice", dispatch_lock=lock)
    handle = backend.submit(plan, request_id=request, attempt=1)
    with pytest.raises(CancellationError):
        backend.request_cancel(handle)


@pytest.mark.parametrize("mode", ["ack", "timeout", "truncated"])
def test_real_bounded_fake_scheduler_invocation(plan, lock, tmp_path, mode):
    executable = tmp_path / "fake-sbatch"
    programs = {
        "ack": "print('345;alpha')",
        "timeout": "import time; time.sleep(2)",
        "truncated": "print('9' * 10000)",
    }
    write_once(executable, f"#!{sys.executable}\n{programs[mode]}\n".encode())
    executable.chmod(0o700)
    supervisor = None
    if os.name == "nt":
        from wavedl_daemon.supervision.windows import WindowsSupervisor

        class PythonFixtureSupervisor:
            """Run the owned Python fixture without relying on POSIX shebangs."""

            def start(self, spec, *, output_limit):
                assert spec.argv[0] == str(executable)
                return WindowsSupervisor().start(
                    replace(spec, argv=(sys.executable, *spec.argv)),
                    output_limit=output_limit,
                )

        supervisor = PythonFixtureSupervisor()
    runner = SchedulerCommands(
        cwd=tmp_path,
        executables={"sbatch": str(executable)},
        budget=WorkBudget(
            timeout_seconds=0.1 if mode == "timeout" else 5, max_output_bytes=32
        ),
        supervisor=supervisor,
    )
    backend = SlurmSubmission(runner, owner="alice", dispatch_lock=lock)
    if mode == "ack":
        assert (
            backend.submit(plan, request_id=str(uuid4()), attempt=1).job_id
            == "345;alpha"
        )
    else:
        with pytest.raises(SubmissionUnknown):
            backend.submit(plan, request_id=str(uuid4()), attempt=1)


def test_storage_error_prevents_scheduler_side_effect(plan, lock, monkeypatch):
    runner = Runner()
    backend = SlurmSubmission(runner, owner="alice", dispatch_lock=lock)

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr("wavedl_daemon.backends.slurm_submission.write_once", fail)
    with pytest.raises(OSError, match="disk full"):
        backend.submit(plan, request_id=str(uuid4()), attempt=1)
    assert not runner.calls


@pytest.mark.parametrize("name", ["manifest", "intent", "script"])
def test_oversized_persisted_input_and_script_refused(plan, lock, name):
    runner = Runner()
    backend = SlurmSubmission(runner, owner="alice", dispatch_lock=lock)
    request = str(uuid4())
    if name == "script":
        plan = replace(plan, script=plan.script + "#" * 1048577)
        with pytest.raises(BudgetExhausted):
            backend.submit(plan, request_id=request, attempt=1)
    else:
        path = Path(plan.run_dir) / (
            "manifest.json" if name == "manifest" else "attempts/1/submission.json"
        )
        path.parent.mkdir(parents=True)
        write_once(path, b"x" * (1048577 if name == "manifest" else 16385))
        if name == "manifest":
            with pytest.raises(BudgetExhausted):
                backend.submit(plan, request_id=request, attempt=1)
        else:
            with pytest.raises(SchedulerUnavailable):
                backend.find_by_request_id(plan, request_id=request, attempt=1)
    assert not runner.calls


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires FIFO support")
@pytest.mark.parametrize("name", ["manifest", "intent"])
def test_fifo_submission_input_refused_without_blocking(plan, lock, name):
    runner = Runner()
    backend = SlurmSubmission(runner, owner="alice", dispatch_lock=lock)
    path = Path(plan.run_dir) / (
        "manifest.json" if name == "manifest" else "attempts/1/submission.json"
    )
    path.parent.mkdir(parents=True)
    os.mkfifo(path)
    if name == "manifest":
        with pytest.raises(ValueError, match="regular"):
            backend.submit(plan, request_id=str(uuid4()), attempt=1)
    else:
        with pytest.raises(SchedulerUnavailable):
            backend.find_by_request_id(plan, request_id=str(uuid4()), attempt=1)
    assert not runner.calls


@pytest.mark.parametrize("size", [0, 65536, 1048576])
def test_persisted_input_reader_retains_exact_bounded_bytes(tmp_path, size):
    path = tmp_path / "persisted.json"
    content = b"x" * size
    write_once(path, content)
    assert slurm_submission._read_bounded(path, size) == content
    if size:
        with pytest.raises(BudgetExhausted):
            slurm_submission._read_bounded(path, size - 1)


def test_submission_manifest_symlink_is_refused_before_scheduler(plan, lock):
    directory = Path(plan.run_dir)
    directory.mkdir(parents=True, mode=0o700)
    target = directory / "private-target.json"
    write_once(target, plan.manifest_json.encode())
    (directory / "manifest.json").symlink_to(target)
    runner = Runner()
    backend = SlurmSubmission(runner, owner="alice", dispatch_lock=lock)
    with pytest.raises((OSError, ValueError)):
        backend.submit(plan, request_id=str(uuid4()), attempt=1)
    assert runner.calls == []
    assert target.read_bytes() == plan.manifest_json.encode()
    assert not (directory / "attempts").exists()


def test_persisted_input_reader_rejects_linked_parent_and_directory(tmp_path):
    owned = tmp_path / "owned"
    owned.mkdir(mode=0o700)
    write_once(owned / "input.json", b"owned")
    alias = tmp_path / "alias"
    alias.symlink_to(owned, target_is_directory=True)
    with pytest.raises((OSError, ValueError)):
        slurm_submission._read_bounded(alias / "input.json", 100)
    with pytest.raises((OSError, ValueError)):
        slurm_submission._read_bounded(owned, 100)


def test_persisted_input_reader_enforces_native_private_owner(tmp_path, monkeypatch):
    from wavedl.core import safe_files

    path = tmp_path / "input.json"
    write_once(path, b"owned")
    calls = []

    def denied(api, handle, *, confidential=False):
        calls.append(handle)
        raise UnsafeOwnership("Untrusted input owner")

    monkeypatch.setattr(safe_files, "_private_owner", denied)
    with pytest.raises(UnsafeOwnership):
        slurm_submission._read_bounded(path, 100)
    assert len(calls) == 1


def test_persisted_input_reader_bounds_growth_and_closes_file(tmp_path, monkeypatch):
    from wavedl.core import safe_files

    path = tmp_path / "input.json"
    write_once(path, b"short")
    opened = []

    def grown_read(stream, count):
        opened.append(stream)
        assert count == 6  # At most limit+1 bytes, even after the size check.
        return b"grown!"

    monkeypatch.setattr(safe_files.SafeFile, "read", grown_read)
    with pytest.raises(BudgetExhausted):
        slurm_submission._read_bounded(path, 5)
    assert len(opened) == 1 and opened[0].handle is None


@pytest.mark.parametrize("changed", ["member", "parent"])
def test_persisted_input_reader_rechecks_owner_after_read_and_closes(
    tmp_path, monkeypatch, changed
):
    from wavedl.core import safe_files

    path = tmp_path / "input.json"
    write_once(path, b"owned")
    native_owner = safe_files._private_owner
    native_read = safe_files.SafeFile.read
    owner_calls, opened = [], []
    read_started = False

    def owner(api, handle, *, confidential=False):
        owner_calls.append(handle)
        if read_started:
            target = owner_calls[1] if changed == "member" else owner_calls[0]
            if handle == target:
                raise UnsafeOwnership("Ownership changed while reading")
        return native_owner(api, handle, confidential=confidential)

    def read(stream, count):
        nonlocal read_started
        read_started = True
        opened.append(stream)
        return native_read(stream, count)

    monkeypatch.setattr(safe_files, "_private_owner", owner)
    monkeypatch.setattr(safe_files.SafeFile, "read", read)
    with pytest.raises(UnsafeOwnership, match="changed while reading"):
        slurm_submission._read_bounded(path, 100)
    assert opened and all(stream.handle is None for stream in opened)
    assert len(owner_calls) == (3 if changed == "member" else 4)


@pytest.mark.skipif(os.name != "posix", reason="Windows pins replacement natively")
def test_persisted_input_reader_checks_identity_after_read(tmp_path, monkeypatch):
    from wavedl.core import safe_files

    path = tmp_path / "input.json"
    write_once(path, b"owned")
    native_read = safe_files.SafeFile.read
    replaced = []

    def replace_after_read(stream, count):
        chunk = native_read(stream, count)
        if not replaced:
            path.unlink()
            path.write_bytes(b"other")
            replaced.append(stream)
        return chunk

    monkeypatch.setattr(safe_files.SafeFile, "read", replace_after_read)
    with pytest.raises(ValueError, match="no longer identifies pinned file"):
        slurm_submission._read_bounded(path, 100)
    assert replaced[0].handle is None
    assert path.read_bytes() == b"other"


@pytest.mark.skipif(os.name != "posix", reason="Native mode ownership contract")
@pytest.mark.parametrize("member", [False, True])
def test_persisted_input_reader_rejects_writable_parent_or_member(tmp_path, member):
    owned = tmp_path / "owned"
    owned.mkdir(mode=0o700)
    path = owned / "input.json"
    write_once(path, b"untrusted-writable")
    (path if member else owned).chmod(0o666 if member else 0o777)
    with pytest.raises(UnsafeOwnership):
        slurm_submission._read_bounded(path, 100)
