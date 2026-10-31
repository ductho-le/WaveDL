"""Scheduler-shaped fake responses; no real cluster commands."""

import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4

import pytest

from wavedl.core.layout import OperationKind, RunLayout, allocate_attempt
from wavedl.core.manifest import RuntimeSelection, build_manifest, write_manifest
from wavedl.core.outcomes import write_outcome_receipt
from wavedl.core.scheduler_attempt import (
    SchedulerBinding,
    SchedulerIdentity,
    coordinate_attempt,
)
from wavedl_daemon.backends.contracts import Handle, ObservationStatus, OutcomeStatus
from wavedl_daemon.backends.slurm_observation import (
    RestartBindingUnavailable,
    SlurmObservation,
)
from wavedl_daemon.execution.budget import BudgetExhausted
from wavedl_daemon.supervision.base import ProcessResult


class Runner:
    def __init__(self, **responses):
        self.responses = responses
        self.calls = []

    def run(self, argv):
        self.calls.append(argv)
        result = self.responses.get(argv[0], "")
        if isinstance(result, Exception):
            raise result
        if isinstance(result, ProcessResult):
            return result
        return ProcessResult(0, result.encode(), b"")


def handle(tmp_path, job="42;alpha", **raw_changes):
    request = str(uuid4())
    raw = {
        "cluster": "alpha",
        "job_name": f"wavedl-{request}",
        "comment": request,
        "restart_count": 0,
        "operation": "train",
    }
    raw.update(raw_changes)
    return Handle(
        "slurm",
        str(uuid4()),
        request,
        1,
        "alice",
        str(tmp_path),
        job_id=job,
        raw_json=json.dumps(raw),
    )


def account(h, state="COMPLETED", code="0:0", restart=0, suffix="", **changes):
    fields = {
        "job": h.job_id.split(";")[0] + suffix,
        "cluster": "alpha",
        "owner": "alice",
        "name": f"wavedl-{h.request_id}",
        "state": state,
        "code": code,
        "restart": str(restart),
        "submit": "2026-10-01T10:00:00",
        "comment": h.request_id,
    }
    fields.update(changes)
    return "|".join(fields.values()) + "\n"


def controller(h, state="RUNNING", restart=0, code="0:0", **changes):
    fields = {
        "JobId": h.job_id.split(";")[0],
        "UserId": "alice(1000)",
        "JobName": f"wavedl-{h.request_id}",
        "Comment": h.request_id,
        "JobState": state,
        "Restarts": str(restart),
        "ExitCode": code,
    }
    fields.update(changes)
    return " ".join(f"{key}={value}" for key, value in fields.items()) + "\n"


def receipt(h, **changes):
    directory = Path(h.run_dir) / "attempts" / "1"
    directory.mkdir(parents=True, exist_ok=True)
    values = {
        "run_id": h.run_id,
        "request_id": h.request_id,
        "attempt": 1,
        "job_id": "42",
        "cluster": "alpha",
        "restart_count": 0,
        "operation": "train",
        "runtime_exit": 0,
        "batch_exit": 0,
        "stage_out_exit": 0,
    }
    values.update(changes)
    return write_outcome_receipt(directory, **values)


@contextmanager
def windows_foreign_write_acl(path):
    """Temporarily grant one owned path access, restoring its original DACL."""
    import ctypes
    from ctypes import wintypes

    api = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    pointer = wintypes.LPVOID
    api.GetNamedSecurityInfoW.argtypes = [
        wintypes.LPWSTR,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.POINTER(pointer),
        ctypes.POINTER(pointer),
        ctypes.POINTER(pointer),
        ctypes.POINTER(pointer),
        ctypes.POINTER(pointer),
    ]
    api.GetNamedSecurityInfoW.restype = wintypes.DWORD
    api.GetSecurityDescriptorControl.argtypes = [
        pointer,
        ctypes.POINTER(wintypes.WORD),
        ctypes.POINTER(wintypes.DWORD),
    ]
    api.GetSecurityDescriptorControl.restype = wintypes.BOOL
    api.SetNamedSecurityInfoW.argtypes = [
        wintypes.LPWSTR,
        ctypes.c_int,
        wintypes.DWORD,
        pointer,
        pointer,
        pointer,
        pointer,
    ]
    api.SetNamedSecurityInfoW.restype = wintypes.DWORD
    kernel.LocalFree.argtypes = [pointer]
    kernel.LocalFree.restype = pointer
    descriptor, dacl = pointer(), pointer()
    code = api.GetNamedSecurityInfoW(
        str(path),
        1,
        4,
        None,
        None,
        ctypes.byref(dacl),
        None,
        ctypes.byref(descriptor),
    )
    if code:
        raise ctypes.WinError(code)
    try:
        control, revision = wintypes.WORD(), wintypes.DWORD()
        if not api.GetSecurityDescriptorControl(
            descriptor, ctypes.byref(control), ctypes.byref(revision)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        # Preserve whether inheritance was protected, not just the ACE list.
        flags = 4 | (0x80000000 if control.value & 0x1000 else 0x20000000)
        try:
            subprocess.run(
                ["icacls", str(path), "/grant", "*S-1-1-0:(W)"],
                check=True,
                capture_output=True,
                timeout=10,
            )
            yield
        finally:
            code = api.SetNamedSecurityInfoW(
                str(path), 1, flags, None, None, dacl, None
            )
            if code:
                raise ctypes.WinError(code)
    finally:
        if kernel.LocalFree(descriptor):
            raise ctypes.WinError(ctypes.get_last_error())


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("RUNNING", ObservationStatus.RUNNING),
        ("PENDING", ObservationStatus.QUEUED),
        ("REQUEUED", ObservationStatus.QUEUED),
        ("SPECIAL_EXIT", ObservationStatus.HELD),
        ("COMPLETED", ObservationStatus.TERMINAL),
        ("FAILED", ObservationStatus.TERMINAL),
    ],
)
def test_accounting_state_is_not_automatically_terminal(tmp_path, state, expected):
    h = handle(tmp_path)
    observation = SlurmObservation(
        Runner(sacct=account(h, state)), owner="alice"
    ).observe(h)
    assert observation.status == expected


def test_root_row_not_batch_extern_or_step(tmp_path):
    h = handle(tmp_path)
    rows = account(h, "RUNNING") + "".join(
        account(h, "FAILED", "9:0", suffix=suffix)
        for suffix in (".batch", ".extern", ".0")
    )
    observation = SlurmObservation(Runner(sacct=rows), owner="alice").observe(h)
    assert observation.status == ObservationStatus.RUNNING
    assert observation.exit_code is None


def test_exit_and_signal_are_preserved_separately(tmp_path):
    h = handle(tmp_path)
    observation = SlurmObservation(
        Runner(sacct=account(h, "FAILED", "0:15")), owner="alice"
    ).observe(h)
    assert observation.exit_code == 0 and observation.signal == 15
    assert observation.batch.status == OutcomeStatus.FAILED


def test_live_requeue_cannot_mix_previous_accounting_exit(tmp_path):
    h = handle(tmp_path)
    receipt(h)
    observer = SlurmObservation(
        Runner(scontrol=controller(h, restart=1), sacct=account(h)), owner="alice"
    )
    assert observer.observe(h).status == ObservationStatus.ERROR
    result = observer.collect(h)
    assert result.runtime.status == result.stage_out.status == OutcomeStatus.UNKNOWN


@pytest.mark.parametrize(
    "rows",
    [lambda h: account(h) + account(h, restart=1), lambda h: account(h) + account(h)],
)
def test_ambiguous_accounting_executions_rejected(tmp_path, rows):
    h = handle(tmp_path)
    assert (
        SlurmObservation(Runner(sacct=rows(h)), owner="alice").observe(h).status
        == ObservationStatus.ERROR
    )


def test_current_controller_running_overrides_accounting_terminal(tmp_path):
    h = handle(tmp_path)
    observer = SlurmObservation(
        Runner(scontrol=controller(h), sacct=account(h)), owner="alice"
    )
    assert observer.observe(h).status == ObservationStatus.RUNNING


@pytest.mark.parametrize(
    "change",
    [
        {"owner": "bob"},
        {"cluster": "other"},
        {"comment": "foreign"},
        {"name": "foreign"},
    ],
)
def test_native_identity_mismatch_rejected(tmp_path, change):
    h = handle(tmp_path)
    assert (
        SlurmObservation(Runner(sacct=account(h, **change)), owner="alice")
        .observe(h)
        .status
        == ObservationStatus.ERROR
    )


@pytest.mark.parametrize(
    "change", [{"comment": "foreign"}, {"job_name": "foreign"}, {"cluster": "other"}]
)
def test_noncanonical_handle_rejected_before_query(tmp_path, change):
    h = handle(tmp_path, **change)
    runner = Runner()
    assert (
        SlurmObservation(runner, owner="alice").observe(h).status
        == ObservationStatus.ERROR
    )
    assert runner.calls == []


@pytest.mark.parametrize("raw_json", ["not-json", "[]", "null"])
def test_corrupted_handle_payload_is_observation_error_without_query(
    tmp_path, raw_json
):
    h = handle(tmp_path)
    # Simulate corruption after the frozen constructor's persistence validation.
    object.__setattr__(h, "raw_json", raw_json)
    runner = Runner()
    observation = SlurmObservation(runner, owner="alice").observe(h)
    assert observation.status == ObservationStatus.ERROR
    assert runner.calls == []


def test_bare_job_number_is_not_durable_identity(tmp_path):
    h = handle(tmp_path, job="42")
    assert (
        SlurmObservation(Runner(), owner="alice").observe(h).status
        == ObservationStatus.ERROR
    )


@pytest.mark.parametrize(
    "failure",
    [
        ProcessResult(1, b"", b"Access denied"),
        ProcessResult(0, b"", b"", timed_out=True),
        ProcessResult(0, b"", b"", stdout_truncated=True),
        BudgetExhausted("too many jobs"),
    ],
)
def test_query_failure_is_not_absence(tmp_path, failure):
    h = handle(tmp_path)
    assert (
        SlurmObservation(Runner(squeue=failure), owner="alice").observe(h).status
        == ObservationStatus.ERROR
    )


def test_accounting_outage_not_equivalent_to_disabled(tmp_path):
    h = handle(tmp_path)
    observer = SlurmObservation(
        Runner(sacct=ProcessResult(1, b"", b"Connection refused")), owner="alice"
    )
    assert observer.observe(h).status == ObservationStatus.ERROR
    disabled = SlurmObservation(
        Runner(sacct=ProcessResult(1, b"", b"Accounting storage is disabled")),
        owner="alice",
    )
    assert disabled.observe(h).status == ObservationStatus.ABSENT


def test_vanished_job_is_absent_not_failed(tmp_path):
    h = handle(tmp_path)
    assert (
        SlurmObservation(Runner(), owner="alice").observe(h).status
        == ObservationStatus.ABSENT
    )


@pytest.mark.parametrize(
    "accounting",
    ["", "42||||||||\n", "garbage", "42|alpha|alice|name| |0:0|0|time|comment\n"],
)
def test_malformed_and_missing_states_fail_closed(tmp_path, accounting):
    h = handle(tmp_path)
    if not accounting:
        accounting = account(h, state=" ")
    assert (
        SlurmObservation(Runner(sacct=accounting), owner="alice").observe(h).status
        == ObservationStatus.ERROR
    )


def test_queries_are_batched_and_full_job_ids_key_results(tmp_path):
    first, second = handle(tmp_path), handle(tmp_path, job="43;alpha")
    runner = Runner(sacct=account(first) + account(second))
    result = SlurmObservation(runner, owner="alice").observe_many((first, second))
    assert set(result) == {"42;alpha", "43;alpha"}
    assert len(runner.calls) == 3
    assert "42,43" in runner.calls[0]


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ("MinJobAge = 120\nAccountingStorageType = accounting_storage/none", False),
        ("MinJobAge = 120\nAccountingStorageType = accounting_storage/slurmdbd", True),
        ("MinJobAge = 120", None),
    ],
)
def test_capabilities_distinguish_empty_probe_from_disabled(tmp_path, config, expected):
    capability = SlurmObservation(Runner(scontrol=config), owner="alice").capabilities()
    assert capability.accounting is expected
    assert capability.min_job_age == 120


def test_independent_receipt_outcomes_and_scheduler_epilogue_failure(tmp_path):
    h = handle(tmp_path)
    receipt(h, stage_out_exit=7)
    observer = SlurmObservation(
        Runner(sacct=account(h, "FAILED", "2:0")), owner="alice"
    )
    result = observer.collect(h)
    assert result.runtime.status == OutcomeStatus.SUCCESS
    assert result.batch.exit_code == 0
    assert result.stage_out.exit_code == 7
    assert observer.observe(h).batch.exit_code == 2


def test_native_failure_stays_independent_of_all_successful_receipt_commands(tmp_path):
    h = handle(tmp_path)
    receipt(h)
    observer = SlurmObservation(
        Runner(sacct=account(h, "FAILED", "2:0")), owner="alice"
    )
    result = observer.collect(h)
    assert all(
        component.status == OutcomeStatus.SUCCESS
        for component in (result.runtime, result.batch, result.stage_out)
    )
    assert observer.observe(h).batch.status == OutcomeStatus.FAILED


@pytest.mark.parametrize("member", ["run", "attempts", "attempt"])
def test_receipt_symlink_ancestry_is_never_followed(tmp_path, member):
    h = handle(tmp_path / "owned")
    receipt(h)
    root = Path(h.run_dir)
    path = {
        "run": root,
        "attempts": root / "attempts",
        "attempt": root / "attempts" / "1",
    }[member]
    target = tmp_path / f"foreign-{member}"
    path.rename(target)
    path.symlink_to(target, target_is_directory=True)
    outcome = SlurmObservation(Runner(sacct=account(h)), owner="alice").collect(h)
    assert outcome.runtime.status == OutcomeStatus.UNKNOWN
    assert str(tmp_path) not in outcome.runtime.detail


@pytest.mark.parametrize("member", ["run", "attempts", "attempt", "receipt"])
def test_group_writable_receipt_ancestry_is_refused(tmp_path, member):
    h = handle(tmp_path)
    file = receipt(h)
    root = Path(h.run_dir)
    path = {
        "run": root,
        "attempts": root / "attempts",
        "attempt": file.parent,
        "receipt": file,
    }[member]
    observer = SlurmObservation(Runner(sacct=account(h)), owner="alice")
    assert observer.collect(h).runtime.status == OutcomeStatus.SUCCESS

    def verify_refused():
        started = time.monotonic()
        outcome = observer.collect(h)
        assert time.monotonic() - started < 5
        assert outcome.runtime.status == OutcomeStatus.UNKNOWN
        assert str(tmp_path) not in outcome.runtime.detail

    if os.name == "nt":
        # chmod cannot express Windows group access. Mutate the actual DACL on
        # this exact owned fixture member; SID spelling is locale-independent.
        with windows_foreign_write_acl(path):
            verify_refused()
        # SUCCESS requires all held ancestry/file handles to pass native DACL
        # ownership validation again after the original DACL is restored.
        assert observer.collect(h).runtime.status == OutcomeStatus.SUCCESS
    else:
        original_mode = path.stat().st_mode
        try:
            path.chmod(original_mode | 0o020)
            verify_refused()
        finally:
            path.chmod(original_mode)
        assert observer.collect(h).runtime.status == OutcomeStatus.SUCCESS


@pytest.mark.parametrize("nested", [False, True])
def test_duplicate_receipt_fields_never_establish_outcome(tmp_path, nested):
    h = handle(tmp_path)
    path = receipt(h)
    content = path.read_text()
    if nested:
        content = content.replace(
            '"exit_code": 0', '"exit_code": 91, "exit_code": 0', 1
        )
    else:
        content = content.rstrip()[:-1] + f', "run_id": "{h.run_id}"}}'
    path.write_text(content)
    assert (
        SlurmObservation(Runner(sacct=account(h)), owner="alice")
        .collect(h)
        .runtime.status
        == OutcomeStatus.UNKNOWN
    )


def test_wrong_current_user_receipt_is_refused(tmp_path, monkeypatch):
    from wavedl.core.safe_files import SafeFile, UnsafeOwnership

    h = handle(tmp_path)
    receipt(h)

    def wrong_owner(self):
        raise UnsafeOwnership("Receipt does not belong to current user")

    monkeypatch.setattr(SafeFile, "verify_private_owner", wrong_owner)
    assert (
        SlurmObservation(Runner(sacct=account(h)), owner="alice")
        .collect(h)
        .runtime.status
        == OutcomeStatus.UNKNOWN
    )


def test_scheduler_wrapper_zero_does_not_prove_training_success(tmp_path):
    h = handle(tmp_path)
    result = SlurmObservation(Runner(sacct=account(h)), owner="alice").collect(h)
    assert result.batch.status == OutcomeStatus.SUCCESS
    assert result.runtime.status == OutcomeStatus.UNKNOWN


@pytest.mark.parametrize(
    "mutation", ["identity", "oversize", "bool_exit", "bad_signal", "running"]
)
def test_invalid_or_stale_receipt_not_collected(tmp_path, mutation):
    h = handle(tmp_path)
    path = receipt(h)
    data = json.loads(path.read_bytes())
    if mutation == "identity":
        data["attempt"] = 2
    elif mutation == "bool_exit":
        data["runtime"]["exit_code"] = True
    elif mutation == "bad_signal":
        data["runtime"]["signal"] = 15
    path.write_text("x" * 16385 if mutation == "oversize" else json.dumps(data))
    observer = SlurmObservation(
        Runner(sacct=account(h, "RUNNING" if mutation == "running" else "COMPLETED")),
        owner="alice",
    )
    assert observer.collect(h).runtime.status == OutcomeStatus.UNKNOWN


def test_wrong_owner_and_invalid_owner(tmp_path):
    h = replace(handle(tmp_path), owner="bob")
    assert (
        SlurmObservation(Runner(), owner="alice").observe(h).status
        == ObservationStatus.ERROR
    )
    with pytest.raises(ValueError):
        SlurmObservation(Runner(), owner="--allusers")


def test_live_identity_can_verify_accounting_without_stored_comments(tmp_path):
    h = handle(tmp_path)
    runner = Runner(
        scontrol=controller(h, state="COMPLETED"), sacct=account(h, comment="")
    )
    assert (
        SlurmObservation(runner, owner="alice").observe(h).status
        == ObservationStatus.TERMINAL
    )
    assert (
        SlurmObservation(Runner(sacct=account(h, comment="")), owner="alice")
        .observe(h)
        .status
        == ObservationStatus.ERROR
    )


def test_held_pending_reason_and_queue_comment_identity(tmp_path):
    h = handle(tmp_path)
    row = f"42|alice|wavedl-{h.request_id}|PENDING|JobHeldUser|{h.request_id}\n"
    assert (
        SlurmObservation(Runner(squeue=row), owner="alice").observe(h).status
        == ObservationStatus.HELD
    )
    assert (
        SlurmObservation(
            Runner(squeue=row.replace(h.request_id, "foreign")), owner="alice"
        )
        .observe(h)
        .status
        == ObservationStatus.ERROR
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"JobState": ""},
        {"Restarts": "invalid"},
        {"Comment": "foreign"},
        {"ExitCode": "invalid", "JobState": "COMPLETED"},
    ],
)
def test_controller_malformed_identity_and_fields_fail_closed(tmp_path, changes):
    h = handle(tmp_path)
    assert (
        SlurmObservation(Runner(scontrol=controller(h, **changes)), owner="alice")
        .observe(h)
        .status
        == ObservationStatus.ERROR
    )


def test_symlink_receipt_not_trusted(tmp_path):
    h = handle(tmp_path)
    original = receipt(h)
    target = tmp_path / "foreign-receipt.json"
    original.rename(target)
    original.symlink_to(target)
    assert (
        SlurmObservation(Runner(sacct=account(h)), owner="alice")
        .collect(h)
        .runtime.status
        == OutcomeStatus.UNKNOWN
    )


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX filesystem FIFO; Windows native pipe rejection tested separately",
)
def test_fifo_receipt_refused_without_blocking(tmp_path):
    h = handle(tmp_path)
    path = Path(h.run_dir) / "attempts" / "1" / "slurm-outcome.json"
    path.parent.mkdir(parents=True)
    os.mkfifo(path)
    assert (
        SlurmObservation(Runner(sacct=account(h)), owner="alice")
        .collect(h)
        .runtime.status
        == OutcomeStatus.UNKNOWN
    )


def test_directory_receipt_is_not_a_valid_outcome(tmp_path):
    """Native filesystem special objects cannot establish scheduler success."""
    h = handle(tmp_path)
    path = Path(h.run_dir) / "attempts" / "1" / "slurm-outcome.json"
    path.mkdir(parents=True)
    assert (
        SlurmObservation(Runner(sacct=account(h)), owner="alice")
        .collect(h)
        .runtime.status
        == OutcomeStatus.UNKNOWN
    )


@pytest.mark.skipif(os.name != "nt", reason="Actual Win32 named pipe primitive")
def test_native_windows_pipe_is_rejected_before_reading():
    """Windows pipes live outside filesystem receipts; native reader fails closed."""
    # Isolate the native operation behind a wall-clock timeout: rejection must
    # not wait for pipe data, a writer or a connection handshake.
    script = r"""
import ctypes
from ctypes import wintypes
from pathlib import Path
from uuid import uuid4
from wavedl.core.safe_files import _windows_api
kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.CreateNamedPipeW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD,
    wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
    wintypes.DWORD, wintypes.LPVOID]
kernel.CreateNamedPipeW.restype = wintypes.HANDLE
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
kernel.CloseHandle.restype = wintypes.BOOL
name = "\\\\.\\pipe\\wavedl-receipt-" + str(uuid4())
pipe = kernel.CreateNamedPipeW(name, 3, 1, 1, 4096, 4096, 0, None)
if pipe == ctypes.c_void_p(-1).value:
    raise ctypes.WinError(ctypes.get_last_error())
try:
    try:
        _windows_api().open(Path(name))
    except ValueError as error:
        assert "disk" in str(error)
    else:
        raise AssertionError("Native pipe was accepted as a disk file")
finally:
    if not kernel.CloseHandle(pipe):
        raise ctypes.WinError(ctypes.get_last_error())
print("native-pipe-refused")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.stdout.strip() == "native-pipe-refused"


def test_capability_query_failure_remains_unknown_with_diagnostics():
    result = SlurmObservation(
        Runner(scontrol=ProcessResult(1, b"", b"Access denied")), owner="alice"
    ).capabilities()
    assert result.accounting is None and result.min_job_age is None
    assert "Access denied" in result.diagnostics[0]


def test_per_handle_requeue_error_preserves_valid_peer(tmp_path):
    first = handle(tmp_path)
    second = handle(tmp_path, job="43;alpha")
    runner = Runner(scontrol=controller(first, restart=1) + controller(second))
    result = SlurmObservation(runner, owner="alice").observe_many((first, second))
    assert result[first.job_id].status == ObservationStatus.ERROR
    assert result[second.job_id].status == ObservationStatus.RUNNING


def test_controller_truncation_does_not_hide_valid_accounting(tmp_path):
    h = handle(tmp_path)
    runner = Runner(
        scontrol=ProcessResult(0, b"partial", b"", stdout_truncated=True),
        sacct=account(h),
    )
    assert (
        SlurmObservation(runner, owner="alice").observe(h).status
        == ObservationStatus.TERMINAL
    )
    runner.responses["sacct"] = ""
    assert (
        SlurmObservation(runner, owner="alice").observe(h).status
        == ObservationStatus.ERROR
    )


def test_empty_controller_literal_is_genuine_absence(tmp_path):
    h = handle(tmp_path)
    assert (
        SlurmObservation(Runner(scontrol="No jobs in the system\n"), owner="alice")
        .observe(h)
        .status
        == ObservationStatus.ABSENT
    )


def test_runtime_query_failure_is_normalized_to_unavailable(tmp_path):
    h = handle(tmp_path)
    assert (
        SlurmObservation(
            Runner(squeue=RuntimeError("scheduler timeout")), owner="alice"
        )
        .observe(h)
        .status
        == ObservationStatus.ERROR
    )


def test_current_accounting_generation_ignores_old_restart_rows(tmp_path):
    h = handle(tmp_path, restart_count=1)
    rows = account(h, "FAILED", "9:0", restart=0) + account(h, "COMPLETED", restart=1)
    result = SlurmObservation(Runner(sacct=rows), owner="alice").observe(h)
    assert result.status == ObservationStatus.TERMINAL and result.exit_code == 0


def test_only_old_accounting_generation_is_not_current_terminal(tmp_path):
    h = handle(tmp_path, restart_count=1)
    result = SlurmObservation(Runner(sacct=account(h)), owner="alice").observe(h)
    assert result.status == ObservationStatus.ABSENT


def test_restart_discovery_requires_native_before_binding(tmp_path, monkeypatch):
    import wavedl.core.scheduler_attempt as module

    h = handle(tmp_path)

    def never(*args, **kwargs):
        pytest.fail("Binding reader must not run without a new native restart")

    monkeypatch.setattr(module, "read_scheduler_binding", never, raising=False)
    observer = SlurmObservation(Runner(scontrol=controller(h)), owner="alice")
    assert observer.discover_restart(h) is None
    observer.runner.responses["scontrol"] = ""
    with pytest.raises(ValueError, match="absent"):
        observer.discover_restart(h)


@pytest.mark.parametrize("alias", ["0", "1", "bad", ""])
def test_restart_aliases_fail_closed_for_observation_and_discovery(tmp_path, alias):
    h = handle(tmp_path)
    observer = SlurmObservation(
        Runner(scontrol=controller(h, RestartCnt=alias)), owner="alice"
    )
    if alias == "0":
        assert observer.observe(h).status == ObservationStatus.RUNNING
        assert observer.discover_restart(h) is None
    else:
        assert observer.observe(h).status == ObservationStatus.ERROR
        with pytest.raises(ValueError, match="restart"):
            observer.discover_restart(h)


def test_restart_discovery_proposes_only_matching_complete_binding(
    tmp_path, monkeypatch
):
    import wavedl.core.scheduler_attempt as module

    h = handle(tmp_path)
    seen = []

    def read(path, identity, *, initial_attempt):
        seen.append((path, identity, initial_attempt))
        return SchedulerBinding(
            identity.run_id,
            identity.request_id,
            identity.cluster,
            identity.job_id,
            identity.owner,
            identity.restart_count,
            3,
        )

    monkeypatch.setattr(module, "read_scheduler_binding", read, raising=False)
    runner = Runner(scontrol=controller(h, restart=2))
    binding, candidate = SlurmObservation(runner, owner="alice").discover_restart(h)
    assert binding.attempt == candidate.attempt == 3
    assert json.loads(candidate.raw_json)["restart_count"] == 2
    assert json.loads(candidate.raw_json)["initial_attempt"] == 1
    assert h.attempt == 1 and json.loads(h.raw_json)["restart_count"] == 0
    assert seen[0][0] == tmp_path and seen[0][2] == 1
    assert runner.calls == [
        ("scontrol", "--clusters", "alpha", "--oneliner", "show", "job", "42")
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"UserId": "bob(999)"},
        {"Comment": "foreign"},
        {"JobName": "foreign"},
        {"JobId": "43"},
        {"Cluster": "other"},
        {"Restarts": "invalid"},
        {"JobState": ""},
    ],
)
def test_restart_discovery_rejects_unverified_native(tmp_path, changes, monkeypatch):
    import wavedl.core.scheduler_attempt as module

    h = handle(tmp_path)
    monkeypatch.setattr(
        module,
        "read_scheduler_binding",
        lambda *args, **kwargs: pytest.fail("Unverified native must not read binding"),
        raising=False,
    )
    with pytest.raises(ValueError):
        SlurmObservation(
            Runner(scontrol=controller(h, restart=1, **changes)), owner="alice"
        ).discover_restart(h)


def test_restart_discovery_incomplete_claim_never_adopts(tmp_path, monkeypatch):
    import wavedl.core.scheduler_attempt as module

    h = handle(tmp_path)

    def missing(*args, **kwargs):
        raise FileNotFoundError("incomplete scheduler claim")

    monkeypatch.setattr(module, "read_scheduler_binding", missing, raising=False)
    with pytest.raises(
        RestartBindingUnavailable, match="binding unavailable"
    ) as caught:
        SlurmObservation(
            Runner(scontrol=controller(h, restart=1)), owner="alice"
        ).discover_restart(h)
    assert caught.value.native_restart_count == 1


@pytest.mark.parametrize("invalid", [True, 0, -1, "1", 2**63])
def test_unavailable_restart_exception_requires_validated_generation(invalid):
    with pytest.raises(ValueError):
        RestartBindingUnavailable(invalid)


@pytest.mark.parametrize(
    "native", ["", ProcessResult(1, b"", b"controller unavailable")]
)
def test_absent_native_or_controller_outage_is_not_confirmed_restart(tmp_path, native):
    h = handle(tmp_path)
    with pytest.raises(ValueError) as caught:
        SlurmObservation(Runner(scontrol=native), owner="alice").discover_restart(h)
    assert not isinstance(caught.value, RestartBindingUnavailable)


@pytest.mark.parametrize("first_value", ["foreign", ""])
@pytest.mark.parametrize("key", ["JobId", "UserId", "Comment", "Restarts", "JobState"])
def test_duplicate_controller_authority_fields_never_last_value_wins(
    tmp_path, monkeypatch, key, first_value
):
    import wavedl.core.scheduler_attempt as module

    h = handle(tmp_path)
    output = f"{key}={first_value} " + controller(h, restart=1)
    monkeypatch.setattr(
        module,
        "read_scheduler_binding",
        lambda *args, **kwargs: pytest.fail(
            "Ambiguous native identity must not read claims"
        ),
    )
    observer = SlurmObservation(Runner(scontrol=output), owner="alice")
    assert observer.observe(h).status == ObservationStatus.ERROR
    with pytest.raises(ValueError, match="Duplicate native") as caught:
        observer.discover_restart(h)
    assert not isinstance(caught.value, RestartBindingUnavailable)


@pytest.mark.parametrize("key", ["Comment", "JobName"])
def test_quoted_native_identity_does_not_gain_authority(tmp_path, key):
    h = handle(tmp_path)
    correct = h.request_id if key == "Comment" else f"wavedl-{h.request_id}"
    output = controller(h, restart=1, **{key: f'"{correct}"'})
    observer = SlurmObservation(Runner(scontrol=output), owner="alice")
    assert observer.observe(h).status == ObservationStatus.ERROR
    with pytest.raises(ValueError, match="identity mismatch"):
        observer.discover_restart(h)


def test_native_field_name_suffix_does_not_gain_identity_authority(tmp_path):
    h = handle(tmp_path)
    output = controller(h, restart=1).replace("JobId=", "Other-JobId=")
    observer = SlurmObservation(Runner(scontrol=output), owner="alice")
    assert observer.observe(h).status == ObservationStatus.ERROR
    with pytest.raises(ValueError, match="Malformed native"):
        observer.discover_restart(h)


@pytest.mark.skipif(
    os.name != "posix", reason="Scheduler binding foundation requires POSIX"
)
def test_restart_discovery_reads_real_private_binding_and_attempt(tmp_path):
    h = handle(tmp_path)
    layout = RunLayout.create(tmp_path, h.run_id)
    h = replace(h, run_dir=str(layout.run_dir))
    write_manifest(
        layout,
        build_manifest(
            layout,
            operation=OperationKind.TRAIN,
            resolved_config={"epochs": 1},
            runtime=RuntimeSelection("/usr/bin/python3", "selected", "cpu", {}),
            submission_cwd=tmp_path,
        ),
    )
    assert allocate_attempt(layout) == 1
    (layout.attempt_dir(1) / "submission.json").write_text(
        json.dumps(
            {
                "request_id": h.request_id,
                "attempt": 1,
                "owner": h.owner,
                "created_at": "2026-10-04T00:00:00Z",
            }
        )
    )
    identity = SchedulerIdentity(h.run_id, h.request_id, "alpha", "42", "alice", 1)
    expected = coordinate_attempt(layout.run_dir, identity, initial_attempt=1)
    observer = SlurmObservation(
        Runner(scontrol=controller(h, restart=1)), owner="alice"
    )
    binding, candidate = observer.discover_restart(h)
    assert binding == expected and candidate.attempt == 2
    corrupted = asdict(expected) | {"request_id": str(uuid4())}
    (layout.attempt_dir(2) / "scheduler_identity.json").write_text(
        json.dumps(corrupted)
    )
    with pytest.raises(ValueError, match="binding unavailable"):
        observer.discover_restart(h)
