"""Submission command outcomes are bounded evidence, never runtime completion."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace

import pytest

from wavedl.core import scheduler_attempt, submission, submission_outcomes
from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.manifest import RuntimeSelection, build_manifest, write_manifest
from wavedl.core.outcomes import MAX_RECEIPT_BYTES, command_outcome
from wavedl.core.scheduler_attempt import SchedulerIdentity, coordinate_attempt
from wavedl.core.submission import create_submission, identity_for_submission
from wavedl.core.submission_outcomes import (
    read_submission_outcome_receipt,
    submission_outcome_name,
    write_submission_outcome_receipt,
)


RUN = "00000000-0000-4000-8000-000000000001"
REQUEST = "00000000-0000-4000-8000-000000000002"
PLAN = "a" * 64


@pytest.fixture
def prepared(tmp_path):
    layout = RunLayout.create(tmp_path / "runs", RUN)
    write_manifest(
        layout,
        build_manifest(
            layout,
            operation=OperationKind.TRAIN,
            resolved_config={},
            runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
            submission_cwd=tmp_path,
        ),
    )
    request = identity_for_submission(
        layout.run_dir,
        request_id=REQUEST,
        backend="slurm",
        owner="alice",
        plan_sha256=PLAN,
    )
    create_submission(layout.run_dir, request)
    identity = SchedulerIdentity(RUN, REQUEST, "cluster-a", "12345", "alice", 0)
    return layout, request, identity


def write(prepared, **changes):
    layout, _request, identity = prepared
    return write_submission_outcome_receipt(
        layout.run_dir,
        **(
            {key: value for key, value in asdict(identity).items() if key != "run_id"}
            | {
                "plan_sha256": PLAN,
                "operation": "train",
                "runtime_exit": 125,
                "batch_exit": 1,
                "stage_out_exit": 125,
            }
            | changes
        ),
    )


def read(prepared, **changes):
    layout, _request, identity = prepared
    return read_submission_outcome_receipt(
        layout.run_dir, identity, **({"plan_sha256": PLAN} | changes)
    )


def test_setup_failure_has_full_provenance_without_attempt_or_journal(prepared):
    layout, request, identity = prepared
    before = layout.manifest.read_bytes()
    path = write(prepared)
    namespace = layout.run_dir / "submissions" / REQUEST
    expected_name = submission_outcome_name(identity)
    assert path.name == expected_name
    assert path.samefile(namespace / expected_name)
    assert path.parent.samefile(namespace)
    assert {member.name for member in namespace.parent.iterdir()} == {REQUEST}
    assert {member.name for member in namespace.iterdir()} == {
        "request.json",
        expected_name,
    }
    receipt = read(prepared)
    assert receipt == {
        "schema_version": 2,
        **asdict(request),
        "cluster": identity.cluster,
        "job_id": identity.job_id,
        "restart_count": 0,
        "operation": "train",
        "attempt": None,
        "runtime": command_outcome(125),
        "batch": command_outcome(1),
        "stage_out": command_outcome(125),
    }
    assert list(layout.attempts.iterdir()) == []
    assert not list(layout.run_dir.rglob("events.jsonl"))
    assert layout.manifest.read_bytes() == before
    with pytest.raises(FileExistsError):
        write(prepared, batch_exit=0)
    assert read(prepared) == receipt


def test_concurrent_publication_has_one_immutable_winner(prepared):
    def publish(_index):
        try:
            write(prepared)
            return True
        except FileExistsError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(publish, range(16))) == 1
    assert read(prepared)["batch"] == command_outcome(1)


@pytest.mark.parametrize("code", [0, 1, 124, 126, 255])
def test_no_runtime_execution_can_be_claimed_without_binding(prepared, code):
    with pytest.raises(ValueError, match="Unbound"):
        write(prepared, runtime_exit=code)
    assert list(prepared[0].attempts.iterdir()) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("attempt", True),
        ("attempt", 0),
        ("attempt", 2**63),
        ("runtime_exit", True),
        ("batch_exit", -1),
        ("stage_out_exit", 256),
        ("restart_count", True),
        ("job_id", "001"),
        ("cluster", "bad cluster"),
        ("owner", "bob"),
        ("plan_sha256", None),
        ("plan_sha256", "b" * 64),
        ("operation", "inference"),
    ],
)
def test_invalid_write_refused_without_publication(prepared, field, value):
    with pytest.raises((ValueError, FileNotFoundError)):
        write(prepared, **{field: value})
    namespace = prepared[0].run_dir / "submissions" / REQUEST
    assert not list(namespace.glob("slurm-outcome-*"))


@pytest.mark.skipif(os.name != "posix", reason="Scheduler proof requires POSIX")
def test_positive_attempt_needs_both_claim_and_attempt_binding(prepared, monkeypatch):
    layout, _request, identity = prepared
    with pytest.raises(FileNotFoundError):
        write(prepared, attempt=1, runtime_exit=0)
    binding = coordinate_attempt(layout.run_dir, identity, initial_attempt=None)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Outcome publication reopened the held run directory")

    monkeypatch.setattr(scheduler_attempt, "open_directory", forbidden)
    monkeypatch.setattr(submission, "open_directory", forbidden)
    with pytest.raises(ValueError, match="binding"):
        write(prepared, attempt=binding.attempt + 1, runtime_exit=0)
    path = write(prepared, attempt=binding.attempt, runtime_exit=0, batch_exit=0)
    assert read(prepared, attempt=binding.attempt)["runtime"] == command_outcome(0)
    with pytest.raises(ValueError):
        read(prepared)
    attempt_proof = layout.attempt_dir(binding.attempt) / "scheduler_identity.json"
    attempt_proof.unlink()
    with pytest.raises(FileNotFoundError):
        read(prepared, attempt=binding.attempt)
    assert path.exists()


@pytest.mark.parametrize("field", ["job_id", "cluster", "restart_count"])
def test_native_scheduler_identity_selects_distinct_receipt(prepared, field):
    layout, _request, identity = prepared
    write(prepared)
    changed = replace(
        identity,
        **{
            field: {"job_id": "12346", "cluster": "cluster-b", "restart_count": 1}[
                field
            ]
        },
    )
    assert submission_outcome_name(changed) != submission_outcome_name(identity)
    with pytest.raises(FileNotFoundError):
        read_submission_outcome_receipt(layout.run_dir, changed, plan_sha256=PLAN)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("schema_version", 1),
        ("restart_count", False),
        ("attempt", 1),
        ("job_id", "12346"),
        ("owner", "bob"),
        ("backend", "local"),
        ("manifest_sha256", "b" * 64),
        ("plan_sha256", "b" * 64),
        ("operation", "test"),
        ("runtime", command_outcome(0)),
        ("batch", command_outcome(0) | {"status": "failed"}),
        ("stage_out", command_outcome(125) | {"extra": 1}),
    ],
)
def test_mutated_receipt_is_never_accepted(prepared, field, value):
    path = write(prepared)
    receipt = json.loads(path.read_bytes())
    receipt[field] = value
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        read(prepared)


def test_duplicate_nested_and_top_level_keys_nonfinite_and_oversize(prepared):
    path = write(prepared)
    original = path.read_bytes()
    for raw in (
        original.replace(
            b'"schema_version": 2', b'"schema_version": 2, "schema_version": 2'
        ),
        original.replace(b'"signal": null', b'"signal": null, "signal": null', 1),
        original.replace(b'"signal": null', b'"signal": NaN', 1),
        original + b" " * MAX_RECEIPT_BYTES,
        b"[]",
    ):
        path.write_bytes(raw)
        with pytest.raises(ValueError):
            read(prepared)


def test_manifest_and_plan_authority_are_exact(prepared):
    write(prepared)
    with pytest.raises(ValueError):
        read(prepared, plan_sha256="b" * 64)
    layout = prepared[0]
    layout.manifest.write_bytes(layout.manifest.read_bytes() + b" ")
    with pytest.raises(ValueError):
        read(prepared)


@pytest.mark.skipif(os.name != "posix", reason="Native POSIX ancestry proof")
def test_symlink_and_private_file_requirements(prepared, tmp_path):
    path = write(prepared)
    target = tmp_path / "receipt"
    target.write_bytes(path.read_bytes())
    target.chmod(0o600)
    path.unlink()
    path.symlink_to(target)
    with pytest.raises((OSError, ValueError)):
        read(prepared)
    path.unlink()
    path.write_bytes(target.read_bytes())
    path.chmod(0o666)
    with pytest.raises(ValueError):
        read(prepared)


def test_actual_cli_records_setup_failure_without_runtime_side_effects(prepared):
    layout, _request, identity = prepared
    command = [
        sys.executable,
        "-m",
        "wavedl.core.submission_outcomes",
        str(layout.run_dir),
    ]
    for name, value in asdict(identity).items():
        command.extend(["--" + name.replace("_", "-"), str(value)])
    command.extend(
        [
            "--plan-sha256",
            PLAN,
            "--operation",
            "train",
            "--runtime-exit",
            "125",
            "--batch-exit",
            "1",
            "--stage-out-exit",
            "125",
        ]
    )
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert read(prepared)["attempt"] is None
    assert read(prepared)["runtime"]["status"] == "unknown"
    assert list(layout.attempts.iterdir()) == []
    assert not list(layout.run_dir.rglob("events.jsonl"))
    repeated = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert repeated.returncode == 2
    assert "FileExistsError" in repeated.stderr


def test_prestart_outcome_never_reads_scheduler_or_reopens_root(prepared, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Prestart outcome tried scheduler allocation or root reopen")

    monkeypatch.setattr(
        submission_outcomes, "read_scheduler_binding_in_directory", forbidden
    )
    monkeypatch.setattr(submission, "open_directory", forbidden)
    write(prepared)
    assert read(prepared)["attempt"] is None


def test_publication_byte_bound_prevents_any_receipt(prepared, monkeypatch):
    monkeypatch.setattr(submission_outcomes, "MAX_RECEIPT_BYTES", 10)
    with pytest.raises(ValueError, match="byte bound"):
        write(prepared)
    assert not list(
        (prepared[0].run_dir / "submissions" / REQUEST).glob("slurm-outcome-*")
    )


@pytest.mark.parametrize("change", ["extra", "missing"])
def test_receipt_field_set_is_exact(prepared, change):
    path = write(prepared)
    receipt = json.loads(path.read_bytes())
    if change == "extra":
        receipt["completion"] = True
    else:
        del receipt["run_directory"]
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="fields"):
        read(prepared)


@pytest.mark.skipif(os.name != "posix", reason="Scheduler proof requires POSIX")
@pytest.mark.parametrize(
    "field", ["owner", "job_id", "cluster", "restart_count", "attempt"]
)
def test_positive_attempt_rejects_foreign_claim_proof(prepared, field):
    layout, _request, identity = prepared
    binding = coordinate_attempt(layout.run_dir, identity, initial_attempt=None)
    claim = (
        layout.run_dir
        / "submissions"
        / REQUEST
        / "launchers"
        / identity.claim_name
        / "binding.json"
    )
    receipt = json.loads(claim.read_bytes())
    receipt[field] = {
        "owner": "bob",
        "job_id": "12346",
        "cluster": "cluster-b",
        "restart_count": 1,
        "attempt": binding.attempt + 1,
    }[field]
    claim.write_text(json.dumps(receipt))
    with pytest.raises((ValueError, FileNotFoundError)):
        write(prepared, attempt=binding.attempt, runtime_exit=0)


@pytest.mark.skipif(os.name != "posix", reason="Native POSIX ancestry-swap proof")
def test_outcome_ancestry_swap_cannot_publish_into_foreign_run(prepared, monkeypatch):
    layout, _request, _identity = prepared
    moved = layout.run_dir.parent / "moved"
    original = submission_outcomes.publish_submission_member

    def swapped(selected, name, content, **kwargs):
        layout.run_dir.rename(moved)
        foreign = layout.run_dir / "submissions" / REQUEST
        foreign.mkdir(parents=True, mode=0o700)
        return original(selected, name, content, **kwargs)

    monkeypatch.setattr(submission_outcomes, "publish_submission_member", swapped)
    with pytest.raises(ValueError):
        write(prepared)
    assert not list(layout.run_dir.rglob("slurm-outcome-*"))
    assert not list(moved.rglob("slurm-outcome-*"))


@pytest.mark.skipif(os.name != "nt", reason="Native Windows held-write capability")
def test_windows_setup_failure_receipt_uses_held_root(prepared):
    write(prepared)
    assert read(prepared)["attempt"] is None


def test_outcome_writer_retries_root_sharing_only_before_publication(
    prepared, monkeypatch
):
    from types import SimpleNamespace

    from wavedl.core import submission, submission_outcomes

    original = submission_outcomes.open_directory
    calls = []

    def contend(path, *, writable=False):
        calls.append(path)
        if len(calls) == 1:
            assert not list(path.rglob("slurm-outcome-*"))
            error = PermissionError(13, "owned Windows sharing")
            error.winerror = 32
            raise error
        return original(path, writable=writable)

    monkeypatch.setattr(
        submission, "os", SimpleNamespace(**(vars(os) | {"name": "nt"}))
    )
    monkeypatch.setattr(submission_outcomes, "open_directory", contend)
    write(prepared)
    assert len(calls) == 2
    assert read(prepared)["attempt"] is None
