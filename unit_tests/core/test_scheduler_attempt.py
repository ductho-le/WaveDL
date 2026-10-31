"""Real process coordination and crash-boundary scheduler identity tests."""

import json
import multiprocessing
import os
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from wavedl.core.layout import OperationKind, RunLayout, allocate_attempt
from wavedl.core.manifest import RuntimeSelection, build_manifest, write_manifest
from wavedl.core.safe_files import open_directory
from wavedl.core.scheduler_attempt import (
    IncompleteSchedulerClaim,
    SchedulerIdentity,
    coordinate_attempt,
    main,
    read_scheduler_binding,
    scheduler_environment_identity,
)


RUN = "00000000-0000-4000-8000-000000000001"
REQUEST = "00000000-0000-4000-8000-000000000002"
pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Scheduler startup binding deliberately requires POSIX"
)


@pytest.fixture
def layout(tmp_path):
    layout = RunLayout.create(tmp_path, RUN)
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
                "request_id": REQUEST,
                "attempt": 1,
                "owner": "alice",
                "created_at": "2026-10-04T00:00:00Z",
            }
        )
    )
    return layout


def identity(restart=0):
    return SchedulerIdentity(RUN, REQUEST, "cluster-a", "12345", "alice", restart)


def test_read_only_binding_proof_has_no_allocation_or_publication(layout, monkeypatch):
    binding = coordinate_attempt(layout.run_dir, identity(1), initial_attempt=1)
    import wavedl.core.scheduler_attempt as startup

    def forbidden(*args, **kwargs):
        pytest.fail("Read-only proof must never allocate or publish")

    monkeypatch.setattr(startup, "allocate_attempt", forbidden)
    monkeypatch.setattr(startup, "_publish", forbidden)
    monkeypatch.setattr(startup, "_mkdir", forbidden)
    assert (
        read_scheduler_binding(layout.run_dir, identity(1), initial_attempt=1)
        == binding
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        read_scheduler_binding(
            layout.run_dir, replace(identity(1), job_id="12346"), initial_attempt=1
        )


def test_scheduler_environment_gate_does_not_change_standalone():
    assert (
        scheduler_environment_identity(
            {"SLURM_JOB_ID": "12345", "SLURM_RESTART_COUNT": "8"}
        )
        is None
    )
    with pytest.raises(ValueError):
        scheduler_environment_identity({"WAVEDL_SCHEDULER_BINDING": "true"})


@pytest.fixture
def writer_layout(tmp_path):
    from wavedl.core.submission import create_submission, identity_for_submission

    selected = RunLayout.create(tmp_path, RUN)
    write_manifest(
        selected,
        build_manifest(
            selected,
            operation=OperationKind.TRAIN,
            resolved_config={"epochs": 1},
            runtime=RuntimeSelection("/usr/bin/python3", "selected", "cpu", {}),
            submission_cwd=tmp_path,
        ),
    )
    receipt = identity_for_submission(
        selected.run_dir,
        request_id=REQUEST,
        backend="slurm",
        owner="alice",
        plan_sha256="a" * 64,
    )
    create_submission(selected.run_dir, receipt)
    return selected


def writer_environment(**changes):
    return {
        "WAVEDL_SCHEDULER_BINDING": "1",
        "WAVEDL_ATTEMPT_ALLOCATION": "writer",
        "WAVEDL_RUN_ID": RUN,
        "WAVEDL_REQUEST_ID": REQUEST,
        "WAVEDL_SCHEDULER_CLUSTER": "cluster-a",
        "WAVEDL_SCHEDULER_OWNER": "alice",
        "SLURM_JOB_ID": "12345",
        "SLURM_RESTART_COUNT": "0",
    } | changes


def test_writer_startup_creates_missing_attempts_parent_after_authority(writer_layout):
    writer_layout.attempts.rmdir()
    binding = coordinate_attempt(
        writer_layout.run_dir, identity(), initial_attempt=None
    )
    assert binding.attempt == 1
    assert writer_layout.attempts.stat().st_mode & 0o777 == 0o700
    assert [path.name for path in writer_layout.attempts.iterdir()] == ["1"]
    assert (
        read_scheduler_binding(writer_layout.run_dir, identity(), initial_attempt=None)
        == binding
    )


def test_readonly_binding_never_creates_missing_attempts_parent(writer_layout):
    writer_layout.attempts.rmdir()
    with pytest.raises(FileNotFoundError):
        read_scheduler_binding(writer_layout.run_dir, identity(), initial_attempt=None)
    assert not writer_layout.attempts.exists()
    assert not (writer_layout.run_dir / "submissions" / REQUEST / "launchers").exists()


@pytest.mark.parametrize("corruption", ["owner", "manifest", "missing_request", "plan"])
def test_foreign_writer_authority_cannot_create_attempts_parent(
    writer_layout, corruption
):
    writer_layout.attempts.rmdir()
    scheduler = identity()
    if corruption == "owner":
        scheduler = replace(scheduler, owner="bob")
    elif corruption == "manifest":
        writer_layout.manifest.write_bytes(writer_layout.manifest.read_bytes() + b" ")
    else:
        request = writer_layout.run_dir / "submissions" / REQUEST / "request.json"
        if corruption == "missing_request":
            request.unlink()
        else:
            value = json.loads(request.read_bytes())
            value["plan_sha256"] = None
            request.write_text(json.dumps(value))
    with pytest.raises((ValueError, FileNotFoundError)):
        coordinate_attempt(writer_layout.run_dir, scheduler, initial_attempt=None)
    assert not writer_layout.attempts.exists()


def test_legacy_scheduler_never_creates_missing_attempts_parent(writer_layout):
    writer_layout.attempts.rmdir()
    with pytest.raises(FileNotFoundError):
        coordinate_attempt(writer_layout.run_dir, identity(), initial_attempt=1)
    assert not writer_layout.attempts.exists()


def test_writer_environment_is_explicit_and_never_consumes_reservation():
    assert scheduler_environment_identity(writer_environment()) == (identity(), None)
    assert scheduler_environment_identity(
        writer_environment(SLURM_RESTART_COUNT="2")
    ) == (identity(2), None)
    for changed in (
        {"WAVEDL_ATTEMPT_ALLOCATION": "reserved"},
        {"WAVEDL_ATTEMPT_ALLOCATION": None},
        {"WAVEDL_SCHEDULER_INITIAL_ATTEMPT": "1"},
        {"WAVEDL_SCHEDULER_INITIAL_ATTEMPT": None},
        {"SLURM_RESTART_COUNT": "00"},
    ):
        with pytest.raises(ValueError):
            scheduler_environment_identity(writer_environment(**changed))


def test_writer_initial_and_restart_allocate_without_native_ack(writer_layout):
    assert list(writer_layout.attempts.iterdir()) == []
    initial = coordinate_attempt(
        writer_layout.run_dir, identity(), initial_attempt=None
    )
    assert initial.attempt == 1
    namespace = writer_layout.run_dir / "submissions" / REQUEST
    assert not (namespace / "submission.json").exists()
    assert (namespace / "launchers" / identity().claim_name / "binding.json").is_file()
    assert not (writer_layout.run_dir / "scheduler_attempts").exists()
    assert (
        read_scheduler_binding(writer_layout.run_dir, identity(), initial_attempt=None)
        == initial
    )
    (writer_layout.attempts / "8").mkdir()
    restarted = coordinate_attempt(
        writer_layout.run_dir, identity(1), initial_attempt=None
    )
    assert restarted.attempt == 9
    assert (
        coordinate_attempt(writer_layout.run_dir, identity(1), initial_attempt=None)
        == restarted
    )


def _writer_participant(path, start, results):
    start.wait(15)
    try:
        binding = coordinate_attempt(
            Path(path), identity(), initial_attempt=None, timeout_seconds=15
        )
        results.put(("ok", asdict(binding)))
    except BaseException as error:
        results.put(("error", type(error).__name__))


def test_twenty_initial_writers_allocate_exactly_one_attempt(writer_layout):
    context = multiprocessing.get_context("spawn")
    start, results = context.Event(), context.Queue()
    processes = [
        context.Process(
            target=_writer_participant,
            args=(str(writer_layout.run_dir), start, results),
        )
        for _ in range(20)
    ]
    try:
        for process in processes:
            process.start()
        start.set()
        answers = [results.get(timeout=20) for _ in processes]
        assert all(kind == "ok" for kind, _ in answers), answers
        assert len({json.dumps(value, sort_keys=True) for _, value in answers}) == 1
        assert answers[0][1]["attempt"] == 1
        assert [path.name for path in writer_layout.attempts.iterdir()] == ["1"]
    finally:
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        results.close()


@pytest.mark.parametrize(
    "changed",
    [
        {"request_id": RUN},
        {"run_id": REQUEST},
        {"owner": "bob"},
        {"backend": "local"},
        {"backend": []},
        {"owner": True},
        {"manifest_sha256": "b" * 64},
        {"plan_sha256": None},
        {"plan_sha256": "A" * 64},
        {"schema_version": True},
        {"unexpected": "field"},
        {"run_directory": "/foreign/run"},
    ],
)
def test_writer_submission_forgery_fails_before_allocation(writer_layout, changed):
    namespace = writer_layout.run_dir / "submissions" / REQUEST
    request = namespace / "request.json"
    request.write_text(json.dumps(json.loads(request.read_text()) | changed))
    with pytest.raises(ValueError):
        coordinate_attempt(writer_layout.run_dir, identity(), initial_attempt=None)
    assert not list(writer_layout.attempts.iterdir())
    assert not (namespace / "launchers").exists()


@pytest.mark.parametrize("target", ["request", "namespace"])
def test_writer_missing_or_aliased_authority_is_refused(
    writer_layout, tmp_path, target
):
    namespace = writer_layout.run_dir / "submissions" / REQUEST
    if target == "request":
        request = namespace / "request.json"
        content = request.read_bytes()
        request.unlink()
        foreign = tmp_path / "outside.json"
        foreign.write_bytes(content)
        request.symlink_to(foreign)
    else:
        moved = tmp_path / "outside-submission"
        namespace.rename(moved)
        namespace.symlink_to(moved, target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        coordinate_attempt(writer_layout.run_dir, identity(), initial_attempt=None)
    assert not list(writer_layout.attempts.iterdir())


def test_writer_none_cannot_reinterpret_legacy_reserved_intent(layout):
    with pytest.raises(FileNotFoundError):
        coordinate_attempt(layout.run_dir, identity(), initial_attempt=None)
    assert [path.name for path in layout.attempts.iterdir()] == ["1"]


def test_writer_missing_receipt_and_incomplete_claim_never_allocate(writer_layout):
    namespace = writer_layout.run_dir / "submissions" / REQUEST
    request = namespace / "request.json"
    content = request.read_bytes()
    request.unlink()
    with pytest.raises(FileNotFoundError):
        coordinate_attempt(writer_layout.run_dir, identity(), initial_attempt=None)
    request.write_bytes(content)
    request.chmod(0o600)
    (namespace / "launchers" / identity().claim_name).mkdir(parents=True, mode=0o700)
    with pytest.raises(IncompleteSchedulerClaim):
        coordinate_attempt(
            writer_layout.run_dir,
            identity(),
            initial_attempt=None,
            timeout_seconds=0.02,
        )
    assert not list(writer_layout.attempts.iterdir())


@pytest.mark.parametrize("kind", ["duplicate", "oversized", "group-writable"])
def test_writer_request_json_bounds_and_private_namespace(writer_layout, kind):
    namespace = writer_layout.run_dir / "submissions" / REQUEST
    request = namespace / "request.json"
    if kind == "duplicate":
        raw = request.read_text().rstrip()
        request.write_text(raw[:-1] + f',"request_id":"{REQUEST}"}}')
    elif kind == "oversized":
        request.write_bytes(b" " * (10 * 1024 + 1))
    else:
        namespace.chmod(0o770)
    with pytest.raises(ValueError):
        coordinate_attempt(writer_layout.run_dir, identity(), initial_attempt=None)
    assert not list(writer_layout.attempts.iterdir())
    assert not (namespace / "launchers").exists()


def test_writer_claim_symlink_and_forged_binding_are_refused(writer_layout, tmp_path):
    namespace = writer_layout.run_dir / "submissions" / REQUEST
    outside = tmp_path / "outside-claims"
    outside.mkdir(mode=0o700)
    (namespace / "launchers").symlink_to(outside, target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        coordinate_attempt(writer_layout.run_dir, identity(), initial_attempt=None)
    assert not list(outside.iterdir())
    assert not list(writer_layout.attempts.iterdir())
    (namespace / "launchers").unlink()
    binding = coordinate_attempt(
        writer_layout.run_dir, identity(), initial_attempt=None
    )
    receipt = namespace / "launchers" / identity().claim_name / "binding.json"
    receipt.write_text(json.dumps(asdict(binding) | {"owner": "bob"}))
    with pytest.raises(ValueError, match="identity mismatch"):
        read_scheduler_binding(writer_layout.run_dir, identity(), initial_attempt=None)
    assert [path.name for path in writer_layout.attempts.iterdir()] == ["1"]


def test_writer_cli_explicitly_allocates_and_returns_full_binding(
    writer_layout, capsys
):
    arguments = [
        str(writer_layout.run_dir),
        "--run-id",
        RUN,
        "--request-id",
        REQUEST,
        "--cluster",
        "cluster-a",
        "--job-id",
        "12345",
        "--owner",
        "alice",
        "--restart-count",
        "0",
        "--writer",
    ]
    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out) == asdict(identity()) | {"attempt": 1}
    with pytest.raises(SystemExit) as error:
        main(arguments + ["--initial-attempt", "1"])
    assert error.value.code == 2
    assert [path.name for path in writer_layout.attempts.iterdir()] == ["1"]


def test_writer_root_swap_at_allocation_barrier_never_creates_foreign_attempt(
    writer_layout, tmp_path, monkeypatch
):
    import wavedl.core.scheduler_attempt as startup

    outside = tmp_path / "foreign-run"
    outside.mkdir(mode=0o700)
    (outside / "attempts").mkdir(mode=0o700)
    original = startup.allocate_attempt_in_directory
    called = False

    def swap(root):
        nonlocal called
        called = True
        writer_layout.run_dir.rename(tmp_path / "owned-renamed-run")
        writer_layout.run_dir.symlink_to(outside, target_is_directory=True)
        return original(root)

    monkeypatch.setattr(startup, "allocate_attempt_in_directory", swap)
    with pytest.raises((ValueError, OSError)):
        coordinate_attempt(writer_layout.run_dir, identity(), initial_attempt=None)
    assert called
    assert list((outside / "attempts").iterdir()) == []
    assert list((tmp_path / "owned-renamed-run" / "attempts").iterdir()) == []


def test_writer_read_only_binding_never_allocates_or_publishes(
    writer_layout, monkeypatch
):
    import wavedl.core.scheduler_attempt as startup

    binding = coordinate_attempt(
        writer_layout.run_dir, identity(), initial_attempt=None
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Read-only writer proof must not mutate runtime directories")

    for name in (
        "allocate_attempt_in_directory",
        "allocate_attempt",
        "_mkdir",
        "_publish",
    ):
        monkeypatch.setattr(startup, name, forbidden)
    assert (
        read_scheduler_binding(writer_layout.run_dir, identity(), initial_attempt=None)
        == binding
    )


def test_read_only_proof_rejects_boolean_selected_receipt(layout):
    binding = coordinate_attempt(layout.run_dir, identity(), initial_attempt=1)
    forged = asdict(binding) | {"attempt": True}
    (layout.attempt_dir(1) / "scheduler_identity.json").write_text(json.dumps(forged))
    with pytest.raises(ValueError):
        read_scheduler_binding(layout.run_dir, identity(), initial_attempt=1)


def test_unrelated_mapping_does_not_change_the_owned_run_view(layout):
    from wavedl.core.manifest import StorageMapping

    data = json.loads(layout.manifest.read_text())
    data["storage_map"] = [
        asdict(StorageMapping("/host/not-this-run", "/execution/not-this-run"))
    ]
    layout.manifest.write_text(json.dumps(data))
    assert (
        coordinate_attempt(layout.run_dir, identity(), initial_attempt=1).attempt == 1
    )


def _participant(path, start, results):
    start.wait(15)
    try:
        binding = coordinate_attempt(
            Path(path), identity(1), initial_attempt=1, timeout_seconds=15
        )
        results.put(("ok", asdict(binding)))
    except BaseException as error:
        results.put(("error", type(error).__name__))


def test_twenty_processes_publish_exactly_one_restart_binding(layout):
    context = multiprocessing.get_context("spawn")
    start, results = context.Event(), context.Queue()
    processes = [
        context.Process(target=_participant, args=(str(layout.run_dir), start, results))
        for _ in range(20)
    ]
    try:
        for process in processes:
            process.start()
        start.set()
        answers = [results.get(timeout=20) for _ in processes]
        assert all(kind == "ok" for kind, _ in answers), answers
        assert len({json.dumps(value, sort_keys=True) for _, value in answers}) == 1
        assert answers[0][1]["attempt"] == 2
        assert sorted(path.name for path in layout.attempts.iterdir()) == ["1", "2"]
    finally:
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        results.close()


def test_initial_reserved_attempt_and_gapped_restart_allocation(layout):
    initial = coordinate_attempt(layout.run_dir, identity(), initial_attempt=1)
    assert initial.attempt == 1
    (layout.attempts / "9").mkdir()
    restarted = coordinate_attempt(layout.run_dir, identity(1), initial_attempt=1)
    assert restarted.attempt == 10
    assert (
        coordinate_attempt(layout.run_dir, identity(1), initial_attempt=1) == restarted
    )


def _crash_before_binding(path):
    from wavedl.core import scheduler_attempt

    original = scheduler_attempt._publish

    def crash(directory, name, content):
        if name == "binding.json":
            os._exit(71)
        return original(directory, name, content)

    scheduler_attempt._publish = crash
    coordinate_attempt(Path(path), identity(1), initial_attempt=1)


def _crash_after_binding(path):
    from wavedl.core import scheduler_attempt

    original = scheduler_attempt._publish

    def crash(directory, name, content):
        original(directory, name, content)
        if name == "binding.json":
            os._exit(72)

    scheduler_attempt._publish = crash
    coordinate_attempt(Path(path), identity(1), initial_attempt=1)


@pytest.mark.parametrize(
    "target,code,complete",
    [(_crash_before_binding, 71, False), (_crash_after_binding, 72, True)],
)
def test_crashed_coordinator_never_reclaims_or_duplicates_attempt(
    layout, target, code, complete
):
    process = multiprocessing.get_context("spawn").Process(
        target=target, args=(str(layout.run_dir),)
    )
    process.start()
    process.join(timeout=15)
    try:
        assert not process.is_alive()
        assert process.exitcode == code
        if complete:
            assert (
                coordinate_attempt(
                    layout.run_dir, identity(1), initial_attempt=1
                ).attempt
                == 2
            )
        else:
            with pytest.raises(IncompleteSchedulerClaim):
                coordinate_attempt(
                    layout.run_dir, identity(1), initial_attempt=1, timeout_seconds=0.02
                )
        assert sorted(path.name for path in layout.attempts.iterdir()) == ["1", "2"]
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", RUN.upper().replace("000001", "ABCDEF")),
        ("request_id", "not-uuid"),
        ("cluster", "../escape"),
        ("owner", "alice\nroot"),
        ("job_id", "0"),
        ("job_id", "01"),
        ("job_id", "123.batch"),
        ("job_id", str(2**63)),
        ("restart_count", True),
        ("restart_count", -1),
    ],
)
def test_invalid_scheduler_identity_is_rejected(field, value):
    with pytest.raises(ValueError):
        replace(identity(), **{field: value})


def test_forged_binding_and_conflicting_initial_reservation_fail_closed(layout):
    coordinate_attempt(layout.run_dir, identity(), initial_attempt=1)
    receipt = (
        layout.run_dir / "scheduler_attempts" / identity().claim_name / "binding.json"
    )
    data = json.loads(receipt.read_bytes())
    receipt.write_text(json.dumps(data | {"request_id": RUN}))
    with pytest.raises(ValueError, match="identity mismatch"):
        coordinate_attempt(layout.run_dir, identity(), initial_attempt=1)
    receipt.write_text(json.dumps(data))
    (layout.attempts / "2").mkdir()
    (layout.attempt_dir(2) / "submission.json").write_text(
        json.dumps(
            {
                "request_id": REQUEST,
                "attempt": 2,
                "owner": "alice",
                "created_at": "2026-10-04T00:00:00Z",
            }
        )
    )
    with pytest.raises(ValueError, match="initial attempt"):
        coordinate_attempt(layout.run_dir, identity(), initial_attempt=2)


def test_preexisting_attempt_identity_is_never_overwritten(layout):
    destination = layout.attempt_dir(1) / "scheduler_identity.json"
    destination.write_text("foreign")
    with pytest.raises(FileExistsError):
        coordinate_attempt(layout.run_dir, identity(), initial_attempt=1)
    assert destination.read_text() == "foreign"
    with pytest.raises(IncompleteSchedulerClaim):
        coordinate_attempt(
            layout.run_dir, identity(), initial_attempt=1, timeout_seconds=0.02
        )


def test_follower_never_elects_or_allocates(layout):
    with pytest.raises(IncompleteSchedulerClaim):
        coordinate_attempt(
            layout.run_dir,
            identity(1),
            initial_attempt=1,
            coordinator=False,
            timeout_seconds=0.02,
        )
    assert [path.name for path in layout.attempts.iterdir()] == ["1"]
    assert not (layout.run_dir / "scheduler_attempts" / identity(1).claim_name).exists()


def test_symlinked_manifest_and_wrong_manifest_run_are_refused(layout):
    original = layout.manifest.read_bytes()
    layout.manifest.write_text(json.dumps(json.loads(original) | {"run_id": REQUEST}))
    with pytest.raises(ValueError, match="manifest"):
        coordinate_attempt(layout.run_dir, identity(), initial_attempt=1)
    target = layout.run_dir / "outside.json"
    target.write_bytes(original)
    layout.manifest.unlink()
    try:
        layout.manifest.symlink_to(target)
    except OSError:
        pytest.skip("Symlink creation unavailable")
    with pytest.raises((ValueError, OSError)):
        coordinate_attempt(layout.run_dir, identity(), initial_attempt=1)


@pytest.mark.skipif(os.name != "posix", reason="POSIX writable ownership guard")
def test_group_writable_run_is_refused(layout):
    layout.run_dir.chmod(0o770)
    with pytest.raises(ValueError):
        coordinate_attempt(layout.run_dir, identity(), initial_attempt=1)


def test_cli_prints_complete_bounded_binding(layout, capsys):
    assert (
        main(
            [
                str(layout.run_dir),
                "--run-id",
                RUN,
                "--request-id",
                REQUEST,
                "--cluster",
                "cluster-a",
                "--job-id",
                "12345",
                "--owner",
                "alice",
                "--restart-count",
                "0",
                "--initial-attempt",
                "1",
            ]
        )
        == 0
    )
    value = json.loads(capsys.readouterr().out)
    assert value == asdict(identity()) | {"attempt": 1}


@pytest.mark.skipif(os.name != "posix", reason="Anchored POSIX allocation")
def test_swapped_run_root_refuses_allocation_without_outside_writes(layout, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "attempts").mkdir()
    with open_directory(layout.attempts) as anchor:
        original = layout.run_dir
        original.rename(tmp_path / "original-run")
        original.symlink_to(outside, target_is_directory=True)
        with pytest.raises((ValueError, OSError)):
            allocate_attempt(layout, attempts_anchor=anchor)
    assert list((outside / "attempts").iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="Anchored POSIX allocation")
def test_symlink_swap_during_allocation_never_writes_to_replacement(
    layout, tmp_path, monkeypatch
):
    outside = tmp_path / "outside"
    outside.mkdir()
    original_mkdir = os.mkdir
    with open_directory(layout.attempts) as anchor:
        swapped = False

        def swap(name, mode=0o777, *, dir_fd=None):
            nonlocal swapped
            if name == "2" and dir_fd == anchor.handle and not swapped:
                swapped = True
                layout.attempts.rename(layout.run_dir / "old-attempts")
                layout.attempts.symlink_to(outside, target_is_directory=True)
            return original_mkdir(name, mode, dir_fd=dir_fd)

        monkeypatch.setattr(os, "mkdir", swap)
        with pytest.raises((ValueError, OSError)):
            allocate_attempt(layout, attempts_anchor=anchor)
        assert swapped
    assert list(outside.iterdir()) == []


def test_wrong_attempt_anchor_is_rejected(layout, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    with (
        open_directory(other) as anchor,
        pytest.raises(ValueError, match="different run"),
    ):
        allocate_attempt(layout, attempts_anchor=anchor)
    assert list(other.iterdir()) == []


@pytest.mark.parametrize(
    "changed",
    [
        {"request_id": RUN},
        {"owner": "bob"},
        {"attempt": True},
        {"attempt": 9},
    ],
)
def test_durable_submission_intent_must_match_startup_identity(layout, changed):
    path = layout.attempt_dir(1) / "submission.json"
    path.write_text(json.dumps(json.loads(path.read_bytes()) | changed))
    with pytest.raises(ValueError, match="submission intent"):
        coordinate_attempt(layout.run_dir, identity(), initial_attempt=1)
    assert not (layout.run_dir / "scheduler_attempts").exists()


def test_missing_submission_proof_fails_before_claim(layout):
    (layout.attempt_dir(1) / "submission.json").unlink()
    with pytest.raises(FileNotFoundError):
        coordinate_attempt(layout.run_dir, identity(), initial_attempt=1)
    assert not (layout.run_dir / "scheduler_attempts").exists()


def test_actual_slurm_submission_receipt_is_valid_startup_proof(tmp_path):
    from unittest.mock import Mock
    from uuid import uuid4

    from wavedl.core.layout import OperationKind
    from wavedl.core.manifest import RuntimeSelection, build_manifest
    from wavedl_daemon.backends.slurm_plan import SlurmPlanner, SlurmSite
    from wavedl_daemon.backends.slurm_submission import SlurmSubmission
    from wavedl_daemon.storage.ownership import DispatchLock
    from wavedl_daemon.supervision.base import ProcessResult

    run_id, request_id = str(uuid4()), str(uuid4())
    manifest = build_manifest(
        RunLayout(tmp_path / "submitted", run_id),
        operation=OperationKind.TRAIN,
        resolved_config={"epochs": 1, "model": "mlp", "data_path": "train.npz"},
        runtime=RuntimeSelection("/usr/bin/python3", "selected", "cpu", {}),
        submission_cwd=tmp_path,
    )
    plan = SlurmPlanner(SlurmSite(partition="cpu", cluster="alpha")).plan(manifest)
    runner = Mock()
    runner.run.return_value = ProcessResult(0, b"123;alpha\n", b"")
    with DispatchLock(tmp_path / "dispatch.lock") as lock:
        backend = SlurmSubmission(runner, owner="alice", dispatch_lock=lock)
        backend.submit(plan, request_id=request_id, attempt=2)
    bound = coordinate_attempt(
        Path(plan.run_dir),
        SchedulerIdentity(
            run_id,
            request_id,
            "alpha",
            "123",
            "alice",
            0,
        ),
        initial_attempt=2,
    )
    assert bound.attempt == 2
    assert runner.run.call_count == 1


@pytest.mark.parametrize(
    "changed",
    [
        {"schema_version": 9},
        {"run_id": 42},
        {"operation": "download"},
        {
            "runtime": {
                "python": "relative",
                "mode": "selected",
                "device": "cpu",
                "environment": {},
            }
        },
        {"resolved_config": []},
        {"resolved_config": {"output_dir": "/other/run"}},
    ],
)
def test_invalid_full_manifest_is_rejected_before_claim(layout, changed):
    value = json.loads(layout.manifest.read_bytes())
    layout.manifest.write_text(json.dumps(value | changed))
    with pytest.raises(ValueError, match="manifest"):
        coordinate_attempt(layout.run_dir, identity(), initial_attempt=1)
    assert not (layout.run_dir / "scheduler_attempts").exists()


def test_two_field_fake_manifest_is_not_execution_authority(layout):
    layout.manifest.write_text(
        json.dumps({"run_id": RUN, "run_directory": str(layout.run_dir)})
    )
    with pytest.raises(ValueError, match="manifest"):
        coordinate_attempt(layout.run_dir, identity(), initial_attempt=1)
    assert not (layout.run_dir / "scheduler_attempts").exists()


@pytest.mark.parametrize("target", ["manifest", "submission", "binding"])
def test_duplicate_json_keys_never_select_execution_identity(layout, target):
    if target == "manifest":
        path = layout.manifest
        key, repeated = "run_id", RUN
    elif target == "submission":
        path = layout.attempt_dir(1) / "submission.json"
        key, repeated = "request_id", REQUEST
    else:
        coordinate_attempt(layout.run_dir, identity(), initial_attempt=1)
        path = (
            layout.run_dir
            / "scheduler_attempts"
            / identity().claim_name
            / "binding.json"
        )
        key, repeated = "run_id", RUN
    raw = path.read_text().rstrip()
    path.write_text(raw[:-1] + f',"{key}":"{repeated}"}}')
    with pytest.raises(ValueError, match="Duplicate"):
        coordinate_attempt(layout.run_dir, identity(), initial_attempt=1)


def test_post_link_flush_failure_requires_followers_own_successful_flush(
    layout, monkeypatch
):
    original_link, original_fsync = os.link, os.fsync
    linked = False

    def link(source, destination, *args, **kwargs):
        nonlocal linked
        original_link(source, destination, *args, **kwargs)
        if destination == "binding.json":
            linked = True

    def fail_after_link(descriptor):
        if linked:
            raise OSError("Directory flush unavailable after visible link")
        return original_fsync(descriptor)

    monkeypatch.setattr(os, "link", link)
    monkeypatch.setattr(os, "fsync", fail_after_link)
    with pytest.raises(OSError, match="flush unavailable"):
        coordinate_attempt(layout.run_dir, identity(1), initial_attempt=1)
    assert linked
    with pytest.raises(OSError, match="flush unavailable"):
        coordinate_attempt(
            layout.run_dir, identity(1), initial_attempt=1, coordinator=False
        )
    flushed = []

    def flush(descriptor):
        flushed.append(descriptor)
        return original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", flush)
    assert (
        coordinate_attempt(
            layout.run_dir, identity(1), initial_attempt=1, coordinator=False
        ).attempt
        == 2
    )
    assert len(flushed) == 5
    assert len(set(flushed)) == 5
    assert sorted(path.name for path in layout.attempts.iterdir()) == ["1", "2"]
