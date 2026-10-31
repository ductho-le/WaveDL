"""Runtime writer allocation and fail-closed Local startup proof tests."""

import json
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from threading import Barrier

import pytest

from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.local_attempt import (
    IncompleteLocalClaim,
    LocalStartupBinding,
    LocalStartupIdentity,
    coordinate_local_attempt,
    read_local_binding,
)
from wavedl.core.manifest import RuntimeSelection, build_manifest, write_manifest
from wavedl.core.submission import (
    create_submission,
    identity_for_submission,
    publish_submission_file,
)


RUN = "00000000-0000-4000-8000-000000000001"
REQUEST = "00000000-0000-4000-8000-000000000002"


@pytest.fixture
def setup(tmp_path):
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
    receipt = identity_for_submission(
        layout.run_dir,
        request_id=REQUEST,
        backend="local",
        owner="owner_token-123",
        plan_sha256="a" * 64,
    )
    namespace = create_submission(layout.run_dir, receipt)
    identity = LocalStartupIdentity(
        **{key: value for key, value in asdict(receipt).items() if key != "backend"},
        supervisor_id="supervisor_token-123",
    )
    publish_submission_file(
        layout.run_dir,
        receipt,
        "local-submission.json",
        json.dumps(
            {
                "schema_version": 1,
                "run_id": identity.run_id,
                "request_id": identity.request_id,
                "owner": identity.owner,
                "supervisor_id": identity.supervisor_id,
                "plan_sha256": identity.plan_sha256,
            }
        ).encode(),
    )
    return layout, namespace, identity


def test_no_attempt_before_writer_and_duplicate_is_idempotent(setup):
    layout, _, identity = setup
    assert list(layout.attempts.iterdir()) == []
    binding = coordinate_local_attempt(layout.run_dir, identity)
    assert binding.attempt == 1
    assert coordinate_local_attempt(layout.run_dir, identity) == binding
    assert read_local_binding(layout.run_dir, identity) == binding
    assert [path.name for path in layout.attempts.iterdir()] == ["1"]


def test_writer_respects_existing_gapped_reservations(setup):
    layout, _, identity = setup
    for number in (1, 8, 15):
        layout.attempt_dir(number).mkdir()
    assert coordinate_local_attempt(layout.run_dir, identity).attempt == 16


def test_two_concurrent_writers_publish_one_attempt(setup):
    layout, _, identity = setup
    start = Barrier(2)

    def participant():
        start.wait(timeout=5)
        return coordinate_local_attempt(layout.run_dir, identity, timeout_seconds=5)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: participant(), range(2)))
    assert results[0] == results[1]
    assert results[0].attempt == 1
    assert len(list(layout.attempts.iterdir())) == 1


def test_abandoned_claim_fails_closed_without_allocation(setup):
    layout, namespace, identity = setup
    (namespace / "startup-claim").mkdir(mode=0o700)
    with pytest.raises(IncompleteLocalClaim):
        coordinate_local_attempt(layout.run_dir, identity, timeout_seconds=0.01)
    assert list(layout.attempts.iterdir()) == []
    assert (namespace / "startup-claim").is_dir()


@pytest.mark.parametrize(
    "field,value",
    [("owner", "foreign"), ("plan_sha256", "b" * 64), ("manifest_sha256", "c" * 64)],
)
def test_foreign_submission_identity_never_allocates(setup, field, value):
    layout, _, identity = setup
    with pytest.raises(ValueError):
        coordinate_local_attempt(layout.run_dir, replace(identity, **{field: value}))
    assert list(layout.attempts.iterdir()) == []


def test_foreign_supervisor_cannot_consume_existing_proof(setup):
    layout, _, identity = setup
    coordinate_local_attempt(layout.run_dir, identity)
    foreign = replace(identity, supervisor_id="foreign-supervisor")
    with pytest.raises(ValueError):
        read_local_binding(layout.run_dir, foreign)
    with pytest.raises(ValueError):
        coordinate_local_attempt(layout.run_dir, foreign)
    assert len(list(layout.attempts.iterdir())) == 1


@pytest.mark.parametrize("target", ["startup.json", "local_identity.json"])
def test_corrupt_or_boolean_proof_is_rejected(setup, target):
    layout, namespace, identity = setup
    binding = coordinate_local_attempt(layout.run_dir, identity)
    path = (
        namespace / target
        if target == "startup.json"
        else layout.attempt_dir(1) / target
    )
    path.write_text(json.dumps(asdict(binding) | {"attempt": True}))
    with pytest.raises(ValueError):
        read_local_binding(layout.run_dir, identity)


def test_read_only_never_allocates_or_publishes(setup, monkeypatch):
    import wavedl.core.local_attempt as startup

    layout, _, identity = setup
    expected = coordinate_local_attempt(layout.run_dir, identity)

    def forbidden(*args, **kwargs):
        pytest.fail("Read-only binding must not mutate")

    for name in ("allocate_attempt_in_directory", "_mkdir", "_publish"):
        monkeypatch.setattr(startup, name, forbidden)
    assert read_local_binding(layout.run_dir, identity) == expected


def test_missing_request_and_missing_attempt_fail_closed(setup):
    layout, namespace, identity = setup
    with pytest.raises(FileNotFoundError):
        read_local_binding(layout.run_dir, identity)
    (namespace / "request.json").unlink()
    with pytest.raises(FileNotFoundError):
        coordinate_local_attempt(layout.run_dir, identity)
    assert list(layout.attempts.iterdir()) == []


@pytest.mark.parametrize("attempt", [True, 0, -1, 2**63])
def test_binding_rejects_invalid_attempt(setup, attempt):
    _, _, identity = setup
    with pytest.raises(ValueError):
        LocalStartupBinding(**asdict(identity), attempt=attempt)


def test_torn_publication_retains_attempt_and_claim(setup, monkeypatch):
    import wavedl.core.local_attempt as startup

    layout, namespace, identity = setup
    publish = startup._publish

    def crash(directory, name, content):
        if name == "startup.json":
            raise OSError("simulated crash before binding publication")
        publish(directory, name, content)

    monkeypatch.setattr(startup, "_publish", crash)
    with pytest.raises(OSError):
        coordinate_local_attempt(layout.run_dir, identity)
    with pytest.raises(IncompleteLocalClaim):
        coordinate_local_attempt(layout.run_dir, identity, timeout_seconds=0.01)
    assert len(list(layout.attempts.iterdir())) == 1
    assert (namespace / "startup-claim").exists()


@pytest.mark.parametrize("mode", ["missing", "foreign", "duplicate"])
def test_invalid_supervisor_fence_never_allocates(setup, mode):
    layout, namespace, identity = setup
    fence = namespace / "local-submission.json"
    if mode == "missing":
        fence.unlink()
    elif mode == "foreign":
        data = json.loads(fence.read_text())
        data["supervisor_id"] = "foreign"
        fence.write_text(json.dumps(data))
    else:
        raw = fence.read_text()
        fence.write_text('{"schema_version":1,' + raw[1:])
    with pytest.raises((ValueError, FileNotFoundError)):
        coordinate_local_attempt(layout.run_dir, identity)
    assert list(layout.attempts.iterdir()) == []


def _process_participant(path, identity_fields, start, results):
    start.wait(10)
    try:
        binding = coordinate_local_attempt(
            Path(path), LocalStartupIdentity(**identity_fields), timeout_seconds=5
        )
        results.put(("ok", asdict(binding)))
    except BaseException as error:
        results.put(("error", type(error).__name__))


def test_real_process_race_elects_one_writer(setup):
    layout, _, identity = setup
    context = multiprocessing.get_context("spawn")
    start, results = context.Event(), context.Queue()
    processes = [
        context.Process(
            target=_process_participant,
            args=(str(layout.run_dir), asdict(identity), start, results),
        )
        for _ in range(4)
    ]
    try:
        for process in processes:
            process.start()
        start.set()
        records = [results.get(timeout=15) for _ in processes]
        for process in processes:
            process.join(10)
            assert process.exitcode == 0
        assert all(record[0] == "ok" for record in records), records
        assert all(record[1] == records[0][1] for record in records)
        assert len(list(layout.attempts.iterdir())) == 1
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
        results.close()
        results.join_thread()


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink attack gate")
def test_symlinked_claim_or_attempt_proof_rejected(setup):
    layout, namespace, identity = setup
    binding = coordinate_local_attempt(layout.run_dir, identity)
    proof = layout.attempt_dir(binding.attempt) / "local_identity.json"
    proof.unlink()
    proof.symlink_to(namespace / "startup.json")
    with pytest.raises((OSError, ValueError)):
        read_local_binding(layout.run_dir, identity)


def test_leading_dash_supervisor_is_preserved(setup):
    _, _, identity = setup
    assert (
        replace(identity, supervisor_id="-url_safe-token").supervisor_id
        == "-url_safe-token"
    )


def test_writer_and_reader_use_one_explicit_root_capability(setup, monkeypatch):
    import wavedl.core.local_attempt as startup
    import wavedl.core.submission as submission

    layout, _, identity = setup
    opened = []
    open_root = startup.open_directory

    def record(path, *, writable=False):
        opened.append((path, writable))
        return open_root(path, writable=writable)

    def forbidden(*args, **kwargs):
        pytest.fail("Submission verification must not reopen held run pathname")

    monkeypatch.setattr(startup, "open_directory", record)
    monkeypatch.setattr(submission, "open_directory", forbidden)
    expected = coordinate_local_attempt(layout.run_dir, identity)
    assert opened == [(layout.run_dir, True)]
    opened.clear()
    assert read_local_binding(layout.run_dir, identity) == expected
    assert opened == [(layout.run_dir, False)]


def test_local_writer_sharing_admission_uses_existing_deadline_before_allocation(
    setup, monkeypatch
):
    from types import SimpleNamespace

    from wavedl.core import local_attempt, submission

    layout, _, identity = setup
    original = local_attempt.open_directory
    calls = []

    def contend(path, *, writable=False):
        calls.append(path)
        if len(calls) == 1:
            assert list(layout.attempts.iterdir()) == []
            error = PermissionError(13, "owned Windows sharing")
            error.winerror = 32
            raise error
        return original(path, writable=writable)

    monkeypatch.setattr(
        submission, "os", SimpleNamespace(**(vars(os) | {"name": "nt"}))
    )
    monkeypatch.setattr(local_attempt, "open_directory", contend)
    result = coordinate_local_attempt(layout.run_dir, identity, timeout_seconds=1)
    assert result.attempt == 1 and len(calls) == 2
    assert len(list(layout.attempts.iterdir())) == 1


def test_local_writer_expired_admission_never_allocates_or_resets_deadline(
    setup, monkeypatch
):
    from types import SimpleNamespace

    from wavedl.core import local_attempt, submission

    layout, namespace, identity = setup
    clock = [0.0]
    error = PermissionError(13, "owned Windows sharing")
    error.winerror = 32
    calls = []

    def denied(*args, **kwargs):
        calls.append(clock[0])
        raise error

    timer = SimpleNamespace(
        monotonic=lambda: clock[0], sleep=lambda s: clock.__setitem__(0, clock[0] + s)
    )
    monkeypatch.setattr(submission, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(submission, "time", timer)
    monkeypatch.setattr(local_attempt, "time", timer)
    monkeypatch.setattr(local_attempt, "open_directory", denied)
    with pytest.raises(PermissionError) as refusal:
        coordinate_local_attempt(layout.run_dir, identity, timeout_seconds=0.03)
    assert refusal.value is error and clock[0] == 0.03 and calls == [0, 0.02]
    assert list(layout.attempts.iterdir()) == []
    assert not (namespace / "startup-claim").exists()
