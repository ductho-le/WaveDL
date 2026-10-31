"""Pre-runtime submission receipts allocate no attempts and retain crash fences."""

import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest

from wavedl.core import submission
from wavedl.core.layout import OperationKind, RunLayout
from wavedl.core.manifest import (
    RuntimeSelection,
    StorageMapping,
    build_manifest,
    write_manifest,
)
from wavedl.core.safe_files import open_directory
from wavedl.core.submission import (
    create_submission,
    identity_for_submission,
    identity_for_submission_directory,
    mkdir_submission_member,
    open_submission,
    open_submission_in_directory,
    publish_submission_file,
    publish_submission_member,
    read_submission,
    read_submission_file,
)


RUN_ID = "00000000-0000-4000-8000-000000000001"
REQUEST_ID = "00000000-0000-4000-8000-000000000002"


def setup_run(tmp_path):
    layout = RunLayout.create(tmp_path / "runs", RUN_ID)
    manifest = build_manifest(
        layout,
        operation=OperationKind.TRAIN,
        resolved_config={},
        runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
        submission_cwd=tmp_path,
    )
    write_manifest(layout, manifest)
    identity = identity_for_submission(
        layout.run_dir,
        request_id=REQUEST_ID,
        backend="local",
        owner="operator",
        plan_sha256="a" * 64,
    )
    return layout, identity


def test_create_read_and_file_publication_do_not_allocate_runtime_attempts(tmp_path):
    layout, identity = setup_run(tmp_path)
    before = layout.manifest.read_bytes()
    namespace = create_submission(layout.run_dir, identity)
    expected_namespace = layout.run_dir / "submissions" / REQUEST_ID
    # A held Windows handle returns the extended DOS namespace spelling.
    # Require the exact existing directory/member, not lexical path spelling.
    assert namespace.name == REQUEST_ID
    assert namespace.samefile(expected_namespace)
    assert namespace.parent.samefile(expected_namespace.parent)
    assert {path.name for path in expected_namespace.parent.iterdir()} == {REQUEST_ID}
    assert read_submission(layout.run_dir, identity) == identity
    assert identity.manifest_sha256 == hashlib.sha256(before).hexdigest()
    published = publish_submission_file(
        layout.run_dir, identity, "submission.sh", b"script"
    )
    assert published.name == "submission.sh"
    assert published.parent.samefile(namespace)
    assert published.samefile(expected_namespace / "submission.sh")
    assert {path.name for path in namespace.iterdir()} == {
        "request.json",
        "submission.sh",
    }
    assert read_submission_file(layout.run_dir, identity, "submission.sh") == b"script"
    assert not list(layout.attempts.iterdir())
    assert layout.manifest.read_bytes() == before
    with pytest.raises(FileExistsError):
        create_submission(layout.run_dir, identity)
    with pytest.raises(FileExistsError):
        publish_submission_file(
            layout.run_dir, identity, "submission.sh", b"replacement"
        )
    assert read_submission_file(layout.run_dir, identity, "submission.sh") == b"script"


def test_parallel_submission_claim_has_exactly_one_winner(tmp_path):
    layout, identity = setup_run(tmp_path)

    def claim():
        try:
            create_submission(layout.run_dir, identity)
            return True
        except FileExistsError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(lambda _n: claim(), range(16))) == 1
    assert read_submission(layout.run_dir, identity) == identity


def test_crashed_empty_claim_is_never_reused(tmp_path):
    layout, identity = setup_run(tmp_path)
    namespace = layout.run_dir / "submissions" / REQUEST_ID
    namespace.mkdir(parents=True, mode=0o700)
    namespace.parent.chmod(0o700)
    with pytest.raises(FileExistsError):
        create_submission(layout.run_dir, identity)
    with pytest.raises(FileNotFoundError):
        read_submission(layout.run_dir, identity)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", RUN_ID.upper().replace("0001", "ABCD")),
        ("request_id", "invalid"),
        ("backend", "unknown"),
        ("owner", "bad owner"),
        ("run_directory", "relative"),
        ("manifest_sha256", None),
        ("manifest_sha256", "A" * 64),
        ("plan_sha256", "short"),
    ],
)
def test_identity_validation(tmp_path, field, value):
    _layout, identity = setup_run(tmp_path)
    with pytest.raises(ValueError):
        replace(identity, **{field: value})


@pytest.mark.parametrize(
    "field", ["run_id", "request_id", "backend", "owner", "plan_sha256"]
)
def test_receipt_cannot_be_read_under_foreign_identity(tmp_path, field):
    layout, identity = setup_run(tmp_path)
    namespace = create_submission(layout.run_dir, identity)
    data = json.loads((namespace / "request.json").read_bytes())
    data[field] = "foreign"
    (namespace / "request.json").write_text(json.dumps(data))
    with pytest.raises(ValueError):
        read_submission(layout.run_dir, identity)


def test_manifest_bytes_are_part_of_identity_and_legacy_history_stays_untouched(
    tmp_path,
):
    layout, identity = setup_run(tmp_path)
    legacy = layout.attempts / "9"
    legacy.mkdir(mode=0o700)
    (legacy / "submission.json").write_bytes(b"legacy")
    create_submission(layout.run_dir, identity)
    assert (legacy / "submission.json").read_bytes() == b"legacy"
    layout.manifest.write_bytes(layout.manifest.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest"):
        read_submission(layout.run_dir, identity)


def test_duplicate_json_fields_and_boolean_version_are_rejected(tmp_path):
    layout, identity = setup_run(tmp_path)
    namespace = create_submission(layout.run_dir, identity)
    path = namespace / "request.json"
    raw = path.read_bytes()
    path.write_bytes(
        raw.replace(b'"schema_version": 1', b'"schema_version": 1, "schema_version": 1')
    )
    with pytest.raises(ValueError, match="Duplicate"):
        read_submission(layout.run_dir, identity)
    path.write_bytes(raw.replace(b'"schema_version": 1', b'"schema_version": true'))
    with pytest.raises(ValueError):
        read_submission(layout.run_dir, identity)


@pytest.mark.parametrize(
    "name", ["../escape", "a/b", "a\\b", "a:b", "request.json", "manifest.json"]
)
def test_file_names_and_bounds_are_validated(tmp_path, name):
    layout, identity = setup_run(tmp_path)
    create_submission(layout.run_dir, identity)
    with pytest.raises(ValueError):
        publish_submission_file(layout.run_dir, identity, name, b"x")


def test_file_byte_bound_checked_before_publication_and_read(tmp_path):
    layout, identity = setup_run(tmp_path)
    namespace = create_submission(layout.run_dir, identity)
    with pytest.raises(ValueError):
        publish_submission_file(layout.run_dir, identity, "file", b"123", limit=2)
    assert not (namespace / "file").exists()
    publish_submission_file(layout.run_dir, identity, "file", b"123")
    with pytest.raises(ValueError):
        read_submission_file(layout.run_dir, identity, "file", limit=2)


@pytest.mark.skipif(os.name != "posix", reason="Native POSIX symlink/permission proof")
def test_no_follow_and_private_namespace(tmp_path):
    layout, identity = setup_run(tmp_path)
    namespace = create_submission(layout.run_dir, identity)
    target = tmp_path / "secret"
    target.write_bytes(b"untouched")
    (namespace / "script").symlink_to(target)
    with pytest.raises((ValueError, OSError)):
        read_submission_file(layout.run_dir, identity, "script")
    with pytest.raises(FileExistsError):
        publish_submission_file(layout.run_dir, identity, "script", b"replace")
    assert target.read_bytes() == b"untouched"
    namespace.chmod(0o777)
    with pytest.raises(ValueError):
        read_submission(layout.run_dir, identity)


def test_declared_execution_view_has_same_identity(tmp_path):
    host_root, execution_root = tmp_path / "host", tmp_path / "execution"
    host = RunLayout.create(host_root, RUN_ID)
    execution = RunLayout.create(execution_root, RUN_ID)
    manifest = build_manifest(
        host,
        operation=OperationKind.TRAIN,
        resolved_config={},
        runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
        submission_cwd=tmp_path,
        storage_map=(StorageMapping(str(host_root), str(execution_root)),),
    )
    write_manifest(host, manifest)
    execution.manifest.write_bytes(host.manifest.read_bytes())
    execution.manifest.chmod(0o600)
    identity = identity_for_submission(
        host.run_dir, request_id=REQUEST_ID, backend="slurm", owner="operator"
    )
    assert (
        identity_for_submission(
            execution.run_dir, request_id=REQUEST_ID, backend="slurm", owner="operator"
        )
        == identity
    )
    create_submission(host.run_dir, identity)
    create_submission(execution.run_dir, identity)
    assert read_submission(execution.run_dir, identity) == identity
    foreign = RunLayout.create(tmp_path / "foreign", RUN_ID)
    foreign.manifest.write_bytes(host.manifest.read_bytes())
    with pytest.raises(ValueError):
        create_submission(foreign.run_dir, identity)


@pytest.mark.skipif(os.name != "nt", reason="Native Windows held-ancestry publication")
def test_native_windows_publication(tmp_path):
    layout, identity = setup_run(tmp_path)
    create_submission(layout.run_dir, identity)
    publish_submission_file(layout.run_dir, identity, "submission.sh", b"native")
    assert read_submission_file(layout.run_dir, identity, "submission.sh") == b"native"


@pytest.mark.skipif(os.name != "posix", reason="Native POSIX ancestry-swap proof")
def test_ancestry_exchange_during_publication_does_not_write_foreign_namespace(
    tmp_path, monkeypatch
):
    layout, identity = setup_run(tmp_path)
    namespace = create_submission(layout.run_dir, identity)
    moved = layout.run_dir.parent / "moved"
    original = submission._publish

    def swapped(directory, name, content):
        layout.run_dir.rename(moved)
        layout.run_dir.mkdir(mode=0o700)
        foreign = layout.run_dir / "submissions" / REQUEST_ID
        foreign.mkdir(parents=True, mode=0o700)
        return original(directory, name, content)

    monkeypatch.setattr(submission, "_publish", swapped)
    with pytest.raises(ValueError):
        publish_submission_file(layout.run_dir, identity, "script", b"never foreign")
    assert not (namespace / "script").exists()
    assert not (moved / "submissions" / REQUEST_ID / "script").exists()


def test_open_submission_holds_verified_namespace(tmp_path):
    layout, identity = setup_run(tmp_path)
    namespace = create_submission(layout.run_dir, identity)
    with open_submission(layout.run_dir, identity, writable=True) as directory:
        assert directory.path == namespace
        directory.verify_private_owner(confidential=True)
        assert (
            json.loads(directory.read_file("request.json"))["request_id"] == REQUEST_ID
        )
        mkdir_submission_member(directory, "claim")
        with directory.open_directory("claim", writable=True) as claim:
            publish_submission_member(claim, "binding.json", b"{}", limit=2)
            assert claim.read_file("binding.json") == b"{}"
            with pytest.raises(FileExistsError):
                publish_submission_member(claim, "binding.json", b"{}", limit=2)
        with pytest.raises(FileExistsError):
            mkdir_submission_member(directory, "claim")


def test_mock_windows_submission_publication_syncs_held_capability_not_reopened_path(
    tmp_path, monkeypatch
):
    from wavedl.core import persistence

    calls = []

    def link_file(source, target):
        calls.append("held-link")
        os.link(tmp_path / source, tmp_path / target, follow_symlinks=False)

    directory = SimpleNamespace(
        path=tmp_path,
        api=object(),
        writable=True,
        guard=lambda: calls.append("guard"),
        sync=lambda: calls.append("held-sync"),
        link_file=link_file,
    )
    monkeypatch.setattr(
        submission,
        "os",
        SimpleNamespace(
            name="nt",
            open=os.open,
            O_WRONLY=os.O_WRONLY,
            O_CREAT=os.O_CREAT,
            O_EXCL=os.O_EXCL,
            fdopen=os.fdopen,
            fsync=os.fsync,
        ),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Held ancestry cannot be independently reopened for flush")

    monkeypatch.setattr(persistence, "sync_directory", forbidden)
    monkeypatch.setattr(persistence, "write_once", forbidden)
    submission._mkdir(directory, "private")
    submission._publish(directory, "binding.json", b"verified")
    assert (tmp_path / "private").is_dir()
    assert (tmp_path / "binding.json").read_bytes() == b"verified"
    assert calls.count("held-sync") == 3
    assert calls.count("held-link") == 1
    assert not list(tmp_path.glob(".binding.json-*"))
    with pytest.raises(FileExistsError):
        submission._publish(directory, "binding.json", b"replacement")
    assert (tmp_path / "binding.json").read_bytes() == b"verified"


def test_mock_windows_readonly_submission_cannot_publish_before_mutation(tmp_path):
    directory = SimpleNamespace(
        path=tmp_path, api=object(), writable=False, guard=lambda: None
    )
    with pytest.raises(PermissionError):
        submission._mkdir(directory, "child")
    with pytest.raises(PermissionError):
        submission._publish(directory, "binding.json", b"data")
    assert list(tmp_path.iterdir()) == []


def test_local_owner_accepts_existing_urlsafe_token_shape(tmp_path):
    _layout, identity = setup_run(tmp_path)
    assert replace(identity, owner="-local_token").owner == "-local_token"
    with pytest.raises(ValueError):
        replace(identity, backend="slurm", owner="-local_token")


def test_caller_held_root_helpers_never_reopen_root(tmp_path, monkeypatch):
    layout, identity = setup_run(tmp_path)
    create_submission(layout.run_dir, identity)
    with open_directory(layout.run_dir, writable=True) as root:

        def forbidden(*_args, **_kwargs):
            pytest.fail("Submission helper reopened the held root pathname")

        monkeypatch.setattr(submission, "open_directory", forbidden)
        assert (
            identity_for_submission_directory(
                root,
                request_id=REQUEST_ID,
                backend="local",
                owner="operator",
                plan_sha256="a" * 64,
            )
            == identity
        )
        with open_submission_in_directory(root, identity, writable=True) as selected:
            publish_submission_member(selected, "binding.json", b"{}")
            assert selected.read_file("binding.json") == b"{}"
        root.guard()
        assert root.handle is not None


def test_writer_root_sharing_retry_precedes_claim_and_preserves_single_publication(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from wavedl.core import submission

    layout, identity = setup_run(tmp_path)
    original = submission.open_directory
    attempts, anchors = [], []

    def contend(path, *, writable=False):
        attempts.append(path)
        if len(attempts) == 1:
            assert not (layout.run_dir / "submissions").exists()
            error = PermissionError(13, "owned sharing contention")
            error.winerror = 32
            raise error
        anchor = original(path, writable=writable)
        anchors.append(anchor)
        return anchor

    monkeypatch.setattr(
        submission, "os", SimpleNamespace(**(vars(os) | {"name": "nt"}))
    )
    monkeypatch.setattr(submission, "open_directory", contend)
    selected = create_submission(layout.run_dir, identity)
    assert selected.joinpath("request.json").is_file()
    assert len(attempts) == 2
    assert all(anchor.handle is None for anchor in anchors)
    assert not list(layout.attempts.iterdir())
    with pytest.raises(FileExistsError):
        create_submission(layout.run_dir, identity)


@pytest.mark.parametrize("platform,code", [("nt", 5), ("posix", 32)])
def test_writer_root_does_not_retry_other_permissions_or_posix_errors(
    monkeypatch, tmp_path, platform, code
):
    import time
    from types import SimpleNamespace

    from wavedl.core import submission

    error = PermissionError(13, "not transient")
    error.winerror = code
    calls = []

    def denied(*args, **kwargs):
        calls.append(True)
        raise error

    monkeypatch.setattr(submission, "os", SimpleNamespace(name=platform))
    with pytest.raises(PermissionError) as refusal:
        submission._open_writable_root(
            tmp_path, deadline=time.monotonic() + 1, opener=denied
        )
    assert refusal.value is error and calls == [True]


def test_writer_root_deadline_retains_original_denial_without_allocating(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from wavedl.core import submission

    clock, calls = [0.0], []
    error = PermissionError(13, "held by another writer")
    error.winerror = 32

    def denied(*args, **kwargs):
        calls.append(clock[0])
        raise error

    def sleep(seconds):
        clock[0] += seconds

    monkeypatch.setattr(submission, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(
        submission, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep)
    )
    with pytest.raises(PermissionError) as refusal:
        submission._open_writable_root(
            tmp_path, deadline=0.05, poll_seconds=0.02, opener=denied
        )
    assert refusal.value is error
    assert calls == [0, 0.02, 0.04] and clock[0] == 0.05
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(TimeoutError):
        submission._open_writable_root(tmp_path, deadline=0.05, opener=denied)
    assert len(calls) == 3
