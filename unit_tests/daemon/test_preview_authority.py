"""Dataset authority checks without importing the ML stack."""

import os
from pathlib import Path

import pytest

from wavedl_daemon.runs.preview_authority import (
    PreviewAuthority,
    PreviewChanged,
    PreviewForbidden,
    PreviewMissing,
    verify_selection,
)


@pytest.fixture
def authority(tmp_path):
    root = tmp_path.resolve() / "datasets"
    root.mkdir()
    (root / "nested").mkdir()
    (root / "nested" / "data.npz").write_bytes(b"dataset")
    return PreviewAuthority({"datasets": root})


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/data.npz",
        "../data.npz",
        "./data.npz",
        "a//data.npz",
        "a/../data.npz",
        "a\\data.npz",
        "C:data.npz",
        "data.npz\0",
        "\ud800.npz",
        "\udfff.npz",
        "data.txt",
    ],
)
def test_rejects_nonordinary_relative_paths(authority, path):
    with pytest.raises(PreviewForbidden):
        authority.resolve_selection("datasets", path)


@pytest.mark.parametrize("root", [None, "unknown", 1, "../datasets"])
def test_rejects_unknown_root(authority, root):
    with pytest.raises(PreviewForbidden):
        authority.resolve_selection(root, "nested/data.npz")


def test_selection_is_absolute_and_matches_native_fingerprint(authority):
    selected = authority.resolve_selection("datasets", "nested/data.npz")
    info = selected.path.stat()
    assert selected.path.is_absolute()
    expected = {
        "st_dev": info.st_dev,
        "st_ino": info.st_ino,
        "st_size": info.st_size,
        "st_mtime_ns": info.st_mtime_ns,
    }
    assert selected.identity == expected
    assert selected.path.read_bytes() == b"dataset"
    assert len(selected.ancestry) == 2
    verify_selection(selected)


def test_public_readonly_dataset_is_authorized(authority):
    root = authority.roots["datasets"]
    (root / "nested" / "data.npz").chmod(0o444)
    assert (
        authority.resolve_selection("datasets", "nested/data.npz").identity["st_size"]
        == 7
    )


def test_missing_file(authority):
    with pytest.raises(PreviewMissing):
        authority.resolve_selection("datasets", "missing.npz")


def test_directory_is_not_a_dataset(authority):
    (authority.roots["datasets"] / "directory.npz").mkdir()
    expected = PreviewForbidden if os.name == "nt" else PreviewChanged
    with pytest.raises(expected):
        authority.resolve_selection("datasets", "directory.npz")


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink/fifo semantics")
def test_rejects_an_escaping_link_and_a_fifo(authority, tmp_path):
    root = authority.roots["datasets"]
    foreign = tmp_path / "foreign.npz"
    foreign.write_bytes(b"foreign")
    (root / "link.npz").symlink_to(foreign)
    with pytest.raises(PreviewForbidden):
        authority.resolve_selection("datasets", "link.npz")
    os.mkfifo(root / "pipe.npz")
    with pytest.raises(PreviewChanged):
        authority.resolve_selection("datasets", "pipe.npz")


def test_root_rename_cannot_redirect_held_selection(authority):
    root = authority.roots["datasets"]
    if os.name == "nt":
        with authority.open_preview_selection(
            "datasets", "nested/data.npz"
        ) as selected:
            with pytest.raises(PermissionError) as denied:
                root.rename(root.with_name("old-datasets"))
            assert denied.value.winerror in (5, 32)
            assert not root.with_name("old-datasets").exists()
            assert selected.path.read_bytes() == b"dataset"
            verify_selection(selected)
        return
    with (
        pytest.raises(PreviewChanged),
        authority.open_preview_selection("datasets", "nested/data.npz") as selected,
    ):
        root.rename(root.with_name("old-datasets"))
        root.mkdir()
        (root / "nested").mkdir()
        (root / "nested" / "data.npz").write_bytes(b"foreign")
        with pytest.raises(PreviewChanged):
            verify_selection(selected)


def test_modified_file_fingerprint_rejected(authority):
    selected = authority.resolve_selection("datasets", "nested/data.npz")
    selected.path.write_bytes(b"modified data")
    with pytest.raises(PreviewChanged):
        verify_selection(selected)


def test_body_exception_preserved_without_reclassification(authority):
    refusal = ValueError("Malformed dataset")
    with (
        pytest.raises(ValueError) as caught,
        authority.open_preview_selection("datasets", "nested/data.npz"),
    ):
        raise refusal
    assert caught.value is refusal


def test_mutated_dataset_rejected_on_context_exit(authority):
    with (
        pytest.raises(PreviewChanged),
        authority.open_preview_selection("datasets", "nested/data.npz") as selected,
    ):
        selected.path.write_bytes(b"changed")


def test_empty_dataset_has_metadata(authority):
    root = authority.roots["datasets"]
    (root / "empty.npz").touch()
    assert authority.resolve_selection("datasets", "empty.npz").identity["st_size"] == 0


def test_rejects_relative_configured_roots():
    with pytest.raises(ValueError):
        PreviewAuthority({"datasets": Path("relative")})
