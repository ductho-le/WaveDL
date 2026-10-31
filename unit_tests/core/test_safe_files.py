"""Native-platform and mocked Win32 contracts for safe metadata consumption."""

import ctypes
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from wavedl.core import safe_files


def test_owned_stream_survives_parent_close_and_large_file(tmp_path):
    data = b"a" * (1024 * 1024 + 7)
    (tmp_path / "large").write_bytes(data)
    with safe_files.open_directory(tmp_path) as directory:
        stream = directory.open_file("large")
    with stream:
        assert stream.stat().st_size == len(data)
        assert stream.read(7) == b"a" * 7
        assert stream.seek(0, 2) == len(data)
        assert stream.read() == b""
        assert stream.seek(-7, 2) == len(data) - 7
        assert stream.read() == b"a" * 7
        stream.guard()
    stream.close()
    with pytest.raises(ValueError, match="closed"):
        stream.read()


@pytest.mark.parametrize("size", [0, -1, True, 1.5, 1024 * 1024 + 1])
def test_owned_stream_rejects_invalid_read_sizes(tmp_path, size):
    (tmp_path / "file").write_bytes(b"data")
    with (
        safe_files.open_directory(tmp_path) as directory,
        directory.open_file("file") as stream,
        pytest.raises(ValueError),
    ):
        stream.read(size)


def test_owned_stream_first_line_preserves_following_record(tmp_path):
    (tmp_path / "events").write_bytes(b"first\nsecond\n")
    with (
        safe_files.open_directory(tmp_path) as directory,
        directory.open_file("events", share_write=True) as stream,
    ):
        assert stream.read_first_line(6) == b"first\n"
        assert stream.read_first_line(7) == b"second\n"
        stream.seek(0)
        with pytest.raises(ValueError, match="exceeds"):
            stream.read_first_line(5)


@pytest.mark.skipif(os.name == "nt", reason="POSIX identity and mode contracts")
def test_owned_stream_replacement_is_anchored_but_guard_rejects(tmp_path):
    path = tmp_path / "file"
    path.write_bytes(b"original")
    with (
        safe_files.open_directory(tmp_path) as directory,
        directory.open_file("file") as stream,
    ):
        path.rename(tmp_path / "old")
        path.write_bytes(b"replacement")
        assert stream.read() == b"original"
        with pytest.raises(ValueError, match="no longer"):
            stream.guard()


@pytest.mark.skipif(os.name == "nt", reason="POSIX identity and mode contracts")
def test_owned_directory_replacement_rejects_guard(tmp_path):
    path = tmp_path / "directory"
    path.mkdir(mode=0o700)
    (path / "file").write_bytes(b"original")
    with safe_files.open_directory(path) as directory:
        stream = directory.open_file("file")
        path.rename(tmp_path / "old")
        path.mkdir(mode=0o700)
        with pytest.raises(ValueError, match="no longer"):
            directory.guard()
    with stream:
        assert stream.read() == b"original"
        with pytest.raises(ValueError):
            stream.guard()


@pytest.mark.skipif(os.name == "nt", reason="POSIX ancestor symlink contract")
def test_guard_rejects_symlink_ancestor_even_when_identity_matches(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    child = parent / "child"
    child.mkdir()
    with safe_files.open_directory(child) as directory:
        moved = tmp_path / "moved"
        parent.rename(moved)
        parent.symlink_to(moved, target_is_directory=True)
        with pytest.raises((ValueError, OSError)):
            directory.guard()


@pytest.mark.skipif(os.name == "nt", reason="POSIX ownership mode contracts")
def test_owned_objects_reject_insecure_mode_and_foreign_uid(tmp_path, monkeypatch):
    path = tmp_path / "file"
    path.write_bytes(b"data")
    path.chmod(0o600)
    with safe_files.open_directory(tmp_path) as directory:
        directory.verify_private_owner()
        with directory.open_file("file") as stream:
            stream.verify_private_owner()
            path.chmod(0o620)
            with pytest.raises(safe_files.UnsafeOwnership):
                stream.verify_private_owner()
            path.chmod(0o600)
            monkeypatch.setattr(
                safe_files.os, "getuid", lambda: stream.stat().st_uid + 1
            )
            with pytest.raises(safe_files.UnsafeOwnership):
                stream.verify_private_owner()


@pytest.mark.skipif(os.name == "nt", reason="POSIX confidential permission contract")
@pytest.mark.parametrize("mode", [0o640, 0o644, 0o610, 0o601, 0o604])
def test_confidential_file_rejects_group_world_access_from_open_handle(tmp_path, mode):
    path = tmp_path / "credential"
    path.write_bytes(b"secret")
    path.chmod(0o600)
    with (
        safe_files.open_directory(tmp_path) as directory,
        directory.open_file(path.name) as stream,
    ):
        stream.verify_private_owner(confidential=True)
        path.chmod(mode)
        stream.verify_private_owner()
        with pytest.raises(safe_files.UnsafeOwnership):
            stream.verify_private_owner(confidential=True)


def test_confidential_option_reaches_windows_handle_helper(monkeypatch):
    from wavedl.core import windows_ownership

    calls = []
    monkeypatch.setattr(
        windows_ownership,
        "verify_windows_private_owner",
        lambda api, handle, *, confidential=False: calls.append(
            (api, handle, confidential)
        ),
    )
    api = object()
    stream = safe_files.SafeFile("unused", 991, api=api)
    stream.verify_private_owner(confidential=True)
    assert calls == [(api, 991, True)]


def test_binary_read_and_byte_bound(tmp_path):
    path = tmp_path / "metadata"
    path.write_bytes(b"\x00\xff\n")
    assert safe_files.read_regular_file(path, limit=3) == b"\x00\xff\n"
    with pytest.raises(ValueError, match="bounded"):
        safe_files.read_regular_file(path, limit=2)
    if os.name == "nt":
        with pytest.raises(PermissionError) as denied:
            safe_files.read_regular_file(tmp_path)
        assert denied.value.winerror == 5
    else:
        with pytest.raises(ValueError):
            safe_files.read_regular_file(tmp_path)


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_invalid_byte_bound(tmp_path, limit):
    with pytest.raises(ValueError):
        safe_files.read_regular_file(tmp_path / "missing", limit=limit)


def test_directory_children_and_reads(tmp_path):
    (tmp_path / "child").mkdir()
    (tmp_path / "child" / "data").write_bytes(b"binary\x00")
    with safe_files.open_directory(tmp_path) as root:
        assert list(root.entries()) == [safe_files.DirectoryEntry("child", True, False)]
        child = root.open_directory("child")
    with child:
        assert child.read_file("data", limit=7) == b"binary\x00"
        with pytest.raises(ValueError):
            child.read_file("../data")
    with pytest.raises(ValueError, match="closed"):
        list(child.entries())


def test_native_symlink_rejected(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"not metadata")
    link = tmp_path / "link"
    try:
        link.symlink_to(target)
    except OSError as exc:
        if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege unavailable")
        raise
    with pytest.raises((ValueError, OSError)):
        safe_files.read_regular_file(link)
    with safe_files.open_directory(tmp_path) as root:
        entries = {entry.name: entry for entry in root.entries()}
        assert entries["link"] == safe_files.DirectoryEntry("link", False, False)
        with pytest.raises((ValueError, OSError)):
            root.read_file("link")
        with pytest.raises((ValueError, OSError)):
            root.open_directory("link")


@pytest.mark.skipif(os.name == "nt", reason="POSIX FIFO contract")
def test_fifo_rejected_without_blocking(tmp_path):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular"):
        safe_files.read_regular_file(fifo)


def _fake_win32(info=None, file_type=1):
    api = safe_files._Win32.__new__(safe_files._Win32)
    calls = []
    api.ctypes = ctypes
    api.info = lambda handle: info or SimpleNamespace(attributes=0)
    api.dll = SimpleNamespace(
        CreateFileW=lambda *args: calls.append(args) or 123,
        CloseHandle=lambda handle: calls.append(("close", handle)) or 1,
        GetFileType=lambda handle: file_type,
    )
    return api, calls


def test_mock_windows_open_contract():
    api, calls = _fake_win32()
    assert api.open(Path("metadata")) == 123
    assert calls == [("metadata", 0x80000000, 1, None, 3, 0x00200000, None)]
    api.close(123)
    assert calls[-1] == ("close", 123)


def test_mock_windows_directory_contract():
    api, calls = _fake_win32(SimpleNamespace(attributes=0x10))
    assert api.open(Path("directory"), directory=True) == 123
    assert calls[0][5] == 0x00200000 | 0x02000000
    assert calls[0][2] == 1  # No write/delete sharing: pin ancestry.


def test_mock_windows_live_journal_shares_writers():
    api, calls = _fake_win32()
    api.open(Path("journal"), share_write=True)
    assert calls[0][2] == 3  # READ | WRITE, but no DELETE.


def test_mock_windows_validation_failure_closes_handle():
    api, calls = _fake_win32(SimpleNamespace(attributes=0x10))
    with pytest.raises(ValueError):
        api.open(Path("directory"))
    assert calls[-1] == ("close", 123)


@pytest.mark.parametrize("file_type", [0, 2, 3])
def test_mock_windows_special_handle_rejected(file_type):
    api, _ = _fake_win32(file_type=file_type)
    with pytest.raises(ValueError, match="disk"):
        safe_files._Win32.info(api, 123)


@pytest.mark.parametrize("attributes", [0x400, 0x40, 0x410])
def test_mock_windows_reparse_and_device_rejected(attributes):
    api, _ = _fake_win32()
    api.Information = lambda: SimpleNamespace(attributes=attributes)
    api.ctypes = SimpleNamespace(byref=lambda info: info)
    api.dll.GetFileInformationByHandle = lambda *_: 1
    with pytest.raises(ValueError, match="Reparse"):
        safe_files._Win32.info(api, 123)


def test_mock_windows_read_failure_preserves_close_ownership(monkeypatch):
    calls = []
    api = SimpleNamespace(
        open=lambda path: 123,
        read=lambda *_: (_ for _ in ()).throw(ValueError("bound")),
        close=lambda handle: calls.append(handle),
    )
    monkeypatch.setattr(safe_files, "_windows_api", lambda: api)
    # Keep pathlib native; changing global os.name breaks Path on macOS/Linux.
    monkeypatch.setattr(safe_files, "os", SimpleNamespace(name="nt"))
    with pytest.raises(ValueError, match="bound"):
        safe_files.read_regular_file("metadata")
    assert calls == [123]


def _read_api(data, *, reported_size=None, directory=False):
    api, _ = _fake_win32()
    api.types = SimpleNamespace(DWORD=ctypes.c_uint32)
    api.info = lambda _: SimpleNamespace(
        attributes=0x10 if directory else 0,
        size_high=0,
        size_low=len(data) if reported_size is None else reported_size,
    )
    pending = bytearray(data)

    def read_file(_handle, buffer, size, count, _overlapped):
        # Exercise short reads as well as binary data and terminating EOF.
        chunk = bytes(pending[: min(size, 2)])
        del pending[: len(chunk)]
        ctypes.memmove(buffer, chunk, len(chunk))
        count._obj.value = len(chunk)
        return 1

    api.dll.ReadFile = read_file
    return api


def test_mock_windows_owned_stream_has_no_total_size_bound():
    api = _read_api(b"abcd", reported_size=1 << 31)
    stream = safe_files.SafeFile(Path("journal"), 123, api=api)
    assert stream.stat().st_size == 1 << 31
    assert stream.read(4) == b"ab"  # Native short reads are preserved.
    assert stream.read(4) == b"cd"
    assert stream.read(4) == b""
    stream.close()


def test_mock_windows_seek_uses_native_position():
    api = _read_api(b"data")
    calls = []
    api.dll.SetFilePointerEx = lambda *args: calls.append(args) or 1
    with safe_files.SafeFile(Path("journal"), 123, api=api) as stream:
        assert stream.seek(0, 2) == 4
        assert calls[-1] == (123, 4, None, 0)
        assert stream.seek(-2, 1) == 2
        with pytest.raises(ValueError):
            stream.seek(-3, 1)
        with pytest.raises(ValueError):
            stream.seek(True)


def test_mock_windows_owned_stream_closes_parent_on_file_close_failure():
    calls = []

    def fail_close(handle):
        calls.append(handle)
        raise OSError("close failure")

    stream = safe_files.SafeFile(
        Path("file"),
        123,
        api=SimpleNamespace(close=fail_close),
        parent=SimpleNamespace(close=lambda: calls.append("parent")),
    )
    with pytest.raises(OSError, match="close failure"):
        stream.close()
    assert calls == [123, "parent"]
    stream.close()
    assert calls == [123, "parent"]


def test_mock_windows_binary_short_reads():
    api = _read_api(b"\xff\x00\nabc")
    assert api.read(123, 6) == b"\xff\x00\nabc"


def test_mock_windows_read_detects_growth_after_size_check():
    api = _read_api(b"grown", reported_size=1)
    with pytest.raises(ValueError, match="exceeds"):
        api.read(123, 4)


def test_mock_windows_read_rejects_oversize_before_read():
    api = _read_api(b"huge")
    api.dll.ReadFile = lambda *_: pytest.fail("ReadFile called for oversized file")
    with pytest.raises(ValueError, match="bounded"):
        api.read(123, 3)


def test_mock_windows_read_rejects_directory_before_read():
    api = _read_api(b"", directory=True)
    api.dll.ReadFile = lambda *_: pytest.fail("ReadFile called for directory")
    with pytest.raises(ValueError, match="regular"):
        api.read(123, 3)


@pytest.mark.parametrize(
    ("data", "limit", "expected"),
    [
        (b"first\n" + b"rest" * 10000, 6, b"first\n"),
        (b"partial", 7, b"partial"),
        (b"", 0, b""),
        (b"a\nsecond", 2, b"a\n"),
    ],
    ids=["large-tail", "partial", "empty", "second-line"],
)
def test_first_line_ignores_total_file_size(tmp_path, data, limit, expected):
    path = tmp_path / "events.jsonl"
    path.write_bytes(data)
    assert safe_files.read_first_line(path, limit=limit) == expected
    api = _read_api(data, reported_size=1 << 31)
    assert api.read_first_line(123, limit) == expected


@pytest.mark.parametrize("data", [b"abcd\n", b"abcd", b"\n"])
def test_first_line_overflow_is_not_truncated(tmp_path, data):
    limit = 0 if data == b"\n" else 3
    path = tmp_path / "events.jsonl"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="record exceeds"):
        safe_files.read_first_line(path, limit=limit)
    with pytest.raises(ValueError, match="record exceeds"):
        _read_api(data).read_first_line(123, limit)


def test_first_line_missing_and_symlink(tmp_path):
    path = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        safe_files.read_first_line(path)
    target = tmp_path / "target"
    target.write_bytes(b"record\n")
    try:
        path.symlink_to(target)
    except OSError as exc:
        if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege unavailable")
        raise
    with pytest.raises((ValueError, OSError)):
        safe_files.read_first_line(path)


@pytest.mark.skipif(os.name == "nt", reason="POSIX FIFO contract")
def test_first_line_fifo_does_not_block(tmp_path):
    path = tmp_path / "fifo"
    os.mkfifo(path)
    with pytest.raises(ValueError, match="regular"):
        safe_files.read_first_line(path)


def test_first_line_can_probe_while_writer_open(tmp_path):
    path = tmp_path / "journal"
    with path.open("ab") as writer:
        writer.write(b"record\n")
        writer.flush()
        assert safe_files.read_first_line(path) == b"record\n"


def test_directory_streams_large_file_digest(tmp_path):
    data = bytes(range(256)) * 1024
    (tmp_path / "weights").write_bytes(data)
    with safe_files.open_directory(tmp_path) as anchor:
        assert anchor.digest_file("weights") == (
            len(data),
            hashlib.sha256(data).hexdigest(),
        )
        with pytest.raises(ValueError):
            anchor.digest_file("../weights")


def test_mock_windows_streaming_digest():
    api = _read_api(b"binary\x00\xffweights")
    assert b"".join(api.chunks(123)) == b"binary\x00\xffweights"


def test_mock_windows_digest_failure_closes_file_handle(tmp_path):
    closed = []
    api = SimpleNamespace(
        open=lambda _: 42,
        final_path=lambda _: tmp_path / "weights",
        chunks=lambda _: (_ for _ in ()).throw(OSError("read failed")),
        close=closed.append,
    )
    anchor = safe_files.SafeDirectory(tmp_path, 1, api=api)
    with pytest.raises(OSError, match="read failed"):
        anchor.digest_file("weights")
    assert closed == [42]
    anchor.close()
    assert closed == [42, 1]


def test_mock_windows_escaped_ancestry_closes_all_handles(monkeypatch, tmp_path):
    handles = {}
    closed = []

    def open_handle(path, *, directory=False):
        assert directory
        handle = len(handles) + 1
        handles[handle] = path
        return handle

    def final_path(handle):
        path = handles[handle]
        return Path("/escaped/member") if handle == 2 else path

    api = SimpleNamespace(open=open_handle, final_path=final_path, close=closed.append)
    monkeypatch.setattr(safe_files, "_windows_api", lambda: api)
    monkeypatch.setattr(safe_files, "os", SimpleNamespace(name="nt"))
    with pytest.raises(ValueError, match="ancestry"):
        safe_files.open_directory(tmp_path)
    assert closed == [2, 1]


def test_mock_windows_child_retains_own_ancestry(monkeypatch, tmp_path):
    handles = {}
    closed = []

    def open_handle(path, *, directory=False):
        assert directory
        handle = len(handles) + 1
        handles[handle] = path
        return handle

    api = SimpleNamespace(
        open=open_handle, final_path=handles.__getitem__, close=closed.append
    )
    monkeypatch.setattr(safe_files, "_windows_api", lambda: api)
    monkeypatch.setattr(safe_files, "os", SimpleNamespace(name="nt"))
    with safe_files.open_directory(tmp_path) as parent:
        child = parent.open_directory("child")
        child_handle = child.handle
    assert child_handle not in closed
    child.close()
    child.close()
    assert sorted(closed) == sorted(handles)


@pytest.mark.skipif(os.name != "nt", reason="Native Windows ancestry race contract")
def test_native_windows_pinned_directory_cannot_be_renamed(tmp_path):
    directory = tmp_path / "pinned"
    directory.mkdir()
    with safe_files.open_directory(directory) as anchor:
        with pytest.raises(OSError):
            directory.rename(tmp_path / "moved")
        assert list(anchor.entries()) == []
    directory.rename(tmp_path / "moved")


@pytest.mark.skipif(os.name != "nt", reason="Native Windows junction contract")
def test_native_windows_junction_rejected(tmp_path):
    import subprocess

    target = tmp_path / "target"
    target.mkdir()
    junction = tmp_path / "junction"
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(target)],
        check=True,
        capture_output=True,
    )
    with pytest.raises(ValueError, match="Reparse"):
        safe_files.open_directory(junction)


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX fsync; Windows held-handle flush is tested by mock Win32 capability",
)
def test_directory_sync_uses_held_posix_descriptor_without_path_reopen(
    tmp_path, monkeypatch
):
    with safe_files.open_directory(tmp_path, writable=True) as directory:
        descriptor = directory.handle
        calls = []
        monkeypatch.setattr(safe_files.os, "fsync", calls.append)
        directory.sync()
        assert calls == [descriptor]


def test_mock_win32_write_capability_denies_write_delete_sharing_and_flushes_held_handle():
    calls, flushed = [], []
    api = safe_files._Win32.__new__(safe_files._Win32)
    api.ctypes = ctypes
    api.dll = SimpleNamespace(
        CreateFileW=lambda *args: calls.append(args) or 77,
        FlushFileBuffers=lambda handle: flushed.append(handle) or True,
    )
    api.info = lambda handle: SimpleNamespace(attributes=0x10)
    assert api.open(Path("directory"), directory=True, writable=True) == 77
    assert calls[0][1] == 0xC0000000
    assert calls[0][2] == 1
    assert calls[0][5] & 0x02000000
    api.sync(77)
    assert flushed == [77]
    with pytest.raises(ValueError):
        api.open(Path("directory"), directory=True, writable=True, share_write=True)
    with pytest.raises(ValueError):
        api.open(Path("file"), writable=True)


def test_mock_windows_writable_child_duplicates_ancestry_without_conflicting_reopen(
    monkeypatch, tmp_path
):
    paths, live, opened, duplicated, flushed = {}, set(), [], [], []

    def opened_handle(path, *, directory=False, share_write=False, writable=False):
        handle = len(paths) + 1
        paths[handle] = path
        live.add(handle)
        opened.append((path, directory, writable, share_write))
        return handle

    def duplicate(original):
        assert original in live
        handle = len(paths) + 1
        paths[handle] = paths[original]
        live.add(handle)
        duplicated.append((original, handle))
        return handle

    def close(handle):
        live.remove(handle)

    api = SimpleNamespace(
        open=opened_handle,
        duplicate=duplicate,
        close=close,
        final_path=paths.__getitem__,
        info=lambda handle: None,
        sync=flushed.append,
    )
    monkeypatch.setattr(safe_files, "_windows_api", lambda: api)
    monkeypatch.setattr(safe_files, "os", SimpleNamespace(name="nt"))
    with safe_files.open_directory(tmp_path, writable=True) as parent:
        assert opened[-1][2] is True
        assert all(not item[2] for item in opened[:-1])
        count = len(opened)
        child = parent.open_directory("claim", writable=True)
        assert opened[count:] == [(tmp_path / "claim", True, True, False)]
        assert duplicated
        child.sync()
        stream = child.open_file("binding.json")
    assert child.handle in live
    assert stream.parent.handle in live
    child.close()
    assert stream.parent.handle in live
    stream.guard()
    stream.close()
    assert live == set()
    assert len(flushed) == 1


def test_mock_windows_readonly_directory_cannot_sync_or_escalate_existing_handle(
    tmp_path,
):
    flushed = []
    api = SimpleNamespace(sync=flushed.append, close=lambda handle: None)
    with (
        safe_files.SafeDirectory(tmp_path, 99, api=api) as directory,
        pytest.raises(PermissionError),
    ):
        directory.sync()
    assert flushed == []


def test_mock_windows_directory_flush_unsupported_fails_closed():
    api = safe_files._Win32.__new__(safe_files._Win32)
    api.ctypes = SimpleNamespace(get_last_error=lambda: 6)
    api.dll = SimpleNamespace(FlushFileBuffers=lambda handle: False)
    with pytest.raises(safe_files.UnsupportedSafeRead):
        api.sync(99)


@pytest.mark.skipif(
    os.name != "nt", reason="Native Windows writable directory publication"
)
def test_native_windows_writable_directory_remains_pinned_and_sync_is_fail_closed(
    tmp_path,
):
    with safe_files.open_directory(tmp_path, writable=True) as directory:
        assert directory.writable
        with pytest.raises(OSError):
            tmp_path.rename(tmp_path.with_name("renamed"))
        try:
            directory.sync()
        except safe_files.UnsupportedSafeRead:
            pytest.skip(
                "This native filesystem does not support durable directory flush"
            )


def test_held_directory_exclusive_link_preserves_bytes_and_existing_target(
    tmp_path, monkeypatch
):
    (tmp_path / "source").write_bytes(b"verified-content")
    (tmp_path / "existing").write_bytes(b"existing-content")
    foreign = tmp_path.with_name(tmp_path.name + "-unrelated-cwd")
    foreign.mkdir()
    with safe_files.open_directory(tmp_path, writable=True) as directory:
        monkeypatch.chdir(foreign)
        directory.link_file("source", "published")
        directory.sync()
        assert (tmp_path / "published").read_bytes() == b"verified-content"
        assert not (foreign / "published").exists()
        assert os.path.samefile(tmp_path / "source", tmp_path / "published")
        with pytest.raises(FileExistsError):
            directory.link_file("source", "existing")
        assert (tmp_path / "existing").read_bytes() == b"existing-content"
        if os.name == "nt":
            with pytest.raises(OSError):
                tmp_path.rename(tmp_path.with_name("link-parent-renamed"))


@pytest.mark.parametrize("member", ["..", "a/b", "a\\b", "C:stream", "bad\0name"])
def test_link_refuses_nonmembers_before_native_operation(tmp_path, member):
    with safe_files.open_directory(tmp_path, writable=True) as directory:
        with pytest.raises(ValueError):
            directory.link_file("source", member)
        with pytest.raises(ValueError):
            directory.link_file(member, "target")


def test_link_requires_explicit_writable_capability(tmp_path):
    with (
        safe_files.open_directory(tmp_path) as directory,
        pytest.raises(PermissionError),
    ):
        directory.link_file("source", "target")


def test_windows_link_uses_safe_source_and_held_directory_without_parent_reopen(
    monkeypatch, tmp_path
):
    calls = []
    api = SimpleNamespace(
        info=lambda handle: calls.append(("guard", handle)),
        link=lambda source, parent, target: calls.append(
            ("link", source, parent, target)
        ),
    )
    directory = safe_files.SafeDirectory(tmp_path, 456, api=api, writable=True)

    class Source:
        handle = 123

        def verify_private_owner(self):
            calls.append("source-owner")

        def guard(self):
            calls.append("source-guard")

        def __enter__(self):
            calls.append("source-enter")
            return self

        def __exit__(self, *args):
            calls.append("source-close")

    def open_source(name):
        assert name == "source"
        return Source()

    def forbidden(*args, **kwargs):
        pytest.fail("Hard-link publication must not reopen the directory path")

    monkeypatch.setattr(directory, "open_file", open_source)
    monkeypatch.setattr(safe_files, "open_directory", forbidden)
    directory.link_file("source", "target")
    assert calls == [
        ("guard", 456),
        "source-enter",
        "source-owner",
        "source-guard",
        ("link", 123, 456, "target"),
        "source-guard",
        "source-close",
        ("guard", 456),
    ]


@pytest.mark.parametrize("status,code", [(0, 0), (-1073741771, 183), (-1, 50)])
def test_native_link_buffer_uses_held_handles_utf16_and_no_replace(
    monkeypatch, status, code
):
    calls = []

    class Function:
        def __init__(self, callback):
            self.callback = callback

        def __call__(self, *args):
            return self.callback(*args)

    def native_link(source, iosb, buffer, length, kind):
        class Information(ctypes.Structure):
            _fields_: ClassVar = [
                ("replace", ctypes.c_uint32),
                ("root", ctypes.c_void_p),
                ("length", ctypes.c_uint32),
                ("name", ctypes.c_uint16 * 1),
            ]

        info = ctypes.cast(buffer, ctypes.POINTER(Information)).contents
        encoded = "target-🌊".encode("utf-16-le")
        assert source == 123 and info.root is None and kind == 11
        assert info.replace == 0 and info.length == len(encoded)
        assert length >= ctypes.sizeof(Information)
        assert (
            ctypes.string_at(
                ctypes.addressof(buffer) + Information.name.offset, info.length
            )
            == encoded
        )
        calls.append("link")
        return status

    native = SimpleNamespace(
        NtSetInformationFile=Function(native_link),
        RtlNtStatusToDosError=Function(lambda value: calls.append(value) or code),
    )
    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **kw: native, raising=False)
    monkeypatch.setattr(
        ctypes, "WinError", lambda value: OSError(value, "native error"), raising=False
    )
    api = safe_files._Win32.__new__(safe_files._Win32)
    api.ctypes = ctypes
    if status:
        expected = safe_files.UnsupportedSafeRead if code == 50 else OSError
        with pytest.raises(expected):
            api.link(123, 456, "target-🌊")
        assert calls == ["link", status]
    else:
        api.link(123, 456, "target-🌊")
        assert calls == ["link"]
    assert native.NtSetInformationFile.restype is ctypes.c_int32
    assert native.RtlNtStatusToDosError.restype is ctypes.c_uint32
