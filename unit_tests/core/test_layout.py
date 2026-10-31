import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from wavedl.core.layout import (
    OperationKind,
    RunLayout,
    allocate_attempt,
    allocate_attempt_in_directory,
)
from wavedl.core.safe_files import DirectoryEntry, SafeDirectory, open_directory


RUN_ID = "00000000-0000-4000-8000-000000000001"


def test_run_is_exclusive_and_operations_are_stable(tmp_path):
    layout = RunLayout.create(tmp_path, RUN_ID)
    assert layout.manifest == layout.run_dir / "manifest.json"
    assert [kind.value for kind in OperationKind] == ["train", "test", "plot"]
    with pytest.raises(FileExistsError):
        RunLayout.create(tmp_path, RUN_ID)


def test_allocation_collision_at_mkdir(tmp_path, monkeypatch):
    """Both writers choose 1 before either mkdir; only one can reserve it."""
    layout = RunLayout.create(tmp_path, RUN_ID)
    barrier = Barrier(2)
    original = Path.mkdir

    def synchronized(path, *args, **kwargs):
        if path == layout.attempt_dir(1):
            barrier.wait(timeout=5)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", synchronized)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(allocate_attempt, layout) for _ in range(2)]
        allocated = sorted(future.result(timeout=10) for future in futures)
    assert allocated == [1, 2]
    assert all((layout.attempt_dir(number) / "rank").is_dir() for number in allocated)


def test_existing_numeric_entries_are_never_reused(tmp_path):
    layout = RunLayout.create(tmp_path, RUN_ID)
    (layout.attempts / "2").mkdir()
    (layout.attempts / "009").mkdir()
    (layout.attempts / "11").write_text("reserved", encoding="utf-8")
    (layout.attempts / "notes").mkdir()
    assert allocate_attempt(layout) == 12


def test_requeue_allocates_second_execution_with_same_job_id(tmp_path, monkeypatch):
    layout = RunLayout.create(tmp_path, RUN_ID)
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    first = allocate_attempt(layout)
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    second = allocate_attempt(layout)
    assert (first, second) == (1, 2)
    assert layout.journal(first) != layout.journal(second)


@pytest.mark.parametrize("attempt", [-1, 0, True])
def test_invalid_attempt(tmp_path, attempt):
    layout = RunLayout.create(tmp_path, RUN_ID)
    with pytest.raises(ValueError):
        layout.attempt_dir(attempt)


def test_retained_run_capability_allocates_without_layout_path_reopen(tmp_path):
    layout = RunLayout.create(tmp_path, RUN_ID)
    (layout.attempts / "2").mkdir()
    (layout.attempts / "009").mkdir()
    (layout.attempts / "11").write_text("reserved")
    with open_directory(layout.run_dir, writable=True) as root:
        assert allocate_attempt_in_directory(root) == 12
        assert allocate_attempt_in_directory(root) == 13
    assert (layout.attempt_dir(12) / "rank").is_dir()


def test_retained_run_capability_creates_missing_attempts(tmp_path):
    run = tmp_path / "run"
    run.mkdir(mode=0o700)
    with open_directory(run, writable=True) as root:
        assert allocate_attempt_in_directory(root) == 1
    assert (run / "attempts" / "1" / "rank").is_dir()


@pytest.mark.skipif(os.name != "posix", reason="POSIX rename and directory FD gate")
def test_rename_after_last_guard_never_allocates_in_foreign_run(tmp_path, monkeypatch):
    layout = RunLayout.create(tmp_path, RUN_ID)
    original_run = tmp_path / "held-original"
    mkdir = os.mkdir
    renamed = False

    def swap_before_numeric_mkdir(name, *args, **kwargs):
        nonlocal renamed
        if name == "1" and kwargs.get("dir_fd") is not None and not renamed:
            renamed = True
            layout.run_dir.rename(original_run)
            mkdir(layout.run_dir, mode=0o700)
            mkdir(layout.run_dir / "attempts", mode=0o700)
        return mkdir(name, *args, **kwargs)

    monkeypatch.setattr(os, "mkdir", swap_before_numeric_mkdir)
    with (
        open_directory(layout.run_dir, writable=True) as root,
        pytest.raises(ValueError),
    ):
        allocate_attempt_in_directory(root)
    assert renamed
    assert list((layout.run_dir / "attempts").iterdir()) == []
    # A post-mkdir guard refuses the detached original reservation. Its number
    # is retained, not reclaimed or redirected into the replacement run.
    assert (original_run / "attempts" / "1").is_dir()
    assert not (original_run / "attempts" / "1" / "rank").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX no-follow gate")
def test_numeric_symlink_is_a_reservation_not_an_allocation_target(tmp_path):
    layout = RunLayout.create(tmp_path, RUN_ID)
    outside = tmp_path / "outside"
    outside.mkdir()
    (layout.attempts / "20").symlink_to(outside, target_is_directory=True)
    with open_directory(layout.run_dir, writable=True) as root:
        assert allocate_attempt_in_directory(root) == 21
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("name", [str(2**63 - 1), "9" * 20])
def test_capability_allocator_rejects_exhausted_or_oversized_numbers(tmp_path, name):
    layout = RunLayout.create(tmp_path, RUN_ID)
    (layout.attempts / name).mkdir()
    with (
        open_directory(layout.run_dir, writable=True) as root,
        pytest.raises(ValueError),
    ):
        allocate_attempt_in_directory(root)
    assert len(list(layout.attempts.iterdir())) == 1


@pytest.mark.skipif(os.name != "posix", reason="POSIX mkdir collision gate")
def test_capability_allocator_concurrent_collision(tmp_path, monkeypatch):
    layout = RunLayout.create(tmp_path, RUN_ID)
    barrier = Barrier(2)
    mkdir = os.mkdir

    def collide(name, *args, **kwargs):
        if name == "1" and kwargs.get("dir_fd") is not None:
            barrier.wait(timeout=5)
        return mkdir(name, *args, **kwargs)

    monkeypatch.setattr(os, "mkdir", collide)

    def participant():
        with open_directory(layout.run_dir, writable=True) as root:
            return allocate_attempt_in_directory(root)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(participant) for _ in range(2)]
        allocated = sorted(future.result(timeout=10) for future in futures)
    assert allocated == [1, 2]


def test_capability_allocator_inventory_is_bounded(tmp_path, monkeypatch):
    layout = RunLayout.create(tmp_path, RUN_ID)

    def excessive_inventory(_):
        for number in range(4097):
            yield DirectoryEntry(f"note-{number}", False, True)

    monkeypatch.setattr(SafeDirectory, "entries", excessive_inventory)
    with (
        open_directory(layout.run_dir, writable=True) as root,
        pytest.raises(ValueError, match="inventory exceeds"),
    ):
        allocate_attempt_in_directory(root)
    assert list(layout.attempts.iterdir()) == []


def test_read_only_capability_cannot_allocate(tmp_path):
    layout = RunLayout.create(tmp_path, RUN_ID)
    with open_directory(layout.run_dir) as root, pytest.raises(PermissionError):
        allocate_attempt_in_directory(root)
    assert list(layout.attempts.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory FD contention gate")
def test_capability_allocator_contention_is_bounded(tmp_path, monkeypatch):
    layout = RunLayout.create(tmp_path, RUN_ID)
    mkdir = os.mkdir
    count = 0

    def always_occupied(name, *args, **kwargs):
        nonlocal count
        if (
            isinstance(name, str)
            and name.isdecimal()
            and kwargs.get("dir_fd") is not None
        ):
            count += 1
            raise FileExistsError("simulated concurrent reservation")
        return mkdir(name, *args, **kwargs)

    monkeypatch.setattr(os, "mkdir", always_occupied)
    with (
        open_directory(layout.run_dir, writable=True) as root,
        pytest.raises(ValueError, match="contention exceeds"),
    ):
        allocate_attempt_in_directory(root)
    assert count == 4096
    assert list(layout.attempts.iterdir()) == []
