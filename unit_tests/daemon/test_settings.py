import os
import subprocess
import sys
from pathlib import Path

import pytest

from wavedl_daemon.storage import filesystems, settings as settings_module
from wavedl_daemon.storage.settings import (
    Settings,
    UnsupportedStorage,
    default_run_dir,
    default_state_dir,
)


def roots(tmp_path):
    return {"state_root": tmp_path / "state", "run_root": tmp_path / "runs"}


def test_default_state_dir_is_absolute_and_independent_of_cwd(tmp_path, monkeypatch):
    """The GUI resolved AppData relative to cwd; application data moved with it."""
    before = default_state_dir()
    monkeypatch.chdir(tmp_path)

    assert default_state_dir() == before
    assert before.is_absolute()
    assert Path("AppData").absolute() != before


def test_filesystem_type_is_answered_for_a_real_path():
    """The per-platform mechanism actually works on this machine."""
    assert filesystems.filesystem_type(Path.home())
    assert filesystems.is_network_filesystem(Path.home()) is False


def test_detection_does_not_depend_on_psutil():
    """psutil is not a declared dependency; detection must not reach for it."""
    script = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"psutil", "PySide6", "torch", "numpy"}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from pathlib import Path
from wavedl_daemon.storage.filesystems import filesystem_type
assert filesystem_type(Path.home())
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        env=os.environ | {"PYTHONPATH": str(Path("src").resolve())},
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr


def test_filesystem_type_walks_up_to_an_existing_ancestor(tmp_path):
    """Roots are validated before they exist."""
    missing = tmp_path / "not" / "created" / "yet"

    assert filesystems.filesystem_type(missing) == filesystems.filesystem_type(tmp_path)


def test_network_state_root_is_refused(tmp_path, monkeypatch):
    """SQLite and flock are both unreliable over the network."""
    paths = roots(tmp_path)
    monkeypatch.setattr(
        settings_module,
        "is_network_filesystem",
        lambda path: path == paths["state_root"],
    )

    with pytest.raises(UnsupportedStorage, match="network filesystem"):
        Settings.resolved(**paths)


def test_network_run_root_is_accepted(tmp_path, monkeypatch):
    """The run root has the opposite requirement: shared storage is the point."""
    paths = roots(tmp_path)
    monkeypatch.setattr(
        settings_module, "is_network_filesystem", lambda path: path == paths["run_root"]
    )

    resolved = Settings.resolved(**paths, shared_execution=True)

    assert resolved.run_root == paths["run_root"]


def test_undetectable_filesystem_is_not_treated_as_a_failure(tmp_path, monkeypatch):
    """None means "the platform will not say", which is not grounds to refuse."""
    monkeypatch.setattr(settings_module, "is_network_filesystem", lambda path: None)
    monkeypatch.setattr(settings_module, "is_ephemeral_filesystem", lambda path: None)

    assert Settings.resolved(**roots(tmp_path), shared_execution=True)


def test_unwritable_root_is_refused(tmp_path):
    """Checked by writing, not by os.access, which lies for root and on ACLs."""
    if os.name == "nt":
        pytest.skip("POSIX mode-bit permissions require a POSIX filesystem")
    if os.geteuid() == 0:
        pytest.skip("root bypasses the permission bits this test relies on")
    locked = tmp_path / "locked"
    locked.mkdir(mode=0o500)

    with pytest.raises(UnsupportedStorage, match=r"not writable|not creatable"):
        Settings.resolved(state_root=locked / "state", run_root=tmp_path / "runs")


def test_shared_execution_refuses_a_run_root_inside_the_state_root(tmp_path):
    """The default layout is local-only; a cluster forces an explicit choice."""
    state = tmp_path / "state"

    with pytest.raises(UnsupportedStorage, match="inside the state root"):
        Settings.resolved(
            state_root=state, run_root=state / "runs", shared_execution=True
        )


def test_shared_execution_refuses_an_ephemeral_run_root(tmp_path, monkeypatch):
    """tmpfs cannot be visible to another machine however the cluster is set up."""
    monkeypatch.setattr(settings_module, "is_ephemeral_filesystem", lambda path: True)

    with pytest.raises(UnsupportedStorage, match="ephemeral"):
        Settings.resolved(**roots(tmp_path), shared_execution=True)


def test_local_execution_accepts_the_nested_default_layout(tmp_path):
    """Without shared execution the defaults are fine, including runs/ nesting."""
    state = tmp_path / "state"

    resolved = Settings.resolved(state_root=state, run_root=state / "runs")

    assert resolved.run_root.is_relative_to(resolved.state_root)
    assert default_run_dir().is_relative_to(default_state_dir())


def test_relative_root_is_refused(tmp_path):
    with pytest.raises(UnsupportedStorage, match="must be absolute"):
        Settings(state_root=Path("state"), run_root=tmp_path / "runs")


def test_identical_roots_are_refused(tmp_path):
    with pytest.raises(UnsupportedStorage, match="distinct"):
        Settings.resolved(state_root=tmp_path / "one", run_root=tmp_path / "one")


def test_symlink_alias_cannot_bypass_root_separation(tmp_path):
    actual = tmp_path / "state"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(UnsupportedStorage, match="distinct"):
        Settings.resolved(state_root=actual, run_root=alias)
    with pytest.raises(UnsupportedStorage, match="inside the state root"):
        Settings.resolved(
            state_root=actual, run_root=alias / "runs", shared_execution=True
        )


def test_normalized_parent_path_cannot_bypass_separation(tmp_path):
    with pytest.raises(UnsupportedStorage, match="distinct"):
        Settings.resolved(
            state_root=tmp_path / "state",
            run_root=tmp_path / "another" / ".." / "state",
        )


def test_linux_mountinfo_unescapes_names_and_resolves_symlinks(tmp_path, monkeypatch):
    import builtins
    import io

    actual = tmp_path / "shared runs"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    mount = str(actual).replace(" ", r"\040")
    mountinfo = f"1 0 0:1 / / rw - ext4 /dev/root rw\n2 1 0:2 / {mount} rw - nfs server:/runs rw\n"
    monkeypatch.setattr(builtins, "open", lambda *a, **kw: io.StringIO(mountinfo))
    monkeypatch.setattr(filesystems.sys, "platform", "linux")
    assert filesystems.filesystem_type(alias / "not-created") == "nfs"
    assert filesystems.is_network_filesystem(alias) is True


def test_overridden_state_root_moves_default_run_root(tmp_path):
    state = tmp_path / "state"
    resolved = Settings.resolved(state_root=state)
    assert resolved.run_root == state / "runs"


def test_relative_environment_default_is_not_cwd_dependent(monkeypatch):
    monkeypatch.setattr(settings_module.sys, "platform", "linux")
    monkeypatch.setenv("XDG_STATE_HOME", "relative")
    assert default_state_dir() == Path.home() / ".local/state/wavedl"
