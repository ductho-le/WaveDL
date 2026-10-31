"""Test-process ownership setup restores native state without weakening policy."""

import ctypes
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "user_owned_pytest", ROOT / "tools/run_user_owned_pytest.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OwnerAPI:
    def __init__(self, failure=None):
        self.owner = 100
        self.events = []
        self.failure = failure

    def open(self):
        self.events.append("open")
        if self.failure == "open":
            raise OSError("open failed")
        return 700

    def query(self, token, kind):
        assert token == 700
        self.events.append(("query", kind))
        if self.failure == "query" and len(self.events) == 2:
            raise OSError("query failed")
        return object(), self.owner if kind == 4 else 200

    def set_owner(self, token, owner):
        assert token == 700
        self.events.append(("set", owner))
        if self.failure == "set" and owner == 200:
            raise OSError("set failed")
        if self.failure == "restore" and owner == 100:
            raise OSError("restore failed")
        if self.failure != "verify":
            self.owner = owner

    def equal(self, left, right):
        return left == right

    def close(self, token):
        assert token == 700
        self.events.append("close")
        if self.failure == "close":
            raise OSError("close failed")


def test_owner_context_sets_actual_user_then_restores_even_after_body_failure():
    api = OwnerAPI()
    with (
        pytest.raises(ValueError, match="pytest failed"),
        MODULE.user_owned_objects(platform="nt", api_factory=lambda: api),
    ):
        assert api.owner == 200
        raise ValueError("pytest failed")
    assert api.owner == 100
    assert api.events == [
        "open",
        ("query", 4),
        ("query", 1),
        ("set", 200),
        ("query", 4),
        ("set", 100),
        ("query", 4),
        "close",
    ]


@pytest.mark.parametrize(
    "failure", ["open", "query", "set", "verify", "restore", "close"]
)
def test_setup_verification_restore_and_close_errors_never_silently_pass(failure):
    api = OwnerAPI(failure)
    with (
        pytest.raises((OSError, RuntimeError)),
        MODULE.user_owned_objects(platform="nt", api_factory=lambda: api),
    ):
        pass
    if failure != "open":
        assert api.events[-1] == "close"
    if failure in {"set", "verify"}:
        assert ("set", 100) in api.events


def test_posix_context_never_loads_native_bindings():
    def forbidden():
        pytest.fail("POSIX must not load Win32 bindings")

    with MODULE.user_owned_objects(platform="posix", api_factory=forbidden):
        pass


def test_private_workspace_protects_only_fresh_root_then_removes_it():
    protected = []

    class PrivateAPI:
        def protect_directory(self, root):
            assert root.name.startswith("wavedl-user-pytest-")
            assert (
                root.is_dir() and not root.is_symlink() and list(root.iterdir()) == []
            )
            protected.append(root)

    with MODULE.private_test_workspace(platform="nt", api_factory=PrivateAPI) as root:
        assert protected == [root]
        (root / "pytest").mkdir()
        (root / "pytest/owned").write_bytes(b"owned")
    assert not root.exists()
    assert root.parent.is_dir()


def test_private_sddl_is_protected_inheritable_and_has_only_trusted_principals():
    assert MODULE.private_sddl("S-1-5-21-123") == (
        "D:P(A;OICI;FA;;;S-1-5-21-123)(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)"
    )
    with pytest.raises(ValueError):
        MODULE.private_sddl("S-1-5)(A;;FA;;;WD)")


@pytest.mark.parametrize("set_error", [0, 5])
def test_typed_private_dacl_binding_targets_only_selected_root_and_frees_buffers(
    set_error,
):
    from ctypes import wintypes

    calls = []
    sid_memory = ctypes.create_unicode_buffer("S-1-5-21-123")

    class Function:
        def __init__(self, callback):
            self.callback = callback

        def __call__(self, *args):
            return self.callback(*args)

    def string_sid(sid, out):
        ctypes.cast(out, ctypes.POINTER(wintypes.LPWSTR))[0] = ctypes.cast(
            sid_memory, wintypes.LPWSTR
        )
        return 1

    def descriptor(sddl, revision, out, size):
        calls.append(("sddl", sddl, revision))
        ctypes.cast(out, ctypes.POINTER(wintypes.LPVOID))[0] = 222
        return 1

    def dacl(sd, present, out, defaulted):
        ctypes.cast(present, ctypes.POINTER(wintypes.BOOL))[0] = 1
        ctypes.cast(out, ctypes.POINTER(wintypes.LPVOID))[0] = 333
        return 1

    def set_security(path, kind, flags, owner, group, acl, sacl):
        calls.append(("target", path, kind, flags, owner, group, acl.value, sacl))
        return set_error

    api = MODULE.TokenOwnerAPI.__new__(MODULE.TokenOwnerAPI)
    api.c = SimpleNamespace(
        byref=ctypes.byref,
        cast=ctypes.cast,
        POINTER=ctypes.POINTER,
        c_int=ctypes.c_int,
        WinError=lambda code: OSError(code, "native failure"),
    )
    api.t = wintypes
    api.kernel = SimpleNamespace(
        LocalFree=Function(lambda p: calls.append(("free", p.value)) and None)
    )
    api.security = SimpleNamespace(
        ConvertSidToStringSidW=Function(string_sid),
        ConvertStringSecurityDescriptorToSecurityDescriptorW=Function(descriptor),
        GetSecurityDescriptorDacl=Function(dacl),
        SetNamedSecurityInfoW=Function(set_security),
    )
    api.open = lambda: 700
    api.query = lambda token, kind: (object(), 111)
    api.close = lambda token: calls.append(("close", token))
    if set_error:
        with pytest.raises(OSError):
            api.protect_directory(Path("fresh-owned-test-root"))
    else:
        api.protect_directory(Path("fresh-owned-test-root"))
    assert calls[0] == ("sddl", MODULE.private_sddl("S-1-5-21-123"), 1)
    assert calls[1] == (
        "target",
        "fresh-owned-test-root",
        1,
        0x80000004,
        None,
        None,
        333,
        None,
    )
    assert calls[2][0] == "free" and calls[2][1] == 222
    assert calls[3][0] == "free"
    assert calls[-1] == ("close", 700)


def test_wrapper_enters_owner_context_before_pytest_main_or_temp_allocation(
    monkeypatch,
):
    monkeypatch.setattr(MODULE, "os", SimpleNamespace(name="posix", getcwd=os.getcwd))
    api = OwnerAPI()
    original_path = sys.path.copy()
    original = MODULE.user_owned_objects
    monkeypatch.setattr(
        MODULE,
        "user_owned_objects",
        lambda: original(platform="nt", api_factory=lambda: api),
    )

    def pytest_main(args):
        assert args == ["unit_tests/core", "--confcutdir=unit_tests/core"]
        assert api.owner == 200
        assert api.events[-1] == ("query", 4)
        assert sys.path[0] == os.getcwd()
        import unit_tests.daemon

        assert unit_tests.daemon.__path__
        return 7

    monkeypatch.setitem(sys.modules, "pytest", SimpleNamespace(main=pytest_main))
    assert MODULE.main(["unit_tests/core", "--confcutdir=unit_tests/core"]) == 7
    assert api.owner == 100 and api.events[-1] == "close"
    assert sys.path == original_path


def test_wrapper_restores_import_path_and_owner_after_pytest_failure(monkeypatch):
    monkeypatch.setattr(MODULE, "os", SimpleNamespace(name="posix", getcwd=os.getcwd))
    api = OwnerAPI()
    original_path = sys.path.copy()
    original = MODULE.user_owned_objects
    monkeypatch.setattr(
        MODULE,
        "user_owned_objects",
        lambda: original(platform="nt", api_factory=lambda: api),
    )

    def pytest_main(args):
        assert sys.path[0] == os.getcwd() and api.owner == 200
        raise RuntimeError("pytest failure")

    monkeypatch.setitem(sys.modules, "pytest", SimpleNamespace(main=pytest_main))
    with pytest.raises(RuntimeError, match="pytest failure"):
        MODULE.main([])
    assert sys.path == original_path
    assert api.owner == 100 and api.events[-1] == "close"


def test_windows_wrapper_forces_fresh_protected_basetemp_before_pytest(monkeypatch):
    api = OwnerAPI()
    original_owner, original_workspace = (
        MODULE.user_owned_objects,
        MODULE.private_test_workspace,
    )
    roots = []

    class PrivateAPI:
        def protect_directory(self, root):
            assert api.owner == 200
            roots.append(root)

    monkeypatch.setattr(MODULE, "os", SimpleNamespace(name="nt", getcwd=os.getcwd))
    monkeypatch.setattr(
        MODULE,
        "user_owned_objects",
        lambda: original_owner(platform="nt", api_factory=lambda: api),
    )
    monkeypatch.setattr(
        MODULE,
        "private_test_workspace",
        lambda: original_workspace(platform="nt", api_factory=PrivateAPI),
    )

    def pytest_main(args):
        assert args == ["unit_tests/core", "--basetemp", str(roots[0] / "pytest")]
        assert roots[0].is_dir() and api.owner == 200
        return 9

    monkeypatch.setitem(sys.modules, "pytest", SimpleNamespace(main=pytest_main))
    assert MODULE.main(["unit_tests/core"]) == 9
    assert not roots[0].exists() and api.owner == 100
    with pytest.raises(ValueError, match="owns the Windows pytest"):
        MODULE.main(["--basetemp=existing-user-folder"])


def test_typed_native_bindings_query_set_compare_and_close(monkeypatch):
    from ctypes import wintypes

    calls = []
    state = {"owner": 100}

    class Function:
        def __init__(self, callback):
            self.callback = callback

        def __call__(self, *args):
            return self.callback(*args)

    def open_token(process, access, out):
        calls.append(("open", process, access))
        ctypes.cast(out, ctypes.POINTER(wintypes.HANDLE))[0] = 700
        return 1

    def query(token, kind, buffer, size, required):
        ctypes.cast(required, ctypes.POINTER(wintypes.DWORD))[0] = ctypes.sizeof(
            ctypes.c_void_p
        )
        if buffer is None:
            return 0
        ctypes.cast(buffer, ctypes.POINTER(wintypes.LPVOID))[0] = (
            state["owner"] if kind == 4 else 200
        )
        return 1

    def set_information(token, kind, pointer, size):
        assert token.value == 700 and kind == 4
        assert size == ctypes.sizeof(ctypes.c_void_p)
        state["owner"] = ctypes.cast(pointer, ctypes.POINTER(wintypes.LPVOID))[0]
        calls.append(("set", state["owner"]))
        return 1

    kernel = SimpleNamespace(
        GetCurrentProcess=Function(lambda: 999),
        CloseHandle=Function(lambda token: calls.append(("close", token.value)) or 1),
    )
    security = SimpleNamespace(
        OpenProcessToken=Function(open_token),
        GetTokenInformation=Function(query),
        SetTokenInformation=Function(set_information),
        EqualSid=Function(lambda left, right: int(left == right)),
    )
    monkeypatch.setattr(
        ctypes,
        "WinDLL",
        lambda name, **kwargs: kernel if name == "kernel32" else security,
        raising=False,
    )
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 122, raising=False)
    monkeypatch.setattr(
        ctypes, "WinError", lambda code: OSError(code, "native failure"), raising=False
    )
    with MODULE.user_owned_objects(platform="nt"):
        assert state["owner"] == 200
    assert state["owner"] == 100
    assert calls == [("open", 999, 0x88), ("set", 200), ("set", 100), ("close", 700)]


def test_foundations_core_and_daemon_use_owned_wrapper_without_weakening_gates():
    value = yaml.safe_load(
        (ROOT / ".github/workflows/daemon-foundations.yml").read_text()
    )
    foundation = value["jobs"]["foundations"]
    steps = foundation["steps"]
    for name, folder, report in (
        ("Shared contracts without parent ML fixtures", "core", "core-foundations.xml"),
        ("Native supervision and durable control", "daemon", "daemon-foundations.xml"),
    ):
        step = next(step for step in steps if step.get("name") == name)
        assert step["run"] == (
            f"python tools/run_user_owned_pytest.py unit_tests/{folder} "
            f"--confcutdir=unit_tests/{folder} --junitxml={report} -q"
        )
        assert not step.get("continue-on-error", False)
        if folder == "daemon":
            assert step["if"] == "${{ !cancelled() }}"
    guard = next(
        step
        for step in steps
        if step.get("name") == "Require native foundations acceptance reports"
    )
    assert guard["if"] == "always()"
    assert foundation["timeout-minutes"] == 15
    verification = next(
        step
        for step in steps
        if step.get("name")
        == "Verify scoped test-process ownership setup and native restoration"
    )
    assert verification["run"] == (
        "python -m pytest unit_tests/test_user_owned_pytest.py --noconftest -q"
    )


@pytest.mark.skipif(os.name != "nt", reason="Native TokenOwner restoration witness")
def test_native_owner_context_creates_user_owned_file_and_restores_token(tmp_path):
    from wavedl.core.safe_files import open_directory

    api = MODULE.TokenOwnerAPI()
    token = api.open()
    try:
        _before_buffer, before = api.query(token, 4)
        with (
            MODULE.user_owned_objects(),
            MODULE.private_test_workspace() as workspace,
        ):
            _user_buffer, user = api.query(token, 1)
            _current_buffer, current = api.query(token, 4)
            assert api.equal(current, user)
            (workspace / "native-user-owned").write_bytes(b"native ownership")
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(b'child ownership')",
                    str(workspace / "native-child-owned"),
                ],
                check=True,
                timeout=10,
            )
            with (
                open_directory(workspace) as root,
                root.open_file("native-user-owned") as file,
            ):
                file.verify_private_owner()
                root.verify_private_owner()
            with (
                open_directory(workspace) as root,
                root.open_file("native-child-owned") as file,
            ):
                file.verify_private_owner()
        _after_buffer, after = api.query(token, 4)
        assert api.equal(before, after)
    finally:
        api.close(token)
