"""Pure policy and lifetime checks; actual Win32 checks require a Windows runner."""

import ctypes
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from wavedl.core.safe_files import UnsafeOwnership, open_directory
from wavedl.core.windows_ownership import (
    _Ace,
    _check_acl,
    _SecurityAPI,
    _verify_with_security_api,
)


@pytest.mark.parametrize(
    "mask",
    [2, 4, 16, 64, 256, 0x10000, 0x40000, 0x80000, 0x02000000, 0x10000000, 0x40000000],
)
def test_arbitrary_principals_may_not_write_delete_or_change_control(mask):
    with pytest.raises(UnsafeOwnership):
        _check_acl(True, [_Ace(0, mask, False)])


@pytest.mark.parametrize("owner,privileged", [(True, False), (False, True)])
def test_owner_and_explicit_system_admin_principals_are_trusted(owner, privileged):
    _check_acl(True, [_Ace(0, 0x10000000, owner, privileged)])


def test_arbitrary_principals_read_access_and_deny_aces_are_safe():
    _check_acl(True, [_Ace(0, 0x80020089, False), _Ace(1, 0x10000000, False)])
    _check_acl(True, [])


@pytest.mark.parametrize("mask", [1, 8, 32, 128, 0x20000, 0x20000000, 0x80000000])
def test_confidential_files_reject_untrusted_read_and_execute_aces(mask):
    _check_acl(True, [_Ace(0, mask, False)])
    with pytest.raises(UnsafeOwnership):
        _check_acl(True, [_Ace(0, mask, False)], confidential=True)
    _check_acl(True, [_Ace(0, mask, True)], confidential=True)
    _check_acl(True, [_Ace(0, mask, False, True)], confidential=True)
    _check_acl(True, [_Ace(1, mask, False)], confidential=True)


def test_confidential_rejection_keeps_descriptor_and_token_cleanup():
    api = FakeSecurity(entries=[_Ace(0, 1, False)])
    with pytest.raises(UnsafeOwnership):
        _verify_with_security_api(api, 713, confidential=True)
    assert api.events[-2:] == ["local-free", "token-close"]


def test_foreign_owner_and_null_dacl_are_rejected():
    for owner, entries in [
        (False, []),
        (False, [_Ace(0, 0x10000000, True)]),
        (True, None),
    ]:
        with pytest.raises(UnsafeOwnership):
            _check_acl(owner, entries)


@pytest.mark.parametrize("kind", [2, 3, 4, 5, 6, 9, 10, 17, 255])
def test_unsupported_ace_layouts_fail_closed_even_if_no_write(kind):
    with pytest.raises(UnsafeOwnership):
        _check_acl(True, [_Ace(kind, 0, True)])


def test_deny_ace_does_not_hide_an_unsafe_allow_ace():
    with pytest.raises(UnsafeOwnership):
        _check_acl(True, [_Ace(1, 0x10000000, False), _Ace(0, 2, False)])


class FakeSecurity:
    def __init__(self, *, failure=None, owner="current", entries=None):
        self.failure, self.owner = failure, owner
        self.acl = [] if entries is None else entries
        self.events = []

    @contextmanager
    def token_user(self):
        self.events.append("token-open")
        try:
            if self.failure == "token":
                raise OSError("Token query failed")
            yield "current"
        finally:
            self.events.append("token-close")

    @contextmanager
    def descriptor(self, handle):
        self.events.append(("descriptor", handle))
        try:
            if self.failure == "descriptor":
                raise OSError("Descriptor query failed")
            yield self.owner, "dacl"
        finally:
            self.events.append("local-free")

    def same_sid(self, first, second):
        return first == second

    def entries(self, dacl, owner):
        if self.failure == "acl":
            raise OSError("ACE query failed")
        return self.acl


@pytest.mark.parametrize("failure", [None, "token", "descriptor", "acl"])
def test_query_lifetimes_release_only_owned_resources_on_every_path(failure):
    api = FakeSecurity(failure=failure)
    if failure:
        with pytest.raises(OSError):
            _verify_with_security_api(api, 712)
    else:
        _verify_with_security_api(api, 712)
    assert api.events[-1] == "token-close"
    if failure != "token":
        assert api.events[-2] == "local-free"
        assert ("descriptor", 712) in api.events
    assert "file-close" not in api.events


def test_security_rejection_releases_descriptor_and_token():
    for api in [FakeSecurity(owner="other"), FakeSecurity(entries=[_Ace(0, 2, False)])]:
        with pytest.raises(UnsafeOwnership):
            _verify_with_security_api(api, 123)
        assert api.events[-2:] == ["local-free", "token-close"]


class Function:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


def native_binding_mock(failure):
    """Fixed-width Win32 types exercise actual binding code on non-Windows too."""
    events = []
    types = SimpleNamespace(
        LPVOID=ctypes.c_void_p,
        HANDLE=ctypes.c_void_p,
        DWORD=ctypes.c_uint32,
        BOOL=ctypes.c_int32,
        BYTE=ctypes.c_uint8,
        WORD=ctypes.c_uint16,
    )
    created = {}

    def set_pointer(output, value):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p)).contents.value = value

    def open_token(process, access, output):
        assert process == 99 and access == 8
        if failure == "open-token":
            return False
        set_pointer(output, 101)
        return True

    def token_info(token, kind, buffer, capacity, returned):
        assert token.value == 101 and kind == 1
        ctypes.cast(returned, ctypes.POINTER(types.DWORD)).contents.value = 32
        if buffer is None or failure == "token-info":
            return False
        ctypes.cast(
            buffer, ctypes.POINTER(created["api"].TokenUser)
        ).contents.sid = 1234
        return True

    def security_info(handle, kind, flags, owner, group, dacl, sacl, descriptor):
        assert handle == 700 and kind == 1 and flags == 5
        set_pointer(descriptor, 8888)
        set_pointer(owner, 4321 if failure == "foreign-owner" else 1234)
        set_pointer(dacl, 0 if failure == "null-dacl" else 9876)
        return 5 if failure == "security-info" else 0

    def acl_info(dacl, output, size, kind):
        assert kind == 2
        if failure == "acl-info":
            return False
        ctypes.cast(output, ctypes.POINTER(created["api"].AclSize)).contents.count = 0
        return True

    def scalar(value):
        return value.value if isinstance(value, ctypes.c_void_p) else value

    advapi = SimpleNamespace(
        **{
            name: Function(callback)
            for name, callback in {
                "OpenProcessToken": open_token,
                "GetTokenInformation": token_info,
                "GetSecurityInfo": security_info,
                "EqualSid": lambda first, second: scalar(first) == scalar(second),
                "IsValidSid": lambda sid: failure != "invalid-sid",
                "GetLengthSid": lambda sid: 12,
                "IsWellKnownSid": lambda sid, kind: False,
                "IsValidAcl": lambda acl: failure != "invalid-acl",
                "GetAclInformation": acl_info,
                "GetAce": lambda *args: False,
            }.items()
        }
    )
    kernel = SimpleNamespace(
        GetCurrentProcess=Function(lambda: 99),
        CloseHandle=Function(
            lambda handle: events.append(("close", handle.value)) or failure != "close"
        ),
        LocalFree=Function(
            lambda pointer: (
                events.append(("free", pointer.value))
                or (pointer.value if failure == "free" else None)
            )
        ),
    )
    facade = SimpleNamespace(
        **{
            name: getattr(ctypes, name)
            for name in (
                "POINTER",
                "byref",
                "cast",
                "c_int",
                "sizeof",
                "Structure",
                "create_string_buffer",
            )
        }
    )
    facade.WinDLL = lambda name, **options: advapi
    facade.get_last_error = lambda: 122
    facade.WinError = lambda code: OSError(code, "Mock Win32 query failure")
    api = _SecurityAPI(SimpleNamespace(ctypes=facade, types=types, dll=kernel))
    created["api"] = api
    return api, events


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "open-token",
        "token-info",
        "security-info",
        "acl-info",
        "invalid-sid",
        "invalid-acl",
        "foreign-owner",
        "null-dacl",
        "close",
        "free",
    ],
)
def test_actual_ctypes_query_wrappers_close_token_and_free_descriptor(failure):
    api, events = native_binding_mock(failure)
    if failure:
        with pytest.raises((OSError, UnsafeOwnership)):
            _verify_with_security_api(api, 700)
    else:
        _verify_with_security_api(api, 700)
    expected = []
    if failure not in {"open-token", "token-info", "invalid-sid"}:
        expected.append(("free", 8888))
    if failure != "open-token":
        expected.append(("close", 101))
    assert events == expected


@pytest.mark.parametrize("kind", [71, 1, 3, 11, 27])
@pytest.mark.parametrize("mask", [1, 2, 0x10000000, 0x80000000])
@pytest.mark.parametrize("actual_owner", [True, False])
def test_native_ace_owner_rights_only_trusts_verified_actual_owner(
    kind, mask, actual_owner
):
    """Fixed-width real ACE parsing distinguishes OW from creator/world/users."""
    api, events = native_binding_mock(None if actual_owner else "foreign-owner")
    storage = ctypes.create_string_buffer(20)
    header = ctypes.cast(storage, ctypes.POINTER(api.AceHeader)).contents
    header.kind, header.size = 0, 20
    address = ctypes.addressof(storage)
    ctypes.cast(address + 4, ctypes.POINTER(api.t.DWORD)).contents.value = mask

    def acl_info(dacl, out, size, information_kind):
        ctypes.cast(out, ctypes.POINTER(api.AclSize)).contents.count = 1
        return True

    def get_ace(dacl, index, out):
        ctypes.cast(out, ctypes.POINTER(api.t.LPVOID)).contents.value = address
        return True

    api.advapi.GetAclInformation = Function(acl_info)
    api.advapi.GetAce = Function(get_ace)
    api.advapi.IsWellKnownSid = Function(lambda sid, expected: expected == kind)
    if actual_owner and kind == 71:
        _verify_with_security_api(api, 700, confidential=True)
    else:
        with pytest.raises(UnsafeOwnership):
            _verify_with_security_api(api, 700, confidential=True)
    assert events[-2:] == [("free", 8888), ("close", 101)]


@pytest.mark.skipif(os.name != "nt", reason="Native Windows mkdir0700 owner-rights ACE")
def test_native_mkdir_700_owner_rights_is_private_to_actual_user(tmp_path):
    selected = tmp_path / "mode700"
    selected.mkdir(mode=0o700)
    (selected / "private-file").write_bytes(b"native owner rights")
    with open_directory(selected) as directory:
        directory.verify_private_owner(confidential=True)
        with directory.open_file("private-file") as file:
            file.verify_private_owner(confidential=True)


@pytest.mark.skipif(
    os.name != "nt", reason="Actual token/SID/ACL checks require Windows"
)
def test_native_current_user_file_owner_and_dacl(tmp_path):
    path = tmp_path / "private-file"
    path.write_bytes(b"private")
    with open_directory(tmp_path) as directory, directory.open_file(path.name) as file:
        file.verify_private_owner()
