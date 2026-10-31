"""Handle-based Windows owner/DACL checks, with no pathname or UID fallback.

The owner's SID must be the process token's actual TokenUser SID. SYSTEM and
Builtin Administrators are explicitly trusted for writes: desktop administrators
already control the process. No other user/group is trusted, even if the process
token happens to be a member. Unsupported ACE layouts fail closed.
Owner Rights (S-1-3-4) ACEs apply only to that already-verified object owner;
they are not a separately trusted user or group.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import ClassVar


_WRITABLE = (
    0x00000002  # FILE_WRITE_DATA / FILE_ADD_FILE
    | 0x00000004  # FILE_APPEND_DATA / FILE_ADD_SUBDIRECTORY
    | 0x00000010  # FILE_WRITE_EA
    | 0x00000040  # FILE_DELETE_CHILD
    | 0x00000100  # FILE_WRITE_ATTRIBUTES
    | 0x00010000  # DELETE
    | 0x00040000  # WRITE_DAC
    | 0x00080000  # WRITE_OWNER
    | 0x02000000  # MAXIMUM_ALLOWED (not an ordinary file ACE permission)
    | 0x10000000  # GENERIC_ALL
    | 0x40000000  # GENERIC_WRITE
)
_READABLE = (
    0x00000001  # FILE_READ_DATA / FILE_LIST_DIRECTORY
    | 0x00000008  # FILE_READ_EA
    | 0x00000020  # FILE_EXECUTE / FILE_TRAVERSE
    | 0x00000080  # FILE_READ_ATTRIBUTES
    | 0x00020000  # READ_CONTROL
    | 0x20000000  # GENERIC_EXECUTE
    | 0x80000000  # GENERIC_READ
)


@dataclass(frozen=True)
class _Ace:
    kind: int
    mask: int
    owner: bool
    privileged: bool = False


def _reject():
    from .safe_files import UnsafeOwnership

    raise UnsafeOwnership("Object must be current-user owned with a private DACL")


def _check_acl(
    owner_matches: bool, entries: list[_Ace] | None, *, confidential=False
) -> None:
    """Conservative pure decision; deny ACEs never cancel unsafe allow ACEs."""
    if not owner_matches or entries is None:
        _reject()
    forbidden = _WRITABLE | _READABLE if confidential else _WRITABLE
    for entry in entries:
        if entry.kind not in (0, 1):  # ACCESS_ALLOWED_ACE / ACCESS_DENIED_ACE
            _reject()
        if (
            entry.kind == 0
            and entry.mask & forbidden
            and not (entry.owner or entry.privileged)
        ):
            _reject()


class _SecurityAPI:
    """Lazy typed Win32 security bindings using the existing handle API."""

    def __init__(self, api):
        self.c, self.t, self.kernel = api.ctypes, api.types, api.dll
        c, t = self.c, self.t
        self.advapi = c.WinDLL("advapi32", use_last_error=True)
        pointer = c.POINTER(t.LPVOID)
        signatures = {
            "OpenProcessToken": ([t.HANDLE, t.DWORD, c.POINTER(t.HANDLE)], t.BOOL),
            "GetTokenInformation": (
                [t.HANDLE, c.c_int, t.LPVOID, t.DWORD, c.POINTER(t.DWORD)],
                t.BOOL,
            ),
            "GetSecurityInfo": (
                [
                    t.HANDLE,
                    c.c_int,
                    t.DWORD,
                    pointer,
                    pointer,
                    pointer,
                    pointer,
                    pointer,
                ],
                t.DWORD,
            ),
            "EqualSid": ([t.LPVOID, t.LPVOID], t.BOOL),
            "IsValidSid": ([t.LPVOID], t.BOOL),
            "GetLengthSid": ([t.LPVOID], t.DWORD),
            "IsWellKnownSid": ([t.LPVOID, c.c_int], t.BOOL),
            "IsValidAcl": ([t.LPVOID], t.BOOL),
            "GetAclInformation": ([t.LPVOID, t.LPVOID, t.DWORD, c.c_int], t.BOOL),
            "GetAce": ([t.LPVOID, t.DWORD, pointer], t.BOOL),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.advapi, name)
            function.argtypes, function.restype = args, result
        self.kernel.GetCurrentProcess.argtypes = []
        self.kernel.GetCurrentProcess.restype = t.HANDLE
        self.kernel.CloseHandle.argtypes = [t.HANDLE]
        self.kernel.CloseHandle.restype = t.BOOL
        self.kernel.LocalFree.argtypes = [t.LPVOID]
        self.kernel.LocalFree.restype = t.LPVOID

        class TokenUser(c.Structure):
            _fields_: ClassVar = [("sid", t.LPVOID), ("attributes", t.DWORD)]

        class AclSize(c.Structure):
            _fields_: ClassVar = [
                ("count", t.DWORD),
                ("used", t.DWORD),
                ("free", t.DWORD),
            ]

        class AceHeader(c.Structure):
            _fields_: ClassVar = [("kind", t.BYTE), ("flags", t.BYTE), ("size", t.WORD)]

        self.TokenUser, self.AclSize, self.AceHeader = TokenUser, AclSize, AceHeader

    def _error(self, code=None):
        raise self.c.WinError(self.c.get_last_error() if code is None else code)

    @contextmanager
    def token_user(self):
        c, t = self.c, self.t
        token = t.HANDLE()
        try:
            if not self.advapi.OpenProcessToken(
                self.kernel.GetCurrentProcess(), 0x0008, c.byref(token)
            ):
                self._error()
            size = t.DWORD()
            result = self.advapi.GetTokenInformation(token, 1, None, 0, c.byref(size))
            if (
                result
                or c.get_last_error() != 122
                or not c.sizeof(self.TokenUser) <= size.value <= 65536
            ):
                self._error()
            buffer = c.create_string_buffer(size.value)
            if not self.advapi.GetTokenInformation(
                token, 1, buffer, size, c.byref(size)
            ):
                self._error()
            user = c.cast(buffer, c.POINTER(self.TokenUser)).contents
            if not user.sid or not self.advapi.IsValidSid(user.sid):
                _reject()
            yield user.sid  # buffer remains alive for the entire comparison
        finally:
            if token.value:
                active_error = sys.exception()
                if not self.kernel.CloseHandle(token) and active_error is None:
                    self._error()

    @contextmanager
    def descriptor(self, handle):
        c, t = self.c, self.t
        owner, dacl, descriptor = t.LPVOID(), t.LPVOID(), t.LPVOID()
        try:
            result = self.advapi.GetSecurityInfo(
                handle,
                1,
                0x00000001 | 0x00000004,
                c.byref(owner),
                None,
                c.byref(dacl),
                None,
                c.byref(descriptor),
            )
            if result:
                self._error(result)
            if (
                not descriptor.value
                or not owner.value
                or not self.advapi.IsValidSid(owner)
            ):
                _reject()
            yield owner, dacl
        finally:
            if descriptor.value:
                active_error = sys.exception()
                if self.kernel.LocalFree(descriptor) and active_error is None:
                    self._error()

    def same_sid(self, first, second):
        return bool(self.advapi.EqualSid(first, second))

    def entries(self, dacl, owner):
        if not dacl.value:
            return None  # NULL DACL grants everyone full access
        if not self.advapi.IsValidAcl(dacl):
            _reject()
        c, t = self.c, self.t
        info = self.AclSize()
        if not self.advapi.GetAclInformation(dacl, c.byref(info), c.sizeof(info), 2):
            self._error()
        entries = []
        for index in range(info.count):
            address = t.LPVOID()
            if not self.advapi.GetAce(dacl, index, c.byref(address)):
                self._error()
            if not address.value:
                _reject()
            header = c.cast(address, c.POINTER(self.AceHeader)).contents
            if header.kind not in (0, 1) or header.size < 16:
                _reject()
            # Standard allow/deny ACE: four-byte header, DWORD mask, SID.
            mask = c.cast(address.value + 4, c.POINTER(t.DWORD)).contents.value
            sid = address.value + 8
            if (
                not self.advapi.IsValidSid(sid)
                or self.advapi.GetLengthSid(sid) > header.size - 8
            ):
                _reject()
            entries.append(
                _Ace(
                    header.kind,
                    mask,
                    self.same_sid(sid, owner)
                    # WinCreatorOwnerRightsSid=71 identifies OWNER RIGHTS,
                    # not CREATOR OWNER (3). The caller verifies actual object
                    # ownership against TokenUser before examining any ACE.
                    or bool(self.advapi.IsWellKnownSid(sid, 71)),
                    bool(
                        self.advapi.IsWellKnownSid(sid, 22)
                        or self.advapi.IsWellKnownSid(sid, 26)
                    ),
                )
            )
        return entries


def _verify_with_security_api(security, handle, *, confidential=False):
    with security.token_user() as user, security.descriptor(handle) as (owner, dacl):
        matches = security.same_sid(owner, user)
        if not matches:
            _reject()
        _check_acl(matches, security.entries(dacl, owner), confidential=confidential)


def verify_windows_private_owner(api, handle, *, confidential=False) -> None:
    """Verify a caller-owned open HANDLE; never close that file/directory handle."""
    _verify_with_security_api(_SecurityAPI(api), handle, confidential=confidential)
