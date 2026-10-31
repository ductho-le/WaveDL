"""Run native pytest with this test process's default object owner set to its user.

Elevated Windows runners can default new files to Administrators ownership.
The runtime deliberately requires actual-user ownership, so tests must create
their fixtures with the corresponding owner. Never change production policy,
privileges, token DACLs or existing filesystem objects here. Only a freshly
created test workspace receives an explicit protected inheritable DACL.

Microsoft contracts:
https://learn.microsoft.com/en-us/windows/win32/secauthz/owner-of-a-new-object
https://learn.microsoft.com/en-us/windows/win32/secauthz/access-rights-for-access-token-objects
"""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path


class TokenOwnerAPI:
    """Minimal typed Win32 bindings, imported only on Windows."""

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        self.c, self.t = ctypes, wintypes
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.security = ctypes.WinDLL("advapi32", use_last_error=True)
        signatures = {
            "OpenProcessToken": (
                [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)],
                wintypes.BOOL,
            ),
            "GetTokenInformation": (
                [
                    wintypes.HANDLE,
                    ctypes.c_int,
                    wintypes.LPVOID,
                    wintypes.DWORD,
                    ctypes.POINTER(wintypes.DWORD),
                ],
                wintypes.BOOL,
            ),
            "SetTokenInformation": (
                [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD],
                wintypes.BOOL,
            ),
            "EqualSid": ([wintypes.LPVOID, wintypes.LPVOID], wintypes.BOOL),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.security, name)
            function.argtypes, function.restype = args, result
        self.kernel.GetCurrentProcess.argtypes = []
        self.kernel.GetCurrentProcess.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL

    def error(self):
        return self.c.WinError(self.c.get_last_error())

    def open(self):
        token = self.t.HANDLE()
        # TOKEN_QUERY | TOKEN_ADJUST_DEFAULT; no privilege adjustment.
        if not self.security.OpenProcessToken(
            self.kernel.GetCurrentProcess(), 0x0008 | 0x0080, self.c.byref(token)
        ):
            raise self.error()
        return token

    def query(self, token, kind):
        size = self.t.DWORD()
        if self.security.GetTokenInformation(token, kind, None, 0, self.c.byref(size)):
            raise RuntimeError("Token query unexpectedly returned no buffer")
        if self.c.get_last_error() != 122 or not 0 < size.value <= 65536:
            raise self.error()
        buffer = self.c.create_string_buffer(size.value)
        if not self.security.GetTokenInformation(
            token, kind, buffer, size.value, self.c.byref(size)
        ):
            raise self.error()
        # Both TOKEN_OWNER and TOKEN_USER start with a pointer to their SID.
        sid = self.c.cast(buffer, self.c.POINTER(self.t.LPVOID)).contents.value
        if not sid:
            raise RuntimeError("Token information has no SID")
        return buffer, sid

    def set_owner(self, token, sid):
        owner = self.t.LPVOID(sid)
        if not self.security.SetTokenInformation(
            token, 4, self.c.byref(owner), self.c.sizeof(owner)
        ):
            raise self.error()

    def equal(self, left, right):
        return bool(self.security.EqualSid(left, right))

    def close(self, token):
        if not self.kernel.CloseHandle(token):
            raise self.error()

    def protect_directory(self, path):
        """Set a protected inheritable DACL on a freshly allocated test root."""
        c, t = self.c, self.t
        signatures = {
            "ConvertSidToStringSidW": ([t.LPVOID, c.POINTER(t.LPWSTR)], t.BOOL),
            "ConvertStringSecurityDescriptorToSecurityDescriptorW": (
                [t.LPCWSTR, t.DWORD, c.POINTER(t.LPVOID), c.POINTER(t.DWORD)],
                t.BOOL,
            ),
            "GetSecurityDescriptorDacl": (
                [t.LPVOID, c.POINTER(t.BOOL), c.POINTER(t.LPVOID), c.POINTER(t.BOOL)],
                t.BOOL,
            ),
            "SetNamedSecurityInfoW": (
                [t.LPWSTR, c.c_int, t.DWORD, t.LPVOID, t.LPVOID, t.LPVOID, t.LPVOID],
                t.DWORD,
            ),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.security, name)
            function.argtypes, function.restype = args, result
        self.kernel.LocalFree.argtypes = [t.LPVOID]
        self.kernel.LocalFree.restype = t.LPVOID
        token = self.open()
        descriptor, sid_text = t.LPVOID(), t.LPWSTR()
        try:
            _user_buffer, user_sid = self.query(token, 1)
            if not self.security.ConvertSidToStringSidW(user_sid, c.byref(sid_text)):
                raise self.error()
            sddl = private_sddl(sid_text.value)
            if not self.security.ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl, 1, c.byref(descriptor), None
            ):
                raise self.error()
            present, defaulted, dacl = t.BOOL(), t.BOOL(), t.LPVOID()
            if not self.security.GetSecurityDescriptorDacl(
                descriptor, c.byref(present), c.byref(dacl), c.byref(defaulted)
            ):
                raise self.error()
            if not present.value or not dacl.value:
                raise RuntimeError("Explicit private test DACL required")
            # SE_FILE_OBJECT, DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION.
            result = self.security.SetNamedSecurityInfoW(
                str(path), 1, 0x80000004, None, None, dacl, None
            )
            if result:
                raise c.WinError(result)
        finally:
            try:
                try:
                    if descriptor and self.kernel.LocalFree(descriptor):
                        raise self.error()
                finally:
                    allocation = c.cast(sid_text, t.LPVOID)
                    if allocation and self.kernel.LocalFree(allocation):
                        raise self.error()
            finally:
                self.close(token)


def private_sddl(user_sid):
    import re

    if not isinstance(user_sid, str) or not re.fullmatch(
        r"S-1-(?:[0-9]+-)*[0-9]+", user_sid
    ):
        raise ValueError("Actual token-user SID required")
    return f"D:P(A;OICI;FA;;;{user_sid})(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)"


@contextmanager
def private_test_workspace(*, platform=None, api_factory=TokenOwnerAPI):
    """Protect only one fresh empty root; never alter its shared temp parent."""
    native = (os.name if platform is None else platform) == "nt"
    with tempfile.TemporaryDirectory(prefix="wavedl-user-pytest-") as temporary:
        root = Path(temporary)
        if root.is_symlink() or not root.is_dir() or list(root.iterdir()):
            raise RuntimeError("Fresh ordinary empty test root required")
        root = root.resolve()
        if native:
            api_factory().protect_directory(root)
        yield root


@contextmanager
def user_owned_objects(*, platform=None, api_factory=TokenOwnerAPI):
    """Restore the captured owner and close its token even after pytest fails."""
    if (os.name if platform is None else platform) != "nt":
        yield
        return
    api = api_factory()
    token = api.open()
    try:
        _original_buffer, original_sid = api.query(token, 4)  # TokenOwner
        _user_buffer, user_sid = api.query(token, 1)  # TokenUser
        try:
            api.set_owner(token, user_sid)
            _verified_buffer, verified_sid = api.query(token, 4)
            if not api.equal(verified_sid, user_sid):
                raise RuntimeError("Test token owner did not become the actual user")
            yield
        finally:
            # Keep all SID buffers alive through restoration; pointer values
            # alone do not own the memory returned by GetTokenInformation.
            api.set_owner(token, original_sid)
            _restored_buffer, restored_sid = api.query(token, 4)
            if not api.equal(restored_sid, original_sid):
                raise RuntimeError("Original test token owner was not restored")
    finally:
        api.close(token)


def main(argv=None):
    # Enter before importing/starting pytest or allocating its basetemp. Child
    # processes inherit the primary token rather than an existing path's owner.
    original_path = sys.path.copy()
    try:
        # Match python -m pytest: a direct tools/ script otherwise exposes only
        # tools/ as sys.path[0], breaking checkout-local test helper imports.
        sys.path.insert(0, os.getcwd())
        with user_owned_objects():
            import pytest

            arguments = list(sys.argv[1:] if argv is None else argv)
            if os.name == "nt":
                if any(
                    arg == "--basetemp" or arg.startswith("--basetemp=")
                    for arg in arguments
                ):
                    raise ValueError(
                        "Wrapper owns the Windows pytest base temporary root"
                    )
                with private_test_workspace() as workspace:
                    return pytest.main(
                        [*arguments, "--basetemp", str(workspace / "pytest")]
                    )
            return pytest.main(arguments)
    finally:
        sys.path[:] = original_path


if __name__ == "__main__":
    raise SystemExit(main())
