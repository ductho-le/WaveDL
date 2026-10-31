# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Windows Job Object supervision, with assignment before first execution.

The Windows-only stdlib modules are loaded lazily. CPython's ``_winapi``
CreateProcess wrapper retains the initial thread handle (unlike Popen), allowing
CREATE_SUSPENDED -> AssignProcessToJobObject -> ResumeThread. No breakaway flags
are enabled. Closing the non-inheritable job handle kills its remaining members.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import threading
import time

from .base import (
    ProcessResult,
    ProcessSpec,
    validate_output_limit,
    validate_seconds,
)


CREATE_SUSPENDED = 0x00000004
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_UNICODE_ENVIRONMENT = 0x00000400
_SPAWN_LOCK = threading.Lock()


class _Win32:
    """Small native adapter; all handles use pointer-width ctypes signatures."""

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise OSError("Windows supervision requires Windows")
        import _winapi
        import msvcrt
        from ctypes import wintypes

        self.winapi = _winapi
        self.msvcrt = msvcrt
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        handle, dword, boolean = wintypes.HANDLE, wintypes.DWORD, wintypes.BOOL
        signatures = {
            "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], handle),
            "SetInformationJobObject": (
                [handle, ctypes.c_int, ctypes.c_void_p, dword],
                boolean,
            ),
            "QueryInformationJobObject": (
                [handle, ctypes.c_int, ctypes.c_void_p, dword, ctypes.c_void_p],
                boolean,
            ),
            "AssignProcessToJobObject": ([handle, handle], boolean),
            "ResumeThread": ([handle], dword),
            "TerminateJobObject": ([handle, wintypes.UINT], boolean),
            "TerminateProcess": ([handle, wintypes.UINT], boolean),
            "GenerateConsoleCtrlEvent": ([dword, dword], boolean),
            "CloseHandle": ([handle], boolean),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.kernel, name)
            function.argtypes = arguments
            function.restype = result

        class BasicLimit(ctypes.Structure):
            _fields_ = (
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", dword),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", dword),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", dword),
                ("SchedulingClass", dword),
            )

        class ExtendedLimit(ctypes.Structure):
            _fields_ = (
                ("BasicLimitInformation", BasicLimit),
                ("IoInfo", ctypes.c_uint64 * 6),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            )

        class Accounting(ctypes.Structure):
            _fields_ = (
                ("TotalUserTime", ctypes.c_int64),
                ("TotalKernelTime", ctypes.c_int64),
                ("ThisPeriodTotalUserTime", ctypes.c_int64),
                ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                ("TotalPageFaultCount", dword),
                ("TotalProcesses", dword),
                ("ActiveProcesses", dword),
                ("TotalTerminatedProcesses", dword),
            )

        self.extended_limit = ExtendedLimit
        self.accounting = Accounting

    @staticmethod
    def _check(result):
        if not result:
            raise ctypes.WinError(ctypes.get_last_error())
        return result

    def create_job(self):
        job = self._check(self.kernel.CreateJobObjectW(None, None))
        try:
            limits = self.extended_limit()
            limits.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE
            self._check(
                self.kernel.SetInformationJobObject(
                    job, 9, ctypes.byref(limits), ctypes.sizeof(limits)
                )
            )
            return job
        except BaseException:
            self.close(job)
            raise

    def spawn(self, spec, flags):
        descriptors = []
        process = thread = None
        try:
            stdout_read, stdout_write = os.pipe()
            descriptors.extend((stdout_read, stdout_write))
            stderr_read, stderr_write = os.pipe()
            descriptors.extend((stderr_read, stderr_write))
            stdin = os.open(os.devnull, os.O_RDONLY)
            descriptors.append(stdin)
            handles = [
                self.msvcrt.get_osfhandle(fd)
                for fd in (stdin, stdout_write, stderr_write)
            ]
            startup = subprocess.STARTUPINFO(
                dwFlags=subprocess.STARTF_USESTDHANDLES,
                hStdInput=handles[0],
                hStdOutput=handles[1],
                hStdError=handles[2],
                lpAttributeList={"handle_list": handles},
            )
            environment = os.environ.copy() if spec.inherit_environment else {}
            if spec.environment is not None:
                environment.update(spec.environment)
            environment["WAVEDL_SUPERVISED"] = "1"
            # Restrict inheritance to the three redirected handles. Never let
            # children inherit the job, process, thread, or pipe read handles.
            with _SPAWN_LOCK:
                try:
                    for handle in handles:
                        os.set_handle_inheritable(handle, True)
                    process, thread, pid, _ = self.winapi.CreateProcess(
                        None,
                        subprocess.list2cmdline(spec.argv),
                        None,
                        None,
                        True,
                        flags,
                        environment,
                        str(spec.cwd),
                        startup,
                    )
                finally:
                    for handle in handles:
                        os.set_handle_inheritable(handle, False)
            for fd in (stdout_write, stderr_write, stdin):
                os.close(fd)
                descriptors.remove(fd)
            descriptors.clear()  # ownership of read ends transfers to caller
            return process, thread, pid, stdout_read, stderr_read
        except BaseException:
            if process is not None:
                try:
                    self.terminate_process(process)
                finally:
                    try:
                        self.close(thread)
                    finally:
                        self.close(process)
            raise
        finally:
            for fd in descriptors:
                os.close(fd)

    def assign(self, job, process):
        self._check(self.kernel.AssignProcessToJobObject(job, process))

    def resume(self, thread):
        if self.kernel.ResumeThread(thread) == 0xFFFFFFFF:
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self, handle):
        self._check(self.kernel.CloseHandle(handle))

    def terminate_process(self, process):
        self._check(self.kernel.TerminateProcess(process, 1))
        self.winapi.WaitForSingleObject(process, self.winapi.INFINITE)

    def terminate_job(self, job):
        self._check(self.kernel.TerminateJobObject(job, 1))

    def active(self, job):
        info = self.accounting()
        self._check(
            self.kernel.QueryInformationJobObject(
                job, 1, ctypes.byref(info), ctypes.sizeof(info), None
            )
        )
        return info.ActiveProcesses

    def exit_code(self, process):
        self.winapi.WaitForSingleObject(process, self.winapi.INFINITE)
        return self.winapi.GetExitCodeProcess(process)

    def leader_running(self, process):
        return self.winapi.WaitForSingleObject(process, 0) == self.winapi.WAIT_TIMEOUT

    def send_break(self, pid):
        if not self.kernel.GenerateConsoleCtrlEvent(1, pid):
            error = ctypes.get_last_error()
            # GUI/headless hosts or an already-exited console group cannot
            # receive CTRL_BREAK. The grace phase still precedes forceful kill.
            if error not in (6, 87):
                raise ctypes.WinError(error)


class _Drain:
    def __init__(self, fd: int, limit: int):
        self.data = bytearray()
        self.truncated = False
        self.error = None
        self.thread = threading.Thread(target=self._read, args=(fd, limit), daemon=True)
        self.thread.start()

    def _read(self, fd: int, limit: int) -> None:
        try:
            while chunk := os.read(fd, 65536):
                remaining = limit - len(self.data)
                self.data.extend(chunk[:remaining])
                self.truncated |= len(chunk) > remaining
        except OSError as error:
            self.error = error
        finally:
            os.close(fd)


class WindowsProcess:
    """One owned job; completion means zero job members and drained streams."""

    def __init__(self, api, job, process, pid, stdout, stderr, output_limit):
        self.pid = pid
        self._api, self._job, self._process = api, job, process
        self._condition = threading.Condition()
        self._deadline = None
        self._cancelled = False
        self._timed_out = False
        self._result = None
        self._error = None
        self._stdout = _Drain(stdout, output_limit)
        self._stderr = _Drain(stderr, output_limit)
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def cancel(self, grace_seconds: float = 5) -> None:
        validate_seconds(grace_seconds, "grace_seconds")
        with self._condition:
            if self._result is not None or self._error is not None:
                return
            if not self._cancelled:
                self._cancelled = True
                self._deadline = time.monotonic() + grace_seconds
                self._condition.notify_all()

    def wait(self, timeout: float | None = None) -> ProcessResult:
        if timeout is not None:
            validate_seconds(timeout, "timeout")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._result is None and self._error is None:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    self._timed_out = True
                    self._cancelled = True
                    self._deadline = time.monotonic()
                    deadline = None
                    self._condition.notify_all()
                else:
                    self._condition.wait(remaining)
            if self._error is not None:
                raise self._error
            return self._result

    def _run(self):
        signalled = killed = False
        error = None
        result = None
        try:
            while self._api.active(self._job):
                with self._condition:
                    if (
                        not self._api.leader_running(self._process)
                        and self._deadline is None
                    ):
                        self._deadline = time.monotonic() + 0.2
                    deadline = self._deadline
                if deadline is not None and not signalled:
                    self._api.send_break(self.pid)
                    signalled = True
                if deadline is not None and not killed and time.monotonic() >= deadline:
                    self._api.terminate_job(self._job)
                    killed = True
                with self._condition:
                    self._condition.wait(0.01)
            returncode = self._api.exit_code(self._process)
            for drain in (self._stdout, self._stderr):
                drain.thread.join()
                if drain.error is not None:
                    raise drain.error
            with self._condition:
                result = ProcessResult(
                    returncode,
                    bytes(self._stdout.data),
                    bytes(self._stderr.data),
                    self._cancelled,
                    self._timed_out,
                    self._stdout.truncated,
                    self._stderr.truncated,
                )
        except BaseException as failure:
            error = failure
            try:
                self._api.terminate_job(self._job)
                self._api.exit_code(self._process)
            except BaseException as cleanup_failure:
                error = BaseExceptionGroup(
                    "supervision and cleanup failed", [failure, cleanup_failure]
                )
        finally:
            # Job close also kills members if an API/drain error interrupted us.
            for handle in (self._job, self._process):
                try:
                    self._api.close(handle)
                except BaseException as close_failure:
                    error = (
                        close_failure
                        if error is None
                        else BaseExceptionGroup(
                            "supervision handle cleanup failed", [error, close_failure]
                        )
                    )
            for drain in (self._stdout, self._stderr):
                drain.thread.join()
            with self._condition:
                self._result = result
                self._error = error
                self._condition.notify_all()


class WindowsSupervisor:
    def __init__(self, *, _api=None):
        self._api = _api  # injectable only for portable ordering/failure tests

    def start(
        self, spec: ProcessSpec, *, output_limit: int = 1048576
    ) -> WindowsProcess:
        validate_output_limit(output_limit)
        if not isinstance(spec, ProcessSpec):
            raise ValueError("spec must be a ProcessSpec")
        api = self._api if self._api is not None else _Win32()
        job = api.create_job()
        process = thread = None
        reads = []
        try:
            process, thread, pid, stdout, stderr = api.spawn(
                spec,
                CREATE_SUSPENDED
                | CREATE_NEW_PROCESS_GROUP
                | CREATE_UNICODE_ENVIRONMENT,
            )
            reads.extend((stdout, stderr))
            api.assign(job, process)
            api.resume(thread)
            api.close(thread)
            thread = None
            supervised = WindowsProcess(
                api, job, process, pid, stdout, stderr, output_limit
            )
            reads.clear()
            return supervised
        except BaseException:
            # Even assignment failure must reap the still-suspended process.
            try:
                if process is not None:
                    api.terminate_process(process)
            finally:
                for fd in reads:
                    os.close(fd)
                try:
                    if thread is not None:
                        api.close(thread)
                finally:
                    try:
                        if process is not None:
                            api.close(process)
                    finally:
                        api.close(job)
            raise
