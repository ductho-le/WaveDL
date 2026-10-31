# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""POSIX process-group supervision with bounded, continuously drained pipes.

Only the direct child is waitable on macOS. Orphaned descendants are reaped by
the OS, not this daemon. Descendants must remain in the inherited process group;
deliberately escaping sessions requires a stronger OS containment mechanism.
"""

from __future__ import annotations

import contextlib
import os
import select
import selectors
import signal
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


class POSIXSupervisor:
    def start(self, spec: ProcessSpec, *, output_limit: int = 1048576) -> POSIXProcess:
        if os.name != "posix":
            raise OSError("POSIX supervision requires POSIX")
        if not isinstance(spec, ProcessSpec):
            raise ValueError("spec must be a ProcessSpec")
        validate_output_limit(output_limit)
        environment = os.environ.copy() if spec.inherit_environment else {}
        if spec.environment is not None:
            environment.update(spec.environment)
        environment["WAVEDL_SUPERVISED"] = "1"
        process = subprocess.Popen(
            spec.argv,
            cwd=spec.cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            shell=False,
        )
        return POSIXProcess(process, output_limit)


class POSIXProcess:
    """A single worker owns signalling, draining, reaping, and finalization."""

    def __init__(self, process: subprocess.Popen, output_limit: int):
        self._process = process
        self._output_limit = output_limit
        self._condition = threading.Condition()
        self._deadline: float | None = None
        self._cancelled = False
        self._timed_out = False
        self._result: ProcessResult | None = None
        self._error: BaseException | None = None
        self._leader_exited = False
        self._reaped = False
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    @property
    def pid(self) -> int:
        return self._process.pid

    def cancel(self, grace_seconds: float = 5) -> None:
        validate_seconds(grace_seconds, "grace_seconds")
        with self._condition:
            if self._result is None and not self._cancelled:
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
                    kill_deadline = time.monotonic() + 0.2
                    if self._deadline is None or self._deadline > kill_deadline:
                        self._deadline = kill_deadline
                    deadline = None
                    self._condition.notify_all()
                else:
                    # A worker failure must never strand an unbounded waiter.
                    # Periodic liveness checks also catch a broken finalizer.
                    if not self._worker.is_alive():
                        self._error = RuntimeError(
                            "supervision worker exited without terminal evidence"
                        )
                        break
                    self._condition.wait(
                        0.05 if remaining is None else min(remaining, 0.05)
                    )
            if self._error is not None:
                raise RuntimeError("process supervision failed") from self._error
            return self._result

    def _signal_group(self, signum: int) -> bool:
        if self._reaped:
            raise RuntimeError("cannot signal a released process-group identity")
        try:
            os.killpg(self.pid, signum)
        except ProcessLookupError:
            return False
        except PermissionError:
            # Darwin rejects signals to a group containing only zombies. EPERM
            # alone is not proof: a live, inaccessible descendant must fail.
            # TERM can exit the leader between this loop's exit observation and
            # immediate KILL escalation. Do not let a stale observation prevent
            # the stronger reserved-leader/all-zombie snapshot from proving exit.
            if self._darwin_group_is_dead():
                self._leader_exited = True
                return False
            raise
        return True

    def _darwin_group_is_dead(self) -> bool:
        if sys.platform != "darwin":
            return False
        snapshot = self._group_snapshot()
        leader_present = False
        for line in snapshot.splitlines():
            fields = line.split()
            if len(fields) != 3:
                return False
            try:
                pid, group = int(fields[0]), int(fields[1])
            except ValueError:
                return False
            if group != self.pid:
                return False
            if not fields[2].startswith("Z"):
                return False
            leader_present |= pid == self.pid
        # The unreaped leader reserves this ID and its session. With no live
        # group members, no member can fork and no foreign session can join it.
        return leader_present

    def _group_snapshot(self) -> str:
        """Darwin ps -g selects exactly this reserved process-group leader."""
        probe = subprocess.Popen(
            ("/bin/ps", "-g", str(self.pid), "-o", "pid=,pgid=,stat="),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
        captured = bytearray()
        deadline = time.monotonic() + 1
        try:
            with selectors.DefaultSelector() as selector:
                os.set_blocking(probe.stdout.fileno(), False)
                selector.register(probe.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("process-group snapshot timed out")
                    for key, _ in selector.select(remaining):
                        chunk = os.read(key.fd, 65537 - len(captured))
                        captured.extend(chunk)
                        if len(captured) > 65536:
                            raise OSError(
                                "process-group snapshot exceeded output limit"
                            )
                        if not chunk:
                            selector.unregister(key.fileobj)
                returncode = probe.wait(max(0, deadline - time.monotonic()))
                if returncode != 0:
                    raise OSError("process-group snapshot failed")
            return captured.decode("ascii")
        finally:
            if probe.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(probe.pid, signal.SIGKILL)
                probe.wait()
            probe.stdout.close()

    def _darwin_leader_has_exited(self) -> bool:
        """Confirm the unreaped owned leader is a zombie without releasing its ID.

        kqueue registration may lose to a very fast child exit. Polling Popen
        would reap the leader too early: live descendants still need signalling
        under its reserved process-group identity.
        """
        if sys.platform != "darwin" or self._reaped:
            return False
        leader_exited = False
        for line in self._group_snapshot().splitlines():
            fields = line.split()
            if len(fields) != 3:
                return False
            try:
                pid, group = int(fields[0]), int(fields[1])
            except ValueError:
                return False
            if group != self.pid:
                return False
            if pid == self.pid:
                leader_exited = fields[2].startswith("Z")
        return leader_exited

    def _observe_exit(self, queue) -> bool:
        if self._leader_exited:
            return True
        if queue is None:
            status = os.waitid(os.P_PID, self.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            self._leader_exited = status is not None and status.si_pid == self.pid
        else:
            events = queue.control(None, 1, 0)
            for event in events:
                if event.flags & select.KQ_EV_ERROR:
                    raise OSError(event.data, "child exit observation failed")
                if event.fflags & select.KQ_NOTE_EXIT:
                    self._leader_exited = True
        return self._leader_exited

    def _run(self) -> None:
        captured = [bytearray(), bytearray()]
        truncated = [False, False]
        terminated = False
        killed = False
        drain_deadline = None
        try:
            with contextlib.ExitStack() as stack:
                queue = None
                if not hasattr(os, "waitid"):
                    queue = stack.enter_context(contextlib.closing(select.kqueue()))
                    try:
                        queue.control(
                            [
                                select.kevent(
                                    self.pid,
                                    filter=select.KQ_FILTER_PROC,
                                    flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                                    fflags=select.KQ_NOTE_EXIT,
                                )
                            ],
                            0,
                            0,
                        )
                    except ProcessLookupError:
                        if not self._darwin_leader_has_exited():
                            raise
                        self._leader_exited = True
                selector = stack.enter_context(selectors.DefaultSelector())
                for index, stream in enumerate(
                    (self._process.stdout, self._process.stderr)
                ):
                    os.set_blocking(stream.fileno(), False)
                    selector.register(stream, selectors.EVENT_READ, index)
                while True:
                    leader_exited = self._observe_exit(queue)
                    with self._condition:
                        # A leader exiting is not evidence its group has exited.
                        if leader_exited and self._deadline is None:
                            self._deadline = time.monotonic() + 0.2
                        deadline = self._deadline
                    if deadline is not None and not terminated:
                        terminated = True
                        group_exists = self._signal_group(signal.SIGTERM)
                        if not group_exists:
                            killed = True
                    if (
                        deadline is not None
                        and not killed
                        and time.monotonic() >= deadline
                    ):
                        self._signal_group(signal.SIGKILL)
                        killed = True
                    if killed and leader_exited and drain_deadline is None:
                        drain_deadline = time.monotonic() + 1
                    if (
                        drain_deadline is not None
                        and selector.get_map()
                        and time.monotonic() >= drain_deadline
                    ):
                        # A setsid descendant can retain inherited pipes after
                        # the owned group is killed. Missing EOF means cleanup is
                        # incomplete: neither return success nor wait indefinitely.
                        raise TimeoutError(
                            "output pipes remain open after process-group cleanup; "
                            "an escaped descendant may still be alive"
                        )
                    if leader_exited and not selector.get_map():
                        # killpg(0) detects descendants even if they closed pipes.
                        if killed or not self._signal_group(0):
                            break
                    delay = 0.02
                    if deadline is not None and not killed:
                        delay = max(0, min(delay, deadline - time.monotonic()))
                    for key, _ in selector.select(delay):
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                            continue
                        index = key.data
                        available = self._output_limit - len(captured[index])
                        captured[index].extend(chunk[:available])
                        if len(chunk) > available:
                            truncated[index] = True
                # Release PID/PGID only once every group signal is complete.
                self._reaped = True
                returncode = self._process.wait()
            with self._condition:
                self._result = ProcessResult(
                    returncode,
                    bytes(captured[0]),
                    bytes(captured[1]),
                    self._cancelled,
                    self._timed_out,
                    truncated[0],
                    truncated[1],
                )
                self._condition.notify_all()
        except BaseException as error:
            try:
                if not self._reaped:
                    # Preserve the original supervision error, but a failing
                    # group-evidence probe must not skip direct-child reaping.
                    with contextlib.suppress(BaseException):
                        self._signal_group(signal.SIGKILL)
                    # Popen.kill() polls (and can reap); direct kill does not.
                    with contextlib.suppress(OSError):
                        os.kill(self.pid, signal.SIGKILL)
                    self._reaped = True
                self._process.wait()
            finally:
                # Pipe teardown can itself fail (e.g. EBADF). Error publication
                # must survive cleanup failures; every waiter receives evidence.
                try:
                    for stream in (self._process.stdout, self._process.stderr):
                        with contextlib.suppress(OSError, ValueError):
                            stream.close()
                finally:
                    with self._condition:
                        self._error = error
                        self._condition.notify_all()
