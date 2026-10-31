"""Daemon-suite collection policy and isolated process fixtures."""

import subprocess
import sys
from contextlib import contextmanager
from queue import Queue
from threading import Thread
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException

from wavedl_daemon.events.broker import EventBroker
from wavedl_daemon.http.app import _http_error, _validation_error


def route_app(*, routes):
    """Mount isolated API groups with the production validation/error boundary."""
    return FastAPI(
        routes=routes,
        exception_handlers={
            HTTPException: _http_error,
            RequestValidationError: _validation_error,
        },
    )


class FakeServices:
    """Complete inert HTTP facade; individual tests replace only their operation."""

    def __init__(self):
        self.daemon_id = "daemon-test"
        self.run_service = Mock()
        self.catalog = Mock()
        self.ingestor = Mock()
        self.event_log = Mock()
        self.broker = EventBroker()
        self.artifacts = Mock()
        self.pool = Mock()

    async def resolve_plan(self, run_id):
        return "11111111-1111-1111-1111-111111111111"

    async def health(self):
        return {}

    async def adopt(self, path, *, attempt=None, idempotency_key):
        raise AssertionError("Unexpected adoption")

    async def inspect(self, root, path):
        raise AssertionError("Unexpected inspection")

    async def preview(self, request):
        raise AssertionError("Unexpected preview")

    async def host_profile(self):
        raise AssertionError("Unexpected host profile")

    async def start(self):
        pass

    async def close(self):
        pass


@pytest.fixture
def fake_services():
    return FakeServices()


@pytest.fixture
def paused_process():
    """Provide a bounded readiness handshake with deterministic cleanup."""

    @contextmanager
    def start(script, expected):
        process = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        ready = Queue()

        def read_ready():
            assert process.stdout is not None
            ready.put(process.stdout.readline().strip())

        reader = Thread(target=read_ready, daemon=True)
        reader.start()
        try:
            assert ready.get(timeout=10) == expected
            yield process
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
            reader.join(timeout=2)
            process.communicate(timeout=2)
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()

    return start
