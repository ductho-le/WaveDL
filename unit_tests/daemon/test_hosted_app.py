"""The assembled HTTP app uses one owner thread and durable replay/artifacts."""

import asyncio
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import httpx
import pytest

from wavedl.core.events import HelloPayload
from wavedl.core.generations import publish_generation
from wavedl.core.layout import RunLayout
from wavedl.core.local_attempt import (
    LocalStartupIdentity,
    coordinate_local_attempt,
    read_local_binding,
)
from wavedl.core.protocol import Envelope
from wavedl.core.submission import (
    create_submission,
    identity_for_submission,
    publish_submission_file,
)
from wavedl_daemon.backends.contracts import (
    BackendObservation,
    ObservationStatus,
    SubmissionHandle,
    SubmissionObservation,
)
from wavedl_daemon.backends.local import LocalBackend
from wavedl_daemon.execution.budget import StorageWorkError
from wavedl_daemon.http.app import create_app
from wavedl_daemon.runs.runtime_service import HostedServices
from wavedl_daemon.storage.settings import Settings


TOKEN = "hosted-test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def test_session_controller_uses_shared_pool_lane(tmp_path):
    from wavedl_daemon.runs.runtime_service import ServiceSession

    settings = Settings(tmp_path / "state", tmp_path / "runs")
    with ServiceSession(settings) as session:
        assert session.controller.context.lane is session.pool.control_lane


@pytest.mark.skipif(os.name != "posix", reason="POSIX state-root confidentiality")
def test_public_state_root_is_refused_without_chmod_or_registry_write(tmp_path):
    settings = Settings(tmp_path / "state", tmp_path / "runs")
    settings.state_root.mkdir(mode=0o755)
    # A private harness umask must not turn this public-state witness into 0700.
    settings.state_root.chmod(0o755)
    assert settings.state_root.stat().st_mode & 0o777 == 0o755
    services = HostedServices(settings)

    async def run():
        with pytest.raises(StorageWorkError):
            await services.start()
        await services.close()

    asyncio.run(run())
    assert settings.state_root.stat().st_mode & 0o777 == 0o755
    assert not (settings.state_root / "registry.db").exists()


class Backend:
    """ASGI contract fake with real immutable writer startup proof, not ML."""

    owner = "test-owner"

    def __init__(self):
        self.submissions = 0
        self.handle = None
        self.startup = None

    def plan(self, manifest, **kwargs):
        return LocalBackend().plan(manifest, **kwargs)

    def submit(self, plan, *, request_id, attempt):
        assert attempt is None and plan.allocation_mode == "writer"
        digest = hashlib.sha256(plan.to_json().encode()).hexdigest()
        identity = identity_for_submission(
            Path(plan.run_dir),
            request_id=request_id,
            backend="local",
            owner=self.owner,
            plan_sha256=digest,
        )
        create_submission(Path(plan.run_dir), identity)
        publish_submission_file(
            Path(plan.run_dir), identity, "plan.json", plan.to_json().encode()
        )
        supervisor_id = "owned-test-supervisor"
        publish_submission_file(
            Path(plan.run_dir),
            identity,
            "local-submission.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": plan.run_id,
                    "request_id": request_id,
                    "owner": self.owner,
                    "supervisor_id": supervisor_id,
                    "plan_sha256": digest,
                }
            ).encode(),
        )
        self.startup = LocalStartupIdentity(
            identity.run_id,
            identity.request_id,
            identity.owner,
            supervisor_id,
            identity.run_directory,
            identity.manifest_sha256,
            digest,
        )
        self.handle = SubmissionHandle(
            "local",
            plan.run_id,
            request_id,
            self.owner,
            plan.run_dir,
            pid=12345,
            supervisor_id=supervisor_id,
        )
        self.submissions += 1
        return self.handle

    def discover_initial(self, handle):
        assert handle == self.handle
        try:
            return read_local_binding(Path(handle.run_dir), self.startup)
        except FileNotFoundError:
            return None

    def runtime_startup(self):
        return coordinate_local_attempt(Path(self.handle.run_dir), self.startup)

    def observe(self, handle):
        if isinstance(handle, SubmissionHandle):
            assert handle == self.handle
            return SubmissionObservation(ObservationStatus.RUNNING)
        assert handle == self.handle.bind_attempt(handle.attempt)
        return BackendObservation(handle.attempt, ObservationStatus.RUNNING)

    def request_cancel(self, handle):
        pass


def test_hosted_lifecycle_push_and_artifacts(tmp_path, monkeypatch):
    def forbidden_allocation(*args, **kwargs):
        pytest.fail("The daemon must not allocate writer-mode runtime attempts")

    monkeypatch.setattr(
        "wavedl_daemon.runs.control.dispatch.allocate_attempt", forbidden_allocation
    )
    settings = Settings(tmp_path / "state", tmp_path / "runs")
    backend = Backend()
    services = HostedServices(settings, backend_factory=lambda lock: {"local": backend})
    app = create_app(
        settings, token=TOKEN, daemon_id="host", build_id="test", services=services
    )
    assert not settings.state_root.exists()

    async def run():
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                base_url="http://localhost:8420",
                headers=AUTH,
            ) as client,
        ):
            payload = {
                "backend": "local",
                "config": {"epochs": 1},
                "runtime": {
                    "python": sys.executable,
                    "mode": "existing",
                    "device": "cpu",
                    "environment": {},
                },
                "operation": "train",
                "topology": {
                    "num_machines": 1,
                    "processes_per_machine": 1,
                    "machine_rank": 0,
                },
            }
            health = await client.get("/api/health")
            assert health.status_code == 200
            assert health.json()["registered_backends"] == ["local"]
            await services.host.call(
                lambda session: session._monitor_error("test-source", "BudgetExhausted")
            )
            health = await client.get("/api/health")
            assert {
                "source": "test-source",
                "error": "BudgetExhausted",
            } in health.json()["monitoring_errors"]
            preview = await client.post(
                "/api/runs/plan", json=payload, headers={"Idempotency-Key": "one"}
            )
            assert preview.status_code == 200, preview.text
            planned = preview.json()
            assert planned["preview"]["allocation_mode"] == "writer"
            attempts_page = await client.get(f"/api/runs/{planned['run_id']}/attempts")
            assert attempts_page.json() == {"attempts": [], "total": 0}
            assert backend.submissions == 0
            assert not (settings.run_root / planned["run_id"]).exists()
            submitted = await client.post(
                "/api/runs", json={"plan_id": planned["plan_id"]}
            )
            assert submitted.status_code == 202, submitted.text
            assert submitted.json()["attempt"] is None
            deadline = time.monotonic() + 3
            while not backend.submissions and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            assert backend.submissions == 1
            unbound = (
                await client.get(f"/api/runs/{planned['run_id']}/attempts")
            ).json()
            assert all(item["attempt"] is None for item in unbound["attempts"])
            assert not list(
                (settings.run_root / planned["run_id"] / "attempts").iterdir()
            )
            binding = backend.runtime_startup()
            attempt = binding.attempt
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                view = await client.get(f"/api/runs/{planned['run_id']}")
                if view.json()["attempt"] == attempt:
                    break
                await asyncio.sleep(0.01)
            assert view.json()["attempt"] == attempt
            hello = Envelope(
                planned["run_id"],
                attempt,
                "hello",
                0,
                "2026-10-03T00:00:00Z",
                asdict(
                    HelloPayload(
                        1,
                        1,
                        "actual-runtime",
                        "3.11",
                        "cpu",
                        ["train"],
                        sys.executable,
                    )
                ),
            )
            response = await client.post(
                f"/api/runs/{planned['run_id']}/events",
                json={"attempt": attempt, "events": [asdict(hello)]},
            )
            assert response.status_code == 200, response.text
            page = await services.event_log.read_after(planned["run_id"])
            assert page == [hello]
            attempts_page = await client.get(f"/api/runs/{planned['run_id']}/attempts")
            assert attempts_page.status_code == 200
            summary = attempts_page.json()
            assert summary["total"] == 1 and summary["attempts"][0]["current"]
            assert summary["attempts"][0]["runtime"]["device"] == "cpu"
            after_end = await client.get(
                f"/api/runs/{planned['run_id']}/attempts?offset=2"
            )
            assert after_end.json() == {"attempts": [], "total": 1}
            for query in (
                "limit=0",
                "limit=101",
                "offset=-1",
                "limit=1&limit=2",
                "extra=x",
            ):
                assert (
                    await client.get(f"/api/runs/{planned['run_id']}/attempts?{query}")
                ).status_code == 400
            layout = RunLayout(settings.run_root / planned["run_id"], planned["run_id"])
            layout.checkpoints.mkdir(exist_ok=True)
            source = tmp_path / "staging"
            source.mkdir()
            (source / "safe report.svg").write_text("<svg/>")
            (source / "checkpoint_selection.json").write_text(
                json.dumps({"schema_version": 1, "kind": "best"})
            )
            generation = publish_generation(layout, source, attempt=attempt, step=0)
            listing = await client.get(f"/api/runs/{planned['run_id']}/artifacts")
            assert listing.status_code == 200, listing.text
            download = await client.get(
                f"/api/runs/{planned['run_id']}/artifacts/{generation.generation}/safe%20report.svg"
            )
            assert download.status_code == 200, download.text
            assert download.content == b"<svg/>"
            assert download.headers["content-type"] == "application/octet-stream"
            assert "attachment" in download.headers["content-disposition"]
            assert "sandbox" in download.headers["content-security-policy"]
            assert "set-cookie" not in download.headers

    asyncio.run(run())
    assert services.host.outstanding == 0
    assert not services.host._thread.is_alive()
