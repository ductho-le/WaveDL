"""Abrupt writer death across publication cuts and authenticated artifact views.

These are assembled local process/filesystem/HTTP probes, not certification of
power-loss durability or a site's filesystem. The child uses core publication;
the consumer uses real authenticated artifact routes with a test-owned layout
resolver, not a full daemon lifecycle or a numerical training backend.
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from wavedl.core.generations import publish_generation, published_generations
from wavedl.core.layout import OperationKind, RunLayout, allocate_attempt
from wavedl.core.manifest import RuntimeSelection, build_manifest, write_manifest
from wavedl_daemon.execution.budget import WorkBudget
from wavedl_daemon.execution.pool import BoundedThreadPool
from wavedl_daemon.http.app import SecurityBoundary
from wavedl_daemon.http.artifacts import artifact_routes
from wavedl_daemon.runs.artifacts import ArtifactStore


CRASH_WRITER = """
import os
import sys
from pathlib import Path
from wavedl.core import generations
from wavedl.core.layout import RunLayout
layout = RunLayout(Path(sys.argv[1]), sys.argv[2])
source, cut = Path(sys.argv[3]), sys.argv[4]
real_copy = generations._copy_file
real_marker = generations.write_once
def copy(incoming, outgoing):
    if cut == "partial-copy":
        with outgoing.open("xb") as stream:
            stream.write(b"partial")
            stream.flush()
            os.fsync(stream.fileno())
        os._exit(73)
    return real_copy(incoming, outgoing)
def marker(path, content):
    if cut == "before-marker":
        os._exit(73)
    real_marker(path, content)
    os._exit(73)
generations._copy_file = copy
generations.write_once = marker
generations.publish_generation(layout, source, attempt=1, step=2)
raise AssertionError("Crash injection was not reached")
"""


@pytest.mark.integration
@pytest.mark.parametrize("cut", ["partial-copy", "before-marker", "after-marker"])
def test_abrupt_publication_crash_preserves_http_inventory_and_retry(tmp_path, cut):
    """No unwind runs; incomplete bytes stay unservable, valid markers survive."""
    layout = RunLayout.create(tmp_path / "runs", str(uuid4()))
    write_manifest(
        layout,
        build_manifest(
            layout,
            operation=OperationKind.TRAIN,
            resolved_config={},
            runtime=RuntimeSelection(sys.executable, "existing", "cpu", {}),
            submission_cwd=tmp_path,
        ),
    )
    attempt = allocate_attempt(layout)
    source = tmp_path / "staging"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"prior weights")
    prior = publish_generation(layout, source, attempt=attempt, step=1)
    prior_marker = (prior.path / "COMPLETE.json").read_bytes()
    (source / "weights.bin").write_bytes(b"replacement weights")
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            CRASH_WRITER,
            str(layout.run_dir),
            layout.run_id,
            str(source),
            cut,
        ],
        env=os.environ
        | {"PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 73, result.stderr.decode(errors="replace")
    interrupted = layout.checkpoints / "gen-000002"
    assert interrupted.is_dir()
    marker = interrupted / "COMPLETE.json"
    assert marker.exists() == (cut == "after-marker")
    interrupted_bytes = (interrupted / "weights.bin").read_bytes()
    assert interrupted_bytes == (
        b"partial" if cut == "partial-copy" else b"replacement weights"
    )
    expected = ["gen-000001"] + (["gen-000002"] if marker.exists() else [])
    assert [item.generation for item in published_generations(layout)] == expected

    async def view(expected, *, include_retry=False):
        # Recreate the consumer and worker pool on each pass: inventory cannot
        # rely on a writer's unwinding or the consumer's pre-crash cache.
        store = ArtifactStore(lambda run_id: layout)
        with BoundedThreadPool(WorkBudget()) as pool:
            app = SecurityBoundary(
                FastAPI(routes=artifact_routes(store, pool)),
                token=b"publication-test-token",
                allowed_hosts=("localhost",),
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://localhost:8420",
                headers={"Authorization": "Bearer publication-test-token"},
            ) as client:
                prefix = f"/api/runs/{layout.run_id}/artifacts"
                denied = await client.get(prefix, headers={"Authorization": ""})
                assert denied.status_code == 401
                listing = await client.get(prefix)
                assert listing.status_code == 200, listing.text
                assert [
                    item["generation"] for item in listing.json()["generations"]
                ] == expected
                old = await client.get(f"{prefix}/gen-000001/weights.bin")
                assert old.status_code == 200
                assert old.content == b"prior weights"
                crashed = await client.get(f"{prefix}/gen-000002/weights.bin")
                assert crashed.status_code == (200 if cut == "after-marker" else 404)
                if crashed.status_code == 200:
                    assert crashed.content == b"replacement weights"
                if include_retry:
                    retry = await client.get(f"{prefix}/gen-000003/weights.bin")
                    assert retry.status_code == 200
                    assert retry.content == b"retry weights"

    asyncio.run(asyncio.wait_for(view(expected), timeout=5))
    (source / "weights.bin").write_bytes(b"retry weights")
    retry = publish_generation(layout, source, attempt=attempt, step=3)
    assert retry.generation == "gen-000003"
    asyncio.run(
        asyncio.wait_for(
            view([*expected, retry.generation], include_retry=True), timeout=5
        )
    )
    assert (prior.path / "COMPLETE.json").read_bytes() == prior_marker
    assert (interrupted / "weights.bin").read_bytes() == interrupted_bytes
    assert marker.exists() == (cut == "after-marker")
