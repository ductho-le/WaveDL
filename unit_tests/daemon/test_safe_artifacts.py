import asyncio
import os
import sqlite3
import threading
from urllib.parse import quote
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from wavedl.core.generations import publish_generation
from wavedl.core.layout import OperationKind, RunLayout, allocate_attempt
from wavedl.core.manifest import RuntimeSelection, build_manifest, write_manifest
from wavedl.core.safe_files import SafeDirectory
from wavedl_daemon.execution.budget import WorkBudget
from wavedl_daemon.execution.pool import BoundedThreadPool
from wavedl_daemon.http.artifacts import _DownloadOwner, artifact_routes
from wavedl_daemon.runs.artifacts import ArtifactStore, artifact_path


@pytest.fixture
def artifact_run(tmp_path):
    layout = RunLayout.create(tmp_path / "runs", str(uuid4()))
    write_manifest(
        layout,
        build_manifest(
            layout,
            operation=OperationKind.TRAIN,
            resolved_config={},
            runtime=RuntimeSelection("/usr/bin/python3", "selected", "cpu", {}),
            submission_cwd=tmp_path,
        ),
    )
    attempt = allocate_attempt(layout)
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "report.svg").write_bytes(b"<svg>active</svg>")
    (staging / "nested").mkdir()
    (staging / "nested" / "weights.bin").write_bytes(b"weights" * 20000)
    (staging / "nested" / "kết quả đẹp.svg").write_bytes(b"<svg>Unicode file</svg>")
    for punctuation in "+@()!,;'$&=":
        (staging / f"model{punctuation}report.svg").write_bytes(
            b"<svg>punctuation</svg>"
        )
    publication = publish_generation(layout, staging, attempt=attempt, step=1)
    return layout, publication, ArtifactStore(lambda run_id: layout)


@pytest.mark.parametrize(
    "path",
    [
        "../secret",
        "/secret",
        "a/../b",
        "a//b",
        "a\\b",
        "C:/file",
        "//host/file",
        "%2e%2e/x",
        "%252e/x",
        "a\0b",
        "a\nb",
        "a\tb",
        "a\x7fb",
    ],
)
def test_download_path_has_one_strict_portable_spelling(path):
    with pytest.raises(ValueError):
        artifact_path(path)


def test_only_complete_verified_generations_are_listed(artifact_run):
    layout, publication, store = artifact_run
    (layout.checkpoints / "gen-000002").mkdir()
    assert store.list(layout.run_id) == [publication]
    (publication.path / "report.svg").write_bytes(b"changed")
    assert store.list(layout.run_id) == []


def test_replacement_after_listing_is_rejected(artifact_run, tmp_path):
    layout, publication, store = artifact_run
    assert store.list(layout.run_id)
    target = publication.path / "report.svg"
    target.unlink()
    secret = tmp_path / "secret"
    secret.write_bytes(b"private")
    target.symlink_to(secret)
    with pytest.raises((ValueError, FileNotFoundError)):
        store.open(layout.run_id, publication.generation, "report.svg")


def test_replacement_after_open_cannot_redirect_stream(artifact_run, tmp_path):
    layout, publication, store = artifact_run
    with store.open(layout.run_id, publication.generation, "report.svg") as stream:
        target = publication.path / "report.svg"
        secret = tmp_path / "secret"
        secret.write_bytes(b"private")
        if os.name == "nt":
            with pytest.raises(PermissionError) as denied:
                target.unlink()
            assert denied.value.winerror == 32
            assert target.read_bytes() == b"<svg>active</svg>"
        else:
            target.unlink()
            target.symlink_to(secret)
        assert stream.read() == b"<svg>active</svg>"
        assert stream.read() == b""
        assert secret.read_bytes() == b"private"


def test_replacement_between_inventory_validation_and_stream_open_is_rejected(
    artifact_run, tmp_path, monkeypatch
):
    layout, publication, store = artifact_run
    original = SafeDirectory.open_file
    secret = tmp_path / "secret"
    secret.write_bytes(b"private")

    def swapped(directory, name, **kwargs):
        if name == "report.svg":
            target = publication.path / name
            target.unlink()
            target.symlink_to(secret)
        return original(directory, name, **kwargs)

    monkeypatch.setattr(SafeDirectory, "open_file", swapped)
    with pytest.raises((ValueError, FileNotFoundError)):
        store.open(layout.run_id, publication.generation, "report.svg")


def test_open_stream_never_sends_bytes_appended_after_validation(artifact_run):
    layout, publication, store = artifact_run
    with store.open(layout.run_id, publication.generation, "report.svg") as stream:
        path = publication.path / "report.svg"
        if os.name == "nt":
            with pytest.raises(PermissionError) as denied, path.open("ab") as target:
                target.write(b"not-in-published-inventory")
            assert denied.value.errno == 13
            assert getattr(denied.value, "winerror", None) in (None, 32)
            assert path.read_bytes() == b"<svg>active</svg>"
        else:
            with path.open("ab") as target:
                target.write(b"not-in-published-inventory")
        assert stream.read() == b"<svg>active</svg>"
        assert stream.read() == b""


def test_download_is_attachment_and_io_runs_off_loop(artifact_run, monkeypatch):
    layout, publication, store = artifact_run
    main_thread = threading.get_ident()
    real_open = store.open

    def opened(*args):
        assert threading.get_ident() != main_thread
        result = real_open(*args)
        real_read = result.read

        def read(size):
            assert threading.get_ident() != main_thread
            return real_read(size)

        result.read = read
        return result

    monkeypatch.setattr(store, "open", opened)

    async def check():
        with BoundedThreadPool(WorkBudget()) as pool:
            app = FastAPI(routes=artifact_routes(store, pool))
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                prefix = f"/api/runs/{layout.run_id}/artifacts"
                listing = await client.get(prefix)
                assert str(layout.run_dir) not in listing.text
                response = await client.get(
                    f"{prefix}/{publication.generation}/report.svg"
                )
                assert response.content == b"<svg>active</svg>"
                assert response.headers["content-type"] == "application/octet-stream"
                assert response.headers["content-disposition"].startswith("attachment;")
                assert "sandbox" in response.headers["content-security-policy"]
                assert response.headers["x-content-type-options"] == "nosniff"
                invalid = await client.get(
                    f"{prefix}/{publication.generation}/%2573ecret"
                )
                assert invalid.status_code == 400

    asyncio.run(check())


def test_pending_io_cancellation_retains_and_then_closes_owned_handle():
    owner = _DownloadOwner()
    started, release, closed = threading.Event(), threading.Event(), threading.Event()

    class Artifact:
        def close(self):
            closed.set()

    owner.artifact = Artifact()

    def blocked():
        started.set()
        assert release.wait(2)

    worker = threading.Thread(target=lambda: owner.run(blocked))
    worker.start()
    assert started.wait(2)
    owner.close()
    assert not closed.is_set()
    release.set()
    worker.join(2)
    assert closed.is_set()


def test_sql_metadata_resolution_stays_on_owner_and_once_per_request(artifact_run):
    layout, publication, _ = artifact_run
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE runs(run_id TEXT PRIMARY KEY, directory TEXT)")
    connection.execute(
        "INSERT INTO runs VALUES (?,?)", (layout.run_id, str(layout.run_dir))
    )
    owner_thread = threading.get_ident()
    resolutions = []

    async def resolve_layout(run_id):
        assert threading.get_ident() == owner_thread
        resolutions.append(run_id)
        row = connection.execute(
            "SELECT directory FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return RunLayout(layout.run_dir, run_id)

    def forbidden_lookup(run_id):
        raise AssertionError("filesystem worker must not access registry lookup")

    async def check():
        with BoundedThreadPool(WorkBudget()) as pool:
            app = FastAPI(
                routes=artifact_routes(
                    ArtifactStore(forbidden_lookup), pool, resolve_layout=resolve_layout
                )
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                prefix = f"/api/runs/{layout.run_id}/artifacts"
                listing, download = await asyncio.gather(
                    client.get(prefix),
                    client.get(f"{prefix}/{publication.generation}/report.svg"),
                )
                assert listing.status_code == download.status_code == 200
                assert resolutions == [layout.run_id, layout.run_id]
                unknown = await client.get(f"/api/runs/{uuid4()}/artifacts")
                assert unknown.status_code == 404

    try:
        asyncio.run(check())
    finally:
        connection.close()


@pytest.mark.parametrize(
    "failure,status",
    [
        (PermissionError("secret path"), 403),
        (OSError("secret path"), 503),
        (KeyError("secret path"), 404),
    ],
)
def test_artifact_http_failure_policy_is_redacted(artifact_run, failure, status):
    layout, _, store = artifact_run

    async def resolve_layout(run_id):
        raise failure

    async def check():
        with BoundedThreadPool(WorkBudget()) as pool:
            app = FastAPI(
                routes=artifact_routes(store, pool, resolve_layout=resolve_layout)
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get(f"/api/runs/{layout.run_id}/artifacts")
                assert response.status_code == status
                assert "secret path" not in response.text

    asyncio.run(check())


def test_encoded_spaces_unicode_and_nested_files_are_downloadable(artifact_run):
    layout, publication, store = artifact_run

    async def check():
        with BoundedThreadPool(WorkBudget()) as pool:
            app = FastAPI(routes=artifact_routes(store, pool))
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                prefix = f"/api/runs/{layout.run_id}/artifacts/{publication.generation}"
                response = await client.get(f"{prefix}/nested/kết quả đẹp.svg")
                assert response.status_code == 200
                assert response.content == b"<svg>Unicode file</svg>"
                assert response.headers["content-type"] == "application/octet-stream"
                assert response.headers["content-disposition"].startswith("attachment;")
                assert "sandbox" in response.headers["content-security-policy"]
                for encoded in (
                    "nested%2Fweights.bin",
                    "nested%5Cweights.bin",
                    "%252e%252e/secret",
                    "%2e%2e/secret",
                    "nested/weights.bin%0A",
                ):
                    invalid = await client.get(f"{prefix}/{encoded}")
                    assert invalid.status_code in {400, 404}
                    assert str(layout.run_dir) not in invalid.text

    asyncio.run(check())


def test_literal_and_encoded_legal_path_punctuation_download_same_inventory_member(
    artifact_run,
):
    layout, publication, store = artifact_run

    async def check():
        with BoundedThreadPool(WorkBudget()) as pool:
            app = FastAPI(routes=artifact_routes(store, pool))
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                prefix = f"/api/runs/{layout.run_id}/artifacts/{publication.generation}"
                for punctuation in "+@()!,;'$&=":
                    member = f"model{punctuation}report.svg"
                    for spelling in (member, quote(member, safe="")):
                        response = await client.get(f"{prefix}/{spelling}")
                        assert response.status_code == 200, (
                            member,
                            spelling,
                            response.text,
                        )
                        assert response.content == b"<svg>punctuation</svg>"
                        assert response.headers["content-disposition"].startswith(
                            "attachment;"
                        )
                        assert (
                            response.headers["content-type"]
                            == "application/octet-stream"
                        )

    asyncio.run(check())
