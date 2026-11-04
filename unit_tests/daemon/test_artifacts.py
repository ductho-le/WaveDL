import asyncio
import contextlib
import os
import threading
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from wavedl_daemon.execution.budget import WorkBudget
from wavedl_daemon.execution.pool import BoundedThreadPool
from wavedl_daemon.http import artifacts as http_artifacts
from wavedl_daemon.http.artifacts import _DownloadOwner, artifact_routes
from wavedl_daemon.runs.artifacts import (
    CHUNK_BYTES,
    OpenedArtifact,
    UnsafeArtifactPath,
    artifact_path,
    list_output,
    open_output,
)
from wavedl_daemon.runs.layout import RunPaths


RUN = str(uuid4())


@pytest.fixture
def paths(tmp_path):
    paths = RunPaths(tmp_path / "runs" / RUN)
    output = paths.output
    (output / "best_checkpoint").mkdir(parents=True)
    (output / "best_checkpoint" / "model.safetensors").write_bytes(b"weights" * 20000)
    (output / "epoch_2_checkpoint").mkdir()
    (output / "epoch_2_checkpoint" / "training_meta.pkl").write_bytes(b"meta")
    (output / "fold_1").mkdir()
    (output / "fold_1" / "kết quả đẹp.svg").write_bytes(b"<svg>Unicode file</svg>")
    (output / "training_history.csv").write_bytes(b"epoch,loss\n1,0.5\n")
    (output / "notes:1.txt").write_bytes(b"unlistable")
    secret = tmp_path / "secret"
    secret.write_bytes(b"private")
    (output / "escape.txt").symlink_to(secret)
    (output / "interrupted_checkpoint").symlink_to(tmp_path, target_is_directory=True)
    return paths


def app_for(paths, pool):
    async def resolve(run_id):
        if run_id != RUN:
            raise KeyError(run_id)
        return paths

    return FastAPI(routes=artifact_routes(pool, resolve))


def client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


@pytest.mark.parametrize(
    "path",
    [
        "",
        ".",
        "../secret",
        "/secret",
        "a/../b",
        "a//b",
        "a/",
        "a\\b",
        "C:/file",
        "a\0b",
        "a\nb",
        "a\x7fb",
        "a%2fb",
    ],
)
def test_names_are_plain_relative_segments(path):
    with pytest.raises(UnsafeArtifactPath):
        artifact_path(path)


def test_listing_holds_regular_files_and_checkpoint_directories(paths):
    inventory = list_output(paths.output)
    assert [(item.path, item.size) for item in inventory.files] == [
        ("best_checkpoint/model.safetensors", 140000),
        ("epoch_2_checkpoint/training_meta.pkl", 4),
        ("fold_1/kết quả đẹp.svg", 23),
        ("training_history.csv", 17),
    ]
    assert inventory.checkpoints == ("best_checkpoint", "epoch_2_checkpoint")
    assert all(item.modified.endswith("+00:00") for item in inventory.files)


def test_a_run_without_output_lists_nothing(tmp_path):
    inventory = list_output(RunPaths(tmp_path / "missing").output)
    assert inventory.files == () and inventory.checkpoints == ()


def test_symlinks_cannot_reach_outside_output(paths):
    with pytest.raises(UnsafeArtifactPath):
        open_output(paths.output, "escape.txt")
    with pytest.raises(UnsafeArtifactPath):
        open_output(paths.output, "interrupted_checkpoint/secret")


def test_a_download_aborts_when_the_file_changes_after_open(paths):
    target = paths.output / "training_history.csv"
    artifact = open_output(paths.output, "training_history.csv")
    with target.open("ab") as stream:
        stream.write(b"2,0.4\n")
    with pytest.raises(ValueError, match="changed"):
        artifact.read()
    artifact.close()

    artifact = open_output(paths.output, "best_checkpoint/model.safetensors")
    os.truncate(paths.output / "best_checkpoint" / "model.safetensors", 10)
    with pytest.raises(ValueError, match="changed"):
        artifact.read()
    artifact.close()


def test_a_rewrite_in_place_between_chunks_is_never_delivered(paths):
    target = paths.output / "rewritten.bin"
    target.write_bytes(b"A" * CHUNK_BYTES * 2)
    artifact = open_output(paths.output, "rewritten.bin")
    assert artifact.read() == b"A" * CHUNK_BYTES
    with target.open("wb") as stream:
        stream.write(b"B" * CHUNK_BYTES * 2)
    with pytest.raises(ValueError, match="changed"):
        artifact.read()
    artifact.close()


def test_an_undecodable_name_is_left_out_of_the_listing(paths, monkeypatch):
    scandir = os.scandir
    real = paths.output / "training_history.csv"

    class Undecodable:
        name = "bad\udcff.txt"  # how POSIX decodes the byte name b"bad\xff.txt"
        path = str(real)

        def is_dir(self, follow_symlinks):
            return False

        def is_file(self, follow_symlinks):
            return True

        def stat(self, follow_symlinks):
            return real.stat()

    @contextlib.contextmanager
    def with_undecodable(directory):
        with scandir(directory) as entries:
            extra = [Undecodable()] if directory == str(paths.output) else []
            yield [*entries, *extra]

    monkeypatch.setattr(os, "scandir", with_undecodable)

    async def check():
        with BoundedThreadPool(WorkBudget()) as pool:
            async with client(app_for(paths, pool)) as http:
                return await http.get(f"/api/runs/{RUN}/artifacts")

    listing = asyncio.run(check())
    assert listing.status_code == 200
    assert "training_history.csv" in [item["path"] for item in listing.json()["files"]]
    assert len(listing.json()["files"]) == 4


def test_head_answers_the_download_headers_without_reading(paths, monkeypatch):
    reads = []
    monkeypatch.setattr(OpenedArtifact, "read", lambda self: reads.append(self) or b"")

    async def check():
        with BoundedThreadPool(WorkBudget()) as pool:
            async with client(app_for(paths, pool)) as http:
                return await http.head(
                    f"/api/runs/{RUN}/artifacts/best_checkpoint/model.safetensors"
                )

    response = asyncio.run(check())
    assert response.status_code == 200
    assert response.headers["content-length"] == "140000"
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["content-disposition"].startswith("attachment;")
    assert response.content == b""
    assert reads == []


def test_listing_and_download_over_http_run_off_the_loop(paths, monkeypatch):
    loop_thread = threading.get_ident()
    workers = []

    def spy(function):
        def call(*args):
            workers.append(threading.get_ident())
            return function(*args)

        return call

    monkeypatch.setattr(http_artifacts, "list_output", spy(list_output))
    monkeypatch.setattr(http_artifacts, "open_output", spy(open_output))

    async def check():
        with BoundedThreadPool(WorkBudget()) as pool:
            async with client(app_for(paths, pool)) as http:
                prefix = f"/api/runs/{RUN}/artifacts"
                listing = await http.get(prefix)
                assert listing.status_code == 200
                body = listing.json()
                assert set(body) == {"files", "checkpoints"}
                assert body["checkpoints"] == ["best_checkpoint", "epoch_2_checkpoint"]
                assert str(paths.run_dir) not in listing.text
                assert listing.headers["cache-control"] == "no-store"
                for item in body["files"]:
                    response = await http.get(f"{prefix}/{item['path']}")
                    assert response.status_code == 200, item
                    assert (
                        response.content == (paths.output / item["path"]).read_bytes()
                    )
                    assert response.headers["content-length"] == str(item["size"])
                    assert (
                        response.headers["content-type"] == "application/octet-stream"
                    )
                    assert response.headers["content-disposition"].startswith(
                        "attachment;"
                    )
                    assert "sandbox" in response.headers["content-security-policy"]
                    assert response.headers["x-content-type-options"] == "nosniff"

    asyncio.run(check())
    assert workers and loop_thread not in workers


def test_http_refuses_traversal_and_unknown_names(paths):
    async def check():
        with BoundedThreadPool(WorkBudget()) as pool:
            async with client(app_for(paths, pool)) as http:
                prefix = f"/api/runs/{RUN}/artifacts"
                for name in (
                    "%2e%2e/secret",
                    "fold_1/%2e%2e/%2e%2e/secret",
                    "best_checkpoint%5Cmodel.safetensors",
                    "notes:1.txt",
                    "escape.txt",
                    "interrupted_checkpoint/secret",
                    "training_history.csv%00",
                ):
                    response = await http.get(f"{prefix}/{name}")
                    assert response.status_code == 400, name
                    assert str(paths.run_dir) not in response.text
                for name in ("missing.csv", "best_checkpoint", "best_checkpoint/x"):
                    assert (await http.get(f"{prefix}/{name}")).status_code == 404
                assert (
                    await http.get(f"/api/runs/{uuid4()}/artifacts")
                ).status_code == 404
                assert (
                    await http.get("/api/runs/not-a-run/artifacts")
                ).status_code == 400

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


@pytest.mark.parametrize(
    "failure,status",
    [
        (PermissionError("secret path"), 403),
        (OSError("secret path"), 503),
        (KeyError("secret path"), 404),
    ],
)
def test_artifact_http_failure_policy_is_redacted(failure, status):
    async def resolve(run_id):
        raise failure

    async def check():
        with BoundedThreadPool(WorkBudget()) as pool:
            async with client(FastAPI(routes=artifact_routes(pool, resolve))) as http:
                response = await http.get(f"/api/runs/{RUN}/artifacts")
                assert response.status_code == status
                assert "secret path" not in response.text

    asyncio.run(check())
