"""Static metadata stays dependency-light; browsing never escapes its roots."""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from wavedl_daemon.execution.budget import BudgetExhausted, WorkBudget
from wavedl_daemon.execution.pool import BoundedThreadPool, WorkDeadlineExceeded
from wavedl_daemon.execution.service_host import ServiceHost
from wavedl_daemon.http.app import create_app
from wavedl_daemon.http.catalog import catalog_routes
from wavedl_daemon.runs.catalog import (
    AsyncCatalogService,
    BrowseForbidden,
    BrowseMissing,
    CatalogService,
)
from wavedl_daemon.storage.settings import Settings


def test_resolve_selected_member_returns_path_without_reading_payload(
    tmp_path, monkeypatch
):
    from wavedl.core.safe_files import SafeFile

    folder = tmp_path / "folder"
    folder.mkdir()
    target = folder / "large-data.npz"
    target.write_bytes(b"never load this payload")
    monkeypatch.setattr(
        SafeFile, "read", lambda *args: pytest.fail("Resolve must not load payload")
    )
    catalog = CatalogService({"data": tmp_path})
    assert catalog.resolve("data", "folder/large-data.npz") == {
        "path": str(target),
        "kind": "file",
        "root": "data",
        "relative_path": "folder/large-data.npz",
    }
    assert catalog.resolve("data", "folder")["kind"] == "directory"
    assert catalog.resolve("data")["path"] == str(tmp_path)
    assert catalog.browse()["roots"] == [{"id": "data"}]


@pytest.mark.parametrize(
    "path",
    [
        "../outside",
        "/absolute",
        "folder/../target",
        "folder//target",
        "folder\\target",
        "C:target",
        "./target",
    ],
)
def test_resolve_refuses_nonordinary_relative_paths(tmp_path, path):
    with pytest.raises(BrowseForbidden):
        CatalogService({"data": tmp_path}).resolve("data", path)


def test_resolve_missing_symlink_special_and_unreadable_are_visible(
    tmp_path, monkeypatch
):
    from wavedl.core.safe_files import SafeDirectory

    catalog = CatalogService({"data": tmp_path})
    with pytest.raises(BrowseMissing):
        catalog.resolve("data", "missing")
    (tmp_path / "link").symlink_to(tmp_path)
    with pytest.raises(BrowseForbidden):
        catalog.resolve("data", "link")
    if hasattr(os, "mkfifo"):
        os.mkfifo(tmp_path / "pipe")
        with pytest.raises(BrowseForbidden):
            catalog.resolve("data", "pipe")
    (tmp_path / "private").write_bytes(b"private")
    monkeypatch.setattr(
        SafeDirectory,
        "open_file",
        lambda *args: (_ for _ in ()).throw(PermissionError("hidden-path")),
    )
    with pytest.raises(BrowseForbidden, match="unreadable"):
        catalog.resolve("data", "private")


def test_resolve_ancestry_swap_never_returns_replacement_path(tmp_path, monkeypatch):
    from wavedl.core.safe_files import SafeDirectory

    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "data").write_bytes(b"data")
    original = SafeDirectory.open_file
    opened, closed, blocked = [], [], []

    def open_and_swap(directory, name, **kwargs):
        file = original(directory, name, **kwargs)
        opened.append(file)
        close = file.close

        def tracked_close():
            close()
            closed.append(file)

        file.close = tracked_close
        try:
            if os.name == "nt":
                with pytest.raises(PermissionError) as denied:
                    folder.rename(tmp_path / "moved")
                assert denied.value.winerror == 32
                blocked.append(True)
                assert not (tmp_path / "moved").exists()
                assert (folder / "data").read_bytes() == b"data"
                raise denied.value
            folder.rename(tmp_path / "moved")
            folder.mkdir()
            return file
        except BaseException:
            # Until this hook returns, ExitStack has not received the resource.
            file.close()
            raise

    monkeypatch.setattr(SafeDirectory, "open_file", open_and_swap)
    with pytest.raises(BrowseForbidden):
        CatalogService({"data": tmp_path}).resolve("data", "folder/data")
    assert len(opened) == 1
    assert closed == opened
    assert all(file.handle is None for file in opened)
    if os.name == "nt":
        assert blocked == [True]
        assert (folder / "data").read_bytes() == b"data"
        (folder / "data").unlink()  # No pinned handle survives resolver refusal.
        assert not (folder / "data").exists()
    else:
        assert (tmp_path / "moved" / "data").read_bytes() == b"data"
        assert not list(folder.iterdir())


def test_resolve_route_is_authenticated_strict_bounded_and_redacted(tmp_path):
    target = tmp_path / "data.npz"
    target.write_bytes(b"test")
    app = create_app(
        Settings(tmp_path / "state", tmp_path / "runs"),
        token="resolve-secret",
        daemon_id="test",
        build_id="test",
    )
    app.router.routes.extend(catalog_routes(CatalogService({"data": tmp_path})))

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost:8420"
        ) as client:
            url = "/api/browse/resolve"
            response = await client.post(url, json={"root": "data", "path": "data.npz"})
            assert response.status_code == 401 and str(tmp_path) not in response.text
            client.headers["Authorization"] = "Bearer resolve-secret"
            response = await client.post(url, json={"root": "data", "path": "data.npz"})
            assert response.status_code == 200
            assert response.json()["path"] == str(target)
            for body in (
                {"root": "data"},
                {"root": "data", "path": 1},
                {"root": "data", "path": "", "extra": True},
            ):
                assert (await client.post(url, json=body)).status_code == 400
            assert (
                await client.post(url, json={"root": "other", "path": ""})
            ).status_code == 403
            assert (
                await client.post(url, json={"root": "data", "path": "missing"})
            ).status_code == 404
            assert (
                await client.post(
                    url,
                    content=b"x" * 8193,
                    headers={"Content-Type": "application/json"},
                )
            ).status_code == 413

    asyncio.run(scenario())


def test_async_resolve_runs_through_shared_bounded_pool(tmp_path):
    (tmp_path / "data").write_bytes(b"data")
    with BoundedThreadPool(WorkBudget()) as pool:
        host = ServiceHost(
            lambda: SimpleNamespace(
                catalog=CatalogService({"data": tmp_path}), pool=pool
            )
        )

        async def scenario():
            await host.start()
            try:
                selected = await AsyncCatalogService(host).resolve("data", "data")
                assert selected["path"] == str(tmp_path / "data")
            finally:
                await host.close()

        asyncio.run(scenario())


def test_model_catalog_is_bounded_isolated_and_not_runtime_evidence():
    catalog = CatalogService({})
    result = catalog.models()
    assert result["source"] == "bundled_metadata"
    assert result["runtime_verified"] is False
    assert "cnn" in {item["id"] for item in result["models"]}
    assert result["categories"]
    assert len(result["models"]) < 1000
    result["models"][0]["id"] = "corrupted"
    assert "corrupted" not in {item["id"] for item in catalog.models()["models"]}


def test_daemon_owned_catalog_preserves_complete_descriptive_payload():
    result = CatalogService({}).models()
    # Frozen complete public payload, including descriptions, categories, and order.
    # This is independent of the retained, unshipped Qt source tree.
    assert len(result["models"]) == 72
    encoded = json.dumps(result, allow_nan=False, sort_keys=True).encode()
    assert hashlib.sha256(encoded).hexdigest() == (
        "0698198e25d6dcacac2b3779280daf94fea3612650a2e732387f182fddc9527b"
    )


def test_model_metadata_ids_match_registered_model_ids():
    from wavedl.models import list_models

    metadata_ids = {model["id"] for model in CatalogService({}).models()["models"]}
    assert len(metadata_ids) == 72
    assert metadata_ids == set(list_models())


def test_model_catalog_imports_neither_torch_nor_gui():
    source = "from wavedl_daemon.runs.catalog import CatalogService; import sys; CatalogService({}).models(); assert not any(name == 'torch' or name.startswith(('torch.', 'PySide6', 'qfluentwidgets', 'wavedl_gui')) for name in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_metadata_is_loaded_as_json_not_executed_as_code(tmp_path):
    source = tmp_path / "models.json"
    source.write_text('{"categories": {}, "models": [{"id": "safe"}]}')
    result = CatalogService({}, model_metadata_path=source).models()
    assert result["models"] == [{"id": "safe"}]
    source.write_text("raise RuntimeError('must not execute')")
    with pytest.raises(json.JSONDecodeError):
        CatalogService({}, model_metadata_path=source).models()


def test_browse_explicit_roots_no_hostpaths_stable_pages_and_specialfiles(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    (root / "folder").mkdir()
    (root / "a.npz").write_bytes(b"sensitive data never read")
    (root / "b.txt").write_bytes(b"contents")
    (root / "linked").symlink_to(tmp_path)
    if hasattr(os, "mkfifo"):
        os.mkfifo(root / "pipe")
    catalog = CatalogService({"data": root})
    assert catalog.browse() == {"roots": [{"id": "data"}]}
    page = catalog.browse("data", limit=2)
    assert page["entries"] == [
        {"name": "folder", "kind": "directory"},
        {"name": "a.npz", "kind": "file"},
    ]
    assert catalog.browse("data", limit=2, offset=2)["entries"] == [
        {"name": "b.txt", "kind": "file"}
    ]
    assert str(root) not in json.dumps(page)
    assert "sensitive" not in json.dumps(page)
    assert catalog.browse("data", "folder")["entries"] == []


@pytest.mark.parametrize(
    "path",
    [
        "..",
        "../outside",
        "/tmp",
        "folder/..",
        "folder//child",
        "folder/",
        r"folder\child",
        "C:drive",
    ],
)
def test_browse_refuses_lexical_escape(tmp_path, path):
    with pytest.raises(BrowseForbidden):
        CatalogService({"data": tmp_path}).browse("data", path)


def test_browse_refuses_symlink_directory(tmp_path):
    (tmp_path / "linked").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(BrowseForbidden):
        CatalogService({"data": tmp_path}).browse("data", "linked")


def test_large_directory_is_not_silently_truncated(tmp_path):
    for name in ("one", "two", "three"):
        (tmp_path / name).touch()
    with pytest.raises(BudgetExhausted):
        CatalogService({"data": tmp_path}, max_entries=2).browse("data", limit=1)


def test_replaced_nested_directory_is_rejected_before_return(tmp_path, monkeypatch):
    from wavedl.core.safe_files import SafeDirectory

    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "secret.txt").touch()
    original = SafeDirectory.entries
    attempted = []

    def entries(anchor):
        yield from original(anchor)
        if anchor.path.samefile(folder):
            attempted.append(True)
            if os.name == "nt":
                with pytest.raises(PermissionError) as denied:
                    folder.rename(tmp_path / "moved")
                assert denied.value.winerror == 32
                assert not (tmp_path / "moved").exists()
                assert (folder / "secret.txt").read_bytes() == b""
                return
            folder.rename(tmp_path / "moved")
            folder.mkdir()

    monkeypatch.setattr(SafeDirectory, "entries", entries)
    if os.name == "nt":
        result = CatalogService({"data": tmp_path}).browse("data", "folder")
        assert [entry["name"] for entry in result["entries"]] == ["secret.txt"]
    else:
        with pytest.raises(BrowseForbidden, match="ancestry changed"):
            CatalogService({"data": tmp_path}).browse("data", "folder")
    assert attempted == [True]


def test_http_auth_query_bounds_unreadable_and_redacted_errors(tmp_path, monkeypatch):
    service = CatalogService({"data": tmp_path}, max_entries=2)
    app = create_app(
        Settings(tmp_path / "state", tmp_path / "runs"),
        token="catalog-test-token",
        daemon_id="test",
        build_id="test",
    )
    app.router.routes.extend(catalog_routes(service))

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost:8420"
        ) as client:
            assert (await client.get("/api/models")).status_code == 401
            client.headers["Authorization"] = "Bearer catalog-test-token"
            assert (await client.get("/api/models")).status_code == 200
            assert (await client.get("/api/browse")).json() == {
                "roots": [{"id": "data"}]
            }
            for query in ("root=unknown", "root=data&path=../outside"):
                assert (await client.get("/api/browse?" + query)).status_code == 403
            assert (
                await client.get("/api/browse?root=data&path=missing")
            ).status_code == 404
            for query in (
                "root=data&limit=0",
                "root=data&limit=201",
                "root=data&extra=1",
                "root=data&root=data",
                "root=data&offset=-1",
            ):
                assert (await client.get("/api/browse?" + query)).status_code == 400

            def fail(*args):
                raise PermissionError("private credential location")

            monkeypatch.setattr(service, "browse", fail)
            response = await client.get("/api/browse?root=data")
            assert response.status_code == 403
            assert "private" not in response.text

    asyncio.run(run())


def test_async_catalog_adapter_runs_filesystem_off_owner_thread(tmp_path, monkeypatch):
    service = CatalogService({"data": tmp_path})
    calls = []
    original = service.browse

    def browse(*args):
        calls.append(threading.get_ident())
        return original(*args)

    monkeypatch.setattr(service, "browse", browse)
    owner_threads = []
    pool = BoundedThreadPool(WorkBudget())

    def factory():
        owner_threads.append(threading.get_ident())
        return SimpleNamespace(catalog=service, pool=pool, close=pool.shutdown)

    host = ServiceHost(factory)

    async def run():
        await host.start()
        adapter = AsyncCatalogService(host)
        assert (await adapter.browse("data"))["entries"] == []
        with pytest.raises(BrowseMissing):
            await adapter.browse("data", "missing")
        assert (await adapter.models())["runtime_verified"] is False
        await host.close()

    asyncio.run(run())
    assert calls[0] != threading.get_ident()
    assert calls[0] != owner_threads[0]


def test_catalog_root_configuration_is_explicit(tmp_path):
    with pytest.raises(ValueError):
        CatalogService({"data": Path("relative")})
    with pytest.raises(ValueError):
        CatalogService({"../bad": tmp_path})
    assert CatalogService({}).browse() == {"roots": []}


@pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0, reason="Requires POSIX non-root permissions"
)
def test_browse_rejects_locked_directory_and_uses_bounded_default(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir(mode=0o700)
    catalog = CatalogService({"data": tmp_path})
    assert catalog.max_entries == 1000
    locked.chmod(0)
    try:
        with pytest.raises(PermissionError):
            catalog.browse("data", "locked")
    finally:
        locked.chmod(0o700)


def test_preview_child_identity_changed_maps_to_documented_refusal(
    tmp_path, monkeypatch
):
    from wavedl_daemon.runs.preview_service import (
        DatasetPreviewService,
        PreviewIdentityChanged,
    )
    from wavedl_daemon.supervision.base import ProcessResult

    (tmp_path / "data.npz").write_bytes(b"selected file")
    observed = []

    def refused(spec, budget, *, token):
        observed.append(spec)
        return ProcessResult(1, b'{"code":"identity_changed"}', b"")

    monkeypatch.setattr("wavedl_daemon.runs.preview_service.run_command", refused)
    with BoundedThreadPool(WorkBudget()) as pool:
        service = DatasetPreviewService(
            sys.executable, Path.cwd(), pool, CatalogService({"data": tmp_path})
        )
        with pytest.raises(PreviewIdentityChanged, match="Dataset identity changed"):
            asyncio.run(
                service.preview({"root": "data", "path": "data.npz", "view": "all"})
            )
    assert len(observed) == 1


def test_catalog_worker_cancellation_retains_owner_and_pool_admission(tmp_path):
    entered, gate = threading.Event(), threading.Event()
    service = CatalogService({"data": tmp_path})

    def block(*args):
        entered.set()
        gate.wait(3)
        return {"entries": []}

    service.browse = block
    budget = WorkBudget(max_workers=1, max_pending=0, timeout_seconds=0.05)
    pool = BoundedThreadPool(budget)
    host = ServiceHost(
        lambda: SimpleNamespace(catalog=service, pool=pool, close=pool.shutdown),
        budget=budget,
    )

    async def run():
        await host.start()
        adapter = AsyncCatalogService(host)
        try:
            with pytest.raises(WorkDeadlineExceeded):
                await adapter.browse("data")
            assert entered.is_set()
            assert pool.outstanding == host.outstanding == 1
            with pytest.raises(BudgetExhausted):
                await adapter.browse("data")
        finally:
            gate.set()
            await host.close()
        assert pool.outstanding == host.outstanding == 0

    asyncio.run(run())
