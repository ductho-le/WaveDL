"""Scoped supervised previews never expose arbitrary arrays or filesystem paths."""

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat
from starlette.testclient import TestClient

from unit_tests.daemon.conftest import route_app
from wavedl.core.dataset_preview import PreviewLimits
from wavedl_daemon.execution.budget import BudgetExhausted, WorkBudget
from wavedl_daemon.execution.pool import BoundedThreadPool, WorkCancelled
from wavedl_daemon.http.app import create_app
from wavedl_daemon.http.preview import preview_routes
from wavedl_daemon.runs.catalog import CatalogService
from wavedl_daemon.runs.preview_authority import PreviewForbidden
from wavedl_daemon.runs.preview_service import (
    DatasetPreviewService,
    PreviewBadRequest,
    PreviewRefused,
    PreviewTooLarge,
    validate_preview,
)
from wavedl_daemon.runs.runtime_service import HostedServices
from wavedl_daemon.storage.settings import Settings


def make_service(tmp_path, pool, **kwargs):
    return DatasetPreviewService(
        sys.executable, Path.cwd(), pool, CatalogService({"data": tmp_path}), **kwargs
    )


def request(view="all", path="data.npz"):
    return {"root": "data", "path": path, "view": view}


def write_dataset(tmp_path):
    np.savez(
        tmp_path / "data.npz",
        inputs=np.arange(24).reshape(6, 4),
        outputs=np.arange(12).reshape(6, 2),
    )


@pytest.mark.parametrize("view", sorted(DatasetPreviewService.ALLOWED_VIEWS))
def test_actual_npz_six_selectors(tmp_path, view):
    write_dataset(tmp_path)
    with BoundedThreadPool(WorkBudget(timeout_seconds=10)) as pool:
        result = asyncio.run(make_service(tmp_path, pool).preview(request(view)))
    assert result["schema_version"] == 1
    assert result["input_shape"] == [6, 4]
    assert result["selection"] == {"root": "data", "path": "data.npz"}
    assert result["view"] == view
    assert set(result["identity"]) == {"st_dev", "st_ino", "st_size", "st_mtime_ns"}
    assert "raw" not in result


@pytest.mark.parametrize("suffix", [".npz", ".mat", ".h5", ".hdf5"])
@pytest.mark.parametrize("rows,sample_cap", [(1, 512), (3, 1)])
def test_one_sample_multi_output_correlation_is_unavailable(
    tmp_path, suffix, rows, sample_cap
):
    """A valid single sampled row cannot yield a correlation matrix."""
    path = tmp_path / ("data" + suffix)
    arrays = {
        "X": np.arange(rows * 4).reshape(rows, 4),
        "Y": np.arange(rows * 2).reshape(rows, 2),
    }
    if suffix == ".npz":
        np.savez(path, **arrays)
    elif suffix == ".mat":
        savemat(path, arrays, do_compression=True)
    else:
        import h5py

        with h5py.File(path, "w") as file:
            for name, array in arrays.items():
                file[name] = array
    with BoundedThreadPool(WorkBudget(timeout_seconds=10)) as pool:
        service = make_service(
            tmp_path,
            pool,
            limits=replace(
                PreviewLimits(), max_sample_rows=sample_cap, max_output_bytes=200_000
            ),
        )
        result = asyncio.run(service.preview(request(path=path.name)))
    assert len(result["sample_indices"]) == 1
    assert result["correlation"] == {"feature_indices": [0, 1], "matrix": None}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"root": "data", "path": "data.npz"},
        {**request(), "limits": {}},
        {**request(), "view": "expression"},
        {**request(), "path": 1},
    ],
)
def test_fixed_schema_rejects_extra_fields_and_unknown_views(tmp_path, body):
    with BoundedThreadPool(WorkBudget()) as pool, pytest.raises(PreviewBadRequest):
        asyncio.run(make_service(tmp_path, pool).preview(body))


@pytest.mark.parametrize(
    "path", ["../outside.npz", "/absolute.npz", "folder//data.npz", "folder\\data.npz"]
)
def test_traversal_rejected(tmp_path, path):
    with BoundedThreadPool(WorkBudget()) as pool, pytest.raises(PreviewForbidden):
        asyncio.run(make_service(tmp_path, pool).preview(request(path=path)))


def test_symlink_never_followed(tmp_path):
    write_dataset(tmp_path)
    (tmp_path / "link.npz").symlink_to(tmp_path / "data.npz")
    with BoundedThreadPool(WorkBudget()) as pool, pytest.raises(ValueError):
        asyncio.run(make_service(tmp_path, pool).preview(request(path="link.npz")))


def test_no_content_cache(tmp_path):
    write_dataset(tmp_path)
    with BoundedThreadPool(WorkBudget(timeout_seconds=10)) as pool:
        service = make_service(tmp_path, pool)
        first = asyncio.run(service.preview(request()))
        np.savez(tmp_path / "data.npz", inputs=np.ones((4, 3)), outputs=np.ones((4, 1)))
        second = asyncio.run(service.preview(request()))
    assert first["identity"] != second["identity"]
    assert second["input_shape"] == [4, 3]


@pytest.mark.parametrize(
    "content",
    [
        b"{}",
        b"[]",
        b'{"schema_version":1,"schema_version":1}',
        b'{"schema_version":NaN}',
        b"invalid",
    ],
)
def test_untrusted_process_json_is_strict(content):
    with pytest.raises(PreviewRefused):
        validate_preview(content, PreviewLimits())


def test_json_byte_cap():
    with pytest.raises(PreviewTooLarge):
        validate_preview(b" " * 100, replace(PreviewLimits(), max_output_bytes=99))


def test_pool_admission_shared(tmp_path):
    ready, release = threading.Event(), threading.Event()
    with BoundedThreadPool(WorkBudget(max_workers=1, max_pending=0)) as pool:

        def occupied(token):
            ready.set()
            release.wait(5)

        handle = pool.submit(occupied)
        assert ready.wait(2)
        try:
            with pytest.raises(BudgetExhausted):
                asyncio.run(make_service(tmp_path, pool).preview(request()))
        finally:
            release.set()
            handle.result(5)


def test_service_import_has_no_numeric_or_gui_dependencies():
    # -S bypasses editable-install .pth files, so expose only the checkout source.
    source_root = Path(__file__).resolve().parents[2] / "src"
    code = "import sys; import wavedl_daemon.runs.preview_service; assert not any(x in sys.modules for x in ['numpy','torch','h5py','scipy','PySide6'])"
    subprocess.run(
        [sys.executable, "-S", "-c", code],
        check=True,
        capture_output=True,
        timeout=10,
        env=os.environ | {"PYTHONPATH": str(source_root)},
    )


def test_http_no_store_and_refusal_codes():
    class Previewer:
        async def preview(self, body):
            if body.get("view") == "bad":
                raise PreviewBadRequest("secret")
            if body.get("view") == "refused":
                raise PreviewRefused("secret")
            return {
                "schema_version": 1,
                "identity": {
                    "st_dev": 1,
                    "st_ino": 1,
                    "st_size": 1,
                    "st_mtime_ns": 1,
                },
                "format": "npz",
                "input_key": "inputs",
                "output_key": "outputs",
                "input_shape": [1, 1],
                "output_shape": [1, 1],
                "sample_indices": [0],
                "input_histograms": [],
                "output_histograms": [],
                "correlation": {"feature_indices": [], "matrix": None},
                "input_statistics": {"feature_indices": [], "means": [], "stds": []},
                "scatter": {"feature_indices": [], "output_index": 0, "x": [], "y": []},
                "samples": [],
                "limits": {},
                "selection": {"root": body["root"], "path": body["path"]},
                "view": body["view"],
            }

    with TestClient(route_app(routes=preview_routes(Previewer()))) as client:
        result = client.post("/api/datasets/preview", json=request())
        assert result.status_code == 200
        assert result.headers["cache-control"] == "no-store"
        assert (
            client.post("/api/datasets/preview", json=request("bad")).status_code == 400
        )
        refused = client.post("/api/datasets/preview", json=request("refused"))
        assert refused.status_code == 400
        assert refused.json() == {"error": "Bad request"}
        assert "secret" not in refused.text


def test_worker_output_shape_mutation_is_rejected(tmp_path):
    write_dataset(tmp_path)
    with BoundedThreadPool(WorkBudget(timeout_seconds=10)) as pool:
        service = make_service(tmp_path, pool)
        result = asyncio.run(service.preview(request()))
        result.pop("selection")
        result.pop("view")
        result["scatter"]["x"][0].append(123)
        with pytest.raises(PreviewRefused):
            validate_preview(json.dumps(result).encode(), service.limits)


def test_multi_sample_multi_output_missing_correlation_is_rejected(tmp_path):
    """Unavailable single-row correlation must not weaken other matrix checks."""
    write_dataset(tmp_path)
    with BoundedThreadPool(WorkBudget(timeout_seconds=10)) as pool:
        service = make_service(tmp_path, pool)
        result = asyncio.run(service.preview(request()))
    result.pop("selection")
    result.pop("view")
    result["correlation"]["matrix"] = None
    with pytest.raises(PreviewRefused):
        validate_preview(json.dumps(result).encode(), service.limits)


def test_request_cancel_waits_for_cooperative_work_and_releases_slot(
    tmp_path, monkeypatch
):
    write_dataset(tmp_path)
    started, stopped = threading.Event(), threading.Event()

    def command(spec, budget, *, token):
        started.set()
        try:
            while True:
                token.wait(0.01)
                token.check()
        finally:
            stopped.set()

    monkeypatch.setattr("wavedl_daemon.runs.preview_service.run_command", command)
    with BoundedThreadPool(WorkBudget(timeout_seconds=10)) as pool:

        async def scenario():
            task = asyncio.create_task(make_service(tmp_path, pool).preview(request()))
            while not started.is_set():
                await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert stopped.is_set()
            assert pool.outstanding == 0

        asyncio.run(scenario())


def test_process_deadline_reaps_child(tmp_path, monkeypatch):
    from wavedl_daemon.runs import preview_service

    write_dataset(tmp_path)
    original = preview_service.run_command
    ready = tmp_path / "preview-child-ready"
    results = []
    script = f"import time; from pathlib import Path; Path({str(ready)!r}).write_text('started'); time.sleep(30)"

    def controlled(spec, budget, *, token):
        result = original(
            replace(spec, argv=(sys.executable, "-c", script)), budget, token=token
        )
        results.append(result)
        return result

    monkeypatch.setattr(preview_service, "run_command", controlled)
    with BoundedThreadPool(
        WorkBudget(max_workers=1, max_pending=0, timeout_seconds=1)
    ) as pool:
        service = DatasetPreviewService(
            sys.executable, tmp_path, pool, CatalogService({"data": tmp_path})
        )
        start = time.monotonic()
        with pytest.raises((BudgetExhausted, WorkCancelled)):
            asyncio.run(service.preview(request()))
        assert time.monotonic() - start < 3
        assert pool.outstanding == 0
    assert ready.read_text() == "started"
    assert len(results) == 1 and results[0].timed_out


def test_pending_preview_cancellation_drains_without_starting_child(
    tmp_path, monkeypatch
):
    """A cancelled queued preview retains admission until it really unwinds."""
    write_dataset(tmp_path)
    started, release = threading.Event(), threading.Event()
    children = []

    def occupied(token):
        started.set()
        assert release.wait(5)

    def command(*args, **kwargs):
        children.append(args)
        pytest.fail("Cancelled pending preview must not start a child")

    monkeypatch.setattr("wavedl_daemon.runs.preview_service.run_command", command)
    with BoundedThreadPool(
        WorkBudget(max_workers=1, max_pending=1, timeout_seconds=10)
    ) as pool:
        blocker = pool.submit(occupied)
        assert started.wait(2)

        async def scenario():
            task = asyncio.create_task(make_service(tmp_path, pool).preview(request()))
            try:
                async with asyncio.timeout(2):
                    while pool.outstanding != 2:
                        await asyncio.sleep(0.01)
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
                assert pool.outstanding == 2
            finally:
                release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
            blocker.result(2)
            assert pool.outstanding == 0
            assert not children
            assert await pool.submit(lambda token: "reused").wait() == "reused"

        asyncio.run(scenario())


def test_hosted_preview_midrequest_mutation_refuses_decoded_child_result(
    tmp_path, monkeypatch
):
    """Even valid child JSON is refused if its held file changes before return."""
    from wavedl_daemon.runs import preview_service

    write_dataset(tmp_path)
    original_command = preview_service.run_command
    original_bytes = (tmp_path / "data.npz").read_bytes()
    decoded, denied = [], []

    def mutate_after_decode(spec, budget, *, token):
        result = original_command(spec, budget, token=token)
        assert result.returncode == 0
        decoded.append(json.loads(result.stdout))
        try:
            with (tmp_path / "data.npz").open("ab") as dataset:
                dataset.write(b"changed-after-child-decode")
        except PermissionError as error:
            if (
                os.name != "nt"
                or error.errno != 13
                or getattr(error, "winerror", None) not in {None, 32}
            ):
                raise
            denied.append(True)
        return result

    monkeypatch.setattr(preview_service, "run_command", mutate_after_decode)
    settings = Settings(tmp_path / "state", tmp_path / "runs")
    services = HostedServices(settings, browse_roots={"data": tmp_path})
    app = create_app(
        settings,
        token="mutation-token",
        daemon_id="initial",
        build_id="preview-test",
        services=services,
    )
    with TestClient(
        app,
        base_url="http://localhost:8420",
        headers={"Authorization": "Bearer mutation-token"},
    ) as client:
        client.headers["X-WaveDL-Daemon-ID"] = client.get("/api/health").json()[
            "daemon_id"
        ]
        result = client.post("/api/datasets/preview", json=request())
        assert decoded[0]["input_shape"] == [6, 4]
        if os.name == "nt":
            assert denied == [True]
            assert result.status_code == 200, result.text
            assert (tmp_path / "data.npz").read_bytes() == original_bytes
            assert result.json()["input_shape"] == [6, 4]
        else:
            assert denied == []
            assert result.status_code == 409, result.text
            assert "scatter" not in result.text
        assert str(tmp_path) not in result.text
        assert services.pool.outstanding == 0


def test_hosted_preview_shared_admission_refuses_then_recovers(tmp_path):
    """The authenticated route shares capacity and returns no stale success."""
    write_dataset(tmp_path)
    settings = Settings(tmp_path / "state", tmp_path / "runs")
    services = HostedServices(
        settings,
        browse_roots={"data": tmp_path},
        budget=WorkBudget(max_workers=1, max_pending=0, timeout_seconds=10),
    )
    app = create_app(
        settings,
        token="admission-token",
        daemon_id="initial",
        build_id="preview-test",
        services=services,
    )
    ready, release = threading.Event(), threading.Event()

    def occupied(token):
        ready.set()
        assert release.wait(5)

    with TestClient(
        app,
        base_url="http://localhost:8420",
        headers={"Authorization": "Bearer admission-token"},
    ) as client:
        client.headers["X-WaveDL-Daemon-ID"] = client.get("/api/health").json()[
            "daemon_id"
        ]
        blocker = services.pool.submit(occupied)
        assert ready.wait(2)
        try:
            result = client.post("/api/datasets/preview", json=request())
            assert result.status_code == 503, result.text
            assert "scatter" not in result.text
            assert str(tmp_path) not in result.text
            assert services.pool.outstanding == 1
        finally:
            release.set()
            blocker.result(2)
        recovered = client.post("/api/datasets/preview", json=request())
        assert recovered.status_code == 200, recovered.text
        assert recovered.json()["input_shape"] == [6, 4]
        assert services.pool.outstanding == 0


def test_hosted_authenticated_preview_real_child_and_identity_boundary(tmp_path):
    write_dataset(tmp_path)
    settings = Settings(tmp_path / "state", tmp_path / "runs")
    services = HostedServices(settings, browse_roots={"data": tmp_path})
    token = "preview-acceptance-token"
    app = create_app(
        settings,
        token=token,
        daemon_id="initial",
        build_id="preview-test",
        services=services,
    )
    with TestClient(app, base_url="http://localhost:8420") as client:
        assert client.post("/api/datasets/preview", json=request()).status_code == 401
        client.headers["Authorization"] = f"Bearer {token}"
        daemon_id = client.get("/api/health").json()["daemon_id"]
        client.headers["X-WaveDL-Daemon-ID"] = daemon_id
        for view in sorted(DatasetPreviewService.ALLOWED_VIEWS):
            result = client.post("/api/datasets/preview", json=request(view))
            assert result.status_code == 200, result.text
            value = result.json()
            assert value["identity"]["st_ino"] == (tmp_path / "data.npz").stat().st_ino
            assert value["input_shape"] == [6, 4]
            assert value["scatter"]["y"] == [0, 2, 4, 6, 8, 10]
            assert result.headers["cache-control"] == "no-store"
            assert token not in result.text
            assert str(tmp_path) not in result.text
        assert (
            client.post(
                "/api/datasets/preview",
                json=request(),
                headers={"Origin": "http://foreign.example"},
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/datasets/preview",
                json=request(),
                headers={"X-WaveDL-Daemon-ID": "foreign-daemon"},
            ).status_code
            == 409
        )
        for body in (
            request(path="../data.npz"),
            {**request(), "root": "other"},
            {**request(), "limits": {}},
        ):
            assert client.post("/api/datasets/preview", json=body).status_code == 400
        (tmp_path / "bad.npz").write_bytes(b"malformed")
        assert (
            client.post(
                "/api/datasets/preview", json=request(path="bad.npz")
            ).status_code
            == 400
        )
        np.savez(
            tmp_path / "object.npz",
            inputs=np.array([{"x": 1}], dtype=object),
            outputs=np.zeros((1, 1)),
        )
        assert (
            client.post(
                "/api/datasets/preview", json=request(path="object.npz")
            ).status_code
            == 400
        )
        np.savez(
            tmp_path / "wide.npz", inputs=np.ones((1, 65537)), outputs=np.zeros((1, 1))
        )
        assert (
            client.post(
                "/api/datasets/preview", json=request(path="wide.npz")
            ).status_code
            == 413
        )


@pytest.mark.parametrize("kind", ["mat", "h5", "hdf5"])
def test_hosted_optional_dataset_formats(tmp_path, kind):
    path = tmp_path / f"dataset.{kind}"
    x, y = np.arange(24).reshape(6, 4), np.arange(12).reshape(6, 2)
    if kind == "mat":
        scipy = pytest.importorskip("scipy.io")
        scipy.savemat(path, {"inputs": x, "outputs": y})
    else:
        h5py = pytest.importorskip("h5py")
        with h5py.File(path, "w") as file:
            file["inputs"], file["outputs"] = x, y
    settings = Settings(tmp_path / "state", tmp_path / "runs")
    services = HostedServices(settings, browse_roots={"data": tmp_path})
    app = create_app(
        settings,
        token="formats-token",
        daemon_id="initial",
        build_id="test",
        services=services,
    )
    with TestClient(
        app,
        base_url="http://localhost:8420",
        headers={"Authorization": "Bearer formats-token"},
    ) as client:
        result = client.post("/api/datasets/preview", json=request(path=path.name))
        assert result.status_code == 200, result.text
        assert result.json()["input_shape"] == [6, 4]


def test_http_disconnect_cancels_and_drains_preview():
    stopped = asyncio.Event()
    started = asyncio.Event()

    class Previewer:
        async def preview(self, body):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

    async def scenario():
        app = route_app(routes=preview_routes(Previewer()))
        messages = asyncio.Queue()
        await messages.put(
            {
                "type": "http.request",
                "body": json.dumps(request()).encode(),
                "more_body": False,
            }
        )
        sent = []

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "http_version": "1.1",
            "scheme": "http",
            "method": "POST",
            "path": "/api/datasets/preview",
            "raw_path": b"/api/datasets/preview",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "server": ("localhost", 8420),
            "client": ("localhost", 12345),
        }
        operation = asyncio.create_task(app(scope, messages.get, send))
        await asyncio.wait_for(started.wait(), 2)
        await messages.put({"type": "http.disconnect"})
        await asyncio.wait_for(operation, 2)
        assert stopped.is_set()
        assert sent[0]["status"] == 503

    asyncio.run(scenario())
