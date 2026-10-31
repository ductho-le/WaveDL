"""Inspection runs in reaped child processes, never the daemon interpreter."""

import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from starlette.testclient import TestClient

from unit_tests.daemon.conftest import route_app
from wavedl.core.dataset_headers import InspectionLimits, InspectionRefused
from wavedl_daemon.execution.budget import WorkBudget
from wavedl_daemon.execution.commands import run_command
from wavedl_daemon.execution.pool import BoundedThreadPool, WorkCancelled
from wavedl_daemon.http.inspection import inspection_routes
from wavedl_daemon.runs.catalog import CatalogService
from wavedl_daemon.runs.inspection import DatasetInspector, validate_metadata
from wavedl_daemon.supervision.base import ProcessSpec


def test_actual_worker_returns_metadata(tmp_path):
    path = tmp_path / "dataset.npz"
    np.savez_compressed(path, inputs=np.zeros((7, 3)), outputs=np.zeros((7, 1)))
    with BoundedThreadPool(WorkBudget(timeout_seconds=10)) as pool:
        inspector = DatasetInspector(sys.executable, Path.cwd(), pool)
        result = asyncio.run(inspector.inspect(str(path)))
    assert {item["name"] for item in result["variables"]} == {"inputs", "outputs"}
    assert result["variables"][0]["sample_count"] == 7


def test_missing_file_is_generic_refusal(tmp_path):
    with BoundedThreadPool(WorkBudget(timeout_seconds=10)) as pool:
        inspector = DatasetInspector(sys.executable, Path.cwd(), pool)
        with pytest.raises(InspectionRefused) as error:
            asyncio.run(inspector.inspect(str(tmp_path / "private-secret.npy")))
    assert "private-secret" not in str(error.value)


def test_real_deadline_stops_descendant_marker_and_releases_slot(tmp_path, monkeypatch):
    from wavedl_daemon.runs import inspection

    marker = tmp_path / "late-marker"
    ready = tmp_path / "worker-started"
    child = f"import time; from pathlib import Path; time.sleep(2); Path({str(marker)!r}).write_text('escaped')"
    script = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
        f"Path({str(ready)!r}).write_text('started'); time.sleep(30)"
    )
    results = []

    def controlled(spec, budget, *, token):
        result = run_command(
            replace(spec, argv=(sys.executable, "-c", script)), budget, token=token
        )
        results.append(result)
        return result

    monkeypatch.setattr(inspection, "run_command", controlled)
    budget = WorkBudget(max_workers=1, max_pending=0, timeout_seconds=1)
    with BoundedThreadPool(budget) as pool:
        inspector = DatasetInspector(sys.executable, tmp_path, pool)
        with pytest.raises((InspectionRefused, WorkCancelled)):
            asyncio.run(inspector.inspect(str(tmp_path / "data.npy")))
        assert pool.outstanding == 0
    assert ready.read_text() == "started"
    assert len(results) == 1 and results[0].timed_out
    time.sleep(2.1)
    assert not marker.exists()


@pytest.mark.parametrize(
    "content",
    [
        b'{"schema_version":NaN}',
        b'{"schema_version":1,"schema_version":1}',
        b"{}",
        b"[]",
        b"not-json",
    ],
)
def test_worker_output_strict_and_finite(content):
    with pytest.raises(InspectionRefused):
        validate_metadata(content, InspectionLimits())


def test_worker_metadata_bounds_and_bool_dimensions():
    value = {
        "schema_version": 1,
        "format": "npy",
        "variables": [
            {
                "name": "array",
                "shape": [True],
                "dtype": "float32",
                "sample_count": True,
                "declared_bytes": 4,
            }
        ],
    }
    with pytest.raises(InspectionRefused):
        validate_metadata(json.dumps(value).encode(), InspectionLimits())
    with pytest.raises(InspectionRefused):
        validate_metadata(b" " * 20, replace(InspectionLimits(), max_output_bytes=10))


def test_selected_interpreter_and_path_validation(tmp_path):
    with BoundedThreadPool(WorkBudget()) as pool:
        with pytest.raises(ValueError):
            DatasetInspector("python", tmp_path, pool)
        inspector = DatasetInspector(sys.executable, tmp_path, pool)
        with pytest.raises(InspectionRefused):
            asyncio.run(inspector.inspect("relative.npy"))
        with pytest.raises(ValueError):
            DatasetInspector(
                sys.executable, tmp_path, pool, budget=WorkBudget(max_output_bytes=10)
            )


def test_inspection_module_without_numerical_or_gui_imports():
    # -S bypasses editable-install .pth files, so expose only the checkout source.
    source_root = Path(__file__).resolve().parents[2] / "src"
    code = "import sys; import wavedl_daemon.runs.inspection; assert not any(x in sys.modules for x in ['numpy','torch','h5py','scipy','PySide6'])"
    subprocess.run(
        [sys.executable, "-S", "-c", code],
        check=True,
        capture_output=True,
        timeout=10,
        env=os.environ | {"PYTHONPATH": str(source_root)},
    )


def test_http_strict_path_only_and_generic_errors():
    class Inspector:
        async def inspect(self, root, path):
            raise InspectionRefused("private secret path")

    app = route_app(routes=inspection_routes(Inspector()))
    with TestClient(app) as client:
        response = client.post(
            "/api/datasets/inspect",
            json={"root": "data", "path": "data.npy", "python": "/evil"},
        )
        assert response.status_code == 400
        response = client.post(
            "/api/datasets/inspect", json={"root": "data", "path": "data.npy"}
        )
        assert response.status_code == 400
        assert response.json() == {"error": "Bad request"}


def test_http_success_is_noncacheable():
    class Inspector:
        async def inspect(self, root, path):
            return {"schema_version": 1, "format": "npy", "variables": []}

    with TestClient(route_app(routes=inspection_routes(Inspector()))) as client:
        response = client.post(
            "/api/datasets/inspect", json={"root": "data", "path": "data.npy"}
        )
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("real_worker", [False, True])
def test_http_disconnect_cancels_and_drains_inspection(
    tmp_path, monkeypatch, real_worker
):
    stopped = asyncio.Event()
    started = asyncio.Event()
    marker = tmp_path / "inspection-child-started"
    completed = []
    script = (
        "import os,time; from pathlib import Path; "
        f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )

    def command(spec, budget, *, token):
        result = run_command(
            ProcessSpec((sys.executable, "-c", script), Path.cwd()), budget, token=token
        )
        completed.append(result)
        return result

    monkeypatch.setattr("wavedl_daemon.runs.inspection.run_command", command)

    class Inspector:
        async def inspect(self, root, path):
            started.set()
            try:
                if real_worker:
                    return await inspector.inspect(root, path)
                await asyncio.Event().wait()
            finally:
                stopped.set()

    async def scenario():
        app = route_app(routes=inspection_routes(Inspector()))
        messages = asyncio.Queue()
        await messages.put(
            {
                "type": "http.request",
                "body": json.dumps({"root": "data", "path": "data.npy"}).encode(),
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
            "path": "/api/datasets/inspect",
            "raw_path": b"/api/datasets/inspect",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "server": ("localhost", 8420),
            "client": ("localhost", 12345),
        }
        operation = asyncio.create_task(app(scope, messages.get, send))
        await asyncio.wait_for(started.wait(), 2)
        if real_worker:
            deadline = time.monotonic() + 2
            while not marker.exists() and time.monotonic() < deadline:
                await asyncio.sleep(0.005)
            assert marker.exists()
        await messages.put({"type": "http.disconnect"})
        await asyncio.wait_for(operation, 2)
        assert stopped.is_set()
        assert sent[0]["status"] == 503
        if real_worker:
            assert completed and completed[0].cancelled
            assert pool.outstanding == 0
            if os.name != "nt":
                with pytest.raises(ProcessLookupError):
                    os.kill(int(marker.read_text()), 0)

    (tmp_path / "data.npy").write_bytes(b"dataset")
    with BoundedThreadPool(WorkBudget(timeout_seconds=5)) as pool:
        inspector = DatasetInspector(
            sys.executable, Path.cwd(), pool, catalog=CatalogService({"data": tmp_path})
        )
        asyncio.run(scenario())


def test_catalog_bound_inspection_rejects_paths_outside_configured_root(
    tmp_path, monkeypatch
):
    from wavedl_daemon.runs import inspection

    original_command = inspection.run_command
    observations = []

    def observed(spec, budget, *, token):
        result = original_command(spec, budget, token=token)
        observations.append(
            {
                "returncode": result.returncode,
                "timed_out": result.timed_out,
                "cancelled": result.cancelled,
                "stdout": result.stdout.decode("utf-8", errors="replace"),
                "stderr": result.stderr.decode("utf-8", errors="replace"),
            }
        )
        return result

    monkeypatch.setattr(inspection, "run_command", observed)
    dataset = tmp_path / "data.npz"
    np.savez(dataset, inputs=np.zeros((2, 1)), outputs=np.zeros((2, 1)))
    npy = tmp_path / "vector.npy"
    np.save(npy, np.zeros((2, 1)))
    with BoundedThreadPool(WorkBudget(timeout_seconds=10)) as pool:
        inspector = DatasetInspector(
            sys.executable,
            Path.cwd(),
            pool,
            catalog=CatalogService({"data": tmp_path}),
        )
        with pytest.raises(InspectionRefused):
            asyncio.run(inspector.inspect("data", "../outside-secret.npy"))
        try:
            result = asyncio.run(inspector.inspect("data", "data.npz"))
            npy_result = asyncio.run(inspector.inspect("data", "vector.npy"))
        except InspectionRefused:
            pytest.fail(
                f"Actual selected worker refused unchanged fixture: {observations!r}"
            )
    assert {item["name"] for item in result["variables"]} == {"inputs", "outputs"}
    assert npy_result["format"] == "npy"


@pytest.mark.skipif(
    os.name != "nt", reason="Native Windows authority/worker identity convention"
)
def test_native_windows_catalog_header_worker_preserves_identity_binding(tmp_path):
    np.save(tmp_path / "native.npy", np.zeros((3, 2), dtype="float32"))
    with BoundedThreadPool(WorkBudget(timeout_seconds=10)) as pool:
        inspector = DatasetInspector(
            sys.executable,
            Path.cwd(),
            pool,
            catalog=CatalogService({"native": tmp_path}),
        )
        result = asyncio.run(inspector.inspect("native", "native.npy"))
        assert result["format"] == "npy"
        assert result["variables"][0]["shape"] == [3, 2]
        assert pool.outstanding == 0


@pytest.mark.skipif(os.name != "posix", reason="Executable wrapper requires POSIX")
def test_catalog_bound_inspection_rejects_replaced_path_before_metadata_acceptance(
    tmp_path,
):
    dataset = tmp_path / "data.npz"
    np.savez(dataset, inputs=np.zeros((2, 1)), outputs=np.zeros((2, 1)))
    worker = tmp_path / "replace-python"
    worker.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, sys\n"
        "path = pathlib.Path(sys.argv[sys.argv.index('--path') + 1])\n"
        "raw, original = path.read_bytes(), path.stat()\n"
        "path.unlink()\n"
        "path.write_bytes(raw)\n"
        "os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))\n"
        "os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
    )
    worker.chmod(0o700)
    with BoundedThreadPool(WorkBudget(timeout_seconds=10)) as pool:
        inspector = DatasetInspector(
            str(worker),
            Path.cwd(),
            pool,
            catalog=CatalogService({"data": tmp_path}),
        )
        with pytest.raises(InspectionRefused):
            asyncio.run(inspector.inspect("data", "data.npz"))
