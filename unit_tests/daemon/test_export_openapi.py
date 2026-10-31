"""Offline OpenAPI export documents wire models without executing services."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
TOOL = ROOT / "tools" / "export_daemon_openapi.py"


def exporter():
    spec = importlib.util.spec_from_file_location("export_daemon_openapi", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_openapi_documents_run_lifecycle():
    module = exporter()
    spec = module.build_openapi()
    paths = spec["paths"]
    assert set(paths) == {
        "/api/health",
        "/api/bootstrap",
        "/api/session",
        "/api/runs/adopt",
        "/api/models",
        "/api/browse",
        "/api/browse/resolve",
        "/api/datasets/inspect",
        "/api/datasets/preview",
        "/api/host",
        "/api/runs/{run_id}/artifacts",
        "/api/runs/{run_id}/artifacts/{generation}/{path}",
        "/api/runs/{run_id}/events",
        "/api/runs/plan",
        "/api/runs",
        "/api/runs/{run_id}",
        "/api/runs/{run_id}/cancel",
        "/api/runs/{run_id}/attempts",
    }
    assert "post" in paths["/api/runs/plan"]
    assert {"get", "post"} <= paths["/api/runs"].keys()
    assert "get" in paths["/api/runs/{run_id}"]
    assert "post" in paths["/api/runs/{run_id}/cancel"]
    assert "get" in paths["/api/runs/{run_id}/attempts"]
    schemas = spec["components"]["schemas"]
    assert {
        "PlanRequest",
        "SubmitRequest",
        "PlanView",
        "RunResponse",
        "RunList",
        "AttemptPage",
    } <= schemas.keys()
    plan = schemas["PlanRequest"]
    assert plan["additionalProperties"] is False
    assert set(plan["required"]) == {
        "backend",
        "operation",
        "config",
        "runtime",
        "topology",
    }
    for name in ("storage_map", "submission_cwd", "checkpoint_source"):
        assert name in plan["properties"]
        assert name not in plan["required"]
    assert not any(path.startswith(("/docs", "/redoc")) for path in paths)
    assert spec["info"] == {
        "title": "WaveDL daemon API",
        "version": module._package_version(),
    }
    assert spec == module.build_openapi()
    rendered = json.dumps(spec)
    for private in (
        str(ROOT.resolve()),
        "offline-openapi-token",
        "offline-openapi-daemon",
        "offline-openapi-build",
    ):
        assert private not in rendered
    assert "servers" not in spec


def test_offline_service_raises_if_called():
    service = exporter()._InertRunService()
    for name in ("plan", "submit", "cancel", "get", "list", "attempts"):
        with pytest.raises(AssertionError, match="OpenAPI generation called"):
            getattr(service, name)()


def test_openapi_has_no_422():
    spec = exporter().build_openapi()
    assert "/api/datasets/inspect" in spec["paths"]
    assert "/api/datasets/preview" in spec["paths"]
    assert "/api/bootstrap" in spec["paths"]
    for methods in spec["paths"].values():
        for operation in methods.values():
            assert "422" not in operation["responses"]


def test_cli_output_is_byte_identical(tmp_path):
    outputs = []
    for seed in ("1", "2"):
        output = tmp_path / f"openapi-{seed}.json"
        result = subprocess.run(
            [sys.executable, str(TOOL), "--output", str(output)],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            check=True,
            timeout=30,
        )
        assert result.stdout == b""
        assert result.stderr == b""
        outputs.append(output.read_bytes())
    result = subprocess.run(
        [sys.executable, str(TOOL)],
        capture_output=True,
        check=True,
        timeout=30,
    )
    assert result.stderr == b""
    assert result.stdout == outputs[0] == outputs[1]
    assert outputs[0] == (
        json.dumps(exporter().build_openapi(), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def test_openapi_documents_daemon_errors_not_fastapi_422():
    spec = exporter().build_openapi()
    operations = {
        (method, path): operation
        for path, methods in spec["paths"].items()
        for method, operation in methods.items()
    }
    assert {key: op["operationId"] for key, op in operations.items()} == {
        ("get", "/api/health"): "get_health",
        ("post", "/api/bootstrap"): "exchange_bootstrap",
        ("post", "/api/session"): "rotate_session",
        ("delete", "/api/session"): "revoke_session",
        ("post", "/api/runs/adopt"): "adopt_run",
        ("get", "/api/models"): "list_models",
        ("get", "/api/browse"): "browse_files",
        ("post", "/api/browse/resolve"): "resolve_browse",
        ("post", "/api/datasets/inspect"): "inspect_dataset",
        ("post", "/api/datasets/preview"): "preview_dataset",
        ("get", "/api/host"): "get_host",
        ("get", "/api/runs/{run_id}/artifacts"): "list_artifacts",
        (
            "get",
            "/api/runs/{run_id}/artifacts/{generation}/{path}",
        ): "download_artifact",
        ("post", "/api/runs/{run_id}/events"): "push_events",
        ("get", "/api/runs/{run_id}/events"): "stream_events",
        ("post", "/api/runs/plan"): "plan_run",
        ("post", "/api/runs"): "submit_run",
        ("get", "/api/runs"): "list_runs",
        ("get", "/api/runs/{run_id}"): "get_run",
        ("post", "/api/runs/{run_id}/cancel"): "cancel_run",
        ("get", "/api/runs/{run_id}/attempts"): "list_run_attempts",
    }
    error = {"$ref": "#/components/schemas/ErrorResponse"}
    for operation in operations.values():
        assert "422" not in operation["responses"]
        for status in ("400", "401", "404", "409", "503"):
            content = operation["responses"][status]["content"]
            assert content["application/json"]["schema"] == error
    for key in (("post", "/api/runs/plan"), ("post", "/api/runs")):
        assert {"408", "413", "415"} <= operations[key]["responses"].keys()
    schemas = spec["components"]["schemas"]
    assert schemas["ErrorResponse"]["required"] == ["error"]
    assert not {"HTTPValidationError", "ValidationError"} & schemas.keys()
