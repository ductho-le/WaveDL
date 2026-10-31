"""Fixture-only acceptance of bounded preview refusal and recovery.

These tests deliberately use valid numeric datasets whose declared or compressed
payload exceeds a fixture-selected cap.  They exercise the authenticated HTTP
route and the real supervised worker, rather than mocking preview responses.
"""

import asyncio
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat
from starlette.testclient import TestClient

from wavedl.core.dataset_preview import PreviewLimits
from wavedl_daemon.execution.budget import WorkBudget
from wavedl_daemon.execution.pool import BoundedThreadPool
from wavedl_daemon.http.app import create_app
from wavedl_daemon.runs.catalog import CatalogService
from wavedl_daemon.runs.preview_service import (
    DatasetPreviewService,
    PreviewTooLarge,
)
from wavedl_daemon.runs.runtime_service import HostedServices
from wavedl_daemon.storage.settings import Settings


def _request(path="data.npz"):
    return {"root": "data", "path": path, "view": "all"}


def _safe_dataset(path):
    np.savez(path, X=np.arange(24).reshape(6, 4), y=np.arange(6).reshape(6, 1))


def test_valid_compressed_npz_refusal_clears_and_recovers(tmp_path):
    """ZIP expansion refusal is sanitized, capacity is released, and retry works."""
    # A highly compressible but valid numeric archive: tiny on disk, >100x when
    # expanded.  This is the same bounded witness used by the core policy tests.
    np.savez_compressed(
        tmp_path / "data.npz",
        X=np.zeros((10_000, 10), dtype=np.float64),
        y=np.zeros((10_000, 1), dtype=np.float64),
    )
    settings = Settings(tmp_path / "state", tmp_path / "runs")
    services = HostedServices(
        settings,
        browse_roots={"data": tmp_path},
        budget=WorkBudget(max_workers=1, max_pending=0, timeout_seconds=10),
    )
    app = create_app(
        settings,
        token="budget-token",
        daemon_id="initial",
        build_id="preview-budget-test",
        services=services,
    )
    with TestClient(
        app,
        base_url="http://localhost:8420",
        headers={"Authorization": "Bearer budget-token"},
    ) as client:
        client.headers["X-WaveDL-Daemon-ID"] = client.get("/api/health").json()[
            "daemon_id"
        ]
        refused = client.post("/api/datasets/preview", json=_request())
        assert refused.status_code == 413
        assert refused.json() == {"error": "Request too large"}
        assert "exceeds" not in refused.text
        assert "data.npz" not in refused.text
        assert services.pool.outstanding == 0

        _safe_dataset(tmp_path / "data.npz")
        recovered = client.post("/api/datasets/preview", json=_request())
        assert recovered.status_code == 200, recovered.text
        assert recovered.json()["input_shape"] == [6, 4]
        assert recovered.json()["output_shape"] == [6, 1]
        assert services.pool.outstanding == 0


@pytest.mark.parametrize(
    ("overrides", "filename"),
    [
        pytest.param({"max_file_bytes": 1}, "data.npz", id="file"),
        pytest.param({"max_decompressed_bytes": 100}, "data.npz", id="decompressed"),
        pytest.param({"max_input_features": 2}, "data.npz", id="features"),
        pytest.param(
            {"max_decompressed_bytes": 1_000}, "data.mat", id="mat-declared-memory"
        ),
    ],
)
def test_fixture_selected_byte_decompressed_and_feature_caps_refuse(
    tmp_path, overrides, filename
):
    """Valid numeric inputs refuse each selected cap without leakage."""
    if filename.endswith(".mat"):
        savemat(
            tmp_path / filename,
            {"X": np.zeros((200, 4)), "y": np.zeros((200, 1))},
            do_compression=True,
        )
    else:
        np.savez(tmp_path / filename, X=np.zeros((20, 4)), y=np.zeros((20, 1)))
    with BoundedThreadPool(WorkBudget(timeout_seconds=10)) as pool:
        service = DatasetPreviewService(
            sys.executable,
            # Foundations intentionally imports from relative PYTHONPATH=src.
            # Keep the real worker's source import rooted at the checkout,
            # independently of the selected temporary dataset authority.
            Path(__file__).resolve().parents[2],
            pool,
            CatalogService({"data": tmp_path}),
            limits=replace(PreviewLimits(), max_output_bytes=200_000, **overrides),
        )
        with pytest.raises(PreviewTooLarge):
            asyncio.run(service.preview(_request(filename)))
        assert pool.outstanding == 0
