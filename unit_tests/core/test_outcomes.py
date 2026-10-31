"""Immutable command receipts retain independent, execution-bound outcomes."""

import json

import pytest

from wavedl.core.outcomes import main, write_outcome_receipt


RUN_ID = "00000000-0000-4000-8000-000000000001"
REQUEST_ID = "00000000-0000-4000-8000-000000000002"


def receipt(directory, **changes):
    values = {
        "run_id": RUN_ID,
        "request_id": REQUEST_ID,
        "attempt": 1,
        "job_id": "123",
        "cluster": "alpha",
        "restart_count": 0,
        "operation": "train",
        "runtime_exit": 0,
        "batch_exit": 0,
        "stage_out_exit": 1,
    }
    return write_outcome_receipt(directory, **(values | changes))


def test_receipt_independent_outcomes_and_no_overwrite(tmp_path):
    path = receipt(tmp_path)
    data = json.loads(path.read_bytes())
    assert data["runtime"]["status"] == "success"
    assert data["stage_out"]["status"] == "failed"
    with pytest.raises(FileExistsError):
        receipt(tmp_path, runtime_exit=1)
    assert json.loads(path.read_bytes()) == data


def test_requeue_separate_receipt_unexecuted_is_unknown(tmp_path):
    initial = receipt(tmp_path)
    restarted = receipt(tmp_path, restart_count=1, runtime_exit=125)
    assert initial != restarted
    assert json.loads(restarted.read_bytes())["runtime"]["exit_code"] is None
    assert json.loads(restarted.read_bytes())["runtime"]["status"] == "unknown"


@pytest.mark.parametrize(
    "changes",
    [
        {"run_id": "invalid"},
        {"attempt": True},
        {"cluster": ""},
        {"job_id": "1;alpha"},
        {"restart_count": -1},
        {"runtime_exit": 256},
        {"operation": "other"},
    ],
)
def test_invalid_receipt_never_published(tmp_path, changes):
    with pytest.raises(ValueError):
        receipt(tmp_path, **changes)
    assert list(tmp_path.iterdir()) == []


def test_helper_cli(tmp_path):
    assert (
        main(
            [
                str(tmp_path),
                "--run-id",
                RUN_ID,
                "--request-id",
                REQUEST_ID,
                "--attempt",
                "1",
                "--job-id",
                "123",
                "--cluster",
                "alpha",
                "--restart-count",
                "0",
                "--operation",
                "test",
                "--runtime-exit",
                "0",
                "--batch-exit",
                "0",
                "--stage-out-exit",
                "0",
            ]
        )
        == 0
    )
    assert (
        json.loads((tmp_path / "slurm-outcome.json").read_bytes())["operation"]
        == "test"
    )
