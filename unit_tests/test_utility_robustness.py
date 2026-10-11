"""Regression tests for production utility failures found in the full audit."""

import argparse
import logging
from pathlib import Path
from unittest.mock import patch

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from scipy.sparse import csc_matrix, csr_matrix

from wavedl.utils.config import load_config, merge_config_with_args, validate_config
from wavedl.utils.constraints import ExpressionConstraint, PhysicsConstrainedLoss
from wavedl.utils.data import (
    NPZSource,
    load_test_data,
    load_training_data,
    prepare_data,
)
from wavedl.utils.distributed import broadcast_value, sync_tensor
from wavedl.utils.losses import LogCoshLoss
from wavedl.utils.metrics import (
    calc_pearson,
    calc_per_target_r2,
    plot_prediction_vs_index,
    plot_scientific_scatter,
)
from wavedl.utils.targets import create_target_scaler


def _write_marker(path):
    Path(path).write_text("unpickled")
    return np.ones((2, 3), dtype=np.float32)


class _PickleProbe:
    def __init__(self, path):
        self.path = str(path)

    def __reduce__(self):
        return _write_marker, (self.path,)


class _CPUAccelerator:
    num_processes = 1
    is_main_process = True
    device = torch.device("cpu")

    @staticmethod
    def wait_for_everyone():
        pass


@pytest.mark.parametrize("method", ["load", "load_mmap", "load_outputs_only"])
def test_npz_pickle_never_executes_without_explicit_trust(tmp_path, method):
    marker = tmp_path / "marker"
    data_path = tmp_path / "unsafe.npz"
    probe = np.array([_PickleProbe(marker)], dtype=object)
    np.savez(data_path, X=probe, Y=probe)
    with pytest.raises(ValueError, match="trusted file"):
        getattr(NPZSource(), method)(data_path)
    assert not marker.exists()


def test_trusted_npz_sparse_loading_remains_supported(tmp_path):
    data_path = tmp_path / "trusted.npz"
    values = np.empty(4, dtype=object)
    for index in range(4):
        values[index] = csr_matrix(np.full((2, 3), index))
    np.savez(data_path, X=values, Y=np.ones((4, 1)))
    x, y = load_training_data(data_path, allow_pickle=True)
    assert x.shape == (4, 2, 3)
    np.testing.assert_array_equal(x[3], np.full((2, 3), 3))
    assert y.shape == (4, 1)


def test_trust_opt_in_is_part_of_cache_identity(tmp_path):
    source = tmp_path / "trusted.npz"
    objects = np.empty(6, dtype=object)
    for index in range(6):
        objects[index] = csr_matrix(np.full((2, 3), index))
    np.savez(source, X=objects, Y=np.arange(6).reshape(-1, 1))
    args = argparse.Namespace(
        data_path=str(source), seed=1, batch_size=2, workers=0, allow_pickle=True
    )
    prepare_data(args, logging.getLogger(__name__), _CPUAccelerator(), str(tmp_path))
    args.allow_pickle = False
    with pytest.raises(ValueError, match="trusted file"):
        prepare_data(
            args, logging.getLogger(__name__), _CPUAccelerator(), str(tmp_path)
        )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, 1e100, 1j])
@pytest.mark.parametrize("key", ["X", "Y"])
def test_inference_rejects_nonfinite_complex_and_float32_overflow(
    tmp_path, invalid, key
):
    arrays = {"X": np.ones((3, 2)), "Y": np.ones((3, 1))}
    arrays[key] = arrays[key].astype(complex if isinstance(invalid, complex) else float)
    arrays[key][0, 0] = invalid
    path = tmp_path / "invalid.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="finite"):
        load_test_data(path)


def test_inference_rejects_sample_mismatch(tmp_path):
    path = tmp_path / "mismatch.npz"
    np.savez(path, X=np.ones((3, 2)), Y=np.ones((2, 1)))
    with pytest.raises(ValueError, match="sample mismatch"):
        load_test_data(path)


@pytest.mark.parametrize("key", ["X", "Y"])
def test_training_rejects_nonfinite_data_before_completing_cache(tmp_path, key):
    data = {"X": np.ones((6, 2, 3)), "Y": np.ones((6, 1))}
    data[key][0] = np.nan
    path = tmp_path / "invalid.npz"
    np.savez(path, **data)
    args = argparse.Namespace(data_path=str(path), seed=1, batch_size=2, workers=0)
    with pytest.raises(ValueError, match="finite real"):
        prepare_data(
            args, logging.getLogger(__name__), _CPUAccelerator(), str(tmp_path)
        )
    assert not (tmp_path / "data_metadata.pkl").exists()


def test_matlab_sparse_inference_supports_group_inputs(tmp_path):
    source = tmp_path / "sparse.mat"
    original = np.arange(12).reshape(4, 3)
    sparse = csc_matrix(original.T)
    with h5py.File(source, "w") as file:
        group = file.create_group("X")
        group["data"], group["ir"], group["jc"] = (
            sparse.data,
            sparse.indices,
            sparse.indptr,
        )
        group.attrs["MATLAB_sparse"] = sparse.shape[0]
        file["Y"] = np.arange(4).reshape(1, -1)
    x, y = load_test_data(source)
    np.testing.assert_array_equal(x[:, 0], original)
    assert y.shape == (4, 1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_logcosh_large_negative_errors_have_finite_loss_and_gradient(dtype):
    pred = torch.tensor([-60000.0, 60000.0], dtype=dtype, requires_grad=True)
    loss = LogCoshLoss()(pred, torch.zeros_like(pred))
    assert torch.isfinite(loss)
    loss.backward()
    torch.testing.assert_close(pred.grad, torch.tensor([-0.5, 0.5], dtype=dtype))
    assert LogCoshLoss()(torch.zeros(1, dtype=dtype), torch.zeros(1, dtype=dtype)) == 0


def test_constraint_clamp_keywords_are_applied():
    pred = torch.tensor([[-2.0], [2.0]], requires_grad=True)
    constraint = ExpressionConstraint("clamp(y0, min=0, max=1) - 1")
    assert constraint(pred).item() == pytest.approx(0.5)
    constraint(pred).backward()
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("expression", ["sin(y0, out=y1)", "clamp(y0, invalid=0)"])
def test_constraint_unknown_keywords_fail_instead_of_silently_ignoring(expression):
    with pytest.raises(ValueError, match="keyword"):
        ExpressionConstraint(expression)


def test_single_value_input_standard_deviation_is_zero():
    constraint = ExpressionConstraint("y0 - x_std")
    assert constraint(torch.zeros(3, 1), torch.ones(3, 1)).item() == 0


def test_constraint_negative_indices_and_fractional_index_rejection():
    pred, inp = torch.ones(2, 1), torch.ones(2, 3)
    assert ExpressionConstraint("y0 - x[-1]")(pred, inp).item() == 0
    with pytest.raises(ValueError, match="integer indices"):
        ExpressionConstraint("x[0.9]")(pred, inp)


@pytest.mark.parametrize("weights", [[], [-1.0], [float("nan")], [float("inf")]])
def test_constraint_weights_must_be_finite_nonnegative(weights):
    with pytest.raises(ValueError, match="weights"):
        PhysicsConstrainedLoss(torch.nn.MSELoss(), weights=weights)


@pytest.mark.parametrize(
    "text", ["- lr: 0.01", "plain string", "1: value", "a_b: 1\na:\n  b: 2"]
)
def test_malformed_configuration_is_rejected(tmp_path, text):
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_config(path)


def test_config_preserves_nargs_lists_and_bool_types(monkeypatch):
    parser = argparse.ArgumentParser()
    parser.add_argument("--constraint", nargs="+", type=str, default=[])
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args([])
    monkeypatch.setattr("sys.argv", ["test"])
    result = merge_config_with_args({"constraint": ["y0 > 0", "y1 < 1"]}, args, parser)
    assert result.constraint == ["y0 > 0", "y1 < 1"]
    assert validate_config({"fresh": "false", "batch_size": 2.5}, parser=parser)
    assert not validate_config({"fresh": False, "batch_size": 2}, parser=parser)


def test_single_process_float_broadcast_preserves_python_precision():
    value = 1.2345678901234567
    assert broadcast_value(value, _CPUAccelerator()) == value


@pytest.mark.parametrize(
    "reduction,operation",
    [("max", torch.distributed.ReduceOp.MAX), ("min", torch.distributed.ReduceOp.MIN)],
)
def test_extrema_sync_uses_correct_collective_and_does_not_mutate_input(
    reduction, operation
):
    accelerator = _CPUAccelerator()
    accelerator.num_processes = 2
    tensor = torch.tensor([2.0, 4.0])
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.all_reduce") as reduce,
    ):
        reduce.side_effect = lambda value, op: value.add_(1)
        result = sync_tensor(tensor, accelerator, reduction)
    assert reduce.call_args.kwargs["op"] == operation
    torch.testing.assert_close(tensor, torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(result, torch.tensor([3.0, 5.0]))


@pytest.mark.parametrize(
    "function", [calc_pearson, calc_per_target_r2, create_target_scaler]
)
def test_invalid_values_cannot_be_reported_as_valid_metrics(function):
    values = np.array([[0.0], [np.nan], [1.0]])
    with pytest.raises(ValueError, match="finite real"):
        function(values) if function is create_target_scaler else function(
            values, values
        )


def test_pearson_is_invariant_to_physical_units():
    values = np.arange(5.0).reshape(-1, 1) * 1e-12
    assert calc_pearson(values, -values) == pytest.approx(-1.0)


def test_scatter_r2_uses_float64_like_reported_metrics():
    values = np.array([1e9, 1e9 + 64, 1e9, 1e9 + 64], dtype=np.float32).reshape(-1, 1)
    figure = plot_scientific_scatter(values, values[::-1])
    try:
        assert "-3.0000" in figure.axes[0].get_title()
    finally:
        plt.close(figure)


def test_prediction_index_plot_preserves_original_sample_order():
    values = np.arange(50.0).reshape(-1, 1)
    figure = plot_prediction_vs_index(values, values, max_samples=10)
    try:
        np.testing.assert_array_equal(
            figure.axes[0].lines[0].get_ydata(), np.arange(10)
        )
    finally:
        plt.close(figure)


def test_dense_batch_sizes_are_bounded_by_memory():
    from wavedl.utils.data import _bounded_batch_size

    assert _bounded_batch_size((1000, 2, 512, 512), np.float32, 256) == 8
    assert _bounded_batch_size((1000, 2), np.float32, 256) == 256
    assert _bounded_batch_size((10, 10, 4096, 4096), np.float32, 256) == 1


def test_short_multistep_training_does_not_decay_lr_before_first_update():
    from wavedl.utils.schedulers import get_scheduler

    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)
    scheduler = get_scheduler("multistep", optimizer, epochs=2)
    assert optimizer.param_groups[0]["lr"] == 0.1
    assert 0 not in scheduler.milestones


@pytest.mark.parametrize("scheduler", ["plateau", "linear_warmup", "onecycle"])
def test_invalid_warmup_combinations_do_not_mutate_optimizer(scheduler):
    from wavedl.utils.schedulers import get_scheduler_with_warmup

    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)
    with pytest.raises(ValueError, match="does not support"):
        get_scheduler_with_warmup(scheduler, optimizer)
    assert optimizer.param_groups[0]["lr"] == 0.1


def test_zero_weight_disables_constraint_evaluation():
    criterion = PhysicsConstrainedLoss(
        torch.nn.MSELoss(), [ExpressionConstraint("y0 / 0")], weights=[0.0]
    )
    assert criterion(torch.ones(1, 1), torch.ones(1, 1)).item() == 0
