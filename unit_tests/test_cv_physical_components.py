"""CV penalties must use physical units from training-only fold scalers."""

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from unit_tests import test_cv_option_fidelity as cv_fixtures
from wavedl.utils.constraints import PhysicsConstrainedLoss
from wavedl.utils.cv_components import build_cv_criterion


cpu_cv = cv_fixtures.cpu_cv


def criterion(targets, **options):
    scaler = StandardScaler().fit(np.asarray(targets, dtype=np.float32))
    return build_cv_criterion(
        loss_name="mse",
        loss_weights=None,
        huber_delta=1.0,
        training_scaler=scaler,
        device=torch.device("cpu"),
        **options,
    )


@pytest.mark.parametrize("reduction", ["mse", "mae"])
def test_fold_training_scaler_controls_physical_penalty(reduction):
    """The same normalized predictions mean different physical values by fold."""
    first = criterion([[2], [6]], constraint=["y0"], constraint_reduction=reduction)
    second = criterion([[20], [60]], constraint=["y0"], constraint_reduction=reduction)
    pred = torch.zeros((2, 1), requires_grad=True)
    target = torch.zeros_like(pred)
    assert isinstance(first, PhysicsConstrainedLoss)
    assert first.output_mean.tolist() == [4]
    assert first.output_std.tolist() == [2]
    assert second.output_mean.tolist() == [40]
    assert second.output_std.tolist() == [20]
    exponent = 2 if reduction == "mse" else 1
    assert first(pred, target).item() == pytest.approx(0.1 * 4**exponent)
    assert second(pred, target).item() == pytest.approx(0.1 * 40**exponent)
    assert first.base_loss(pred, target).item() == 0
    first(pred, target).backward()
    assert torch.all(pred.grad > 0)


@pytest.mark.parametrize("weight", [0.0, 0.1, 2.0])
def test_input_dependent_constraint_honors_authored_weight(weight):
    """Inputs are supplied to penalties without changing normalized base loss."""
    loss = criterion([[2], [6]], constraint=["y0 - x_mean"], constraint_weight=[weight])
    pred = torch.ones((2, 1))
    target = torch.zeros_like(pred)
    inputs = torch.full((2, 1, 4), 3.0)
    assert loss(pred, target, inputs).item() == pytest.approx(1 + weight * 9)
    assert loss.base_loss(pred, target).item() == 1


def test_constraint_file_uses_fold_physical_predictions(tmp_path):
    """An explicit model-side Python constraint sees inputs and physical output."""
    path = tmp_path / "constraint.py"
    path.write_text(
        "def constraint(pred, inputs=None):\n"
        "    return pred[:, 0] - inputs.flatten(1).mean(1)\n"
    )
    loss = criterion([[2], [6]], constraint_file=str(path), constraint_weight=[0.5])
    assert loss(
        torch.zeros((2, 1)), torch.zeros((2, 1)), torch.ones((2, 4))
    ).item() == pytest.approx(4.5)


def test_unconstrained_weighted_loss_preserves_device_and_weights():
    """No constraints keep the existing weighted base loss and device buffers."""
    scaler = StandardScaler().fit([[1, 10], [3, 30]])
    loss = build_cv_criterion(
        loss_name="weighted_mse",
        loss_weights=[100, 1],
        huber_delta=1,
        training_scaler=scaler,
        device=torch.device("cpu"),
    )
    assert not isinstance(loss, PhysicsConstrainedLoss)
    assert loss.weights.tolist() == [100, 1]
    assert loss.weights.device == torch.device("cpu")
    assert loss(torch.ones((2, 2)), torch.zeros((2, 2))).item() == 50.5


@pytest.mark.parametrize("pretrained", [False, True])
def test_cli_pretrained_selection_reaches_each_fresh_fold(
    cpu_cv, monkeypatch, pretrained
):
    """Default false must not inherit a registry model's pretrained true default."""
    from wavedl.models.registry import MODEL_REGISTRY

    original = MODEL_REGISTRY["cnn"]
    selections = []

    class SelectedModel(original):
        def __init__(self, in_shape, out_size, pretrained=True, **kwargs):
            selections.append(pretrained)
            super().__init__(in_shape, out_size, **kwargs)

    monkeypatch.setitem(MODEL_REGISTRY, "cnn", SelectedModel)
    cpu_cv(*(["--pretrained"] if pretrained else []))
    assert selections == [pretrained, pretrained]


@pytest.mark.parametrize("use_file", [False, True])
def test_cli_fold_constraints_use_training_only_scalers_and_base_validation(
    cpu_cv, monkeypatch, tmp_path, use_file
):
    """Actual CPU training gets penalties, but validation keeps the base loss."""
    from wavedl.utils import cross_validation

    original = cross_validation.train_fold
    penalties, scalers = [], []

    def observe(**options):
        loss = options["criterion"]
        assert isinstance(loss, PhysicsConstrainedLoss)
        np.testing.assert_allclose(loss.output_mean.cpu(), options["scaler"].mean_)
        np.testing.assert_allclose(loss.output_std.cpu(), options["scaler"].scale_)
        assert loss.weights == [0.5]
        calls = []

        def capture(_module, inputs):
            assert len(inputs) == 3
            calls.append(inputs[2].detach().clone())

        hook = loss.register_forward_pre_hook(capture)
        try:
            result = original(**options)
        finally:
            hook.remove()
        assert calls
        penalties.append(len(calls))
        scalers.append(loss.output_mean.detach().cpu().clone())
        return result

    monkeypatch.setattr(cross_validation, "train_fold", observe)
    if use_file:
        path = tmp_path / "constraint-cli.py"
        path.write_text(
            "def constraint(pred, inputs=None):\n"
            "    return pred[:, 0] - inputs.flatten(1).mean(1)\n"
        )
        flags = ["--constraint_file", str(path)]
    else:
        flags = ["--constraint", "y0 - x_mean"]
    results = cpu_cv(*flags, "--constraint_weight", "0.5", "--lr", "0.0001")
    assert penalties == [1, 1], "Penalties must run only for the one training batch"
    assert not torch.equal(scalers[0], scalers[1])
    assert all(result["criterion"] == "PhysicsConstrainedLoss" for result in results)
