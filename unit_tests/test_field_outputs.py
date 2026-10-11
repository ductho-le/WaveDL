"""Focused tests for dense spatial model outputs."""

import pytest
import torch

from wavedl.models import UNetField, UNetRegression, build_model, get_model


@pytest.mark.parametrize(
    ("in_shape", "out_size"),
    [
        ((16,), 2),
        ((16, 20), 3),
        ((8, 8, 12), 1),
    ],
)
def test_unet_field_default_shape_and_gradients(in_shape, out_size):
    model = UNetField(
        in_shape=in_shape,
        out_size=out_size,
        base_channels=2,
        depth=2,
        dropout_rate=0.0,
    )
    x = torch.randn(1, 1, *in_shape, requires_grad=True)

    output = model(x)

    assert model.supports_spatial_output is True
    assert model.requires_spatial_output is True
    assert model.out_shape == (out_size, *in_shape)
    assert output.shape == (1, out_size, *in_shape)

    output.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert model.head.weight.grad is not None
    assert torch.isfinite(model.head.weight.grad).all()


@pytest.mark.parametrize(
    ("in_shape", "out_shape"),
    [((17,), (2, 11)), ((17, 21), (2, 11, 13)), ((9, 11, 13), (2, 5, 7, 9))],
)
def test_unet_field_interpolates_to_requested_output_shape(in_shape, out_shape):
    model = UNetField(
        in_shape=in_shape,
        out_size=2,
        out_shape=out_shape,
        base_channels=2,
        depth=2,
        dropout_rate=0.0,
    )

    output = model(torch.randn(2, 1, *in_shape))

    assert output.shape == (2, *out_shape)
    assert model.head.kernel_size == (1,) * len(in_shape)
    output.square().mean().backward()
    assert torch.isfinite(model.head.weight.grad).all()


def test_unet_field_is_exported_and_registered():
    assert get_model("unet_field") is UNetField

    model = build_model(
        "unet_field",
        in_shape=(16, 16),
        out_size=1,
        base_channels=2,
        depth=2,
    )
    assert isinstance(model, UNetField)


@pytest.mark.parametrize(
    "out_shape", [(2, 8), (3, 8, 8), (2, 0, 8), (2, 8.5, 8), (2, True, 8)]
)
def test_unet_field_rejects_noncanonical_output_shapes(out_shape):
    with pytest.raises(ValueError):
        UNetField(
            in_shape=(8, 8),
            out_size=2,
            out_shape=out_shape,
            base_channels=2,
            depth=2,
        )


@pytest.mark.parametrize("model_class", [UNetField, UNetRegression])
@pytest.mark.parametrize("in_shape", [(4,), (4, 4), (4, 4, 4)])
def test_unet_minimum_grid_supports_single_sample_and_gradients(model_class, in_shape):
    """The deepest GroupNorm must remain valid on a one-cell bottleneck."""
    model = model_class(
        in_shape=in_shape,
        out_size=2,
        base_channels=2,
        depth=2,
        dropout_rate=0.0,
    )
    inputs = torch.randn(1, 1, *in_shape, requires_grad=True)
    output = model(inputs)

    expected_shape = (1, 2, *in_shape) if model_class is UNetField else (1, 2)
    assert output.shape == expected_shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert torch.isfinite(inputs.grad).all()
    assert inputs.grad.abs().sum() > 0

    model.eval()
    with torch.no_grad():
        assert torch.isfinite(model(inputs.detach())).all()


@pytest.mark.parametrize("model_class", [UNetField, UNetRegression])
@pytest.mark.parametrize(
    ("invalid_config", "message"),
    [
        ({"depth": -1}, "depth must be a nonnegative integer"),
        ({"depth": 1.5}, "depth must be a nonnegative integer"),
        ({"depth": True}, "depth must be a nonnegative integer"),
        ({"base_channels": 0}, "base_channels must be a positive integer"),
        ({"base_channels": -1}, "base_channels must be a positive integer"),
        ({"out_size": 0}, "out_size must be a positive integer"),
        ({"out_size": True}, "out_size must be a positive integer"),
        ({"in_shape": (16, 0)}, "in_shape dimensions must be positive integers"),
        ({"in_shape": (16, 16.5)}, "in_shape dimensions must be positive integers"),
        ({"in_shape": (16,) * 4}, "Unsupported dimensionality"),
        ({"in_shape": (3, 4)}, "requires each spatial axis >= 4"),
        (
            {"in_shape": (1, 1), "base_channels": 1, "depth": 0},
            "at least two values per group",
        ),
    ],
)
def test_unet_rejects_invalid_config_before_forward(
    model_class, invalid_config, message
):
    config = {"in_shape": (16, 16), "out_size": 2, "base_channels": 2, "depth": 2}
    config.update(invalid_config)
    with pytest.raises(ValueError, match=message):
        model_class(**config)
