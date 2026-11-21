"""Focused tests for dense spatial model outputs."""

import pytest
import torch

from wavedl.models import (
    InversionNet,
    InversionNetResize,
    UNetField,
    UNetRegression,
    build_model,
    get_model,
)
from wavedl.models.inversionnet import DeconvBlock2d, ResizeConvBlock2d


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


@pytest.mark.parametrize(
    ("model_name", "model_class"),
    [
        ("inversionnet", InversionNet),
        ("inversionnet_resize", InversionNetResize),
    ],
)
def test_inversionnet_variants_are_exported_registered_and_distinct(
    model_name, model_class
):
    assert get_model(model_name) is model_class
    assert model_class.supports_spatial_output is True
    assert model_class.requires_spatial_output is True
    assert model_class.supported_input_channels is None

    config = model_class.get_default_config()
    config["base_channels"] = 2
    model = build_model(model_name, in_shape=(16, 16), out_size=2, **config)
    assert isinstance(model, model_class)

    if model_class is InversionNet:
        assert any(isinstance(m, DeconvBlock2d) for m in model.decoder_stages)
        assert not any(isinstance(m, ResizeConvBlock2d) for m in model.decoder_stages)
    else:
        assert any(isinstance(m, ResizeConvBlock2d) for m in model.decoder_stages)
        assert not any(
            isinstance(m, torch.nn.ConvTranspose2d)
            for m in model.decoder_stages.modules()
        )


@pytest.mark.parametrize("model_class", [InversionNet, InversionNetResize])
@pytest.mark.parametrize(
    ("in_shape", "in_channels", "out_shape"),
    [
        ((16, 16), 1, (1, 16, 16)),
        ((96, 16), 5, (1, 24, 24)),
        ((16, 96), 3, (2, 12, 18)),
        ((17, 21), 2, (3, 11, 13)),
        ((4, 4), 1, (1, 8, 8)),
    ],
)
def test_inversionnet_arbitrary_2d_shapes_channels_and_gradients(
    model_class, in_shape, in_channels, out_shape
):
    """Both variants handle arbitrary 2D input/output geometries at batch_size=1."""
    model = model_class(
        in_shape=in_shape,
        out_size=out_shape[0],
        out_shape=out_shape,
        in_channels=in_channels,
        base_channels=2,
        dropout_rate=0.0,
    )
    x = torch.randn(1, in_channels, *in_shape, requires_grad=True)
    out = model(x)

    assert model.out_shape == out_shape
    assert out.shape == (1, *out_shape)
    assert torch.isfinite(out).all()

    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Unused trainable parameter {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient {name}"

    model.eval()
    with torch.no_grad():
        det1 = model(x.detach())
        det2 = model(x.detach())
    torch.testing.assert_close(det1, det2)


@pytest.mark.parametrize("model_class", [InversionNet, InversionNetResize])
@pytest.mark.parametrize("norm", ["gn", "bn", "none"])
@pytest.mark.parametrize("output_activation", [None, "tanh"])
def test_inversionnet_norm_and_activation_options(model_class, norm, output_activation):
    model = model_class(
        in_shape=(16, 20),
        out_size=2,
        out_shape=(2, 12, 14),
        in_channels=3,
        base_channels=2,
        norm=norm,
        output_activation=output_activation,
    )
    x = torch.randn(2, 3, 16, 20)
    out = model(x)
    assert out.shape == (2, 2, 12, 14)
    assert torch.isfinite(out).all()
    if output_activation == "tanh":
        assert out.min() >= -1.0 and out.max() <= 1.0


@pytest.mark.parametrize("upsample_mode", ["bilinear", "nearest"])
def test_inversionnet_resize_upsample_modes(upsample_mode):
    model = InversionNetResize(
        in_shape=(16, 16),
        out_size=1,
        out_shape=(1, 10, 12),
        base_channels=2,
        upsample_mode=upsample_mode,
    )
    out = model(torch.randn(1, 1, 16, 16))
    assert out.shape == (1, 1, 10, 12)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("model_class", [InversionNet, InversionNetResize])
@pytest.mark.parametrize(
    ("invalid_config", "message"),
    [
        ({"in_shape": (16,)}, "requires 2D input"),
        ({"in_shape": (8, 8, 8)}, "requires 2D input"),
        ({"in_channels": 0}, "in_channels must be a positive integer"),
        ({"in_channels": True}, "in_channels must be a positive integer"),
        ({"base_channels": 0}, "base_channels must be a positive integer"),
        ({"bottleneck_grid": 0}, "bottleneck_grid must be a positive integer"),
        ({"dropout_rate": -0.1}, "dropout_rate"),
        ({"dropout_rate": 1.0}, "dropout_rate"),
        ({"dropout_rate": True}, "dropout_rate"),
        ({"norm": "invalid"}, "Unsupported norm"),
        ({"output_activation": "sigmoid"}, "Unsupported output_activation"),
        ({"out_shape": (2, 8)}, "3 dimensions"),
        ({"out_shape": (3, 8, 8)}, "must match out_size"),
        ({"out_shape": (2, 0, 8)}, "positive integers"),
    ],
)
def test_inversionnet_rejects_invalid_config(model_class, invalid_config, message):
    config = {"in_shape": (16, 16), "out_size": 2, "base_channels": 2}
    config.update(invalid_config)
    with pytest.raises(ValueError, match=message):
        model_class(**config)


def test_inversionnet_resize_rejects_invalid_upsample_mode():
    with pytest.raises(ValueError, match="Unsupported upsample_mode"):
        InversionNetResize(
            in_shape=(16, 16), out_size=1, base_channels=2, upsample_mode="cubic"
        )


def test_inversionnet_rejects_channel_mismatch_in_forward():
    model = InversionNet(in_shape=(16, 16), out_size=1, in_channels=3, base_channels=2)
    with pytest.raises(ValueError, match="Input channel mismatch"):
        model(torch.randn(1, 1, 16, 16))


@pytest.mark.parametrize("model_name", ["inversionnet", "inversionnet_resize"])
def test_inversionnet_onnx_export_roundtrip(tmp_path, model_name):
    onnxruntime = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")

    from wavedl.test import export_to_onnx
    from wavedl.utils.targets import SpatialTargetScaler

    model = build_model(
        model_name,
        in_shape=(17, 21),
        out_size=2,
        out_shape=(2, 12, 14),
        in_channels=3,
        base_channels=2,
    ).eval()
    scaler = SpatialTargetScaler().fit(torch.randn(4, 2, 12, 14).numpy() * 5.0 + 10.0)
    sample = torch.randn(2, 3, 17, 21)
    onnx_path = tmp_path / f"{model_name}.onnx"

    export_to_onnx(
        model=model,
        sample_input=sample[:1],
        output_path=str(onnx_path),
        model_name=model_name,
        scaler=scaler,
        include_denorm=True,
    )

    session = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    ort_out = session.run(None, {"input": sample.numpy()})[0]
    with torch.no_grad():
        torch_out = scaler.inverse_transform(model(sample).numpy())
    torch.testing.assert_close(
        torch.from_numpy(ort_out),
        torch.from_numpy(torch_out.astype("float32")),
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.parametrize("model_name", ["inversionnet", "inversionnet_resize"])
def test_inversionnet_bn_handles_batch_size_one_in_train_mode(model_name):
    model = build_model(
        model_name,
        in_shape=(32, 32),
        out_size=1,
        out_shape=(1, 16, 16),
        in_channels=2,
        base_channels=4,
        norm="bn",
    ).train()
    x = torch.randn(1, 2, 32, 32)
    out = model(x)
    assert out.shape == (1, 1, 16, 16)
    out.sum().backward()


def test_inversionnet_resize_default_width_onnx_validation(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from wavedl.test import export_to_onnx
    from wavedl.utils.preprocessing import MinMaxTargetScaler

    torch.manual_seed(0)
    model = build_model(
        "inversionnet_resize",
        in_shape=(64, 32),
        out_size=1,
        out_shape=(1, 32, 32),
        in_channels=5,
    ).eval()
    sample = torch.randn(1, 5, 64, 32)
    scaler = MinMaxTargetScaler(bounds=(1500.0, 4500.0), target_shape=(1, 32, 32))

    assert export_to_onnx(
        model=model,
        sample_input=sample,
        output_path=str(tmp_path / "inv_resize_raw.onnx"),
        model_name="inversionnet_resize",
        validate=True,
    )
    assert export_to_onnx(
        model=model,
        sample_input=sample,
        output_path=str(tmp_path / "inv_resize_denorm.onnx"),
        model_name="inversionnet_resize",
        scaler=scaler,
        include_denorm=True,
        validate=True,
    )
