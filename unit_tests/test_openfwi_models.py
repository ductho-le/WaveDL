"""CPU tests for the upstream-compatible OpenFWI model implementations."""

import importlib.util
import os
import pickle
from argparse import Namespace
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
from torch import nn

from wavedl.models import BaseModel, InversionNet, VelocityGAN, build_model, get_model
from wavedl.models.openfwi import (
    ConvBlock,
    ConvBlock_Tanh,
    DeconvBlock,
    Discriminator,
    load_openfwi_checkpoint,
    replace_legacy,
)


IN_SHAPE = (1000, 70)
OUT_SHAPE = (1, 70, 70)
SMALL_DIMS = {"dim1": 2, "dim2": 4, "dim3": 8, "dim4": 16, "dim5": 32}
# Optional references: a checkout of lanl/OpenFWI at the pinned commit and the
# official FlatVel-A InversionNet checkpoint. Tests skip when unavailable.
OPENFWI_SRC = Path(os.environ.get("OPENFWI_SRC", "/tmp/openfwi_src"))
OFFICIAL_CHECKPOINT = Path(
    os.environ.get(
        "OPENFWI_CHECKPOINT",
        Path.home() / "Datasets/OpenFWI/pretrained_official/InversionNet/fva_l1.pth",
    )
)


@pytest.fixture(scope="module")
def upstream_network():
    """Load the pinned, optional test reference without changing sys.path."""
    path = OPENFWI_SRC / "network.py"
    if not path.is_file():
        pytest.skip("Pinned upstream OpenFWI network.py is not available")
    spec = importlib.util.spec_from_file_location("openfwi_test_reference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_same_state(actual, expected):
    """Require identical parameter and BatchNorm-buffer names and values."""
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("name", "model_class"),
    [("inversionnet", InversionNet), ("velocitygan", VelocityGAN)],
)
def test_registry_and_model_contract(name, model_class):
    assert get_model(name) is model_class
    model = build_model(
        name,
        in_shape=IN_SHAPE,
        in_channels=5,
        out_size=1,
        out_shape=OUT_SHAPE,
        pretrained=True,  # Ignored: this must not trigger a network download.
        **SMALL_DIMS,
    )
    assert isinstance(model, BaseModel)
    assert model.supports_spatial_output and model.requires_spatial_output
    assert model.supported_input_channels == (5,)
    assert model.in_channels == 5
    assert model.in_shape == IN_SHAPE
    assert model.out_shape == OUT_SHAPE
    assert "convblock1.layers.0.weight" in model.state_dict()
    assert not any(key.startswith("generator.") for key in model.state_dict())


@pytest.mark.slow
@pytest.mark.parametrize("name", ["inversionnet", "velocitygan"])
def test_default_model_output_shape(name):
    model = build_model(name, in_shape=IN_SHAPE, out_size=1).eval()
    with torch.no_grad():
        output = model(torch.randn(2, 5, *IN_SHAPE))
    assert output.shape == (2, *OUT_SHAPE)
    assert torch.isfinite(output).all()
    assert output.min() >= -1 and output.max() <= 1
    assert isinstance(model.deconv6.layers[-1], nn.Tanh)


def test_default_parameter_counts():
    # Counts independently verified against lanl/OpenFWI@48754806.
    model = InversionNet(IN_SHAPE, 1)
    assert model.count_parameters() == 24_409_123
    discriminator = Discriminator()
    assert sum(p.numel() for p in discriminator.parameters() if p.requires_grad) == (
        1_180_003
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", ["inversionnet", "velocitygan", "discriminator"])
def test_upstream_initialization_and_forward_are_identical(upstream_network, name):
    torch.manual_seed(57)
    if name == "discriminator":
        reference = upstream_network.Discriminator()
        torch.manual_seed(57)
        model = Discriminator()
        inputs = torch.randn(2, *OUT_SHAPE)
    else:
        reference = upstream_network.InversionNet()
        torch.manual_seed(57)
        model = build_model(name, in_shape=IN_SHAPE, out_size=1)
        inputs = torch.randn(2, 5, *IN_SHAPE)
    assert_same_state(model.state_dict(), reference.state_dict())
    model.load_state_dict(reference.state_dict(), strict=True)
    model.eval()
    reference.eval()
    with torch.no_grad():
        actual = model(inputs)
        expected = reference(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("block_class", "kwargs", "input_shape"),
    [
        (ConvBlock, {"norm": "bn", "dropout": 0.3}, (2, 3, 8, 8)),
        (ConvBlock, {"norm": "in", "relu_slop": 0.1}, (2, 3, 8, 8)),
        (ConvBlock, {"norm": "none"}, (2, 3, 8, 8)),
        (ConvBlock_Tanh, {"norm": "bn"}, (2, 3, 8, 8)),
        (ConvBlock_Tanh, {"norm": "none"}, (2, 3, 8, 8)),
        (DeconvBlock, {"norm": "bn", "output_padding": 1}, (2, 3, 8, 8)),
        (DeconvBlock, {"norm": "none"}, (2, 3, 8, 8)),
    ],
)
def test_building_blocks_match_upstream(
    upstream_network, block_class, kwargs, input_shape
):
    reference = getattr(upstream_network, block_class.__name__)(3, 4, **kwargs).eval()
    model = block_class(3, 4, **kwargs).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    inputs = torch.randn(input_shape)
    with torch.no_grad():
        torch.testing.assert_close(model(inputs), reference(inputs), rtol=0, atol=0)
    if kwargs.get("dropout"):
        assert model.layers[-1].p == 0.8  # Preserve upstream's fixed probability.


def test_velocitygan_generator_and_discriminator_are_separate():
    model = VelocityGAN(IN_SHAPE, 1, **SMALL_DIMS)
    generator_state = model.state_dict()
    generator_parameters = {id(p) for p in model.parameters()}
    first = model.build_discriminator()
    second = model.build_discriminator()
    assert isinstance(first, Discriminator)
    assert isinstance(second, Discriminator)
    assert first is not second
    assert not generator_parameters.intersection(id(p) for p in first.parameters())
    assert not {id(p) for p in first.parameters()}.intersection(
        id(p) for p in second.parameters()
    )
    assert {id(p) for p in model.parameters()} == generator_parameters
    assert_same_state(model.state_dict(), generator_state)
    assert not any(isinstance(module, Discriminator) for module in model.modules())
    with torch.no_grad():
        assert first.eval()(torch.randn(2, *OUT_SHAPE)).shape == (2, 1)


def test_gradient_flow():
    model = InversionNet(IN_SHAPE, 1, **SMALL_DIMS)
    inputs = torch.randn(2, 5, *IN_SHAPE, requires_grad=True)
    output = model(inputs)
    output.square().mean().backward()
    assert output.shape == (2, *OUT_SHAPE)
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert inputs.grad.abs().sum() > 0
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("model_class", [InversionNet, VelocityGAN])
@pytest.mark.parametrize(
    ("invalid_config", "message"),
    [
        ({"in_shape": (1000, 69)}, "in_shape"),
        ({"in_shape": (64, 64)}, "in_shape"),
        ({"in_shape": (1000,)}, "in_shape"),
        ({"in_channels": 1}, "in_channels=5"),
        ({"in_channels": 5.0}, "in_channels=5"),
        ({"in_channels": True}, "in_channels=5"),
        ({"out_size": 2}, "out_size=1"),
        ({"out_shape": (1, 64, 64)}, "out_shape"),
        ({"out_shape": (70, 70)}, "out_shape"),
        ({"out_shape": (2, 70, 70)}, "out_shape"),
        ({"out_shape": (True, 70, 70)}, "out_shape"),
        ({"out_shape": (1, 70.0, 70)}, "out_shape"),
        ({"out_shape": 70}, "out_shape"),
        ({"sample_spatial": 0.5}, "sample_spatial=1.0"),
        ({"sample_spatial": True}, "sample_spatial=1.0"),
    ],
)
def test_bad_geometry_is_rejected(model_class, invalid_config, message):
    config = {"in_shape": IN_SHAPE, "out_size": 1, **SMALL_DIMS}
    config.update(invalid_config)
    with pytest.raises(ValueError, match=message):
        model_class(**config)


def legacy_state_dict(state_dict):
    """Use the keys emitted by upstream's actual legacy building blocks."""
    result = OrderedDict()
    for key, value in state_dict.items():
        block_name = key.split(".", 1)[0]
        if block_name.startswith("deconv") and block_name.endswith("_1"):
            legacy_name = "Deconv2DwithBN"
        else:
            # The legacy Tanh block also stored its Sequential as Conv2DwithBN.
            legacy_name = "Conv2DwithBN"
        result[key.replace("layers", legacy_name)] = value
    return result


@pytest.mark.parametrize("model_class", [InversionNet, VelocityGAN])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("payload", ["bare", "namespace", "dict"])
def test_checkpoint_round_trip(tmp_path, legacy, payload, model_class):
    original = InversionNet(IN_SHAPE, 1, **SMALL_DIMS)
    state_dict = original.state_dict()
    if legacy:
        state_dict = legacy_state_dict(state_dict)
    args = {"dataset": "flatvel-a", "sample_spatial": 1.0}
    checkpoint = state_dict
    expected_metadata = {}
    if payload != "bare":
        checkpoint = {
            "model": state_dict,
            "epoch": 17,
            "args": Namespace(**args) if payload == "namespace" else args,
            "optimizer": {},
        }
        expected_metadata = {"epoch": 17, "args": args}
    path = tmp_path / "upstream.pth"
    torch.save(checkpoint, path)
    restored = model_class(IN_SHAPE, 1, **SMALL_DIMS)
    prior_safe_globals = torch.serialization.get_safe_globals().copy()
    metadata = load_openfwi_checkpoint(restored, path)
    assert metadata == expected_metadata
    assert set(torch.serialization.get_safe_globals()) == set(prior_safe_globals)
    assert_same_state(restored.state_dict(), original.state_dict())
    assert all(value.device.type == "cpu" for value in restored.state_dict().values())


def test_legacy_conversion_matches_upstream(upstream_network):
    state = legacy_state_dict(InversionNet(IN_SHAPE, 1, **SMALL_DIMS).state_dict())
    assert_same_state(replace_legacy(state), upstream_network.replace_legacy(state))


def test_checkpoint_loading_is_strict(tmp_path):
    model = InversionNet(IN_SHAPE, 1, **SMALL_DIMS)
    state = model.state_dict()
    state.pop("convblock1.layers.0.weight")
    path = tmp_path / "missing-key.pth"
    torch.save({"model": state}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_openfwi_checkpoint(model, path)


class UnsupportedMetadata:
    """A harmless test class that must not be allow-listed by the loader."""


def test_checkpoint_loader_rejects_other_pickle_globals(tmp_path):
    model = InversionNet(IN_SHAPE, 1, **SMALL_DIMS)
    path = tmp_path / "unsupported-global.pth"
    torch.save({"model": model.state_dict(), "args": UnsupportedMetadata()}, path)
    prior_safe_globals = torch.serialization.get_safe_globals().copy()
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        load_openfwi_checkpoint(model, path)
    assert set(torch.serialization.get_safe_globals()) == set(prior_safe_globals)


@pytest.mark.parametrize(
    "checkpoint", [[], {"model": []}, {"model": {}, "args": "not a namespace"}]
)
def test_checkpoint_loader_rejects_invalid_structure(tmp_path, checkpoint):
    path = tmp_path / "invalid.pth"
    torch.save(checkpoint, path)
    model = InversionNet(IN_SHAPE, 1, **SMALL_DIMS)
    with pytest.raises(ValueError, match="checkpoint"):
        load_openfwi_checkpoint(model, path)


@pytest.mark.slow
@pytest.mark.skipif(
    not OFFICIAL_CHECKPOINT.is_file(),
    reason="Official InversionNet weights unavailable",
)
def test_official_checkpoint_loads_strictly():
    model = InversionNet(IN_SHAPE, 1).eval()
    metadata = load_openfwi_checkpoint(model, OFFICIAL_CHECKPOINT)
    assert isinstance(metadata.get("args"), dict)
    assert "epoch" in metadata
    with torch.no_grad():
        output = model(torch.zeros(1, 5, *IN_SHAPE))
    assert output.shape == (1, *OUT_SHAPE)
    assert torch.isfinite(output).all()


@pytest.mark.slow
def test_official_velocitygan_checkpoint_loads_strictly():
    path = Path(
        os.environ.get(
            "OPENFWI_VELOCITYGAN_CHECKPOINT",
            Path.home()
            / "Datasets/OpenFWI/pretrained_official/VelocityGAN/flatvel_a_l1_480.pth",
        )
    )
    if not path.is_file():
        pytest.skip("Official VelocityGAN weights unavailable")
    model = VelocityGAN(IN_SHAPE, 1).eval()
    metadata = load_openfwi_checkpoint(model, path)
    assert isinstance(metadata.get("args"), dict)
    assert "epoch" in metadata
    with torch.no_grad():
        output = model(torch.zeros(1, 5, *IN_SHAPE))
    assert output.shape == (1, *OUT_SHAPE)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("name", ["inversionnet", "velocitygan"])
def test_openfwi_optimizer_groups_decay_every_parameter(name):
    """Upstream AdamW decays all parameters, including biases and BatchNorm."""
    model = build_model(
        name, in_shape=(1000, 70), out_size=1, in_channels=5, out_shape=(1, 70, 70)
    )
    groups = model.get_optimizer_groups(1e-4, 1e-4)
    assert len(groups) == 1
    assert groups[0]["weight_decay"] == 1e-4
    assert {id(p) for p in groups[0]["params"]} == {id(p) for p in model.parameters()}
