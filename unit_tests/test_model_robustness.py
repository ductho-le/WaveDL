"""Regression coverage for model numerics and frozen feature extractors."""

import pytest
import torch
from torch import nn

from wavedl.models import build_model
from wavedl.models._pretrained_utils import (
    DropPath,
    GRN1d,
    GRN2d,
    GRN3d,
    adapt_input_channels,
    freeze_backbone,
)
from wavedl.models.mamba import SelectiveSSM
from wavedl.models.wavenet import GatedResidualBlock, WaveNetBase


def _reference_scan(x, delta, a, b, c, d):
    transition = torch.exp(delta.unsqueeze(-1) * a)
    forcing = delta.unsqueeze(-1) * b.unsqueeze(2) * x.unsqueeze(-1)
    state = x.new_zeros(x.shape[0], x.shape[2], a.numel())
    outputs = []
    for t in range(x.shape[1]):
        state = transition[:, t] * state + forcing[:, t]
        outputs.append((c[:, t].unsqueeze(1) * state).sum(-1) + d * x[:, t])
    return torch.stack(outputs, dim=1)


@pytest.mark.parametrize("length", [1, 9, 64, 257, 513])
@pytest.mark.parametrize("chunked", [False, True])
def test_selective_scan_realistic_decay_values_and_gradients(length, chunked):
    """Ordinary initialized decays must not erase later states or gradients."""
    torch.manual_seed(73)
    inputs = [
        torch.randn(2, length, 4, dtype=torch.float64),
        torch.rand(2, length, 4, dtype=torch.float64) + 0.5,
        -torch.arange(1, 4, dtype=torch.float64),
        torch.randn(2, length, 3, dtype=torch.float64),
        torch.randn(2, length, 3, dtype=torch.float64),
        torch.randn(4, dtype=torch.float64),
    ]
    inputs = [x.requires_grad_() for x in inputs]
    ssm = SelectiveSSM(2, d_state=3, chunk_size=31)
    scan = ssm._chunked_selective_scan if chunked else ssm._selective_scan_single
    actual = scan(*inputs)
    expected = _reference_scan(*inputs)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    probe = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * probe).sum(), inputs)
    expected_grads = torch.autograd.grad((expected * probe).sum(), inputs)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-9, rtol=1e-9)


def test_selective_scan_zero_transitions_keep_current_input():
    ssm = SelectiveSSM(1, d_state=1, chunk_size=3)
    x = torch.ones(1, 8, 2)
    delta = torch.ones_like(x)
    states = torch.ones(1, 8, 1)
    for scan in [ssm._selective_scan_single, ssm._chunked_selective_scan]:
        result = scan(x, delta, torch.tensor([-1000.0]), states, states, torch.zeros(2))
        torch.testing.assert_close(result, x)


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf")])
def test_drop_path_rejects_invalid_probability(probability):
    with pytest.raises(ValueError, match="drop_prob"):
        DropPath(probability)


def test_drop_path_probability_one_has_zero_finite_gradient():
    layer = DropPath(1.0)
    x = torch.ones(3, 4, requires_grad=True)
    actual = layer(x)
    torch.testing.assert_close(actual, torch.zeros_like(x))
    actual.sum().backward()
    torch.testing.assert_close(x.grad, torch.zeros_like(x))
    layer.eval()
    torch.testing.assert_close(layer(x), x)


@pytest.mark.parametrize(
    "grn,shape", [(GRN1d, (1024,)), (GRN2d, (32, 32)), (GRN3d, (8, 8, 16))]
)
def test_grn_half_precision_norm_does_not_overflow(grn, shape):
    layer = grn(2).half()
    x = torch.full((1, 2, *shape), 3000.0, dtype=torch.float16, requires_grad=True)
    result = layer(x)
    torch.testing.assert_close(result, x)
    result.float().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.gamma.grad).all()


def test_input_adaptation_preserves_padding_subclass_and_tensor_properties():
    from timm.layers import Conv2dSame

    original = Conv2dSame(3, 4, kernel_size=3, stride=2, bias=True).double().eval()
    original.requires_grad_(False)
    adapted = adapt_input_channels(original, new_in_channels=1)
    assert type(adapted) is type(original)
    assert adapted.weight.dtype == original.weight.dtype
    assert adapted.weight.device == original.weight.device
    assert not adapted.training and not adapted.weight.requires_grad
    torch.testing.assert_close(adapted.bias, original.bias)
    x = torch.randn(2, 1, 6, 8, dtype=torch.float64)
    # RGB averaging retains the pre-existing grayscale initialization contract.
    expected = (
        original(x.expand(-1, 3, -1, -1)) - original.bias[None, :, None, None]
    ) / 3
    expected += original.bias[None, :, None, None]
    torch.testing.assert_close(adapted(x), expected)


def test_input_adaptation_preserves_existing_channels():
    original = nn.Conv2d(3, 4, 1)
    adapted = adapt_input_channels(original, new_in_channels=3)
    torch.testing.assert_close(adapted.weight, original.weight)
    torch.testing.assert_close(adapted.bias, original.bias)


def test_freeze_utility_does_not_match_nested_fc_names():
    backbone = nn.ModuleDict(
        {"features": nn.ModuleDict({"fc1": nn.Linear(3, 3)}), "fc": nn.Linear(3, 1)}
    )
    freeze_backbone(backbone, exclude_patterns=["fc"])
    assert all(not p.requires_grad for p in backbone["features"].parameters())
    assert all(p.requires_grad for p in backbone["fc"].parameters())


@pytest.mark.parametrize(
    "name,shape,head",
    [
        ("regnet_y_400mf", (64, 64), "fc"),
        ("resnet3d_18", (8, 32, 32), "fc"),
        ("resnet18_pretrained", (64, 64), "fc"),
        ("efficientnet_b0", (64, 64), "classifier"),
        ("mobilenet_v3_small", (64, 64), "classifier"),
        ("fastvit_t8", (64, 64), None),
    ],
)
def test_frozen_backbone_weights_modes_buffers_and_head_gradients(name, shape, head):
    model = build_model(name, shape, 2, pretrained=False, freeze_backbone=True)
    for training in [True, False, True]:
        model.train(training)
        assert model.training is training
        for path, module in model.backbone.named_modules():
            is_head = head is not None and (path == head or path.startswith(head + "."))
            assert module.training is (training and is_head), path
    for path, param in model.backbone.named_parameters():
        assert param.requires_grad is (
            head is not None and path.startswith(head + ".")
        ), path
    buffers = {name: value.clone() for name, value in model.backbone.named_buffers()}
    result = model(torch.randn(2, 1, *shape))
    result.square().mean().backward()
    for name, value in model.backbone.named_buffers():
        torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
    regression_head = model.backbone.get_submodule(head) if head else model.head
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in regression_head.parameters()
    )


@pytest.mark.parametrize("kernel_size", [1, 2, 3, 4])
@pytest.mark.parametrize("dilation", [1, 2, 3])
def test_wavenet_preserves_length_for_even_and_odd_kernels(kernel_size, dilation):
    block = GatedResidualBlock(4, 6, kernel_size, dilation, dropout=0)
    x = torch.randn(2, 4, 17, requires_grad=True)
    residual, skip = block(x)
    assert residual.shape == (2, 4, 17)
    assert skip.shape == (2, 6, 17)
    (residual.mean() + skip.mean()).backward()
    assert torch.isfinite(x.grad).all()


def test_wavenet_reports_convolutional_receptive_field():
    model = WaveNetBase(
        (32,), 1, channels=4, skip_channels=8, n_layers=3, kernel_size=3
    )
    assert model.receptive_field == 1 + (3 - 1) * (1 + 2 + 4)


@pytest.mark.parametrize(
    "name",
    ["convnext_v2_tiny", "unireplknet_tiny", "s4d_small", "wavenet_small", "tcn_small"],
)
def test_variant_default_config_round_trip(name):
    from wavedl.models import get_model

    model_class = get_model(name)
    config = model_class.get_default_config()
    model = model_class((32,), 2, **config).eval()
    with torch.no_grad():
        result = model(torch.randn(1, 1, 32))
    assert result.shape == (1, 2)
    assert torch.isfinite(result).all()
    # A returned configuration must not mutate later model defaults.
    for value in config.values():
        if isinstance(value, list):
            value.clear()
    assert all(
        value
        for value in model_class.get_default_config().values()
        if isinstance(value, list)
    )


def test_wavenet_has_no_unused_trainable_parameters():
    model = WaveNetBase((17,), 2, channels=4, skip_channels=8, n_layers=2)
    model(torch.randn(2, 1, 17)).square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )
    assert "blocks.1.residual_conv.weight" in model.state_dict()
