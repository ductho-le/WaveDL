"""Published checkpoint layouts must not silently change model behavior."""

import pytest
import torch

from wavedl.models import build_model
from wavedl.models.mamba import SelectiveSSM


def test_wavenet_legacy_optimizer_restores_and_keeps_unused_weights_fixed():
    model = build_model("wavenet_small", in_shape=(32,), out_size=2)
    residual = model.blocks[-1].residual_conv
    residual.requires_grad_(True)
    legacy = torch.optim.AdamW(model.get_optimizer_groups(0.001))
    model(torch.randn(2, 1, 32)).square().mean().backward()
    legacy.step()
    state = legacy.state_dict()
    legacy.zero_grad(set_to_none=True)
    residual.requires_grad_(False)
    current = torch.optim.AdamW(model.get_optimizer_groups(0.001))
    current.load_state_dict(state)
    before = residual.weight.detach().clone()
    model(torch.randn(2, 1, 32)).square().mean().backward()
    current.step()
    torch.testing.assert_close(residual.weight, before, rtol=0, atol=0)
    assert residual.weight.grad is None


@pytest.mark.parametrize("strict", [True, False])
def test_unversioned_mamba_checkpoint_requires_explicit_retraining(strict):
    model = SelectiveSSM(8, d_state=4)
    legacy = model.state_dict()
    del legacy["scan_version"]
    with pytest.raises(RuntimeError, match="original WaveDL version"):
        model.load_state_dict(legacy, strict=strict)


def test_versioned_mamba_checkpoint_preserves_predictions():
    model = SelectiveSSM(8, d_state=4).eval()
    restored = SelectiveSSM(8, d_state=4).eval()
    restored.load_state_dict(model.state_dict())
    x = torch.randn(2, 129, 8)
    torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)


def test_maxvit_legacy_partitions_preserve_checkpoint_predictions(monkeypatch):
    from wavedl.models.maxvit import MaxViTBase

    with monkeypatch.context() as legacy:
        legacy.setattr(
            MaxViTBase, "_compute_compatible_size", lambda self, shape: (224, 224)
        )
        original = build_model(
            "maxvit_tiny", in_shape=(64, 64), out_size=2, pretrained=False
        ).eval()
    restored = build_model(
        "maxvit_tiny", in_shape=(64, 64), out_size=2, pretrained=False
    ).eval()
    restored.load_state_dict(original.state_dict())
    x = torch.randn(1, 1, 64, 64)
    with torch.inference_mode():
        torch.testing.assert_close(restored(x), original(x), rtol=0, atol=0)


@pytest.mark.parametrize(
    "name", ["unireplknet_tiny", "unireplknet_small", "unireplknet_base"]
)
def test_unireplknet_legacy_3d_kernels_preserve_checkpoint_predictions(name):
    # Small stage widths keep the test affordable while exercising every kernel.
    kwargs = {
        "in_shape": (32, 32, 32),
        "out_size": 2,
        "depths": [1, 1, 1, 1],
        "dims": [2, 2, 2, 2],
    }
    original = build_model(name, **kwargs, kernel_sizes=[31, 29, 17, 13]).eval()
    restored = build_model(name, **kwargs).eval()
    restored.load_state_dict(original.state_dict())
    x = torch.randn(1, 1, 32, 32, 32)
    with torch.inference_mode():
        torch.testing.assert_close(restored(x), original(x), rtol=0, atol=0)
