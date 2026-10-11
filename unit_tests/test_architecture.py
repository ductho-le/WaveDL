"""
Optimized Unit Tests for Model Architectures
=============================================

Fast, comprehensive tests using combined assertions and efficient design.

**Optimizations**:
    - Combined tests: 5+ separate tests → 1 comprehensive test
    - Smaller batch sizes (1 instead of 2)
    - Fewer parametrized variations (2 batch sizes vs 3)
    - Skip 3D tests by default (use -m slow)
    - Model building is the bottleneck - test more per build

**Test Coverage**:
    - Model instantiation
    - Forward pass correctness
    - Gradient flow
    - Numerical stability (NaN/Inf)
    - Eval mode determinism
    - Parameter counting
    - Batch size flexibility

Run fast tests:
    pytest unit_tests/test_architecture.py -v

Run all tests including 3D:
    pytest unit_tests/test_architecture.py -v -m slow

Author: Ductho Le (ductho.le@outlook.com)
"""

from typing import Any

import pytest
import torch

from wavedl.models.cnn import CNN
from wavedl.models.registry import build_model, get_model, list_models


# Mark all tests in this module as slow (run with -m slow)
pytestmark = pytest.mark.slow


# ==============================================================================
# MODEL DIMENSIONALITY MAPPING
# ==============================================================================

MODEL_DIMS = {
    "tcn": [1],
    "efficientnet": [2],
    "mobilenet": [2],
    "regnet": [2],
    "swin": [2],
    "resnet3d": [3],
    "mc3": [3],
    "vit": [1, 2],
    "resnet": [1, 2, 3],
    "unet": [1, 2, 3],
    "cnn": [1, 2],
    "convnext": [1, 2],
    "densenet": [1, 2],
    # New models
    "convnext_v2": [1, 2],  # Skip 3D for faster tests
    "mamba": [1],
    "vim": [2],
    "maxvit": [2],
    "fastvit": [2],
    "caformer": [2],
    "poolformer": [2],
    "efficientvit": [2],
    "unireplknet": [1, 2],
    # v1.8+ models
    "wavenet": [1],
    "s4d": [1],
}

DEFAULT_DIMS = [2]


def get_supported_dims(model_name: str) -> list[int]:
    """Get supported input dimensionalities for a model."""
    model_lower = model_name.lower()

    if model_lower.endswith("_pretrained"):
        return [2]

    if model_lower.startswith("resnet3d") or model_lower.startswith("mc3"):
        return [3]

    for prefix, dims in MODEL_DIMS.items():
        if model_lower.startswith(prefix):
            return dims

    return DEFAULT_DIMS


def get_primary_dim(model_name: str) -> int:
    """Get the primary (first) supported dimension for a model."""
    return get_supported_dims(model_name)[0]


def get_test_config(model_name: str, dim: int | None = None) -> tuple:
    """Get appropriate test configuration for a model."""
    if dim is None:
        dim = get_primary_dim(model_name)

    model_lower = model_name.lower()

    # Models requiring larger input sizes (timm models with window attention)
    large_input_models = ["maxvit", "fastvit", "caformer", "poolformer", "efficientvit"]
    needs_large_input = any(model_lower.startswith(p) for p in large_input_models)

    in_shape: tuple[int, ...]
    if dim == 1:
        in_shape = (256,)
    elif dim == 2:
        if needs_large_input:
            in_shape = (224, 224)
        else:
            in_shape = (64, 64)
    else:
        in_shape = (16, 64, 64)

    # Several families default to pretrained=True without a _pretrained suffix.
    # Architecture checks must run offline and test construction/gradients
    # independently of downloaded weights.
    kwargs: dict[str, Any] = {"pretrained": False}

    if model_lower.startswith("vit"):
        kwargs["patch_size"] = 8 if dim == 2 else 16
    elif model_lower.startswith("unet"):
        kwargs["depth"] = 3

    return in_shape, kwargs


# ==============================================================================
# FAST COMPREHENSIVE TESTS
# ==============================================================================


class TestAllModels:
    """
    Fast comprehensive tests for ALL registered models.

    Each test covers multiple aspects to minimize model rebuild overhead.
    """

    @pytest.mark.parametrize("model_name", list_models())
    def test_model_complete(self, model_name):
        """
        Combined test: instantiation, forward, gradient, determinism, params.

        Tests 5+ aspects in one model build - much faster than separate tests.
        """
        in_shape, kwargs = get_test_config(model_name)
        config = get_model(model_name).get_default_config()
        config.update(kwargs)
        model = build_model(model_name, in_shape=in_shape, out_size=3, **config)

        # 1. Has parameters
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count > 0, f"{model_name}: No trainable parameters"

        # 2. Forward pass (eval mode)
        model.eval()
        x = torch.randn(1, 1, *in_shape)
        with torch.no_grad():
            out = model(x)

        expected_shape = (
            (1, *model.out_shape) if model.requires_spatial_output else (1, 3)
        )
        assert out.shape == expected_shape, f"{model_name}: Output shape mismatch"
        assert not torch.isnan(out).any(), f"{model_name}: Output contains NaN"
        assert not torch.isinf(out).any(), f"{model_name}: Output contains Inf"

        # 3. Determinism check
        with torch.no_grad():
            out2 = model(x)
        assert torch.allclose(out, out2), f"{model_name}: Not deterministic"

        # 4. Gradient flow
        model.train()
        x_grad = torch.randn(1, 1, *in_shape, requires_grad=True)
        out = model(x_grad)
        loss = out.sum()
        loss.backward()

        assert x_grad.grad is not None, f"{model_name}: No gradient on input"
        assert torch.isfinite(x_grad.grad).all(), (
            f"{model_name}: Non-finite input gradient"
        )
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None, (
                    f"{model_name}: Unused trainable parameter {name}"
                )
                assert torch.isfinite(parameter.grad).all(), (
                    f"{model_name}: Non-finite gradient {name}"
                )

    @pytest.mark.parametrize("model_name", list_models())
    def test_model_output_shape(self, model_name):
        """Test output shape for different out_size values."""
        in_shape, kwargs = get_test_config(model_name)

        for out_size in [1, 5]:  # 2 sizes instead of 4
            model = build_model(
                model_name, in_shape=in_shape, out_size=out_size, **kwargs
            )
            model.eval()

            x = torch.randn(1, 1, *in_shape)
            with torch.no_grad():
                out = model(x)

            model_lower = model_name.lower()
            if model_lower.startswith("unet"):
                assert out.shape[0] == 1
                assert out.shape[1] == out_size
            else:
                assert out.shape == (1, out_size), (
                    f"{model_name}: Expected {(1, out_size)}, got {out.shape}"
                )

    @pytest.mark.parametrize("model_name", list_models())
    def test_model_batch_sizes(self, model_name):
        """Test batch size flexibility."""
        in_shape, kwargs = get_test_config(model_name)
        model = build_model(model_name, in_shape=in_shape, out_size=3, **kwargs)
        model.eval()

        for batch_size in [1, 4]:  # 2 sizes instead of 3
            x = torch.randn(batch_size, 1, *in_shape)
            with torch.no_grad():
                out = model(x)
            assert out.shape[0] == batch_size, (
                f"{model_name}: Batch {batch_size} failed"
            )


# ==============================================================================
# MULTI-DIMENSIONALITY TESTS (1D/2D by default, 3D is slow)
# ==============================================================================


class TestMultiDimensionality:
    """Tests for models supporting multiple input dimensionalities."""

    @pytest.mark.parametrize(
        "model_name",
        [m for m in list_models() if 1 in get_supported_dims(m)],
    )
    def test_1d_input(self, model_name):
        """Test models that support 1D input."""
        in_shape, kwargs = get_test_config(model_name, dim=1)
        model = build_model(model_name, in_shape=in_shape, out_size=3, **kwargs)
        model.eval()

        x = torch.randn(1, 1, *in_shape)
        with torch.no_grad():
            out = model(x)

        assert out.shape[0] == 1, f"{model_name}: 1D forward fail"
        assert not torch.isnan(out).any(), f"{model_name}: 1D NaN"

    @pytest.mark.parametrize(
        "model_name",
        [m for m in list_models() if 2 in get_supported_dims(m)],
    )
    def test_2d_input(self, model_name):
        """Test models that support 2D input."""
        in_shape, kwargs = get_test_config(model_name, dim=2)
        model = build_model(model_name, in_shape=in_shape, out_size=3, **kwargs)
        model.eval()

        x = torch.randn(1, 1, *in_shape)
        with torch.no_grad():
            out = model(x)

        assert out.shape[0] == 1, f"{model_name}: 2D forward fail"
        assert not torch.isnan(out).any(), f"{model_name}: 2D NaN"

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "model_name",
        [m for m in list_models() if 3 in get_supported_dims(m)],
    )
    def test_3d_input(self, model_name):
        """Test models that support 3D input (slow - run with -m slow)."""
        in_shape, kwargs = get_test_config(model_name, dim=3)
        model = build_model(model_name, in_shape=in_shape, out_size=3, **kwargs)
        model.eval()

        x = torch.randn(1, 1, *in_shape)
        with torch.no_grad():
            out = model(x)

        assert out.shape[0] == 1, f"{model_name}: 3D forward fail"


# ==============================================================================
# BASEMODEL INTERFACE TESTS
# ==============================================================================


class TestBaseModelInterface:
    """Tests for BaseModel interface compliance."""

    @pytest.mark.parametrize("model_name", list_models()[:5])  # Just first 5
    def test_interface_methods(self, model_name):
        """Test BaseModel interface methods."""
        in_shape, kwargs = get_test_config(model_name)
        model = build_model(model_name, in_shape=in_shape, out_size=3, **kwargs)

        # count_parameters
        total = model.count_parameters(trainable_only=False)
        trainable = model.count_parameters(trainable_only=True)
        assert total > 0
        assert trainable <= total

        # parameter_summary
        summary = model.parameter_summary()
        assert "total_parameters" in summary
        assert "trainable_parameters" in summary
        assert (
            summary["total_parameters"]
            == summary["trainable_parameters"] + summary["frozen_parameters"]
        )


# ==============================================================================
# MODEL-SPECIFIC QUICK TESTS
# ==============================================================================


class TestCNNSpecific:
    """Quick CNN-specific tests."""

    def test_cnn_basic(self):
        model = CNN(in_shape=(256,), out_size=3)
        x = torch.randn(1, 1, 256)
        out = model(x)
        assert out.shape == (1, 3)


class TestPretrainedModels:
    """Comprehensive pretrained model tests - ALL pretrained models."""

    @pytest.mark.parametrize(
        "model_name", [m for m in list_models() if "pretrained" in m.lower()]
    )  # Test ALL pretrained models
    def test_pretrained_builds(self, model_name):
        """Test that pretrained models build and forward correctly."""
        in_shape, kwargs = get_test_config(model_name)
        model = build_model(model_name, in_shape=in_shape, out_size=3, **kwargs)
        assert model is not None

        x = torch.randn(1, 1, *in_shape)
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any()
        assert out.shape == (1, 3)

    @pytest.mark.parametrize(
        "model_name", [m for m in list_models() if "pretrained" in m.lower()]
    )
    def test_pretrained_single_channel_input(self, model_name):
        """Test that pretrained models properly handle single-channel input."""
        in_shape, kwargs = get_test_config(model_name)
        model = build_model(model_name, in_shape=in_shape, out_size=3, **kwargs)

        # Verify model expects 1 channel (not 3)
        x = torch.randn(1, 1, *in_shape)  # Single channel
        model.eval()
        with torch.no_grad():
            out = model(x)

        # Should not fail and output should be valid
        assert out.shape == (1, 3)
        assert not torch.isnan(out).any()


class TestFreezeBackbone:
    """Tests for freeze_backbone functionality."""

    @pytest.mark.parametrize(
        "model_name", [m for m in list_models() if "pretrained" in m.lower()][:3]
    )  # Sample 3 for speed
    def test_freeze_backbone(self, model_name):
        """Test that freeze_backbone properly freezes parameters."""
        in_shape, kwargs = get_test_config(model_name)
        kwargs["freeze_backbone"] = True
        kwargs["pretrained"] = False  # Speed up test

        model = build_model(model_name, in_shape=in_shape, out_size=3, **kwargs)

        # Count frozen vs trainable
        frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        trainable = sum(1 for p in model.parameters() if p.requires_grad)

        # Most params should be frozen (backbone), few trainable (head)
        assert frozen > 0, "No parameters frozen"
        assert trainable > 0, "No trainable parameters (head should be trainable)"
        assert frozen > trainable, "More trainable than frozen - freeze may have failed"


class TestModelRegistry:
    """Registry sanity checks."""

    def test_list_models(self):
        models = list_models()
        assert len(models) >= 71  # 69 base - 6 pruned + 8 added
        assert "cnn" in models
        assert "resnet18" in models
        # Check existing models
        assert "maxvit_tiny" in models
        assert "fastvit_t8" in models
        assert "caformer_s18" in models
        assert "convnext_v2_tiny" in models
        assert "mamba_1d" in models
        # v1.6.1 models
        assert "efficientvit_m1" in models
        assert "efficientvit_b1" in models
        assert "unireplknet_tiny" in models
        # v1.8 new models
        assert "efficientnet_b4" in models
        assert "efficientnet_b7" in models
        assert "wavenet" in models
        assert "wavenet_small" in models
        assert "wavenet_large" in models
        assert "s4d" in models
        assert "s4d_small" in models
        assert "s4d_large" in models
        # Removed models should not be present
        assert "efficientnet_b1" not in models
        assert "efficientvit_m0" not in models
        assert "efficientvit_b0" not in models
        assert "efficientvit_b3" not in models
        assert "efficientvit_l1" not in models

    def test_build_model(self):
        model = build_model("cnn", in_shape=(256,), out_size=3)
        assert model is not None


# ==============================================================================
# EDGE CASES (Reduced)
# ==============================================================================


class TestModelEdgeCases:
    """Edge case tests (reduced set for speed)."""

    def test_single_output_dimension(self):
        model = CNN(in_shape=(256,), out_size=1)
        x = torch.randn(1, 1, 256)
        out = model(x)
        assert out.shape == (1, 1)

    def test_large_output_dimension(self):
        model = CNN(in_shape=(256,), out_size=100)
        x = torch.randn(1, 1, 256)
        out = model(x)
        assert out.shape == (1, 100)

    def test_model_state_dict_save_load(self):
        model = CNN(in_shape=(256,), out_size=3)
        state = model.state_dict()
        model2 = CNN(in_shape=(256,), out_size=3)
        model2.load_state_dict(state)

        x = torch.randn(1, 1, 256)
        model.eval()
        model2.eval()
        with torch.no_grad():
            out1 = model(x)
            out2 = model2(x)
        assert torch.allclose(out1, out2)
