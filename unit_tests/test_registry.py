"""
Unit Tests for Model Registry
==============================

Comprehensive tests for:
- register_model: decorator functionality
- get_model: model retrieval
- list_models: listing available models
- build_model: model instantiation factory

Author: Ductho Le (ductho.le@outlook.com)
"""

import pytest
import torch
import torch.nn as nn

from wavedl.models.base import BaseModel
from wavedl.models.registry import (
    MODEL_REGISTRY,
    build_model,
    get_model,
    list_models,
    register_model,
)


# ==============================================================================
# TEST FIXTURES
# ==============================================================================
@pytest.fixture(autouse=True)
def clean_registry():
    """Store and restore registry state around each test."""
    # Store original registry state
    original_registry = MODEL_REGISTRY.copy()

    yield

    # Restore original registry (remove any test-added models)
    MODEL_REGISTRY.clear()
    MODEL_REGISTRY.update(original_registry)


# ==============================================================================
# REGISTER MODEL TESTS
# ==============================================================================
class TestRegisterModel:
    """Tests for the @register_model decorator."""

    def test_basic_registration(self):
        """Test that a model can be registered with a name."""

        @register_model("test_model_basic")
        class TestModel(nn.Module):
            def __init__(self, in_shape, out_size, **kwargs):
                super().__init__()
                self.fc = nn.Linear(10, out_size)

            def forward(self, x):
                return self.fc(x)

        assert "test_model_basic" in MODEL_REGISTRY
        assert MODEL_REGISTRY["test_model_basic"] == TestModel

    def test_case_insensitive_registration(self):
        """Test that model names are stored lowercase."""

        @register_model("TestModelCaps")
        class TestModelCaps(nn.Module):
            def __init__(self, in_shape, out_size, **kwargs):
                super().__init__()

            def forward(self, x):
                return x

        assert "testmodelcaps" in MODEL_REGISTRY
        assert "TestModelCaps" not in MODEL_REGISTRY

    def test_duplicate_registration_raises(self):
        """Test that registering duplicate name raises ValueError."""

        @register_model("duplicate_test")
        class FirstModel(nn.Module):
            def __init__(self, in_shape, out_size, **kwargs):
                super().__init__()

            def forward(self, x):
                return x

        with pytest.raises(ValueError, match="already registered"):

            @register_model("duplicate_test")
            class SecondModel(nn.Module):
                def __init__(self, in_shape, out_size, **kwargs):
                    super().__init__()

                def forward(self, x):
                    return x

    def test_decorator_returns_class(self):
        """Test that decorator returns the original class unchanged."""

        @register_model("decorator_return_test")
        class OriginalClass(nn.Module):
            class_attr = "test_value"

            def __init__(self, in_shape, out_size, **kwargs):
                super().__init__()

            def forward(self, x):
                return x

        assert OriginalClass.class_attr == "test_value"
        assert OriginalClass.__name__ == "OriginalClass"


# ==============================================================================
# GET MODEL TESTS
# ==============================================================================
class TestGetModel:
    """Tests for the get_model function."""

    def test_get_registered_model(self):
        """Test retrieving a registered model."""

        @register_model("get_test_model")
        class GetTestModel(nn.Module):
            def __init__(self, in_shape, out_size, **kwargs):
                super().__init__()

            def forward(self, x):
                return x

        ModelClass = get_model("get_test_model")
        assert ModelClass == GetTestModel

    def test_case_insensitive_retrieval(self):
        """Test that retrieval is case-insensitive."""

        @register_model("case_test")
        class CaseTestModel(nn.Module):
            def __init__(self, in_shape, out_size, **kwargs):
                super().__init__()

            def forward(self, x):
                return x

        # All these should work
        assert get_model("case_test") == CaseTestModel
        assert get_model("CASE_TEST") == CaseTestModel
        assert get_model("Case_Test") == CaseTestModel

    def test_unregistered_model_raises(self):
        """Test that getting unregistered model raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            get_model("nonexistent_model_xyz")

    def test_error_message_lists_available(self):
        """Test that error message lists available models."""

        @register_model("available_model_1")
        class AvailableModel1(nn.Module):
            def __init__(self, in_shape, out_size, **kwargs):
                super().__init__()

            def forward(self, x):
                return x

        with pytest.raises(ValueError) as excinfo:
            get_model("nonexistent")

        # Error message should mention available models
        assert "available_model_1" in str(excinfo.value).lower() or "Available" in str(
            excinfo.value
        )


# ==============================================================================
# LIST MODELS TESTS
# ==============================================================================
class TestListModels:
    """Tests for the list_models function."""

    def test_returns_list(self):
        """Test that list_models returns a list."""
        result = list_models()
        assert isinstance(result, list)

    def test_includes_registered_models(self):
        """Test that registered models appear in list."""

        @register_model("list_test_model")
        class ListTestModel(nn.Module):
            def __init__(self, in_shape, out_size, **kwargs):
                super().__init__()

            def forward(self, x):
                return x

        models = list_models()
        assert "list_test_model" in models

    def test_returns_sorted_list(self):
        """Test that list is sorted alphabetically."""

        @register_model("zzz_model")
        class ZzzModel(nn.Module):
            def __init__(self, in_shape, out_size, **kwargs):
                super().__init__()

            def forward(self, x):
                return x

        @register_model("aaa_model")
        class AaaModel(nn.Module):
            def __init__(self, in_shape, out_size, **kwargs):
                super().__init__()

            def forward(self, x):
                return x

        models = list_models()

        assert "aaa_model" in models
        assert "zzz_model" in models
        assert models.index("aaa_model") < models.index("zzz_model")

    def test_idempotent_same_class_and_failed_import_rollback(self, tmp_path):
        """Re-registering the exact same class is idempotent; failed imports roll back."""
        from wavedl.models.registry import import_model_modules

        @register_model("idempotent_model_test")
        class IdempotentModel(nn.Module):
            pass

        # Re-registering the exact same class object is a safe no-op
        assert (
            register_model("idempotent_model_test")(IdempotentModel) is IdempotentModel
        )

        broken_file = tmp_path / "broken_custom_model.py"
        broken_file.write_text(
            "from wavedl.models import register_model\n"
            "@register_model('transient_broken_model')\n"
            "class TransientModel:\n"
            "    pass\n"
            "raise RuntimeError('boom during import')\n",
            encoding="utf-8",
        )
        with pytest.raises(RuntimeError, match="boom during import"):
            import_model_modules([str(broken_file)])

        assert "transient_broken_model" not in MODEL_REGISTRY


# ==============================================================================
# BUILD MODEL TESTS
# ==============================================================================
class TestBuildModel:
    """Tests for the build_model factory function."""

    def test_build_creates_instance(self):
        """Test that build_model creates a model instance."""

        @register_model("build_test")
        class BuildTestModel(BaseModel):
            def __init__(self, in_shape, out_size, **kwargs):
                super().__init__(in_shape, out_size)
                self.fc = nn.Linear(in_shape[0] * in_shape[1], out_size)

            def forward(self, x):
                return self.fc(x.flatten(1))

        model = build_model("build_test", in_shape=(64, 64), out_size=5)

        assert isinstance(model, BuildTestModel)
        assert model.in_shape == (64, 64)
        assert model.out_size == 5

    def test_build_with_1d_shape(self):
        """Test building model with 1D input shape."""

        @register_model("build_1d_test")
        class Build1DModel(BaseModel):
            def __init__(self, in_shape, out_size, **kwargs):
                super().__init__(in_shape, out_size)
                self.fc = nn.Linear(in_shape[0], out_size)

            def forward(self, x):
                return self.fc(x.flatten(1))

        model = build_model("build_1d_test", in_shape=(256,), out_size=3)

        assert model.in_shape == (256,)
        assert model.out_size == 3

    def test_build_with_3d_shape(self):
        """Test building model with 3D input shape."""

        @register_model("build_3d_test")
        class Build3DModel(BaseModel):
            def __init__(self, in_shape, out_size, **kwargs):
                super().__init__(in_shape, out_size)

            def forward(self, x):
                return torch.zeros(x.size(0), self.out_size)

        model = build_model("build_3d_test", in_shape=(16, 32, 32), out_size=4)

        assert model.in_shape == (16, 32, 32)

    def test_build_passes_kwargs(self):
        """Test that kwargs are passed to model constructor."""

        @register_model("build_kwargs_test")
        class BuildKwargsModel(BaseModel):
            def __init__(self, in_shape, out_size, custom_param=10, **kwargs):
                super().__init__(in_shape, out_size)
                self.custom_param = custom_param

            def forward(self, x):
                return torch.zeros(x.size(0), self.out_size)

        model = build_model(
            "build_kwargs_test", in_shape=(64, 64), out_size=5, custom_param=42
        )

        assert model.custom_param == 42

    def test_build_unregistered_raises(self):
        """Test that building unregistered model raises ValueError."""
        with pytest.raises(ValueError):
            build_model("totally_fake_model", in_shape=(64, 64), out_size=5)


# ==============================================================================
# INTEGRATION WITH ACTUAL MODELS
# ==============================================================================
class TestRegistryIntegration:
    """Integration tests with actual WaveDL models."""

    def test_cnn_is_registered(self):
        """Test that CNN is properly registered."""
        # Import to trigger registration

        models = list_models()
        assert "cnn" in models

    def test_build_cnn(self):
        """Test building CNN through registry."""
        from wavedl.models.cnn import CNN

        model = build_model("cnn", in_shape=(64, 64), out_size=5)

        assert isinstance(model, CNN)
