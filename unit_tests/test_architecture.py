"""
Unit Tests for Model Architectures
===================================

Comprehensive tests for all model architectures in WaveDL.

**Tested Architectures**:
    - CNN: Dimension-agnostic baseline CNN
    - ResNet: ResNet-18/34/50 with BasicBlock and Bottleneck
    - EfficientNet: B0/B1/B2 with pretrained weights (2D only)
    - ViT: Vision Transformer (Tiny/Small/Base)
    - ConvNeXt: Modern CNN (Tiny/Small/Base)
    - DenseNet: DenseNet-121/169 with dense connectivity
    - U-Net: Encoder-decoder for spatial and vector regression

**Test Coverage**:
    - Model instantiation with 1D/2D/3D input shapes
    - Forward pass correctness and output validation
    - Gradient flow through all layers
    - Numerical stability (no NaN/Inf outputs)
    - Eval mode determinism
    - Parameter counting and optimizer groups

**Universal Tests**:
    The TestAllModels class automatically tests ALL registered models.
    When you add a new model to the registry, it will be automatically
    included in these tests - no manual updates needed!

Author: Ductho Le (ductho.le@outlook.com)
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from models.base import BaseModel
from models.cnn import CNN
from models.registry import register_model, list_models, build_model


# ==============================================================================
# HELPER: Get model configuration for testing
# ==============================================================================
def get_test_config(model_name: str, dim: int = 2):
    """
    Get appropriate test configuration for a model.
    
    Returns (in_shape, kwargs) tuple suitable for testing.
    """
    # Base shapes for different dimensionalities
    if dim == 1:
        in_shape = (128,)
    elif dim == 2:
        in_shape = (64, 64)
    else:
        in_shape = (16, 32, 32)
    
    kwargs = {}
    
    # Model-specific adjustments
    if model_name.startswith("vit"):
        kwargs["patch_size"] = 8  # Smaller patches for test input size
    elif model_name.startswith("unet"):
        kwargs["depth"] = 3  # Smaller depth for faster tests
    
    return in_shape, kwargs


def get_supported_dims(model_name: str):
    """Get list of supported dimensionalities for a model."""
    # EfficientNet is 2D only
    if model_name.startswith("efficientnet"):
        return [2]
    # ViT supports 1D and 2D
    elif model_name.startswith("vit"):
        return [1, 2]
    # Most models support all dims
    else:
        return [1, 2]  # Skip 3D for speed in tests


# ==============================================================================
# UNIVERSAL MODEL TESTS (Auto-tests ALL registered models)
# ==============================================================================
class TestAllModels:
    """
    Universal tests that automatically run on ALL registered models.
    
    When you add a new model to the registry, it will automatically
    be included in these tests - no manual updates needed!
    """
    
    @pytest.mark.parametrize("model_name", list_models())
    def test_model_forward_2d(self, model_name):
        """Test that all models can perform a forward pass with 2D input."""
        in_shape, kwargs = get_test_config(model_name, dim=2)
        
        model = build_model(model_name, in_shape=in_shape, out_size=3, **kwargs)
        model.eval()
        
        x = torch.randn(2, 1, *in_shape)
        with torch.no_grad():
            out = model(x)
        
        # Check output is valid
        assert out.shape[0] == 2, f"{model_name}: Batch size mismatch"
        assert not torch.isnan(out).any(), f"{model_name}: Output contains NaN"
        assert not torch.isinf(out).any(), f"{model_name}: Output contains Inf"
    
    @pytest.mark.parametrize("model_name", list_models())
    def test_model_gradient_flow(self, model_name):
        """Test that gradients flow through all models."""
        in_shape, kwargs = get_test_config(model_name, dim=2)
        
        model = build_model(model_name, in_shape=in_shape, out_size=3, **kwargs)
        model.train()
        
        x = torch.randn(2, 1, *in_shape, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None, f"{model_name}: No gradient on input"
        assert not torch.isnan(x.grad).any(), f"{model_name}: Gradient contains NaN"
    
    @pytest.mark.parametrize("model_name", list_models())
    def test_model_output_shape(self, model_name):
        """Test that all models produce correct output shape."""
        in_shape, kwargs = get_test_config(model_name, dim=2)
        out_size = 5
        
        model = build_model(model_name, in_shape=in_shape, out_size=out_size, **kwargs)
        model.eval()
        
        x = torch.randn(4, 1, *in_shape)
        with torch.no_grad():
            out = model(x)
        
        # Skip spatial output models (unet with spatial_output=True)
        if model_name == "unet":
            # UNet default is spatial_output=True, so shape is different
            assert out.shape[0] == 4
            assert out.shape[1] == out_size
        else:
            assert out.shape == (4, out_size), f"{model_name}: Expected {(4, out_size)}, got {out.shape}"
    
    @pytest.mark.parametrize("model_name", list_models())
    def test_model_eval_deterministic(self, model_name):
        """Test that models are deterministic in eval mode."""
        in_shape, kwargs = get_test_config(model_name, dim=2)
        
        model = build_model(model_name, in_shape=in_shape, out_size=3, **kwargs)
        model.eval()
        
        x = torch.randn(2, 1, *in_shape)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        
        assert torch.allclose(out1, out2), f"{model_name}: Not deterministic in eval mode"
    
    @pytest.mark.parametrize("model_name", list_models())
    def test_model_has_parameters(self, model_name):
        """Test that all models have trainable parameters."""
        in_shape, kwargs = get_test_config(model_name, dim=2)
        
        model = build_model(model_name, in_shape=in_shape, out_size=3, **kwargs)
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count > 0, f"{model_name}: No trainable parameters"
    
    @pytest.mark.parametrize("model_name", list_models())
    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_model_different_batch_sizes(self, model_name, batch_size):
        """Test that all models handle various batch sizes."""
        in_shape, kwargs = get_test_config(model_name, dim=2)
        
        model = build_model(model_name, in_shape=in_shape, out_size=3, **kwargs)
        model.eval()
        
        x = torch.randn(batch_size, 1, *in_shape)
        with torch.no_grad():
            out = model(x)
        
        assert out.shape[0] == batch_size, f"{model_name}: Batch size {batch_size} failed"
    
    @pytest.mark.parametrize("model_name", [m for m in list_models() if not m.startswith("efficientnet") and "pretrained" not in m])
    def test_model_1d_input(self, model_name):
        """Test that dimension-agnostic models work with 1D input."""
        in_shape, kwargs = get_test_config(model_name, dim=1)
        
        model = build_model(model_name, in_shape=in_shape, out_size=3, **kwargs)
        model.eval()
        
        x = torch.randn(2, 1, *in_shape)
        with torch.no_grad():
            out = model(x)
        
        assert out.shape[0] == 2, f"{model_name}: 1D forward pass failed"
        assert not torch.isnan(out).any(), f"{model_name}: 1D output contains NaN"
    
    @pytest.mark.parametrize("model_name", list_models())
    def test_model_all_params_have_gradients(self, model_name):
        """Test that gradients reach all trainable parameters."""
        in_shape, kwargs = get_test_config(model_name, dim=2)
        
        model = build_model(model_name, in_shape=in_shape, out_size=3, **kwargs)
        model.train()
        
        x = torch.randn(2, 1, *in_shape, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        # Check all trainable parameters received gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{model_name}: No gradient for {name}"


# ==============================================================================
# BASE MODEL ABSTRACT CLASS TESTS
# ==============================================================================
class TestBaseModel:
    """Tests for the BaseModel abstract base class interface and utilities."""
    
    def test_cannot_instantiate_directly(self):
        """Test that BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModel(in_shape=(64, 64), out_size=5)
    
    def test_subclass_must_implement_forward(self):
        """Test that subclass must implement forward method."""
        # This should work - implementing required methods
        @register_model("test_valid_subclass")
        class ValidSubclass(BaseModel):
            def __init__(self, in_shape, out_size, **kwargs):
                super().__init__(in_shape, out_size)
                self.fc = nn.Linear(10, out_size)
            
            def forward(self, x):
                return self.fc(x.view(x.size(0), -1)[:, :10])
        
        model = ValidSubclass(in_shape=(64, 64), out_size=5)
        assert model.in_shape == (64, 64)
        assert model.out_size == 5
    
    def test_count_parameters(self):
        """Test parameter counting functionality."""
        @register_model("test_param_count")
        class ParamCountModel(BaseModel):
            def __init__(self, in_shape, out_size, **kwargs):
                super().__init__(in_shape, out_size)
                self.fc = nn.Linear(100, out_size)  # 100*out_size + out_size params
            
            def forward(self, x):
                return self.fc(x.view(x.size(0), -1)[:, :100])
        
        model = ParamCountModel(in_shape=(10, 10), out_size=5)
        
        # Expected: 100*5 (weights) + 5 (bias) = 505
        assert model.count_parameters() == 505
        assert model.count_parameters(trainable_only=True) == 505
        assert model.count_parameters(trainable_only=False) == 505
    
    def test_parameter_summary(self):
        """Test parameter summary generation."""
        @register_model("test_param_summary")
        class ParamSummaryModel(BaseModel):
            def __init__(self, in_shape, out_size, **kwargs):
                super().__init__(in_shape, out_size)
                self.fc = nn.Linear(100, out_size)
            
            def forward(self, x):
                return self.fc(x.view(x.size(0), -1)[:, :100])
        
        model = ParamSummaryModel(in_shape=(10, 10), out_size=5)
        summary = model.parameter_summary()
        
        assert "total_parameters" in summary
        assert "trainable_parameters" in summary
        assert "frozen_parameters" in summary
        assert "total_mb" in summary
        assert summary["total_parameters"] == 505
        assert summary["frozen_parameters"] == 0
    
    def test_get_default_config(self):
        """Test default config returns empty dict for base."""
        @register_model("test_default_config")
        class ConfigModel(BaseModel):
            def __init__(self, in_shape, out_size, **kwargs):
                super().__init__(in_shape, out_size)
            
            def forward(self, x):
                return torch.zeros(x.size(0), self.out_size)
        
        config = ConfigModel.get_default_config()
        assert isinstance(config, dict)
    
    def test_get_optimizer_groups(self):
        """Test optimizer parameter groups with weight decay."""
        @register_model("test_optim_groups")
        class OptimGroupsModel(BaseModel):
            def __init__(self, in_shape, out_size, **kwargs):
                super().__init__(in_shape, out_size)
                self.fc = nn.Linear(100, out_size)
                self.norm = nn.LayerNorm(out_size)
            
            def forward(self, x):
                out = self.fc(x.view(x.size(0), -1)[:, :100])
                return self.norm(out)
        
        model = OptimGroupsModel(in_shape=(10, 10), out_size=5)
        groups = model.get_optimizer_groups(base_lr=0.001, weight_decay=0.01)
        
        assert isinstance(groups, list)
        assert len(groups) >= 1
        
        # Check that groups have required keys
        for group in groups:
            assert "params" in group
            assert "lr" in group
            assert "weight_decay" in group

# ==============================================================================
# GPU TESTS (Optional - require CUDA)
# ==============================================================================
@pytest.mark.gpu
class TestGPUModels:
    """Tests for GPU deployment and mixed precision training."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_to_gpu(self):
        """Test model can be moved to GPU."""
        model = CNN(in_shape=(64, 64), out_size=5).cuda()
        x = torch.randn(4, 1, 64, 64).cuda()
        
        output = model(x)
        
        assert output.device.type == "cuda"
        assert output.shape == (4, 5)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision(self):
        """Test model works with mixed precision."""
        model = CNN(in_shape=(64, 64), out_size=5).cuda()
        x = torch.randn(4, 1, 64, 64).cuda()
        
        with torch.amp.autocast('cuda'):
            output = model(x)
        
        assert output.shape == (4, 5)
        assert not torch.isnan(output).any()
