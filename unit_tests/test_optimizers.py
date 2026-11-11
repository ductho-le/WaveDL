"""
Unit Tests for Optimizers
=========================

Tests for the optimizer factory and parameter grouping utilities.

Author: Ductho Le (ductho.le@outlook.com)
"""

import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.optim as optim


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavedl.utils.optimizers import (
    get_optimizer,
    get_optimizer_with_param_groups,
    list_optimizers,
)


class SimpleModel(nn.Module):
    """Simple model for testing optimizers."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.bn = nn.BatchNorm1d(20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = torch.relu(self.bn(self.fc1(x)))
        return self.fc2(x)


class TestListOptimizers:
    """Tests for list_optimizers function."""

    def test_returns_list(self):
        """list_optimizers should return a list."""
        result = list_optimizers()
        assert isinstance(result, list)

    def test_contains_expected_optimizers(self):
        """list_optimizers should contain all expected optimizer names."""
        result = list_optimizers()
        expected = ["adamw", "adam", "sgd", "nadam", "radam", "rmsprop"]
        for opt_name in expected:
            assert opt_name in result


class TestGetOptimizer:
    """Tests for get_optimizer factory function."""

    @pytest.fixture
    def model(self):
        return SimpleModel()

    @pytest.mark.parametrize(
        "opt_name", ["adamw", "adam", "sgd", "nadam", "radam", "rmsprop"]
    )
    def test_all_optimizers_instantiate(self, model, opt_name):
        """All registered optimizers should instantiate without error."""
        optimizer = get_optimizer(opt_name, model.parameters(), lr=1e-3)
        assert isinstance(optimizer, optim.Optimizer)

    def test_adamw_default(self, model):
        """AdamW should be correctly configured."""
        optimizer = get_optimizer(
            "adamw", model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        assert isinstance(optimizer, optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == 1e-3
        assert optimizer.param_groups[0]["weight_decay"] == 1e-4

    def test_sgd_with_momentum(self, model):
        """SGD should accept momentum parameter."""
        optimizer = get_optimizer("sgd", model.parameters(), lr=1e-2, momentum=0.9)
        assert isinstance(optimizer, optim.SGD)
        assert optimizer.param_groups[0]["momentum"] == 0.9

    def test_sgd_with_nesterov(self, model):
        """SGD should accept nesterov parameter."""
        optimizer = get_optimizer(
            "sgd", model.parameters(), lr=1e-2, momentum=0.9, nesterov=True
        )
        assert optimizer.param_groups[0]["nesterov"] is True

    def test_adam_with_betas(self, model):
        """Adam variants should accept betas parameter."""
        betas = (0.85, 0.95)
        optimizer = get_optimizer("adam", model.parameters(), lr=1e-3, betas=betas)
        assert optimizer.param_groups[0]["betas"] == betas

    def test_unknown_optimizer_raises_error(self, model):
        """Unknown optimizer name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown optimizer"):
            get_optimizer("unknown_opt", model.parameters(), lr=1e-3)

    def test_case_insensitive(self, model):
        """Optimizer names should be case insensitive."""
        opt1 = get_optimizer("AdamW", model.parameters(), lr=1e-3)
        opt2 = get_optimizer("adamw", model.parameters(), lr=1e-3)
        assert type(opt1) == type(opt2)


class TestGetOptimizerWithParamGroups:
    """Tests for get_optimizer_with_param_groups utility."""

    @pytest.fixture
    def model(self):
        return SimpleModel()

    def test_creates_param_groups(self, model):
        """Should create separate decay and no-decay parameter groups."""
        optimizer = get_optimizer_with_param_groups(
            "adamw", model, lr=1e-3, weight_decay=1e-4
        )

        assert len(optimizer.param_groups) == 2
        assert {g["weight_decay"] for g in optimizer.param_groups} == {0.0, 1e-4}

    def test_no_decay_on_bias(self, model):
        """Bias and normalization parameters should have zero weight decay."""
        optimizer = get_optimizer_with_param_groups(
            "adamw", model, lr=1e-3, weight_decay=1e-4
        )

        groups_by_wd = {
            group["weight_decay"]: {id(p) for p in group["params"]}
            for group in optimizer.param_groups
        }
        assert 0.0 in groups_by_wd
        assert 1e-4 in groups_by_wd

        assert id(model.fc1.bias) in groups_by_wd[0.0]
        assert id(model.fc2.bias) in groups_by_wd[0.0]
        assert id(model.bn.weight) in groups_by_wd[0.0]
        assert id(model.bn.bias) in groups_by_wd[0.0]
        assert id(model.fc1.weight) in groups_by_wd[1e-4]
        assert id(model.fc2.weight) in groups_by_wd[1e-4]

    def test_norm_modules_without_norm_in_attribute_name(self):
        """Norm modules named e.g. 'downsample.1' or 'stage' must still skip decay."""

        class CustomNamedNormModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(8, 8)
                self.stage_scale = nn.GroupNorm(2, 8)
                self.post_block = nn.LayerNorm(8)
                self.pos_embed = nn.Parameter(torch.zeros(1, 4, 8))

        m = CustomNamedNormModel()
        optimizer = get_optimizer_with_param_groups(
            "adamw", m, lr=1e-3, weight_decay=1e-2
        )
        groups_by_wd = {
            group["weight_decay"]: {id(p) for p in group["params"]}
            for group in optimizer.param_groups
        }
        assert id(m.proj.weight) in groups_by_wd[1e-2]
        assert id(m.stage_scale.weight) in groups_by_wd[0.0]
        assert id(m.post_block.weight) in groups_by_wd[0.0]
        assert id(m.pos_embed) in groups_by_wd[0.0]

    def test_custom_no_decay_keywords(self, model):
        """Should accept custom no_decay_keywords."""
        optimizer = get_optimizer_with_param_groups(
            "adamw",
            model,
            lr=1e-3,
            weight_decay=1e-4,
            no_decay_keywords=["bias", "fc2"],
        )
        assert isinstance(optimizer, optim.Optimizer)


class TestOptimizerStep:
    """Tests for optimizer step functionality."""

    @pytest.fixture
    def model(self):
        return SimpleModel()

    @pytest.mark.parametrize(
        "opt_name", ["adamw", "adam", "sgd", "nadam", "radam", "rmsprop"]
    )
    def test_optimizer_step_updates_params(self, model, opt_name):
        """Optimizer step should update model parameters."""
        optimizer = get_optimizer(opt_name, model.parameters(), lr=0.1)

        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Forward and backward pass
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Check parameters changed
        params_changed = False
        for p_init, p_new in zip(initial_params, model.parameters()):
            if not torch.allclose(p_init, p_new):
                params_changed = True
                break

        assert params_changed, "Optimizer step should update parameters"
