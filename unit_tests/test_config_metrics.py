"""
Unit Tests for Configuration and Metrics Utilities
=====================================================

Tests for config.py and metrics.py utility functions to improve coverage.

**Tested Components**:
    - config.py: load_config, _flatten_config, validate_config, create_default_config,
                 merge_config_with_args, save_config
    - metrics.py: MetricTracker, calc_pearson, calc_per_target_r2, get_lr

Author: Ductho Le (ductho.le@outlook.com)
"""

import argparse
import os
import tempfile

import numpy as np
import pytest
import torch


# ==============================================================================
# CONFIG MODULE TESTS
# ==============================================================================


class TestFlattenConfig:
    """Tests for config flattening utility."""

    def test_flat_config_unchanged(self):
        """Test that already-flat configs pass through unchanged."""
        from wavedl.utils.config import _flatten_config

        config = {"lr": 0.001, "epochs": 100, "model": "cnn"}
        result = _flatten_config(config)

        assert result == config

    def test_nested_config_flattened(self):
        """Test that nested configs are flattened with underscores."""
        from wavedl.utils.config import _flatten_config

        config = {
            "optimizer": {"lr": 0.001, "weight_decay": 0.01},
            "model": "cnn",
        }
        result = _flatten_config(config)

        assert result["optimizer_lr"] == 0.001
        assert result["optimizer_weight_decay"] == 0.01
        assert result["model"] == "cnn"
        assert "optimizer" not in result

    def test_deeply_nested_config(self):
        """Test that deeply nested configs are fully flattened."""
        from wavedl.utils.config import _flatten_config

        config = {"a": {"b": {"c": 1}}}
        result = _flatten_config(config)

        assert result == {"a_b_c": 1}

    def test_empty_config(self):
        """Test that empty configs return empty dict."""
        from wavedl.utils.config import _flatten_config

        assert _flatten_config({}) == {}


class TestLoadConfig:
    """Tests for YAML config loading."""

    def test_load_valid_yaml(self):
        """Test loading a valid YAML config file."""
        from wavedl.utils.config import load_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("model: cnn\nlr: 0.001\nepochs: 100\n")
            f.flush()

            try:
                config = load_config(f.name)
                assert config["model"] == "cnn"
                assert config["lr"] == 0.001
                assert config["epochs"] == 100
            finally:
                try:
                    os.unlink(f.name)
                except PermissionError:
                    pass  # Windows file locking - cleanup later

    def test_load_empty_yaml(self):
        """Test loading an empty YAML file returns empty dict."""
        from wavedl.utils.config import load_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()

            try:
                config = load_config(f.name)
                assert config == {}
            finally:
                try:
                    os.unlink(f.name)
                except PermissionError:
                    pass  # Windows file locking - cleanup later

    def test_load_nested_yaml(self):
        """Test that nested YAML is flattened."""
        from wavedl.utils.config import load_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("optimizer:\n  lr: 0.001\n  weight_decay: 0.01\n")
            f.flush()

            try:
                config = load_config(f.name)
                assert config["optimizer_lr"] == 0.001
                assert config["optimizer_weight_decay"] == 0.01
            finally:
                try:
                    os.unlink(f.name)
                except PermissionError:
                    pass  # Windows file locking - cleanup later

    def test_load_missing_file_raises(self):
        """Test that loading missing file raises FileNotFoundError."""
        from wavedl.utils.config import load_config

        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")


class TestValidateConfig:
    """Tests for config validation."""

    def test_valid_config_no_warnings(self):
        """Test that valid config produces no warnings."""
        from wavedl.utils.config import validate_config

        config = {
            "model": "cnn",
            "loss": "mse",
            "optimizer": "adamw",
            "scheduler": "plateau",
            "lr": 0.001,
            "epochs": 100,
            "batch_size": 128,
        }
        warnings = validate_config(config)
        assert warnings == []

    def test_invalid_model_produces_warning(self):
        """Test that invalid model name produces warning."""
        from wavedl.utils.config import validate_config

        config = {"model": "nonexistent_model_xyz"}
        warnings = validate_config(config)
        assert len(warnings) == 1
        assert "Invalid model" in warnings[0]

    def test_invalid_optimizer_produces_warning(self):
        """Test that invalid optimizer name produces warning."""
        from wavedl.utils.config import validate_config

        config = {"optimizer": "invalid_opt"}
        warnings = validate_config(config)
        assert len(warnings) == 1
        assert "Invalid optimizer" in warnings[0]

    def test_out_of_range_lr_produces_warning(self):
        """Test that out-of-range LR produces warning."""
        from wavedl.utils.config import validate_config

        config = {"lr": 15.0}  # > 10 (new bound for OneCycleLR compatibility)
        warnings = validate_config(config)
        assert len(warnings) == 1
        assert "Learning rate" in warnings[0]

    def test_zero_epochs_produces_warning(self):
        """Test that zero epochs produces warning."""
        from wavedl.utils.config import validate_config

        config = {"epochs": 0}
        warnings = validate_config(config)
        assert len(warnings) == 1

    def test_grad_accum_steps_zero_produces_warning(self):
        """Test that grad_accum_steps=0 produces warning."""
        from wavedl.utils.config import validate_config

        config = {"grad_accum_steps": 0}
        warnings = validate_config(config)
        assert len(warnings) == 1
        assert "Gradient accumulation" in warnings[0]

    def test_grad_accum_steps_negative_produces_warning(self):
        """Test that negative grad_accum_steps produces warning."""
        from wavedl.utils.config import validate_config

        config = {"grad_accum_steps": -1}
        warnings = validate_config(config)
        assert len(warnings) == 1
        assert "Gradient accumulation" in warnings[0]

    def test_grad_accum_steps_valid_no_warning(self):
        """Test that valid grad_accum_steps produces no warning."""
        from wavedl.utils.config import validate_config

        config = {"grad_accum_steps": 4}
        warnings = validate_config(config)
        assert warnings == []

    def test_grad_accum_steps_above_max_produces_warning(self):
        """Test that grad_accum_steps above 256 produces warning."""
        from wavedl.utils.config import validate_config

        config = {"grad_accum_steps": 999}
        warnings = validate_config(config)
        assert len(warnings) == 1
        assert "Gradient accumulation" in warnings[0]

    @pytest.mark.parametrize("patience", [1, 1000, 100000, 10**9])
    def test_large_patience_is_valid(self, patience):
        """A large patience (the usual way to disable early stopping) is valid."""
        from wavedl.utils.config import validate_config

        assert validate_config({"patience": patience}) == []

    @pytest.mark.parametrize("patience", [0, -1, float("inf"), float("nan")])
    def test_non_positive_or_non_finite_patience_produces_warning(self, patience):
        """Patience must be a positive finite number."""
        from wavedl.utils.config import validate_config

        warnings = validate_config({"patience": patience})
        assert len(warnings) == 1
        assert "Patience must be positive" in warnings[0]

    def test_cli_patience_100000_passes_training_validation(self, monkeypatch):
        """`wavedl-train --patience 100000` must not be rejected as invalid."""
        from wavedl.train import parse_args
        from wavedl.utils.config import validate_config

        argv = ["wavedl-train", "--model", "cnn", "--data_path", "x.npz"]
        monkeypatch.setattr("sys.argv", [*argv, "--patience", "100000"])
        args, parser = parse_args()

        assert args.patience == 100000
        assert validate_config(vars(args), parser=parser) == []


class TestCreateDefaultConfig:
    """Tests for default config creation."""

    def test_returns_dict(self):
        """Test that create_default_config returns a dict."""
        from wavedl.utils.config import create_default_config

        config = create_default_config()
        assert isinstance(config, dict)

    def test_contains_required_keys(self):
        """Test that default config contains all required keys."""
        from wavedl.utils.config import create_default_config

        config = create_default_config()
        required = [
            "model",
            "batch_size",
            "lr",
            "epochs",
            "loss",
            "optimizer",
            "scheduler",
        ]
        for key in required:
            assert key in config

    def test_default_values_are_valid(self):
        """Test that default values pass validation."""
        from wavedl.utils.config import create_default_config, validate_config

        config = create_default_config()
        warnings = validate_config(config)
        assert warnings == []


class TestSaveConfig:
    """Tests for config saving."""

    def test_save_and_reload(self):
        """Test that saved config can be reloaded."""
        from wavedl.utils.config import load_config, save_config

        args = argparse.Namespace(model="cnn", lr=0.001, epochs=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "config.yaml")
            save_config(args, output_path)

            # Reload and verify
            config = load_config(output_path)
            assert config["model"] == "cnn"
            assert config["lr"] == 0.001
            assert config["epochs"] == 100

    def test_excludes_specified_keys(self):
        """Test that excluded keys are not saved."""
        from wavedl.utils.config import load_config, save_config

        args = argparse.Namespace(model="cnn", secret_key="password", lr=0.001)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "config.yaml")
            save_config(args, output_path, exclude_keys=["secret_key"])

            config = load_config(output_path)
            assert "secret_key" not in config
            assert config["model"] == "cnn"


class TestMergeConfigWithArgs:
    """Tests for config merging with CLI args."""

    def test_config_values_applied(self):
        """Test that config values are applied to args."""
        from wavedl.utils.config import merge_config_with_args

        config = {"model": "resnet18", "lr": 0.01}
        args = argparse.Namespace(model="cnn", lr=0.001, epochs=100)

        merged = merge_config_with_args(config, args)
        assert merged.model == "resnet18"
        assert merged.lr == 0.01
        assert merged.epochs == 100  # unchanged

    def test_unknown_keys_ignored(self):
        """Test that unknown config keys are ignored."""
        from wavedl.utils.config import merge_config_with_args

        config = {"unknown_key": "value", "model": "cnn"}
        args = argparse.Namespace(model="resnet18")

        merged = merge_config_with_args(config, args, ignore_unknown=True)
        assert merged.model == "cnn"
        assert not hasattr(merged, "unknown_key")


class TestGradAccumStepsYAMLIntegration:
    """Integration tests for grad_accum_steps loaded from YAML config."""

    def test_yaml_grad_accum_steps_applied(self):
        """Test that grad_accum_steps from YAML is applied to args."""
        from wavedl.utils.config import load_config, merge_config_with_args

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("model: cnn\ngrad_accum_steps: 4\n")
            f.flush()

            try:
                config = load_config(f.name)
                args = argparse.Namespace(model="cnn", grad_accum_steps=1)
                merged = merge_config_with_args(config, args)
                assert merged.grad_accum_steps == 4
            finally:
                try:
                    os.unlink(f.name)
                except PermissionError:
                    pass

    def test_yaml_grad_accum_steps_validated(self):
        """Test that invalid grad_accum_steps in YAML produces warning."""
        from wavedl.utils.config import load_config, validate_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("model: cnn\ngrad_accum_steps: 0\n")
            f.flush()

            try:
                config = load_config(f.name)
                warnings = validate_config(config)
                assert any("Gradient accumulation" in w for w in warnings)
            finally:
                try:
                    os.unlink(f.name)
                except PermissionError:
                    pass

    def test_yaml_grad_accum_steps_no_spurious_unknown_key_warning(self):
        """Test that grad_accum_steps in YAML does not produce unknown key warning."""
        from wavedl.utils.config import load_config, validate_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("model: cnn\ngrad_accum_steps: 4\n")
            f.flush()

            try:
                config = load_config(f.name)
                warnings = validate_config(config)
                unknown_warnings = [w for w in warnings if "Unknown config key" in w]
                assert unknown_warnings == []
            finally:
                try:
                    os.unlink(f.name)
                except PermissionError:
                    pass


# ==============================================================================
# METRICS MODULE TESTS
# ==============================================================================


class TestMetricTracker:
    """Tests for MetricTracker utility class."""

    def test_initial_state(self):
        """Test tracker starts with zero values."""
        from wavedl.utils.metrics import MetricTracker

        tracker = MetricTracker()
        assert tracker.avg == 0.0
        assert tracker.sum == 0.0
        assert tracker.count == 0

    def test_single_update(self):
        """Test single value update."""
        from wavedl.utils.metrics import MetricTracker

        tracker = MetricTracker()
        tracker.update(5.0)

        assert tracker.avg == 5.0
        assert tracker.sum == 5.0
        assert tracker.count == 1

    def test_multiple_updates(self):
        """Test averaging across multiple updates."""
        from wavedl.utils.metrics import MetricTracker

        tracker = MetricTracker()
        tracker.update(2.0)
        tracker.update(4.0)
        tracker.update(6.0)

        assert tracker.avg == 4.0
        assert tracker.sum == 12.0
        assert tracker.count == 3

    def test_weighted_update(self):
        """Test update with sample count (n)."""
        from wavedl.utils.metrics import MetricTracker

        tracker = MetricTracker()
        tracker.update(5.0, n=10)  # mean of 5.0 for 10 samples

        assert tracker.avg == 5.0
        assert tracker.sum == 50.0
        assert tracker.count == 10

    def test_reset(self):
        """Test reset clears all values."""
        from wavedl.utils.metrics import MetricTracker

        tracker = MetricTracker()
        tracker.update(10.0)
        tracker.reset()

        assert tracker.avg == 0.0
        assert tracker.sum == 0.0
        assert tracker.count == 0

    def test_repr(self):
        """Test string representation."""
        from wavedl.utils.metrics import MetricTracker

        tracker = MetricTracker()
        tracker.update(5.0)

        repr_str = repr(tracker)
        assert "avg" in repr_str.lower() or "5.0" in repr_str


class TestCalcPearson:
    """Tests for Pearson correlation calculation."""

    def test_perfect_correlation(self):
        """Test perfect positive correlation."""
        from wavedl.utils.metrics import calc_pearson

        y_true = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y_pred = y_true.copy()

        corr = calc_pearson(y_true, y_pred)
        assert corr == pytest.approx(1.0, abs=1e-6)

    def test_negative_correlation(self):
        """Test perfect negative correlation."""
        from wavedl.utils.metrics import calc_pearson

        y_true = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y_pred = np.array([[5.0], [4.0], [3.0], [2.0], [1.0]])

        corr = calc_pearson(y_true, y_pred)
        assert corr == pytest.approx(-1.0, abs=1e-6)

    def test_multi_target(self):
        """Test with multiple targets."""
        from wavedl.utils.metrics import calc_pearson

        y_true = np.array([[1.0, 5.0], [2.0, 4.0], [3.0, 3.0], [4.0, 2.0], [5.0, 1.0]])
        y_pred = y_true.copy()

        corr = calc_pearson(y_true, y_pred)
        assert corr == pytest.approx(1.0, abs=1e-6)


class TestCalcPerTargetR2:
    """Tests for per-target R² calculation."""

    def test_perfect_predictions(self):
        """Test R²=1 for perfect predictions."""
        from wavedl.utils.metrics import calc_per_target_r2

        y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_pred = y_true.copy()

        r2_per_target = calc_per_target_r2(y_true, y_pred)
        assert len(r2_per_target) == 2
        assert all(r2 == pytest.approx(1.0, abs=1e-6) for r2 in r2_per_target)

    def test_returns_array(self):
        """Test that return is numpy array."""
        from wavedl.utils.metrics import calc_per_target_r2

        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.1], [2.1], [3.1]])

        r2 = calc_per_target_r2(y_true, y_pred)
        assert isinstance(r2, np.ndarray)


class TestGetLR:
    """Tests for learning rate extraction from optimizer."""

    def test_extract_lr(self):
        """Test extracting LR from optimizer."""
        from wavedl.utils.metrics import get_lr

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        lr = get_lr(optimizer)
        assert lr == pytest.approx(0.01)

    def test_after_scheduler_step(self):
        """Test LR extraction after scheduler step."""
        from wavedl.utils.metrics import get_lr

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        scheduler.step()
        lr = get_lr(optimizer)
        assert lr == pytest.approx(0.05)
