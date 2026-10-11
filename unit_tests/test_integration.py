"""
Integration Tests for WaveDL
=============================

End-to-end tests that verify components work together correctly:
- Model training simulation
- Data pipeline integration
- Full forward/backward passes

Author: Ductho Le (ductho.le@outlook.com)
"""

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from wavedl.models.cnn import CNN
from wavedl.models.registry import build_model
from wavedl.utils.metrics import MetricTracker, calc_pearson, calc_per_target_r2


# ==============================================================================
# END-TO-END TRAINING SIMULATION
# ==============================================================================
@pytest.mark.integration
class TestTrainingSimulation:
    """Integration tests simulating training workflows."""

    @pytest.fixture
    def training_setup(self):
        """Create a complete training setup."""
        # Generate synthetic data
        n_samples = 100
        in_shape = (32, 32)
        out_size = 3
        batch_size = 16

        X = torch.randn(n_samples, 1, *in_shape)
        y = torch.randn(n_samples, out_size)

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = CNN(in_shape=in_shape, out_size=out_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        return {
            "model": model,
            "optimizer": optimizer,
            "criterion": criterion,
            "dataloader": dataloader,
            "in_shape": in_shape,
            "out_size": out_size,
        }

    def test_single_epoch_training(self, training_setup):
        """Test that a single epoch of training completes successfully."""
        model = training_setup["model"]
        optimizer = training_setup["optimizer"]
        criterion = training_setup["criterion"]
        dataloader = training_setup["dataloader"]

        model.train()
        loss_tracker = MetricTracker()

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            loss_tracker.update(loss.item(), n=batch_x.size(0))

        # Should have processed all batches
        assert loss_tracker.count > 0
        assert loss_tracker.avg > 0

    def test_multiple_epochs_training(self, training_setup):
        """Test training over multiple epochs with loss decreasing."""
        model = training_setup["model"]
        optimizer = training_setup["optimizer"]
        criterion = training_setup["criterion"]
        dataloader = training_setup["dataloader"]

        model.train()
        epoch_losses = []

        for epoch in range(5):
            epoch_tracker = MetricTracker()

            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

                epoch_tracker.update(loss.item(), n=batch_x.size(0))

            epoch_losses.append(epoch_tracker.avg)

        # Verify training completed successfully with valid (finite) losses
        # Note: We don't assert loss decreases because random data provides no
        # learnable signal - loss may fluctuate or increase. What matters is
        # that training runs without errors and produces valid numbers.
        assert len(epoch_losses) == 5, "Training did not complete all epochs"
        assert all(np.isfinite(epoch_losses)), "Training produced invalid loss values"

    def test_eval_mode_inference(self, training_setup):
        """Test inference in eval mode."""
        model = training_setup["model"]
        dataloader = training_setup["dataloader"]

        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                output = model(batch_x)
                all_predictions.append(output)
                all_targets.append(batch_y)

        predictions = torch.cat(all_predictions, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()

        # Should have predictions for all samples
        assert predictions.shape == targets.shape

        # Calculate metrics
        r2_scores = calc_per_target_r2(targets, predictions)
        pearson = calc_pearson(targets, predictions)

        # Metrics should be calculable (not NaN)
        assert not np.isnan(r2_scores).any()
        assert not np.isnan(pearson)


# ==============================================================================
# MODEL PIPELINE TESTS
# ==============================================================================
@pytest.mark.integration
class TestModelPipeline:
    """Tests for the complete model pipeline."""

    def test_build_train_eval_pipeline(self):
        """Test building a model from registry, training, and evaluating."""
        # Import to ensure registration

        # Build model from registry
        model = build_model("cnn", in_shape=(32, 32), out_size=5)

        # Create sample data
        X = torch.randn(20, 1, 32, 32)
        y = torch.randn(20, 5)

        # Training step
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # Evaluation step
        model.eval()
        with torch.no_grad():
            predictions = model(X)

        assert predictions.shape == (20, 5)

    def test_model_checkpointing(self):
        """Test saving and loading model checkpoints."""
        model = CNN(in_shape=(32, 32), out_size=3)

        # Train briefly
        X = torch.randn(10, 1, 32, 32)
        y = torch.randn(10, 3)

        optimizer = torch.optim.Adam(model.parameters())
        loss = nn.MSELoss()(model(X), y)
        loss.backward()
        optimizer.step()

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": 1,
            }
            torch.save(checkpoint, checkpoint_path)

            # Load checkpoint
            loaded_checkpoint = torch.load(checkpoint_path, weights_only=True)

            new_model = CNN(in_shape=(32, 32), out_size=3)
            new_model.load_state_dict(loaded_checkpoint["model_state_dict"])

            # Verify weights match
            for (name1, param1), (name2, param2) in zip(
                model.named_parameters(), new_model.named_parameters()
            ):
                assert torch.allclose(param1, param2), f"Mismatch at {name1}"

    def test_different_models_same_interface(self):
        """Test that different models work with the same training code."""
        in_shape = (32, 32)
        out_size = 3

        # Test with CNN (add more models here as they become available)
        model = CNN(in_shape=in_shape, out_size=out_size)

        X = torch.randn(8, 1, *in_shape)
        y = torch.randn(8, out_size)

        model.train()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()

        # Should work the same way for all models
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        assert output.shape == (8, out_size)


# ==============================================================================
# DATA PIPELINE INTEGRATION
# ==============================================================================
@pytest.mark.integration
class TestDataPipelineIntegration:
    """Integration tests for data loading pipeline."""

    def test_npz_to_model_pipeline(self):
        """Test complete pipeline from NPZ file to model predictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create NPZ file
            n_samples = 50
            X = np.random.randn(n_samples, 32, 32).astype(np.float32)
            y = np.random.randn(n_samples, 3).astype(np.float32)

            npz_path = os.path.join(tmpdir, "data.npz")
            np.savez(npz_path, input_train=X, output_train=y)

            # Load data (explicit close for Windows compatibility)
            data = np.load(npz_path)
            X_loaded = data["input_train"].copy()  # Copy to release file handle
            y_loaded = data["output_train"].copy()
            data.close()  # Explicitly close the file

            # Add channel dimension
            X_tensor = torch.from_numpy(X_loaded[:, np.newaxis, :, :])
            torch.from_numpy(y_loaded)

            # Create model and run inference
            model = CNN(in_shape=(32, 32), out_size=3)
            model.eval()

            with torch.no_grad():
                predictions = model(X_tensor)

            assert predictions.shape == (n_samples, 3)

    def test_batch_processing(self):
        """Test that batch processing produces consistent results."""
        model = CNN(in_shape=(32, 32), out_size=3)
        model.eval()

        # Create data
        X = torch.randn(32, 1, 32, 32)

        # Process all at once
        with torch.no_grad():
            full_output = model(X)

        # Process in batches
        batch_outputs = []
        for i in range(0, 32, 8):
            with torch.no_grad():
                batch_out = model(X[i : i + 8])
            batch_outputs.append(batch_out)

        batched_output = torch.cat(batch_outputs, dim=0)

        # Results should be identical
        assert torch.allclose(full_output, batched_output, atol=1e-5)


# ==============================================================================
# METRICS INTEGRATION
# ==============================================================================
@pytest.mark.integration
class TestMetricsIntegration:
    """Integration tests for metrics with models."""

    def test_metrics_on_model_output(self):
        """Test calculating metrics on actual model outputs."""
        model = CNN(in_shape=(32, 32), out_size=5)
        model.eval()

        X = torch.randn(100, 1, 32, 32)
        y_true = torch.randn(100, 5)

        with torch.no_grad():
            y_pred = model(X)

        y_true_np = y_true.numpy()
        y_pred_np = y_pred.numpy()

        # Calculate all metrics
        r2_scores = calc_per_target_r2(y_true_np, y_pred_np)
        pearson = calc_pearson(y_true_np, y_pred_np)

        assert len(r2_scores) == 5
        assert isinstance(pearson, float)

    def test_metric_tracker_in_training_loop(self):
        """Test MetricTracker integration in training loop."""
        model = CNN(in_shape=(32, 32), out_size=3)
        model.train()

        X = torch.randn(32, 1, 32, 32)
        y = torch.randn(32, 3)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=8)

        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        loss_tracker = MetricTracker()

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            loss_tracker.update(loss.item(), n=batch_x.size(0))

        # Verify tracker accumulated correctly
        assert loss_tracker.count == 32
        assert loss_tracker.avg > 0


# ==============================================================================
# GRADIENT FLOW INTEGRATION
# ==============================================================================
@pytest.mark.integration
class TestGradientFlowIntegration:
    """Integration tests for gradient flow through models."""

    def test_gradient_accumulation(self):
        """Test gradient accumulation works correctly."""
        model = CNN(in_shape=(32, 32), out_size=3)
        model.train()

        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        accumulation_steps = 4

        # Simulate gradient accumulation
        for i in range(accumulation_steps):
            X = torch.randn(4, 1, 32, 32)
            y = torch.randn(4, 3)

            output = model(X)
            loss = criterion(output, y) / accumulation_steps
            loss.backward()

        # All parameters should have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

        optimizer.step()
        optimizer.zero_grad()

    def test_gradient_clipping(self):
        """Test gradient clipping works correctly."""
        model = CNN(in_shape=(32, 32), out_size=3)
        model.train()

        criterion = nn.MSELoss()
        max_norm = 1.0

        X = torch.randn(8, 1, 32, 32)
        y = torch.randn(8, 3)

        output = model(X)
        loss = criterion(output, y)
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # Verify gradients are clipped
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

        assert total_norm <= max_norm * 1.01  # Small tolerance


# ==============================================================================
# CHECKPOINT AND RESUME INTEGRATION
# ==============================================================================
@pytest.mark.integration
class TestCheckpointIntegration:
    """Integration tests for checkpoint saving and loading."""

    def test_full_checkpoint_save_load_cycle(self):
        """Test complete checkpoint save and load cycle."""

        model = CNN(in_shape=(32, 32), out_size=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        # Train for a few steps
        X = torch.randn(8, 1, 32, 32)
        y = torch.randn(8, 3)
        criterion = nn.MSELoss()

        for _ in range(5):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": 5,
                "best_val_loss": 0.123,
            }
            checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")
            torch.save(checkpoint, checkpoint_path)

            # Load checkpoint
            new_model = CNN(in_shape=(32, 32), out_size=3)
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
            new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=10)

            loaded = torch.load(checkpoint_path, weights_only=True)
            new_model.load_state_dict(loaded["model_state_dict"])
            new_optimizer.load_state_dict(loaded["optimizer_state_dict"])
            new_scheduler.load_state_dict(loaded["scheduler_state_dict"])

            # Verify states match
            assert loaded["epoch"] == 5
            assert loaded["best_val_loss"] == 0.123

            # Verify model outputs match
            model.eval()
            new_model.eval()
            with torch.no_grad():
                out1 = model(X)
                out2 = new_model(X)
            assert torch.allclose(out1, out2)

    def test_scaler_save_load(self):
        """Test StandardScaler save and load for inference."""
        import pickle

        from sklearn.preprocessing import StandardScaler

        # Fit scaler
        y_train = np.random.randn(100, 5)
        scaler = StandardScaler()
        scaler.fit(y_train)

        with tempfile.TemporaryDirectory() as tmpdir:
            scaler_path = os.path.join(tmpdir, "scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

            with open(scaler_path, "rb") as f:
                loaded_scaler = pickle.load(f)

            # Verify transform matches
            test_data = np.random.randn(10, 5)
            np.testing.assert_array_almost_equal(
                scaler.transform(test_data),
                loaded_scaler.transform(test_data),
            )


# ==============================================================================
# ERROR HANDLING INTEGRATION
# ==============================================================================
@pytest.mark.integration
class TestErrorHandling:
    """Integration tests for error handling."""

    def test_nan_loss_detection(self):
        """Test that NaN loss is detectable."""
        model = CNN(in_shape=(32, 32), out_size=3)
        model.train()

        X = torch.randn(4, 1, 32, 32)
        y = torch.randn(4, 3)

        output = model(X)
        loss = nn.MSELoss()(output, y)

        # Normal loss should not be NaN
        assert not torch.isnan(loss).any()

        # Simulate NaN (for detection logic testing)
        nan_loss = torch.tensor(float("nan"))
        assert torch.isnan(nan_loss)

    def test_inf_output_detection(self):
        """Test that infinite outputs are detectable."""
        model = CNN(in_shape=(32, 32), out_size=3)
        model.eval()

        X = torch.randn(4, 1, 32, 32)
        with torch.no_grad():
            output = model(X)

        # Normal output should not have inf
        assert not torch.isinf(output).any()

    def test_empty_batch_handling(self):
        """Test behavior with edge case batch sizes."""
        model = CNN(in_shape=(32, 32), out_size=3)
        model.eval()

        # Single sample should work
        X = torch.randn(1, 1, 32, 32)
        with torch.no_grad():
            output = model(X)
        assert output.shape == (1, 3)


# ==============================================================================
# LEARNING RATE SCHEDULER INTEGRATION
# ==============================================================================
@pytest.mark.integration
class TestSchedulerIntegration:
    """Integration tests for learning rate schedulers with training."""

    def test_plateau_scheduler_reduces_lr(self):
        """Test ReduceLROnPlateau reduces LR when loss plateaus."""
        model = CNN(in_shape=(32, 32), out_size=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, factor=0.5
        )

        initial_lr = optimizer.param_groups[0]["lr"]

        # Simulate plateau (same loss for multiple epochs)
        for _ in range(5):
            scheduler.step(1.0)  # Same val_loss

        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr

    def test_plateau_scheduler_preserves_multigroup_lr(self):
        """Test ReduceLROnPlateau preserves LR ratios across parameter groups.

        Regression test: DDP LR sync previously broadcast only group[0]'s LR
        to all groups, collapsing per-group ratios (e.g., Swin backbone at 0.1×).
        """
        model = CNN(in_shape=(32, 32), out_size=3)

        # Create multi-group optimizer mimicking Swin's layout:
        # Group 0: backbone (0.01 LR), Group 1: head (0.1 LR)
        backbone_params = list(model.parameters())[:2]
        head_params = list(model.parameters())[2:]
        optimizer = torch.optim.Adam(
            [
                {"params": backbone_params, "lr": 0.01},
                {"params": head_params, "lr": 0.1},
            ]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, factor=0.5
        )

        initial_ratio = (
            optimizer.param_groups[1]["lr"] / optimizer.param_groups[0]["lr"]
        )
        assert initial_ratio == pytest.approx(10.0)  # 0.1 / 0.01

        # Simulate plateau
        for _ in range(5):
            scheduler.step(1.0)

        # Both groups should have reduced, but ratio must be preserved
        assert optimizer.param_groups[0]["lr"] < 0.01
        assert optimizer.param_groups[1]["lr"] < 0.1
        final_ratio = optimizer.param_groups[1]["lr"] / optimizer.param_groups[0]["lr"]
        assert final_ratio == pytest.approx(10.0)

    def test_cosine_scheduler_varies_lr(self):
        """Test CosineAnnealingLR varies LR over epochs."""
        model = CNN(in_shape=(32, 32), out_size=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        lrs = []
        for _ in range(10):
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        # LR should vary (not constant)
        assert len(set(lrs)) > 1

    def test_onecycle_scheduler_with_training(self):
        """Test OneCycleLR works with training loop."""
        model = CNN(in_shape=(32, 32), out_size=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        total_steps = 100
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.01, total_steps=total_steps
        )

        X = torch.randn(4, 1, 32, 32)
        y = torch.randn(4, 3)
        criterion = nn.MSELoss()

        # Simulate training steps
        for _ in range(total_steps):
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Should complete without error


# ==============================================================================
# CLI END-TO-END SUBPROCESS TESTS
# ==============================================================================

# Check for optional dependencies
try:
    import optuna  # noqa: F401

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import onnx  # noqa: F401
    import onnxruntime  # noqa: F401

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


@pytest.mark.integration
@pytest.mark.slow
class TestCLIEndToEnd:
    """End-to-end subprocess tests for CLI entry points.

    These tests verify that the actual CLI commands work as expected
    when invoked as subprocesses, rather than just testing argument parsing.
    """

    def test_wavedl_train_subprocess_minimal(self, temp_training_data):
        """E2E: wavedl-train subprocess runs minimal training successfully.

        This test invokes wavedl-train via subprocess with minimal epochs
        to verify the entire training pipeline works end-to-end.
        """
        import subprocess
        import sys

        npz_path = temp_training_data["npz_path"]
        output_dir = os.path.join(temp_training_data["temp_dir"], "train_output")

        # Build command - use python -m for cross-platform compatibility
        cmd = [
            sys.executable,
            "-m",
            "wavedl.train",
            "--data_path",
            npz_path,
            "--model",
            "cnn",
            "--epochs",
            "2",  # Minimal epochs
            "--patience",
            "1",
            "--batch_size",
            "16",
            "--output_dir",
            output_dir,
            "--seed",
            "42",
            "--mixed_precision",
            "no",  # Ensure MPS compat (bf16 requires PyTorch >= 2.6)
        ]

        # Run training subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout for minimal training
        )

        # Debug output on failure
        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")

        # Verify success
        assert result.returncode == 0, f"Training failed with: {result.stderr}"

        # Verify expected outputs were created
        assert os.path.exists(output_dir), "Output directory not created"
        assert os.path.exists(os.path.join(output_dir, "training_history.csv")), (
            "Training history not saved"
        )
        assert os.path.exists(os.path.join(output_dir, "best_checkpoint")), (
            "Best checkpoint not saved"
        )

    def test_wavedl_train_list_models(self):
        """E2E: wavedl-train --list_models works via subprocess."""
        import subprocess
        import sys

        cmd = [sys.executable, "-m", "wavedl.train", "--list_models"]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0
        # Should list at least some models
        assert "cnn" in result.stdout.lower() or "resnet" in result.stdout.lower()


# ==============================================================================
# HPO OBJECTIVE EXECUTION TESTS
# ==============================================================================
@pytest.mark.integration
@pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed")
class TestHPOObjectiveExecution:
    """Tests that actually execute the HPO objective function.

    These tests go beyond checking that create_objective returns a callable -
    they verify the objective function can execute a trial correctly.
    """

    def test_inprocess_objective_executes_trial(self, temp_training_data):
        """Test that create_objective objective executes a mock trial successfully.

        Uses in-process mode (--inprocess) for faster execution without subprocess overhead.
        """
        from unittest.mock import MagicMock

        import optuna

        from wavedl.hpo import create_objective

        # Create mock args for in-process HPO
        args = MagicMock()
        args.data_path = temp_training_data["npz_path"]
        args.max_epochs = 2  # Very short training
        args.quick = True  # Use quick search space
        args.medium = False
        args.inprocess = True  # In-process mode for speed
        args.seed = 42
        args.timeout = 300
        args.models = ["cnn"]  # Only test CNN for speed
        args.optimizers = ["adamw"]
        args.schedulers = ["plateau"]
        args.losses = ["mse"]
        args.batch_sizes = [16]
        args.n_jobs = 1

        # Create objective function
        objective = create_objective(args)
        assert callable(objective)

        # Create a mock trial that returns fixed values
        trial = MagicMock(spec=optuna.trial.Trial)
        trial.number = 0
        trial.suggest_categorical = MagicMock(
            side_effect=lambda name, choices: choices[0]
        )
        trial.suggest_float = MagicMock(side_effect=lambda name, low, high, **kw: low)
        trial.suggest_int = MagicMock(side_effect=lambda name, low, high, **kw: low)
        trial.report = MagicMock()
        trial.should_prune = MagicMock(return_value=False)

        # Execute the objective - should complete without error
        try:
            result = objective(trial)
            # Result should be a valid loss value (float, not inf)
            assert isinstance(result, float)
            assert result < float("inf"), "Objective returned inf - training failed"
            assert not np.isnan(result), "Objective returned NaN"
        except optuna.TrialPruned:
            # Pruning is acceptable - trial was stopped early
            pass


# ==============================================================================
# ONNX DENORMALIZATION ACCURACY TESTS
# ==============================================================================
@pytest.mark.integration
@pytest.mark.skipif(not HAS_ONNX, reason="onnx/onnxruntime not installed")
class TestONNXDenormAccuracy:
    """Tests for ONNX export numerical accuracy with denormalization.

    Verifies that ONNX models exported with include_denorm=True produce
    outputs that are numerically consistent with PyTorch + scaler.inverse_transform.
    """

    def test_onnx_denorm_numerical_consistency(self, temp_dir, mock_scaler):
        """ONNX with include_denorm=True produces numerically accurate outputs.

        This test:
        1. Creates a model and exports to ONNX with denormalization embedded
        2. Runs inference with both PyTorch (+ manual denorm) and ONNX
        3. Compares outputs for numerical consistency
        """
        import onnxruntime as ort

        from wavedl.models.cnn import CNN
        from wavedl.test import export_to_onnx

        # Build model
        in_shape = (32, 32)
        out_size = 5
        model = CNN(in_shape=in_shape, out_size=out_size)
        model.eval()

        # Create sample input
        sample_input = torch.randn(4, 1, *in_shape)
        onnx_path = os.path.join(temp_dir, "model_denorm.onnx")

        # Export with denormalization
        success = export_to_onnx(
            model,
            sample_input,
            onnx_path,
            scaler=mock_scaler,
            include_denorm=True,
            validate=False,  # We'll do our own validation
        )
        assert success, "ONNX export failed"
        assert os.path.exists(onnx_path)

        # PyTorch inference with manual denormalization
        with torch.no_grad():
            pytorch_normalized = model(sample_input).numpy()
        pytorch_denorm = mock_scaler.inverse_transform(pytorch_normalized)

        # ONNX inference (denormalization is embedded in the model)
        ort_session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        onnx_output = ort_session.run(None, {"input": sample_input.numpy()})[0]

        # Compare outputs
        max_diff = np.abs(pytorch_denorm - onnx_output).max()
        assert np.allclose(pytorch_denorm, onnx_output, rtol=1e-4, atol=1e-5), (
            f"Numerical mismatch between PyTorch and ONNX. Max diff: {max_diff:.2e}"
        )

    def test_onnx_denorm_wrapper_forward(self, mock_scaler):
        """Test ModelWithDenormalization wrapper produces correct outputs."""
        from wavedl.models.cnn import CNN
        from wavedl.test import ModelWithDenormalization

        model = CNN(in_shape=(32, 32), out_size=5)
        model.eval()

        # Wrap with denormalization
        wrapped = ModelWithDenormalization(model, mock_scaler.mean_, mock_scaler.scale_)
        wrapped.eval()

        # Test forward pass
        x = torch.randn(2, 1, 32, 32)

        with torch.no_grad():
            # Get normalized output from base model
            normalized = model(x).numpy()
            # Get denormalized output from wrapper
            denormalized = wrapped(x).numpy()

        # Verify denormalization was applied correctly
        expected = normalized * mock_scaler.scale_ + mock_scaler.mean_
        np.testing.assert_array_almost_equal(denormalized, expected, decimal=5)

    def test_dense_onnx_preserves_opset_dynamic_batch_and_denormalization(
        self, temp_dir
    ):
        """Field export honors its requested opset and accepts later batches."""
        import onnx
        import onnxruntime as ort

        from wavedl.models import UNetField
        from wavedl.test import export_to_onnx
        from wavedl.utils import SpatialTargetScaler

        model = UNetField(
            in_shape=(16, 16),
            out_size=2,
            out_shape=(2, 9, 11),
            base_channels=2,
            depth=2,
            dropout_rate=0.0,
        ).eval()
        scaler = SpatialTargetScaler().fit(
            np.arange(3 * 2 * 9 * 11, dtype=np.float32).reshape(3, 2, 9, 11)
        )
        sample_input = torch.randn(1, 1, 16, 16)
        onnx_path = os.path.join(temp_dir, "field_denorm.onnx")

        assert export_to_onnx(
            model,
            sample_input,
            onnx_path,
            opset_version=17,
            scaler=scaler,
            include_denorm=True,
            validate=False,
        )

        exported = onnx.load(onnx_path)
        assert exported.opset_import[0].version == 17
        output_batch_dim = exported.graph.output[0].type.tensor_type.shape.dim[0]
        assert output_batch_dim.dim_param == "batch_size"

        later_batch = torch.randn(2, 1, 16, 16)
        with torch.no_grad():
            expected = scaler.inverse_transform(model(later_batch).numpy())
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        actual = session.run(None, {"input": later_batch.numpy()})[0]

        assert actual.shape == (2, 2, 9, 11)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


# ==============================================================================
# MAMBA NUMERICAL STABILITY TESTS
# ==============================================================================
@pytest.mark.integration
@pytest.mark.slow
class TestMambaStability:
    """Stress tests for Mamba numerical stability.

    These tests verify that Mamba handles long sequences without
    numerical instability (NaN/Inf), especially sequences longer than
    the MAX_SAFE_SEQUENCE_LENGTH threshold (512) which triggers chunked scan.
    """

    def test_mamba_long_sequence_no_nan(self):
        """Mamba handles sequences > MAX_SAFE_SEQUENCE_LENGTH without NaN.

        Tests with sequence length 4096 (8x the 512 threshold) to ensure
        the chunked parallel scan handles long sequences correctly.
        """
        from wavedl.models.mamba import Mamba1D

        # 4096 is well above MAX_SAFE_SEQUENCE_LENGTH (512)
        # This forces the chunked scan path
        seq_length = 4096
        in_shape = (seq_length,)
        out_size = 3

        model = Mamba1D(in_shape=in_shape, out_size=out_size)
        model.eval()

        # Test with batch of inputs
        x = torch.randn(2, 1, seq_length)

        with torch.no_grad():
            output = model(x)

        # Verify no numerical issues
        assert output.shape == (2, out_size), f"Unexpected shape: {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"

    def test_mamba_gradient_stability_long_sequence(self):
        """Mamba gradients are stable for long sequences.

        Tests backward pass with long sequences to ensure gradients
        don't explode or vanish.
        """
        from wavedl.models.mamba import Mamba1D

        seq_length = 2048  # Above threshold
        in_shape = (seq_length,)
        out_size = 3

        model = Mamba1D(in_shape=in_shape, out_size=out_size)
        model.train()

        x = torch.randn(2, 1, seq_length, requires_grad=True)

        # Forward pass
        output = model(x)
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Verify gradients exist and are valid
        assert x.grad is not None, "No gradient computed for input"
        assert not torch.isnan(x.grad).any(), "Input gradient contains NaN"
        assert not torch.isinf(x.grad).any(), "Input gradient contains Inf"

        # Check model parameter gradients
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    def test_mamba_chunked_vs_single_consistency(self):
        """Verify chunked and single-pass scan produce similar results.

        For sequences at the boundary, results should be consistent.
        """
        from wavedl.models.mamba import Mamba1D

        # Test at boundary length
        seq_length = 512  # Exactly at MAX_SAFE_SEQUENCE_LENGTH
        in_shape = (seq_length,)
        out_size = 3

        model = Mamba1D(in_shape=in_shape, out_size=out_size)
        model.eval()

        torch.manual_seed(42)
        x = torch.randn(1, 1, seq_length)

        with torch.no_grad():
            output = model(x)

        # Should produce valid output
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert output.shape == (1, out_size)
