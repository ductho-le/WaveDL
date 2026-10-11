"""
Unit Tests for Physical Constraints
===================================

Tests for constraint modules: expression parsing, bounds, hard transforms,
reparameterization, and combined loss wrapper.

Author: Ductho Le (ductho.le@outlook.com)
"""

import os
import sys
import tempfile

import pytest
import torch
import torch.nn as nn


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavedl.utils.constraints import (
    ExpressionConstraint,
    FileConstraint,
    PhysicsConstrainedLoss,
    build_constraints,
)


# ==============================================================================
# ExpressionConstraint Tests
# ==============================================================================
class TestExpressionConstraint:
    """Tests for soft expression constraints."""

    def test_simple_expression(self):
        """Simple subtraction expression should work."""
        constraint = ExpressionConstraint("y0 - y1")
        pred = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
        result = constraint(pred)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_expression_with_violation(self):
        """Expression should return non-zero for violations."""
        constraint = ExpressionConstraint("y0 - y1")
        pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # y0 != y1
        result = constraint(pred)
        assert result.item() > 0

    def test_product_constraint(self):
        """Product constraint y0 - y1*y2 should work."""
        constraint = ExpressionConstraint("y0 - y1 * y2")
        # y0 = 6, y1 = 2, y2 = 3 → y0 - y1*y2 = 0
        pred = torch.tensor([[6.0, 2.0, 3.0]])
        result = constraint(pred)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_expression_applies_per_pixel_to_dense_channels(self):
        """Field channel symbols should retain their spatial dimensions."""
        constraint = ExpressionConstraint("y0 - y1")
        pred = torch.ones(2, 2, 3, 4)

        assert constraint(pred).item() == pytest.approx(0.0, abs=1e-6)

    def test_math_functions(self):
        """Math functions (sin, cos, exp) should work."""
        constraint = ExpressionConstraint("sin(y0)")
        pred = torch.tensor([[0.0]])  # sin(0) = 0
        result = constraint(pred)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_input_aggregates(self):
        """Input aggregates (x_mean) should work."""
        constraint = ExpressionConstraint("y0 - x_mean")
        pred = torch.tensor([[5.0]])
        inputs = torch.tensor([[5.0, 5.0, 5.0]])  # mean = 5
        result = constraint(pred, inputs)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_unsafe_function_raises(self):
        """Unsafe functions should raise ValueError."""
        with pytest.raises(ValueError, match="Unsafe function"):
            ExpressionConstraint("eval(y0)")

    def test_unknown_variable_raises(self):
        """Unknown variables should raise ValueError."""
        constraint = ExpressionConstraint("unknown_var")
        pred = torch.tensor([[1.0]])
        with pytest.raises(ValueError, match="Unknown variable"):
            constraint(pred)

    def test_output_index_out_of_range(self):
        """Accessing y10 when only 3 outputs should raise."""
        constraint = ExpressionConstraint("y10")
        pred = torch.tensor([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="out of range"):
            constraint(pred)

    def test_reduction_mae(self):
        """MAE reduction should use abs instead of square."""
        constraint = ExpressionConstraint("y0", reduction="mae")
        pred = torch.tensor([[-1.0], [1.0]])  # violations: 1, 1
        result = constraint(pred)
        assert result.item() == pytest.approx(1.0, abs=1e-6)

    def test_comparison_greater_than(self):
        """Greater than constraint: y0 > 0."""
        constraint = ExpressionConstraint("y0 > 0", reduction="mae")
        # y0=5: satisfied, violation = 0
        assert constraint(torch.tensor([[5.0]])).item() == pytest.approx(0.0, abs=1e-6)
        # y0=-2: violated, violation = 2
        assert constraint(torch.tensor([[-2.0]])).item() == pytest.approx(2.0, abs=1e-6)

    def test_comparison_less_than(self):
        """Less than constraint: y0 < 1."""
        constraint = ExpressionConstraint("y0 < 1", reduction="mae")
        # y0=0.5: satisfied
        assert constraint(torch.tensor([[0.5]])).item() == pytest.approx(0.0, abs=1e-6)
        # y0=3: violated, violation = 2
        assert constraint(torch.tensor([[3.0]])).item() == pytest.approx(2.0, abs=1e-6)

    def test_comparison_ordering(self):
        """Ordering constraint: y0 < y1."""
        constraint = ExpressionConstraint("y0 < y1", reduction="mae")
        # y0=1, y1=5: satisfied
        assert constraint(torch.tensor([[1.0, 5.0]])).item() == pytest.approx(
            0.0, abs=1e-6
        )
        # y0=5, y1=3: violated, violation = 2
        assert constraint(torch.tensor([[5.0, 3.0]])).item() == pytest.approx(
            2.0, abs=1e-6
        )


# ==============================================================================
# FileConstraint Tests
# ==============================================================================
class TestFileConstraint:
    """Tests for file-based constraints."""

    def test_load_constraint_file(self):
        """Should load and execute constraint from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("import torch\n")
            f.write("def constraint(pred, inputs=None):\n")
            f.write("    return pred[:, 0] - pred[:, 1]\n")
            f.flush()

            try:
                constraint = FileConstraint(f.name)
                pred = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
                result = constraint(pred)
                assert result.item() == pytest.approx(0.0, abs=1e-6)
            finally:
                try:
                    os.unlink(f.name)
                except PermissionError:
                    pass  # Windows file locking - cleanup later

    def test_missing_function_raises(self):
        """File without constraint function should raise."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("# Empty file\n")
            f.write("x = 1\n")
            f.flush()

            try:
                with pytest.raises(ValueError, match="must define"):
                    FileConstraint(f.name)
            finally:
                try:
                    os.unlink(f.name)
                except PermissionError:
                    pass  # Windows file locking - cleanup later


# ==============================================================================
# PhysicsConstrainedLoss Tests
# ==============================================================================
class TestPhysicsConstrainedLoss:
    """Tests for combined loss wrapper."""

    def test_base_loss_only(self):
        """Without constraints, should equal base loss."""
        base = nn.MSELoss()
        constrained = PhysicsConstrainedLoss(base)
        pred = torch.tensor([[1.0, 2.0]])
        target = torch.tensor([[1.0, 2.0]])
        result = constrained(pred, target)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_with_constraint(self):
        """With constraint, loss should include penalty."""
        base = nn.MSELoss()
        constraint = ExpressionConstraint("y0 - y1")
        constrained = PhysicsConstrainedLoss(base, [constraint], weights=[1.0])
        pred = torch.tensor([[1.0, 2.0]])  # violation: 1-2 = -1
        target = torch.tensor([[1.0, 2.0]])
        result = constrained(pred, target)
        # MSE = 0, constraint = (-1)² = 1
        assert result.item() == pytest.approx(1.0, abs=1e-6)

    def test_multiple_weights(self):
        """Multiple weights should apply to respective constraints."""
        base = nn.MSELoss()
        c1 = ExpressionConstraint("y0")  # penalty = y0²
        c2 = ExpressionConstraint("y1")  # penalty = y1²
        constrained = PhysicsConstrainedLoss(base, [c1, c2], weights=[0.5, 2.0])
        pred = torch.tensor([[1.0, 1.0]])
        target = torch.tensor([[0.0, 0.0]])
        result = constrained(pred, target)
        # MSE = (1² + 1²)/2 = 1, c1 = 0.5*1² = 0.5, c2 = 2.0*1² = 2.0
        expected = 1.0 + 0.5 + 2.0
        assert result.item() == pytest.approx(expected, abs=1e-5)

    def test_dense_channel_scaler_is_broadcast_in_physical_space(self):
        """Spatial scaler statistics must denormalize every map pixel."""
        constraint = ExpressionConstraint("y0 - y1")
        constrained = PhysicsConstrainedLoss(
            nn.MSELoss(),
            [constraint],
            weights=[1.0],
            output_mean=torch.tensor([[[10.0]], [[5.0]]]),
            output_std=torch.tensor([[[2.0]], [[3.0]]]),
        )
        pred = torch.zeros(2, 2, 3, 4)

        result = constrained(pred, torch.zeros_like(pred))

        assert result.item() == pytest.approx(25.0, abs=1e-6)

    def test_large_physical_mean_preserves_float32_round_trip_and_gradients(self):
        """Do not discard a large mean's residual before adding predictions."""
        constrained = PhysicsConstrainedLoss(
            nn.MSELoss(),
            [ExpressionConstraint("y0 - 1000000000")],
            weights=[1.0],
            output_mean=[[[1.0e9 + 32]]],
            output_std=[[[32.0]]],
        )
        pred = torch.tensor([-1.0, 1.0]).reshape(2, 1, 1, 1).requires_grad_()
        physical = constrained._denormalize(pred)
        torch.testing.assert_close(
            physical.flatten(), torch.tensor([1.0e9, 1.0e9 + 64]), rtol=0, atol=0
        )
        assert all(buffer.dtype == torch.float32 for buffer in constrained.buffers())
        loss = constrained(pred, pred.detach())
        assert loss.item() == pytest.approx(2048.0)
        loss.backward()
        assert torch.isfinite(pred.grad).all()


# ==============================================================================
# Factory Function Tests
# ==============================================================================
class TestFactoryFunctions:
    """Tests for build_* factory functions."""

    def test_build_constraints_expressions(self):
        """build_constraints should handle expressions."""
        constraints = build_constraints(expressions=["y0 - y1"])
        assert len(constraints) == 1
        assert isinstance(constraints[0], ExpressionConstraint)

    def test_build_constraints_comparison(self):
        """build_constraints should handle comparison expressions."""
        constraints = build_constraints(expressions=["y0 > 0", "y0 < 1"])
        assert len(constraints) == 2


# ==============================================================================
# Gradient Flow Tests
# ==============================================================================
class TestGradientFlow:
    """Tests for gradient flow through constraints."""

    def test_expression_constraint_gradients(self):
        """Gradients should flow through expression constraints."""
        constraint = ExpressionConstraint("y0 * y1")
        pred = torch.tensor([[2.0, 3.0]], requires_grad=True)
        result = constraint(pred)
        result.backward()
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape

    def test_combined_loss_gradients(self):
        """Gradients should flow through combined loss."""
        base = nn.MSELoss()
        constraint = ExpressionConstraint("y0")
        loss = PhysicsConstrainedLoss(base, [constraint], weights=[0.1])

        pred = torch.tensor([[1.0, 2.0]], requires_grad=True)
        target = torch.tensor([[0.0, 0.0]])
        result = loss(pred, target)
        result.backward()
        assert pred.grad is not None
