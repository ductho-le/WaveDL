"""
Physical Constraints for Training
=================================

Soft constraint enforcement via penalty-based loss terms.

Usage:
    # Expression constraints
    wavedl-train --constraint "y0 > 0" --constraint_weight 0.1

    # Complex constraints via Python file
    wavedl-train --constraint_file my_constraint.py

Author: Ductho Le (ductho.le@outlook.com)
Version: 2.0.0
"""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F


if TYPE_CHECKING:
    from collections.abc import Callable


# ==============================================================================
# SAFE EXPRESSION PARSING
# ==============================================================================
SAFE_FUNCTIONS: dict[str, Callable] = {
    "sin": torch.sin,
    "cos": torch.cos,
    "tan": torch.tan,
    "exp": torch.exp,
    # log/sqrt are domain-protected: clamp the argument to a small positive floor
    # so denormalized predictions <= 0 (common before convergence) yield a finite
    # penalty/gradient instead of NaN/-inf, which would poison training via backprop.
    "log": lambda t: torch.log(torch.clamp(t, min=1e-12)),
    "sqrt": lambda t: torch.sqrt(torch.clamp(t, min=1e-12)),
    "abs": torch.abs,
    "relu": F.relu,
    "sigmoid": torch.sigmoid,
    "softplus": F.softplus,
    "tanh": torch.tanh,
    "min": torch.minimum,
    "max": torch.maximum,
    "pow": torch.pow,
    "clamp": torch.clamp,
}

INPUT_AGGREGATES: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "x_mean": lambda x: x.mean(dim=tuple(range(1, x.ndim))),
    "x_sum": lambda x: x.sum(dim=tuple(range(1, x.ndim))),
    "x_max": lambda x: x.amax(dim=tuple(range(1, x.ndim))),
    "x_min": lambda x: x.amin(dim=tuple(range(1, x.ndim))),
    "x_std": lambda x: x.std(dim=tuple(range(1, x.ndim)), correction=0),
    "x_energy": lambda x: (x**2).sum(dim=tuple(range(1, x.ndim))),
}


# ==============================================================================
# SOFT CONSTRAINTS
# ==============================================================================
class ExpressionConstraint(nn.Module):
    """
    Soft constraint via string expression.

    Parses mathematical expressions using Python's AST for safe evaluation.
    Supports output variables (y0, y1, ...), input aggregates (x_mean, ...),
    and whitelisted math functions.

    Example:
        >>> constraint = ExpressionConstraint("y0 - y1 * y2")
        >>> penalty = constraint(predictions, inputs)

        >>> constraint = ExpressionConstraint("sin(y0) + cos(y1)")
        >>> penalty = constraint(predictions, inputs)
    """

    def __init__(self, expression: str, reduction: str = "mse"):
        """
        Args:
            expression: Mathematical expression to evaluate (should equal 0)
            reduction: How to reduce violations - 'mse' or 'mae'
        """
        super().__init__()
        if reduction not in {"mse", "mae"}:
            raise ValueError("Constraint reduction must be 'mse' or 'mae'.")
        self.expression = expression
        self.reduction = reduction
        self._tree = ast.parse(expression, mode="eval")
        self._validate(self._tree)

    def _validate(self, tree: ast.Expression) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant):
                if not isinstance(node.value, (int, float)) or not math.isfinite(
                    node.value
                ):
                    raise ValueError(
                        "Constraint constants must be finite real numbers."
                    )
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id not in SAFE_FUNCTIONS:
                    raise ValueError(
                        f"Unsafe function '{node.func.id}' in constraint. "
                        f"Allowed: {list(SAFE_FUNCTIONS.keys())}"
                    )
                allowed_keywords = {"min", "max"} if node.func.id == "clamp" else set()
                for keyword in node.keywords:
                    if keyword.arg not in allowed_keywords:
                        raise ValueError(
                            f"Unsupported keyword '{keyword.arg}' for {node.func.id}."
                        )
            elif isinstance(node, ast.Call):
                raise ValueError("Only direct function calls supported (e.g., sin(x))")

    def _eval(
        self, node: ast.AST, pred: torch.Tensor, inputs: torch.Tensor | None
    ) -> torch.Tensor:
        """Recursively evaluate AST node."""
        if isinstance(node, ast.Constant):
            return torch.tensor(node.value, device=pred.device, dtype=pred.dtype)
        elif isinstance(node, ast.Name):
            name = node.id
            # Output variable: y0, y1, ...
            if name.startswith("y") and name[1:].isdigit():
                idx = int(name[1:])
                if idx >= pred.shape[1]:
                    raise ValueError(
                        f"Output index {idx} out of range. "
                        f"Model has {pred.shape[1]} outputs."
                    )
                return pred[:, idx]
            # Input aggregate: x_mean, x_sum, ...
            elif name in INPUT_AGGREGATES:
                if inputs is None:
                    raise ValueError(
                        f"Constraint uses '{name}' but inputs not provided."
                    )
                value = INPUT_AGGREGATES[name](inputs)
                if pred.ndim > 2:
                    value = value.reshape(value.shape[0], *([1] * (pred.ndim - 2)))
                return value
            else:
                raise ValueError(
                    f"Unknown variable '{name}'. "
                    f"Use y0, y1, ... for outputs or {list(INPUT_AGGREGATES.keys())} for inputs."
                )
        elif isinstance(node, ast.BinOp):
            left = self._eval(node.left, pred, inputs)
            right = self._eval(node.right, pred, inputs)
            ops = {
                ast.Add: torch.add,
                ast.Sub: torch.sub,
                ast.Mult: torch.mul,
                ast.Div: torch.div,
                ast.Pow: torch.pow,
                ast.Mod: torch.remainder,
            }
            op_fn = ops.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op_fn(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval(node.operand, pred, inputs)
            if isinstance(node.op, ast.USub):
                return -operand
            elif isinstance(node.op, ast.UAdd):
                return operand
            else:
                raise ValueError(
                    f"Unsupported unary operator: {type(node.op).__name__}"
                )
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct function calls supported (e.g., sin(x))")
            func_name = node.func.id
            if func_name not in SAFE_FUNCTIONS:
                raise ValueError(f"Unsafe function: {func_name}")
            args = [self._eval(arg, pred, inputs) for arg in node.args]
            kwargs = {
                keyword.arg: self._eval(keyword.value, pred, inputs)
                for keyword in node.keywords
                if keyword.arg is not None
            }
            return SAFE_FUNCTIONS[func_name](*args, **kwargs)
        elif isinstance(node, ast.Compare):
            # Comparison operators: y0 > 0, y0 < 1, y0 >= 0, y0 <= 1
            # Returns penalty (violation amount) when constraint is not satisfied
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise ValueError(
                    "Only single comparisons supported (e.g., 'y0 > 0', not 'y0 > 0 > y1')"
                )
            left = self._eval(node.left, pred, inputs)
            right = self._eval(node.comparators[0], pred, inputs)
            op = node.ops[0]

            # Return violation amount (0 if satisfied, positive if violated)
            if isinstance(
                op, (ast.Gt, ast.GtE)
            ):  # y0 > right → penalize if y0 <= right
                return F.relu(right - left)
            elif isinstance(
                op, (ast.Lt, ast.LtE)
            ):  # y0 < right → penalize if y0 >= right
                return F.relu(left - right)
            elif isinstance(op, ast.Eq):  # y0 == right → penalize difference
                return torch.abs(left - right)
            elif isinstance(op, ast.NotEq):  # y0 != right → not useful as constraint
                raise ValueError(
                    "'!=' is not a valid constraint. Use '==' for equality constraints."
                )
            else:
                raise ValueError(
                    f"Unsupported comparison operator: {type(op).__name__}"
                )
        elif isinstance(node, ast.Subscript):
            # Input indexing: x[0], x[0,5], x[0,5,10]
            if not isinstance(node.value, ast.Name) or node.value.id != "x":
                raise ValueError(
                    "Subscript indexing only supported for 'x' (inputs). "
                    "Use x[i], x[i,j], or x[i,j,k]."
                )
            if inputs is None:
                raise ValueError("Constraint uses 'x[...]' but inputs not provided.")

            # Parse indices from the slice
            indices = self._parse_subscript_indices(node.slice)

            # Auto-squeeze channel dimension for single-channel inputs
            # This allows x[i,j] syntax for (B, 1, H, W) inputs instead of x[c,i,j]
            inputs_for_indexing = inputs
            if inputs.ndim >= 3 and inputs.shape[1] == 1:
                inputs_for_indexing = inputs.squeeze(1)  # (B, 1, H, W) → (B, H, W)

            # Validate dimensions match
            # inputs shape: (batch, dim1) or (batch, dim1, dim2) or (batch, dim1, dim2, dim3)
            input_ndim = inputs_for_indexing.ndim - 1  # Exclude batch dimension
            if len(indices) != input_ndim:
                raise ValueError(
                    f"Input has {input_ndim}D shape (after channel squeeze), but got {len(indices)} indices. "
                    f"Use x[i] for 1D, x[i,j] for 2D, x[i,j,k] for 3D inputs."
                )

            # Extract the value at the specified indices (for entire batch)
            if len(indices) == 1:
                value = inputs_for_indexing[:, indices[0]]
            elif len(indices) == 2:
                value = inputs_for_indexing[:, indices[0], indices[1]]
            elif len(indices) == 3:
                value = inputs_for_indexing[:, indices[0], indices[1], indices[2]]
            else:
                raise ValueError("Only 1D, 2D, or 3D input indexing supported.")

            if pred.ndim > 2:
                value = value.reshape(value.shape[0], *([1] * (pred.ndim - 2)))
            return value
        elif isinstance(node, ast.Expression):
            return self._eval(node.body, pred, inputs)
        else:
            raise ValueError(f"Unsupported AST node type: {type(node).__name__}")

    def _parse_subscript_indices(self, slice_node: ast.AST) -> list[int]:
        """Parse subscript indices from AST slice node."""

        def parse_index(node: ast.AST) -> int:
            sign = 1
            if isinstance(node, ast.UnaryOp) and isinstance(
                node.op, (ast.USub, ast.UAdd)
            ):
                sign = -1 if isinstance(node.op, ast.USub) else 1
                node = node.operand
            if not isinstance(node, ast.Constant) or type(node.value) is not int:
                raise ValueError("Only constant integer indices supported in x[...].")
            return sign * node.value

        if isinstance(slice_node, (ast.Constant, ast.UnaryOp)):
            # Single index: x[0]
            return [parse_index(slice_node)]
        elif isinstance(slice_node, ast.Tuple):
            # Multiple indices: x[0,5] or x[0,5,10]
            return [parse_index(elt) for elt in slice_node.elts]
        else:
            raise ValueError(
                f"Unsupported subscript type: {type(slice_node).__name__}. "
                "Use x[0], x[0,5], or x[0,5,10]."
            )

    def forward(
        self, pred: torch.Tensor, inputs: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute constraint violation penalty.

        Args:
            pred: Model predictions shaped ``(N,T)`` or ``(N,C,H,W)``
            inputs: Model inputs of shape (N, ...) for input-dependent constraints

        Returns:
            Scalar penalty value
        """
        violation = self._eval(self._tree, pred, inputs)
        if self.reduction == "mse":
            return (violation**2).mean()
        else:  # mae
            return violation.abs().mean()

    def __repr__(self) -> str:
        return (
            f"ExpressionConstraint('{self.expression}', reduction='{self.reduction}')"
        )


class FileConstraint(nn.Module):
    """
    Load constraint function from Python file.

    The file must define a function `constraint(pred, inputs=None)` that
    returns per-sample violation values.

    Example file (my_constraint.py):
        import torch

        def constraint(pred, inputs=None):
            # Monotonicity: y0 < y1 < y2
            diffs = pred[:, 1:] - pred[:, :-1]
            return torch.relu(-diffs).sum(dim=1)

    Usage:
        >>> constraint = FileConstraint("my_constraint.py")
        >>> penalty = constraint(predictions, inputs)
    """

    def __init__(self, file_path: str, reduction: str = "mse"):
        """
        Args:
            file_path: Path to Python file containing constraint function
            reduction: How to reduce violations - 'mse' or 'mae'
        """
        super().__init__()
        if reduction not in {"mse", "mae"}:
            raise ValueError("Constraint reduction must be 'mse' or 'mae'.")
        self.file_path = file_path
        self.reduction = reduction

        # Load module from file
        module_name = f"constraint_module_{id(self)}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load constraint file: {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        if not callable(getattr(module, "constraint", None)):
            raise ValueError(
                f"Constraint file must define 'constraint(pred, inputs)' function: {file_path}"
            )

        self._constraint_fn = module.constraint

    def forward(
        self, pred: torch.Tensor, inputs: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Evaluate constraint from loaded function."""
        violation = self._constraint_fn(pred, inputs)
        if self.reduction == "mse":
            return (violation**2).mean()
        else:
            return violation.abs().mean()

    def __repr__(self) -> str:
        return f"FileConstraint('{self.file_path}')"


# ==============================================================================
# COMBINED LOSS WRAPPER
# ==============================================================================
class PhysicsConstrainedLoss(nn.Module):
    """
    Combine base loss with constraint penalties.

    Total Loss = Base Loss + Σ(weight_i × constraint_i)

    Constraints are evaluated in **physical space** (denormalized) while
    the base loss is computed in normalized space for stable training.

    Example:
        >>> base_loss = nn.MSELoss()
        >>> constraints = [ExpressionConstraint("y0 - y1*y2")]
        >>> criterion = PhysicsConstrainedLoss(
        ...     base_loss,
        ...     constraints,
        ...     weights=[0.1],
        ...     output_mean=[10, 5, 50],
        ...     output_std=[2, 1, 10],
        ... )
        >>> loss = criterion(pred, target, inputs)
    """

    def __init__(
        self,
        base_loss: nn.Module,
        constraints: list[nn.Module] | None = None,
        weights: list[float] | None = None,
        output_mean: torch.Tensor | list[float] | None = None,
        output_std: torch.Tensor | list[float] | None = None,
    ):
        """
        Args:
            base_loss: Base loss function (e.g., MSELoss)
            constraints: List of constraint modules
            weights: Weight for each constraint. If shorter than constraints,
                     last weight is repeated.
            output_mean: Per-target/channel means. Shape ``(T,)`` or ``(C,1,1)``.
            output_std: Per-target/channel scales. Shape ``(T,)`` or ``(C,1,1)``.
        """
        super().__init__()
        self.base_loss = base_loss
        self.constraints = nn.ModuleList(constraints or [])
        self.weights = [0.1] if weights is None else list(weights)
        if not self.weights or any(
            not math.isfinite(weight) or weight < 0 for weight in self.weights
        ):
            raise ValueError(
                "Constraint weights must be finite, non-negative, and non-empty."
            )
        if (output_mean is None) != (output_std is None):
            raise ValueError("Provide both output_mean and output_std together.")

        # Store scaler as buffers (moves to correct device automatically)
        if output_mean is not None:
            mean64 = torch.as_tensor(
                output_mean, dtype=torch.float64, device="cpu"
            ).detach()
            mean32 = mean64.float()
            # Retain the rounding residual before moving to devices (including
            # MPS) that do not support float64. Add the residual to the small
            # standardized contribution before adding the large physical mean.
            self.register_buffer("output_mean", mean32)
            self.register_buffer("output_mean_residual", (mean64 - mean32).float())
        else:
            self.register_buffer("output_mean", None)
            self.register_buffer("output_mean_residual", None)

        if output_std is not None:
            self.register_buffer(
                "output_std",
                torch.as_tensor(output_std, dtype=torch.float32, device="cpu").detach(),
            )
        else:
            self.register_buffer("output_std", None)

    def _denormalize(self, pred: torch.Tensor) -> torch.Tensor:
        """Convert normalized predictions to physical values."""
        if self.output_mean is None or self.output_std is None:
            return pred
        output_mean = self.output_mean
        output_residual = self.output_mean_residual
        output_std = self.output_std
        # Legacy/custom scalers commonly store channel statistics as (C,).
        # Reshape them for channels-first dense fields; vector behavior is
        # unchanged. SpatialTargetScaler already stores (C,1,1), which also
        # broadcasts correctly after the optional batch dimension below.
        if pred.ndim > 2 and output_mean.ndim == 1:
            shape = (1, -1, *([1] * (pred.ndim - 2)))
            output_mean = output_mean.reshape(shape)
            output_residual = output_residual.reshape(shape)
            output_std = output_std.reshape(shape)
        elif output_mean.ndim == pred.ndim - 1:
            output_mean = output_mean.unsqueeze(0)
            output_residual = output_residual.unsqueeze(0)
            output_std = output_std.unsqueeze(0)
        return (pred * output_std + output_residual) + output_mean

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            pred: Model predictions (normalized)
            target: Ground truth targets (normalized)
            inputs: Model inputs (for input-dependent constraints)

        Returns:
            Combined loss value
        """
        # Base loss in normalized space (stable gradients)
        loss = self.base_loss(pred, target)

        # Denormalize for constraint evaluation (physical units)
        pred_physical = self._denormalize(pred)

        for i, constraint in enumerate(self.constraints):
            weight = self.weights[i] if i < len(self.weights) else self.weights[-1]
            if weight == 0:
                continue
            penalty = constraint(pred_physical, inputs)
            loss = loss + weight * penalty

        return loss

    def __repr__(self) -> str:
        has_scaler = self.output_mean is not None
        return f"PhysicsConstrainedLoss(base={self.base_loss}, constraints={len(self.constraints)}, denormalize={has_scaler})"


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================
def build_constraints(
    expressions: list[str] | None = None,
    file_path: str | None = None,
    reduction: str = "mse",
) -> list[nn.Module]:
    """
    Build soft constraint modules from CLI arguments.

    Args:
        expressions: Expression constraints (e.g., ["y0 - y1*y2", "y0 > 0"])
        file_path: Path to Python constraint file
        reduction: Reduction mode for penalties

    Returns:
        List of constraint modules
    """
    constraints: list[nn.Module] = []

    for expr in expressions or []:
        constraints.append(ExpressionConstraint(expr, reduction))

    if file_path:
        constraints.append(FileConstraint(file_path, reduction))

    return constraints
