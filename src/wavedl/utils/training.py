"""
Shared Training Utilities
=========================

Execution policies shared by standard training and cross-validation.

Author: Ductho Le (ductho.le@outlook.com)
"""

import torch


def enable_deterministic_mode() -> None:
    """Prefer reproducible kernels over cuDNN autotuning and TF32 speedups."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def rescale_accumulated_gradients(
    model: torch.nn.Module, samples: int, expected: int
) -> None:
    """Correct a partial accumulation window to its true sample-mean gradient."""
    if samples != expected:
        factor = expected / samples
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(factor)
