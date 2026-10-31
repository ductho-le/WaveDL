# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Platform process supervision without training or GUI dependencies."""

from .base import ProcessResult, ProcessSpec, SupervisedProcess, Supervisor


__all__ = ["ProcessResult", "ProcessSpec", "SupervisedProcess", "Supervisor"]
