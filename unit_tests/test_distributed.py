"""
Unit Tests for Distributed Training Utilities
==============================================

Comprehensive tests for:
- broadcast_early_stop: early stopping synchronization
- broadcast_value: scalar value broadcasting
- sync_tensor: tensor synchronization
- get_world_info: world information retrieval
- print_rank0: rank-0 only printing

Author: Ductho Le (ductho.le@outlook.com)
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from wavedl.utils.distributed import (
    broadcast_early_stop,
    broadcast_value,
    get_world_info,
    print_rank0,
    sync_tensor,
)


# ==============================================================================
# BROADCAST EARLY STOP TESTS
# ==============================================================================
class TestBroadcastEarlyStop:
    """Tests for the broadcast_early_stop function."""

    def test_returns_true_when_should_stop(self, mock_accelerator):
        """Test returns True when should_stop is True."""
        result = broadcast_early_stop(True, mock_accelerator)
        assert result is True

    def test_returns_false_when_should_not_stop(self, mock_accelerator):
        """Test returns False when should_stop is False."""
        result = broadcast_early_stop(False, mock_accelerator)
        assert result is False

    def test_single_process_no_broadcast(self, mock_accelerator):
        """Test that single-process mode doesn't call dist.broadcast."""
        mock_accelerator.num_processes = 1

        with patch("torch.distributed.broadcast") as mock_broadcast:
            result = broadcast_early_stop(True, mock_accelerator)
            mock_broadcast.assert_not_called()

        assert result is True

    def test_multi_process_calls_broadcast(self, mock_accelerator_multi_gpu):
        """Test that multi-process mode calls dist.broadcast."""
        # Need to handle the tensor being on the right device
        mock_accelerator_multi_gpu.device = torch.device("cpu")

        with patch("torch.distributed.broadcast") as mock_broadcast:
            broadcast_early_stop(True, mock_accelerator_multi_gpu)
            mock_broadcast.assert_called_once()

    def test_creates_correct_tensor(self, mock_accelerator):
        """Test that correct tensor is created for broadcasting."""
        # Run and verify no exceptions
        result = broadcast_early_stop(True, mock_accelerator)
        assert isinstance(result, bool)


# ==============================================================================
# BROADCAST VALUE TESTS
# ==============================================================================
class TestBroadcastValue:
    """Tests for the broadcast_value function."""

    def test_broadcasts_integer(self, mock_accelerator):
        """Test broadcasting integer values."""
        result = broadcast_value(42, mock_accelerator)

        assert result == 42
        assert isinstance(result, int)

    def test_broadcasts_float(self, mock_accelerator):
        """Test broadcasting float values."""
        result = broadcast_value(3.14, mock_accelerator)

        assert result == pytest.approx(3.14, rel=1e-6)
        assert isinstance(result, float)

    def test_preserves_integer_type(self, mock_accelerator):
        """Test that integers are returned as integers."""
        result = broadcast_value(100, mock_accelerator)

        assert isinstance(result, int)
        assert result == 100

    def test_preserves_float_type(self, mock_accelerator):
        """Test that floats are returned as floats."""
        result = broadcast_value(1.5, mock_accelerator)

        assert isinstance(result, float)

    def test_single_process_no_broadcast(self, mock_accelerator):
        """Test that single-process mode doesn't call dist.broadcast."""
        mock_accelerator.num_processes = 1

        with patch("torch.distributed.broadcast") as mock_broadcast:
            result = broadcast_value(42, mock_accelerator)
            mock_broadcast.assert_not_called()

        assert result == 42

    def test_zero_value(self, mock_accelerator):
        """Test broadcasting zero values."""
        result_int = broadcast_value(0, mock_accelerator)
        result_float = broadcast_value(0.0, mock_accelerator)

        assert result_int == 0
        assert result_float == 0.0

    def test_negative_value(self, mock_accelerator):
        """Test broadcasting negative values."""
        result = broadcast_value(-42, mock_accelerator)

        assert result == -42

    def test_large_value(self, mock_accelerator):
        """Test broadcasting large values."""
        result = broadcast_value(10**15, mock_accelerator)

        assert result == 10**15


# ==============================================================================
# SYNC TENSOR TESTS
# ==============================================================================
class TestSyncTensor:
    """Tests for the sync_tensor function."""

    def test_sum_reduction(self, mock_accelerator):
        """Test tensor synchronization with sum reduction."""
        tensor = torch.tensor([1.0, 2.0, 3.0])

        # Mock accelerator.reduce to return the tensor unchanged (single process)
        mock_accelerator.reduce = MagicMock(return_value=tensor)

        result = sync_tensor(tensor, mock_accelerator, reduction="sum")

        mock_accelerator.reduce.assert_called_once()
        assert result is tensor

    def test_mean_reduction(self, mock_accelerator):
        """Test tensor synchronization with mean reduction."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        mock_accelerator.reduce = MagicMock(return_value=tensor)

        sync_tensor(tensor, mock_accelerator, reduction="mean")

        mock_accelerator.reduce.assert_called_once()
        call_args = mock_accelerator.reduce.call_args
        assert (
            call_args.kwargs.get("reduction") == "mean"
            or call_args[1].get("reduction") == "mean"
        )

    def test_invalid_reduction_raises(self, mock_accelerator):
        """Test that invalid reduction type raises ValueError."""
        tensor = torch.tensor([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Invalid reduction"):
            sync_tensor(tensor, mock_accelerator, reduction="invalid")

    def test_valid_reductions(self, mock_accelerator):
        """Test all valid reduction types."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        mock_accelerator.reduce = MagicMock(return_value=tensor)

        valid_reductions = ["sum", "mean", "max", "min"]

        for reduction in valid_reductions:
            result = sync_tensor(tensor, mock_accelerator, reduction=reduction)
            assert result is not None

    def test_default_reduction_is_sum(self, mock_accelerator):
        """Test that default reduction is sum."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        mock_accelerator.reduce = MagicMock(return_value=tensor)

        sync_tensor(tensor, mock_accelerator)

        call_args = mock_accelerator.reduce.call_args
        reduction_arg = call_args.kwargs.get("reduction") or call_args[1].get(
            "reduction", "sum"
        )
        assert reduction_arg == "sum"


# ==============================================================================
# GET WORLD INFO TESTS
# ==============================================================================
class TestGetWorldInfo:
    """Tests for the get_world_info function."""

    def test_returns_dict(self, mock_accelerator):
        """Test that returns a dictionary."""
        result = get_world_info(mock_accelerator)

        assert isinstance(result, dict)

    def test_contains_required_keys(self, mock_accelerator):
        """Test that result contains all required keys."""
        result = get_world_info(mock_accelerator)

        required_keys = ["world_size", "rank", "local_rank", "is_main", "device"]
        for key in required_keys:
            assert key in result

    def test_correct_values_single_process(self, mock_accelerator):
        """Test correct values for single process setup."""
        result = get_world_info(mock_accelerator)

        assert result["world_size"] == 1
        assert result["rank"] == 0
        assert result["local_rank"] == 0
        assert result["is_main"] is True

    def test_correct_values_multi_process(self, mock_accelerator_multi_gpu):
        """Test correct values for multi-process setup."""
        result = get_world_info(mock_accelerator_multi_gpu)

        assert result["world_size"] == 4
        assert result["is_main"] is True  # Our fixture is rank 0

    def test_device_is_string(self, mock_accelerator):
        """Test that device is returned as string."""
        result = get_world_info(mock_accelerator)

        assert isinstance(result["device"], str)


# ==============================================================================
# PRINT RANK0 TESTS
# ==============================================================================
class TestPrintRank0:
    """Tests for the print_rank0 function."""

    def test_prints_on_main_process(self, mock_accelerator):
        """Test that message is printed on main process."""
        mock_accelerator.is_main_process = True

        with patch("builtins.print") as mock_print:
            print_rank0("test message", mock_accelerator)
            mock_print.assert_called_once_with("test message")

    def test_silent_on_non_main_process(self, mock_accelerator):
        """Test that message is not printed on non-main process."""
        mock_accelerator.is_main_process = False

        with patch("builtins.print") as mock_print:
            print_rank0("test message", mock_accelerator)
            mock_print.assert_not_called()

    def test_uses_logger_when_provided(self, mock_accelerator):
        """Test that logger is used when provided."""
        mock_accelerator.is_main_process = True
        mock_logger = MagicMock()

        print_rank0("test message", mock_accelerator, logger=mock_logger)

        mock_logger.info.assert_called_once_with("test message")

    def test_logger_silent_on_non_main(self, mock_accelerator):
        """Test that logger is not called on non-main process."""
        mock_accelerator.is_main_process = False
        mock_logger = MagicMock()

        print_rank0("test message", mock_accelerator, logger=mock_logger)

        mock_logger.info.assert_not_called()


# ==============================================================================
# INTEGRATION-STYLE TESTS
# ==============================================================================
class TestDistributedIntegration:
    """Integration-style tests for distributed utilities."""

    def test_early_stop_flow(self, mock_accelerator):
        """Test typical early stopping flow."""
        patience = 5
        patience_counter = 0
        best_loss = float("inf")

        # Simulate training loop
        losses = [1.0, 0.9, 0.8, 0.85, 0.86, 0.87, 0.88, 0.89]

        for loss in losses:
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1

            should_stop = patience_counter >= patience

            # Broadcast decision
            stop = broadcast_early_stop(should_stop, mock_accelerator)

            if stop:
                break

        # Should have stopped after patience was exhausted
        assert patience_counter >= patience

    def test_value_sync_for_hyperparameters(self, mock_accelerator):
        """Test value synchronization for hyperparameters."""
        # Simulate syncing computed hyperparameters
        learning_rate = 0.001
        batch_size = 128
        num_samples = 10000

        synced_lr = broadcast_value(learning_rate, mock_accelerator)
        synced_bs = broadcast_value(batch_size, mock_accelerator)
        synced_ns = broadcast_value(num_samples, mock_accelerator)

        assert synced_lr == pytest.approx(learning_rate)
        assert synced_bs == batch_size
        assert synced_ns == num_samples

    def test_world_info_for_logging(self, mock_accelerator):
        """Test world info usage for logging."""
        info = get_world_info(mock_accelerator)

        # Simulate creating a log message
        log_msg = (
            f"[Rank {info['rank']}/{info['world_size']}] Training on {info['device']}"
        )

        assert "Rank 0" in log_msg
        assert "cpu" in log_msg.lower()


# ==============================================================================
# EDGE CASES
# ==============================================================================
class TestDistributedEdgeCases:
    """Tests for edge cases in distributed utilities."""

    def test_broadcast_very_large_integer(self, mock_accelerator):
        """Test broadcasting very large integers."""
        large_int = 10**18  # Within int64 range
        result = broadcast_value(large_int, mock_accelerator)

        assert result == large_int

    def test_broadcast_very_small_float(self, mock_accelerator):
        """Test broadcasting very small floats."""
        small_float = 1e-30
        result = broadcast_value(small_float, mock_accelerator)

        assert result == pytest.approx(small_float, rel=1e-6)

    def test_sync_empty_tensor(self, mock_accelerator):
        """Test syncing empty tensor."""
        tensor = torch.tensor([])
        mock_accelerator.reduce = MagicMock(return_value=tensor)

        # Should not raise
        result = sync_tensor(tensor, mock_accelerator)
        assert result.numel() == 0

    def test_sync_multidimensional_tensor(self, mock_accelerator):
        """Test syncing multidimensional tensor."""
        tensor = torch.randn(4, 3, 32, 32)
        mock_accelerator.reduce = MagicMock(return_value=tensor)

        result = sync_tensor(tensor, mock_accelerator)

        assert result.shape == (4, 3, 32, 32)


def test_cpu_barrier_omits_device_ids_and_preserves_accelerator(monkeypatch):
    from types import SimpleNamespace

    from accelerate import Accelerator

    from wavedl.utils import distributed

    accelerator = SimpleNamespace(
        num_processes=2,
        device=torch.device("cpu"),
        local_process_index=1,
        wait_for_everyone=MagicMock(),
    )
    original_attributes = vars(accelerator).copy()
    original_accelerate_barrier = Accelerator.wait_for_everyone
    barrier = MagicMock()
    monkeypatch.setattr(distributed.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed.dist, "barrier", barrier)
    # Reproduce the host capability that caused upstream to select MPS, while
    # keeping the actual process-group device explicitly on CPU.
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    distributed.wait_for_everyone(accelerator)

    barrier.assert_called_once_with()
    accelerator.wait_for_everyone.assert_not_called()
    assert vars(accelerator) == original_attributes
    assert Accelerator.wait_for_everyone is original_accelerate_barrier


@pytest.mark.parametrize(
    ("device", "processes", "available", "initialized"),
    [
        ("cpu", 1, True, True),
        ("cpu", 2, True, False),
        ("cpu", 2, False, False),
        ("cuda:0", 2, True, True),
        ("mps", 2, True, True),
    ],
)
def test_non_cpu_group_wait_delegates_to_accelerate(
    monkeypatch, device, processes, available, initialized
):
    from types import SimpleNamespace

    from wavedl.utils import distributed

    accelerator = SimpleNamespace(
        num_processes=processes,
        device=torch.device(device),
        wait_for_everyone=MagicMock(),
    )
    barrier = MagicMock()
    monkeypatch.setattr(distributed.dist, "is_available", lambda: available)
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: initialized)
    monkeypatch.setattr(distributed.dist, "barrier", barrier)

    distributed.wait_for_everyone(accelerator)

    accelerator.wait_for_everyone.assert_called_once_with()
    barrier.assert_not_called()


def test_barrier_helper_is_exported():
    from wavedl.utils import distributed, wait_for_everyone

    assert wait_for_everyone is distributed.wait_for_everyone
