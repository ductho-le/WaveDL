"""
Unit Tests for Strict-JSON Metric Emission
===========================================

Tests that the structured metrics line emitted for daemon/browser consumption is
valid JSON under RFC 8259, which forbids the bare ``Infinity``/``NaN`` tokens
that Python's ``json`` module emits by default.

``best_val_loss`` is initialised to ``float("inf")`` and is only updated during
checkpoint handling, so the first epoch of every fresh run carries a non-finite
value. A diverged run additionally produces ``NaN`` losses and metrics. Python
round-trips both, but browser ``JSON.parse`` and other strict parsers reject
them, so the payload must be sanitised before serialisation.

**Tested Components**:
    - train.py: _finite_or_none

Author: Ductho Le (ductho.le@outlook.com)
"""

import json

import numpy as np
import pytest


# ==============================================================================
# SANITISER TESTS
# ==============================================================================


class TestFiniteOrNone:
    """Tests for the non-finite float sanitiser."""

    def test_infinity_becomes_none(self):
        """Positive infinity is replaced with None."""
        from wavedl.train import _finite_or_none

        assert _finite_or_none(float("inf")) is None

    def test_negative_infinity_becomes_none(self):
        """Negative infinity is replaced with None."""
        from wavedl.train import _finite_or_none

        assert _finite_or_none(float("-inf")) is None

    def test_nan_becomes_none(self):
        """NaN is replaced with None."""
        from wavedl.train import _finite_or_none

        assert _finite_or_none(float("nan")) is None

    def test_finite_float_unchanged(self):
        """Ordinary floats pass through untouched."""
        from wavedl.train import _finite_or_none

        assert _finite_or_none(0.25) == 0.25

    def test_int_and_str_unchanged(self):
        """Non-float scalars pass through untouched."""
        from wavedl.train import _finite_or_none

        assert _finite_or_none(7) == 7
        assert _finite_or_none("cnn") == "cnn"
        assert _finite_or_none(None) is None

    def test_numpy_non_finite_becomes_none(self):
        """NumPy floating scalars are handled, not just Python floats."""
        from wavedl.train import _finite_or_none

        assert _finite_or_none(np.float32("nan")) is None
        assert _finite_or_none(np.float64("inf")) is None

    def test_numpy_finite_becomes_python_float(self):
        """Finite NumPy scalars are converted so json.dumps can serialise them."""
        from wavedl.train import _finite_or_none

        result = _finite_or_none(np.float32(1.5))
        assert result == pytest.approx(1.5)
        assert isinstance(result, float)

    def test_nested_dict_sanitised(self):
        """Non-finite values nested in dicts are replaced."""
        from wavedl.train import _finite_or_none

        result = _finite_or_none({"a": float("inf"), "b": {"c": float("nan")}})

        assert result == {"a": None, "b": {"c": None}}

    def test_nested_list_sanitised(self):
        """Non-finite values nested in lists are replaced."""
        from wavedl.train import _finite_or_none

        result = _finite_or_none([1.0, float("nan"), [float("inf")]])

        assert result == [1.0, None, [None]]

    def test_tuple_becomes_list(self):
        """Tuples are normalised to lists so the result is JSON-serialisable."""
        from wavedl.train import _finite_or_none

        assert _finite_or_none((1.0, float("inf"))) == [1.0, None]


# ==============================================================================
# STRICT SERIALISATION TESTS
# ==============================================================================


class TestStrictJsonSerialisation:
    """Tests that sanitised payloads serialise under strict JSON rules."""

    def _first_epoch_metrics(self):
        """Metrics as emitted on epoch 1 of a fresh run, before any checkpoint."""
        return {
            "epoch": 1,
            "total_epochs": 100,
            "train_loss": 0.42,
            "val_loss": 0.51,
            "best_val_loss": float("inf"),  # not yet updated on epoch 1
            "r2": 0.8,
            "pearson": 0.9,
            "grad_norm": 1.2,
            "lr": 0.001,
            "mae_avg": 0.3,
            "mae_per_param": [0.1, 0.2],
            "epoch_time": 12.5,
            "total_time": 12.5,
            "patience_counter": 0,
            "max_patience": 20,
        }

    def test_raw_payload_is_not_strict_json(self):
        """Guard: the unsanitised payload is exactly what strict parsers reject."""
        raw = json.dumps(self._first_epoch_metrics())

        assert "Infinity" in raw
        with pytest.raises(ValueError):
            json.loads(raw, parse_constant=_reject_constant)

    def test_sanitised_payload_serialises_strictly(self):
        """allow_nan=False succeeds once the payload is sanitised."""
        from wavedl.train import _finite_or_none

        encoded = json.dumps(
            _finite_or_none(self._first_epoch_metrics()), allow_nan=False
        )

        assert "Infinity" not in encoded
        assert "NaN" not in encoded

    def test_sanitised_payload_survives_strict_parse(self):
        """A strict decoder accepts the sanitised payload and sees null."""
        from wavedl.train import _finite_or_none

        encoded = json.dumps(
            _finite_or_none(self._first_epoch_metrics()), allow_nan=False
        )
        decoded = json.loads(encoded, parse_constant=_reject_constant)

        assert decoded["best_val_loss"] is None
        assert decoded["epoch"] == 1
        assert decoded["val_loss"] == pytest.approx(0.51)

    def test_diverged_run_serialises_strictly(self):
        """A diverged run producing NaN losses still emits valid JSON."""
        from wavedl.train import _finite_or_none

        metrics = self._first_epoch_metrics()
        metrics.update(
            train_loss=float("nan"),
            val_loss=float("nan"),
            r2=float("nan"),
            mae_per_param=[float("nan"), 0.2],
        )

        decoded = json.loads(
            json.dumps(_finite_or_none(metrics), allow_nan=False),
            parse_constant=_reject_constant,
        )

        assert decoded["train_loss"] is None
        assert decoded["r2"] is None
        assert decoded["mae_per_param"] == [None, 0.2]


def _reject_constant(name):
    """Emulate a strict (non-Python) JSON parser rejecting Infinity/NaN tokens."""
    raise ValueError(f"non-finite token rejected by strict parser: {name}")
