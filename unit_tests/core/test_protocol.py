import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from wavedl.core.protocol import (
    MAX_JSON_DEPTH,
    Envelope,
    ProtocolError,
    decode,
    encode,
    finite_or_none,
    strict_loads,
)


RUN_ID = "00000000-0000-4000-8000-000000000001"


def event(**changes):
    values = {
        "run_id": RUN_ID,
        "attempt": 1,
        "type": "hello",
        "seq": 0,
        "ts": "2026-09-06T00:00:00Z",
        "payload": {},
    }
    return Envelope(**(values | changes))


def test_round_trip_and_unknown_fields():
    """Extra envelope fields survive decoding as ignored extensions."""
    original = event(payload={"future": {"label": "café"}})
    wire = asdict(original) | {"future_envelope": True}
    assert decode(json.dumps(wire)) == original
    assert decode(encode(original)) == original
    assert encode(original).endswith(b"\n")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", 2),
        ("version", True),
        ("protocol", "other"),
        ("seq", -1),
        ("seq", True),
        ("attempt", -1),
        ("attempt", 0),
        ("ts", "yesterday"),
        ("ts", "2026-02-30T00:00:00Z"),
        ("run_id", "not-a-uuid"),
        ("payload", []),
    ],
)
def test_invalid_known_field(field, value):
    """Invalid known fields raise the shared exception, without coercion."""
    wire = asdict(event()) | {field: value}
    with pytest.raises(ProtocolError):
        decode(json.dumps(wire))


@pytest.mark.parametrize("wire", [b"{", b"\xff\n", b"[]", b"{}"])
def test_malformed_or_missing_envelope(wire):
    with pytest.raises(ProtocolError):
        decode(wire)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_input_is_rejected(token):
    wire = encode(event(payload={"loss": 0})).replace(
        b'"loss":0', f'"loss":{token}'.encode()
    )
    with pytest.raises(ProtocolError):
        decode(wire)


def test_recursive_sanitizing_produces_strict_json():
    """Epoch-one infinity and nested non-finite metrics become null."""
    payload = {
        "best_val_loss": float("inf"),
        "nested": [float("nan"), {"negative": -float("inf")}],
    }
    expected = {"best_val_loss": None, "nested": [None, {"negative": None}]}
    assert finite_or_none(payload) == expected
    wire = encode(event(payload=payload))
    assert decode(wire).payload == expected
    assert b"Infinity" not in wire and b"NaN" not in wire
    assert payload["best_val_loss"] == float("inf")


def test_numpy_scalars_are_sanitized_without_importing_numpy():
    """np.float32 is not a float subclass; an isinstance(float) test misses it."""
    numpy = pytest.importorskip("numpy")

    payload = {
        "loss": numpy.float32("nan"),
        "best": numpy.float64("inf"),
        "lr": numpy.float32(0.5),
        "epoch": numpy.int32(3),
    }
    sanitized = finite_or_none(payload)

    assert sanitized == {"loss": None, "best": None, "lr": 0.5, "epoch": 3}
    assert isinstance(sanitized["lr"], float)
    assert isinstance(sanitized["epoch"], int)
    assert decode(encode(event(payload=payload))).payload == sanitized


def test_booleans_and_strings_survive_sanitizing():
    """bool is an Integral subclass; it must not be widened to 0/1."""
    assert finite_or_none({"ok": True, "name": "cnn", "n": 7, "z": None}) == {
        "ok": True,
        "name": "cnn",
        "n": 7,
        "z": None,
    }


def test_oversized_integer_stays_inside_the_error_boundary():
    """int() past sys.get_int_max_str_digits() raises a bare ValueError."""
    wire = encode(event(payload={"loss": 0})).replace(
        b'"loss":0', b'"loss":' + b"9" * (sys.get_int_max_str_digits() + 1)
    )

    with pytest.raises(ProtocolError):
        decode(wire)


def test_shared_import_loads_no_ml_gui_or_daemon():
    """A fresh interpreter can import contracts with every heavy import blocked."""
    script = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "torch", "numpy", "PySide6", "wavedl_daemon", "wavedl_gui"
        } or fullname in {"wavedl.models", "wavedl.utils"}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from wavedl.core.protocol import Envelope
import wavedl
assert callable(wavedl.__getattr__)
assert "build_model" in dir(wavedl)
"""
    env = os.environ | {"PYTHONPATH": str(Path("src").resolve())}
    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_deeply_nested_wire_is_a_protocol_error():
    wire = b"[" * 2000 + b"0" + b"]" * 2000
    with pytest.raises(ProtocolError):
        strict_loads(wire)


def test_bytearray_json_preserves_mutable_transport_input():
    wire = bytearray(json.dumps({"message": "sóng 波 🌊"}, ensure_ascii=False).encode())
    original = wire[:]
    assert strict_loads(wire) == {"message": "sóng 波 🌊"}
    assert wire == original


@pytest.mark.parametrize(
    "wire", [bytearray(b'"\xff"'), bytearray(b"[" * 33 + b"0" + b"]" * 33)]
)
def test_bytearray_rejects_invalid_utf8_and_excessive_depth(wire):
    with pytest.raises(ProtocolError):
        strict_loads(wire)


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("object_layers", [False, True])
def test_json_container_depth_has_an_explicit_boundary(binary, object_layers):
    opening = '{"next":[' if object_layers else "["
    closing = "]}" if object_layers else "]"
    layers = MAX_JSON_DEPTH // 2 if object_layers else MAX_JSON_DEPTH
    wire = opening * layers + "0" + closing * layers
    strict_loads(wire.encode() if binary else wire)
    excessive = opening + wire + closing
    with pytest.raises(ProtocolError, match="JSON nesting exceeds supported depth"):
        strict_loads(excessive.encode() if binary else excessive)


@pytest.mark.parametrize(
    "value", ["[{" * 100, '\\"[{' * 100, "\\\\[{" * 100, '"]}\\\\\\"[{']
)
def test_quoted_brackets_and_escapes_do_not_count_as_json_depth(value):
    assert strict_loads(json.dumps({value: [value]})) == {value: [value]}


@pytest.mark.parametrize("value", [None, True, False, 0, 1.5, "plain", [], {}])
def test_json_depth_check_preserves_root_values(value):
    assert strict_loads(json.dumps(value)) == value


@pytest.mark.parametrize("wire", ['{"x": [}', '["unterminated\\', "][", '{"x": 0'])
def test_json_depth_scan_does_not_replace_syntax_validation(wire):
    with pytest.raises(ProtocolError):
        strict_loads(wire)


def test_cyclic_payload_is_a_protocol_error():
    payload = {}
    payload["cycle"] = payload
    with pytest.raises(ProtocolError):
        encode(event(payload=payload))
