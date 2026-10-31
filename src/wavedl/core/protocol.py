"""Strict UTF-8 JSON envelopes. Sequence numbers belong to one attempt."""

from __future__ import annotations

import json
import math
import numbers
import re
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta
from uuid import UUID


# Bound container nesting independently of CPython's recursion limit. This uses
# the browser's 32-level safety budget, not its root-at-zero node-depth counting.
MAX_JSON_DEPTH = 32


class ProtocolError(ValueError):
    """A record violates the supported protocol."""


class MalformedJSON(ProtocolError):
    """Invalid UTF-8 or JSON syntax; readers may retain a bounded diagnostic."""


def finite_or_none(value: object) -> object:
    """Replace non-finite reals with ``None`` so the payload is strict JSON.

    Dispatch is on ``numbers``, not ``float``, because NumPy's scalar types are
    not uniformly float subclasses: ``np.float64`` is one, ``np.float32`` is
    not. An ``isinstance(value, float)`` test therefore passes
    ``np.float32("nan")`` straight through to an encoder that then refuses it
    with "Object of type float32 is not JSON serializable". NumPy registers its
    scalar types in the ``numbers`` ABCs on import, so this catches them
    without importing NumPy into the contracts package -- when NumPy is absent
    there are no NumPy values for the check to miss.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)  # np.int32 is Integral but not JSON-serialisable
    if isinstance(value, numbers.Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(item) for item in value]
    return value


def _reject_constant(token: str):
    raise ProtocolError(f"Non-finite JSON number: {token}")


def _finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ProtocolError(f"JSON number out of range: {token}")
    return value


def _bounded_int(token: str) -> int:
    """Convert an integer literal, keeping the failure inside ProtocolError.

    CPython caps int/str conversion at ``sys.get_int_max_str_digits()`` (4300
    by default, CVE-2020-10735), and ``int()`` past that limit raises a bare
    ``ValueError`` that is neither a ``JSONDecodeError`` nor a ``UnicodeError``
    -- so an oversized literal on the wire would escape the decoder's error
    boundary and reach the caller as an unclassified crash.
    """
    try:
        return int(token)
    except ValueError as exc:
        raise MalformedJSON(f"JSON integer rejected: {exc}") from exc


def _check_json_depth(data: str) -> None:
    """Bound container nesting before decoding, ignoring quoted/escaped syntax."""
    depth = 0
    quoted = False
    escaped = False
    for character in data:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in "[{":
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise MalformedJSON("JSON nesting exceeds supported depth")
        elif character in "]}":
            depth -= 1
    # JSON syntax, including mismatched containers/strings, remains json.loads'
    # responsibility; this scan only supplies its interpreter-independent bound.


def strict_loads(data: bytes | bytearray | str) -> object:
    try:
        if isinstance(data, (bytes, bytearray)):
            data = data.decode("utf-8")
        _check_json_depth(data)
        return json.loads(
            data,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
            parse_int=_bounded_int,
        )
    except ProtocolError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MalformedJSON(str(exc)) from exc
    except (ArithmeticError, ValueError, RecursionError) as exc:
        # Any remaining numeric conversion failure stays inside the boundary.
        raise MalformedJSON(str(exc)) from exc


@dataclass(frozen=True)
class Envelope:
    run_id: str
    attempt: int
    type: str
    seq: int
    ts: str
    payload: dict[str, object]
    protocol: str = "wavedl-jsonl"
    version: int = 1

    def __post_init__(self):
        if self.protocol != "wavedl-jsonl":
            raise ProtocolError("Unsupported protocol")
        if type(self.version) is not int or self.version != 1:
            raise ProtocolError("Unsupported version")
        for name, minimum in (("attempt", 1), ("seq", 0)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ProtocolError(f"Invalid {name}")
        try:
            if not isinstance(self.run_id, str):
                raise ValueError("run_id must be a UUID string")
            UUID(self.run_id)
        except ValueError as exc:
            raise ProtocolError("Invalid run_id") from exc
        if not isinstance(self.type, str) or not self.type:
            raise ProtocolError("Invalid event type")
        if not isinstance(self.payload, dict):
            raise ProtocolError("payload must be an object")
        if not all(isinstance(key, str) for key in self.payload):
            raise ProtocolError("payload keys must be strings")
        if not isinstance(self.ts, str) or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)",
            self.ts,
        ):
            raise ProtocolError("ts must be UTC RFC 3339")
        try:
            stamp = datetime.fromisoformat(self.ts.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ProtocolError("Invalid timestamp") from exc
        if stamp.utcoffset() != timedelta(0):
            raise ProtocolError("ts must be UTC")


def decode(data: bytes | str) -> Envelope:
    value = strict_loads(data)
    if not isinstance(value, dict):
        raise ProtocolError("Envelope must be an object")
    names = {item.name for item in fields(Envelope)}
    if not names <= value.keys():
        raise ProtocolError(f"Missing envelope fields: {sorted(names - value.keys())}")
    try:
        return Envelope(**{name: value[name] for name in names})
    except (TypeError, ValueError) as exc:
        raise ProtocolError(str(exc)) from exc


def encode(envelope: Envelope) -> bytes:
    try:
        data = json.dumps(
            finite_or_none(asdict(envelope)),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        decode(data)
    except (
        TypeError,
        ValueError,
        UnicodeError,
        ArithmeticError,
        RecursionError,
    ) as exc:
        raise ProtocolError(str(exc)) from exc
    return data + b"\n"
