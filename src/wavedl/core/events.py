"""Typed payloads and the resume rollback marker; no projection side effects."""

from __future__ import annotations

import math
import types
import warnings
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Literal, Self, get_args, get_origin, get_type_hints

from wavedl.core.protocol import Envelope, ProtocolError


class UnknownEventWarning(RuntimeWarning):
    """An extension event was retained but has no local payload schema."""


def _typed(value: object, annotation: object) -> object:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is types.UnionType:
        for option in args:
            try:
                return _typed(value, option)
            except ProtocolError:
                continue
        raise ProtocolError("Value does not match any allowed type")
    if origin is Literal:
        if any(type(value) is type(item) and value == item for item in args):
            return value
    elif origin is list and isinstance(value, list):
        return [_typed(item, args[0]) for item in value]
    elif origin is dict and isinstance(value, dict):
        return {
            _typed(key, args[0]): _typed(item, args[1]) for key, item in value.items()
        }
    elif annotation is object:
        return value
    elif annotation is float and type(value) in (int, float):
        try:
            if math.isfinite(value):
                return float(value)
        except OverflowError as exc:
            raise ProtocolError("Numeric value is out of range") from exc
    elif isinstance(annotation, type) and is_dataclass(annotation):
        return annotation.from_dict(value)
    elif type(value) is annotation:
        return value
    raise ProtocolError(f"Expected {annotation}, got {type(value).__name__}")


class Payload:
    @classmethod
    def from_dict(cls, value: object) -> Self:
        if not isinstance(value, dict):
            raise ProtocolError("Payload must be an object")
        annotations = get_type_hints(cls)
        try:
            result = cls(
                **{
                    item.name: _typed(value[item.name], annotations[item.name])
                    for item in fields(cls)
                    if item.name in value
                }
            )
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"Invalid {cls.__name__}: {exc}") from exc
        for name in (
            "step",
            "size",
            "protocol_min",
            "protocol_max",
            "source_attempt",
            "source_seq",
        ):
            number = getattr(result, name, None)
            minimum = (
                1 if name in {"protocol_min", "protocol_max", "source_attempt"} else 0
            )
            if number is not None and number < minimum:
                raise ProtocolError(f"Invalid {name}")
        if (
            isinstance(result, HelloPayload)
            and result.protocol_min > result.protocol_max
        ):
            raise ProtocolError("Reversed protocol range")
        if isinstance(result, SupersedePayload) and not result.generation:
            raise ProtocolError("Missing restored generation")
        return result


@dataclass(frozen=True)
class HelloPayload(Payload):
    protocol_min: int
    protocol_max: int
    runtime_version: str
    python: str
    device: Literal["cpu", "mps", "cuda"]
    capabilities: list[str]
    interpreter: str
    container: str | None = None
    allocated_devices: list[str] = field(default_factory=list)
    prefix: str | None = None
    base_prefix: str | None = None


@dataclass(frozen=True)
class StatePayload(Payload):
    state: Literal[
        "starting", "running", "stopping", "completed", "failed", "cancelled"
    ]
    message: str = ""


@dataclass(frozen=True)
class MetricPayload(Payload):
    name: str
    value: float | None
    step: int
    unit: str = ""
    metrics: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: object) -> Self:
        result = super().from_dict(value)
        for name in (
            "train_loss",
            "val_loss",
            "best_val_loss",
            "lr",
            "r2",
            "pearson",
            "patience",
            "patience_ctr",
            "epoch_time",
            "elapsed_time",
            "grad_norm",
            "mae_avg",
            "val_r2",
            "val_pearson",
            "val_mae_avg",
            "total_time",
        ):
            if name in result.metrics:
                _typed(result.metrics[name], float | None)
        for name in ("epoch", "total_epochs", "patience_counter", "max_patience"):
            if name in result.metrics:
                _typed(result.metrics[name], int)
        if "mae_per_param" in result.metrics:
            _typed(result.metrics["mae_per_param"], list[float | None])
        for name, metric in result.metrics.items():
            if name.startswith("MAE_Phys_P"):
                _typed(metric, float | None)
        if "per_param_mae" in result.metrics:
            _typed(result.metrics["per_param_mae"], dict[str, float | None])
        return result


@dataclass(frozen=True)
class ArtifactPayload(Payload):
    kind: Literal["checkpoint", "plot", "history", "other"]
    path: str
    sha256: str
    size: int
    label: str = ""
    generation: str | None = None
    step: int | None = None


@dataclass(frozen=True)
class ExitPayload(Payload):
    code: int
    status: Literal["completed", "failed", "cancelled"]
    message: str = ""
    artifacts: list[ArtifactPayload] = field(default_factory=list)


@dataclass(frozen=True)
class SupersedePayload(Payload):
    source_attempt: int
    step: int
    generation: str
    reason: str = "resume"
    source_seq: int | None = None


@dataclass(frozen=True)
class LogPayload(Payload):
    stream: Literal["stdout", "stderr", "system"]
    level: Literal["debug", "info", "warning", "error"]
    message: str


@dataclass(frozen=True)
class WarningPayload(Payload):
    code: str
    message: str
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ErrorPayload(WarningPayload):
    recoverable: bool = False


PAYLOADS = {
    "hello": HelloPayload,
    "state": StatePayload,
    "metric": MetricPayload,
    "artifact": ArtifactPayload,
    "exit": ExitPayload,
    "supersede": SupersedePayload,
    "log": LogPayload,
    "warning": WarningPayload,
    "error": ErrorPayload,
}


def decode_payload(envelope: Envelope) -> Payload | None:
    schema = PAYLOADS.get(envelope.type)
    if schema is None:
        warnings.warn(
            f"Unknown event type: {envelope.type}", UnknownEventWarning, stacklevel=2
        )
        return None
    return schema.from_dict(envelope.payload)


def validate_supersession(
    payload: SupersedePayload,
    known_steps: set[tuple[int, int]],
    *,
    known_attempts: set[int] | None = None,
) -> None:
    if (payload.source_attempt, payload.step) not in known_steps and not (
        payload.step == 0 and payload.source_attempt in (known_attempts or set())
    ):
        raise ProtocolError("Supersession references an absent step")
