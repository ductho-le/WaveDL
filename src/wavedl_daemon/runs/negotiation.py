# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Attempt-scoped runtime handshake validation with caller-supplied clocks.

Scheduler waiting never starts a deadline. A current positive RUNNING observation
is required to certify expiry; observation loss is not a protocol verdict. This
tracker performs no I/O and its monotonic start is never persisted across boots.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from uuid import UUID

from wavedl.core.events import HelloPayload, decode_payload
from wavedl.core.protocol import Envelope, ProtocolError, decode
from wavedl_daemon.backends.contracts import BackendObservation, ObservationStatus


class StaleNegotiation(ValueError):
    """An event or observation belongs to a different attempt/run."""


class NegotiationStatus(StrEnum):
    WAITING = "waiting"
    ACCEPTED = "accepted"
    FAILED = "failed"


def _clock(value: float) -> float:
    try:
        valid = type(value) in {int, float} and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("Monotonic clock must be a finite nonnegative number")
    return float(value)


def canonical_event(event: Envelope) -> str:
    """Validate strict finite JSON even for directly constructed envelopes."""
    if not isinstance(event, Envelope):
        raise ProtocolError("Runtime event must be an Envelope")
    try:
        encoded = json.dumps(
            asdict(event), allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        checked = decode(encoded)
        decode_payload(checked)
        return encoded
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ProtocolError(f"Invalid runtime event: {exc}") from exc


def checked_hello(
    event: Envelope, *, protocol_min: int = 1, protocol_max: int = 1
) -> tuple[str, HelloPayload, int]:
    encoded = canonical_event(event)
    if event.type != "hello" or event.seq != 0:
        raise ProtocolError("First runtime event must be hello at seq 0")
    hello = HelloPayload.from_dict(event.payload)
    lower = max(protocol_min, hello.protocol_min)
    upper = min(protocol_max, hello.protocol_max)
    if lower > upper or not lower <= event.version <= upper:
        raise ProtocolError("Runtime hello has an incompatible protocol range")
    return encoded, hello, event.version


@dataclass(frozen=True)
class NegotiationResult:
    status: NegotiationStatus
    reason: str
    hello_json: str | None = None
    protocol_version: int | None = None
    running_since: float | None = None
    handshake_timed_out: bool = False


class RuntimeNegotiation:
    """Enforce first-event hello and retain an immutable accepted provenance."""

    def __init__(
        self,
        run_id: str,
        attempt: int,
        *,
        timeout_seconds: float = 30,
        protocol_min: int = 1,
        protocol_max: int = 1,
    ):
        UUID(run_id)
        if type(attempt) is not int or attempt < 1:
            raise ValueError("attempt must be a positive integer")
        if (
            type(protocol_min) is not int
            or type(protocol_max) is not int
            or not 1 <= protocol_min <= protocol_max
        ):
            raise ValueError("Invalid daemon protocol range")
        self.timeout_seconds = _clock(timeout_seconds)
        if self.timeout_seconds == 0:
            raise ValueError("Handshake deadline must be positive")
        self.run_id, self.attempt = run_id, attempt
        self.protocol_min, self.protocol_max = protocol_min, protocol_max
        self._running_since: float | None = None
        self._last_clock: float | None = None
        self._observation: BackendObservation | None = None
        self._event_json: str | None = None
        self._result = NegotiationResult(
            NegotiationStatus.WAITING, "Awaiting runtime hello"
        )

    @property
    def result(self) -> NegotiationResult:
        return self._result

    def _time(self, now: float) -> float:
        value = _clock(now)
        if self._last_clock is not None and value < self._last_clock:
            raise ValueError("Monotonic clock moved backwards")
        self._last_clock = value
        return value

    def _fail(self, reason: str, *, timed_out: bool = False) -> NegotiationResult:
        self._result = NegotiationResult(
            NegotiationStatus.FAILED,
            reason,
            self._result.hello_json,
            self._result.protocol_version,
            self._running_since,
            timed_out,
        )
        return self._result

    def reject(self, reason: str) -> NegotiationResult:
        """Latch an external approval-policy failure without fabricating hello."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Runtime rejection requires a visible reason")
        if self._result.status == NegotiationStatus.FAILED:
            return self._result
        return self._fail(reason)

    def observe_backend(
        self, observation: BackendObservation | None, *, now: float
    ) -> NegotiationResult:
        if observation is not None and observation.attempt != self.attempt:
            raise StaleNegotiation("Observation is not for the active attempt")
        now = self._time(now)
        self._observation = observation
        if self._result.status != NegotiationStatus.WAITING:
            return self._result
        if observation is not None and observation.status == ObservationStatus.RUNNING:
            if self._running_since is None:
                self._running_since = now
            if now - self._running_since >= self.timeout_seconds:
                return self._fail(
                    "Running backend has no runtime hello before handshake deadline",
                    timed_out=True,
                )
        self._result = NegotiationResult(
            NegotiationStatus.WAITING,
            "Awaiting runtime hello"
            if self._running_since is not None
            else "Awaiting backend execution and runtime hello",
            running_since=self._running_since,
        )
        return self._result

    def accept_event(
        self, event: Envelope, *, now: float | None = None
    ) -> NegotiationResult:
        if not isinstance(event, Envelope):
            return self._fail("Runtime event must be an Envelope")
        if event.run_id != self.run_id or event.attempt != self.attempt:
            raise StaleNegotiation("Event is not for the full active run/attempt")
        if now is not None:
            self.observe_backend(self._observation, now=now)
        if self._result.status == NegotiationStatus.FAILED:
            return self._result
        try:
            encoded = canonical_event(event)
            if self._event_json is not None:
                if event.type == "hello":
                    if encoded != self._event_json:
                        return self._fail("Conflicting repeated runtime hello")
                elif event.seq == 0:
                    return self._fail("Runtime seq 0 cannot replace hello")
                return self._result
            encoded, _, version = checked_hello(
                event,
                protocol_min=self.protocol_min,
                protocol_max=self.protocol_max,
            )
        except ProtocolError as exc:
            return self._fail(str(exc))
        self._event_json = encoded
        self._result = NegotiationResult(
            NegotiationStatus.ACCEPTED,
            "Runtime hello accepted",
            json.dumps(event.payload, allow_nan=False, sort_keys=True),
            version,
            self._running_since,
        )
        return self._result
