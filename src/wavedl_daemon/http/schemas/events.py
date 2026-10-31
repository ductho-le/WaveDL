# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Frozen producer push contract; protocol decoding owns envelope validation."""

from __future__ import annotations

from pydantic import ConfigDict, Field, JsonValue

from wavedl_daemon.http.schemas.common import WireModel


class EventPushRequest(WireModel):
    # Producers may be a different installed version. Historically unknown batch
    # keys were ignored; forbidding them would break the frozen push contract.
    model_config = ConfigDict(extra="ignore")

    attempt: int = Field(ge=1, le=2**63 - 1)
    # Scalars and malformed objects must reach decode/reject_runtime_events (409),
    # not request-model validation (400), which would skip durable rejection.
    events: list[JsonValue] = Field(min_length=1, max_length=1024)


class EventAck(WireModel):
    attempt: int
    seq: int
