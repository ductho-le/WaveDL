# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Standalone run registration wire schemas."""

from __future__ import annotations

from typing import Annotated

from pydantic import MISSING, Field

from wavedl_daemon.http.schemas.common import WireModel


class AdoptRequest(WireModel):
    path: str
    attempt: Annotated[int, Field(ge=1)] | MISSING = MISSING


class AdoptionResponse(WireModel):
    plan_id: str
    run_id: str
    request_id: str
    operation: str
    attempt: int
    backend: str
    state: str
    uncertainty: bool
    submission_requested: bool
    cancellation_requested: bool
    backend_status: str
    adopted: bool
