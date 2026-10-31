# Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""Health, launch capability exchange, and session wire contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import MISSING

from wavedl_daemon.http.schemas.common import WireModel


class MonitoringError(WireModel):
    source: str
    error: str


class PlanError(WireModel):
    plan_id: str
    error: str


class HealthResponse(WireModel):
    # Hosted diagnostics are absent, rather than null, in the service-less app.
    registered_backends: list[str] | MISSING = MISSING
    monitoring_errors: list[MonitoringError] | MISSING = MISSING
    filesystem_work_outstanding: int | MISSING = MISSING
    restart_watch_refusals: int | MISSING = MISSING
    restart_monitoring_errors: list[PlanError] | MISSING = MISSING
    dispatch_errors: list[PlanError] | MISSING = MISSING
    daemon_id: str
    build_id: str
    protocol_min: int
    protocol_max: int


class BootstrapRequest(WireModel):
    code: str


class SessionGrantResponse(WireModel):
    access_token: str
    token_type: Literal["Bearer"]
    expires_at: int
    daemon_id: str
