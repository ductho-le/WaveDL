"""Dependency-free supersession-aware telemetry history.

Journal artifacts are observations, not proof of published checkpoint validity.
Only an independently verified generation may acknowledge an absent boundary.
Fresh checkpoint restores include source_seq for causal replay. Legacy markers
without it retain step-only compatibility and cannot exclude delayed same-step data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

from wavedl.core.events import ArtifactPayload, MetricPayload, SupersedePayload
from wavedl.core.protocol import Envelope, ProtocolError


@dataclass(frozen=True)
class MetricRecord:
    run_id: str
    attempt: int
    seq: int
    step: int
    name: str
    value: float | None
    unit: str = ""
    metrics: dict[str, object] = field(default_factory=dict)
    source_generation: str | None = None
    active: bool = True


def checked_metric(payload: dict[str, object]) -> MetricPayload:
    result = MetricPayload.from_dict(payload)
    try:
        json.dumps(result.metrics, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ProtocolError("Metric details must be finite JSON") from exc
    return result


def validate_boundary(
    event: Envelope,
    payload: SupersedePayload,
    known_steps: set[tuple[int, int]],
    known_attempts: set[int],
    published: set[tuple[int, int, str]],
) -> None:
    if payload.source_attempt >= event.attempt:
        raise ProtocolError("Supersession must restore an earlier attempt")
    if (
        (payload.source_attempt, payload.step) not in known_steps
        and (payload.source_attempt, payload.step, payload.generation) not in published
        and not (payload.step == 0 and payload.source_attempt in known_attempts)
    ):
        raise ProtocolError("Supersession references an absent step")


class MetricHistory:
    """Fold one run; resumed points replace the abandoned tail, never freeze it."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.records: list[MetricRecord] = []
        self.known_attempts: set[int] = set()
        self.published: set[tuple[int, int, str]] = set()
        self.generations: dict[int, str] = {}
        self.cursors: dict[int, int] = {}
        self.lineage: dict[int, tuple[int, int, int | None]] = {}
        self.head_attempt = 0

    def _restore(self, attempt: int) -> None:
        bounds: dict[int, tuple[int | None, int | None]] = {}
        bound = source_seq = None
        while True:
            bounds[attempt] = (bound, source_seq)
            if attempt not in self.lineage:
                break
            attempt, step, source_seq = self.lineage[attempt]
            bound = step if bound is None else min(bound, step)
        restored = {}
        for record in self.records:
            if record.attempt not in bounds:
                continue
            bound, source_seq = bounds[record.attempt]
            if bound is not None and record.step > bound:
                continue
            if source_seq is not None and record.seq > source_seq:
                continue
            key = (record.step, record.name)
            prior = restored.get(key)
            if prior is None or (record.attempt, record.seq) > (
                prior.attempt,
                prior.seq,
            ):
                restored[key] = record
        identities = {(record.attempt, record.seq) for record in restored.values()}
        self.records = [
            replace(record, active=(record.attempt, record.seq) in identities)
            for record in self.records
        ]

    def acknowledge_generation(
        self, source_attempt: int, step: int, generation: str
    ) -> None:
        payload = SupersedePayload.from_dict(
            {"source_attempt": source_attempt, "step": step, "generation": generation}
        )
        self.published.add((payload.source_attempt, payload.step, payload.generation))
        self.known_attempts.add(source_attempt)

    def apply(self, event: Envelope) -> None:
        if event.run_id != self.run_id:
            raise ProtocolError("Metric history received a foreign run")
        if event.seq <= self.cursors.get(event.attempt, -1):
            return
        if event.type == "metric":
            payload = checked_metric(event.payload)
            active = event.attempt >= self.head_attempt
            self.records = [
                replace(record, active=False)
                if active
                and record.step == payload.step
                and record.name == payload.name
                else record
                for record in self.records
            ]
            self.records.append(
                MetricRecord(
                    self.run_id,
                    event.attempt,
                    event.seq,
                    payload.step,
                    payload.name,
                    payload.value,
                    payload.unit,
                    payload.metrics,
                    self.generations.get(event.attempt),
                    active,
                )
            )
            self.known_attempts.add(event.attempt)
            self.head_attempt = max(self.head_attempt, event.attempt)
        elif event.type == "supersede":
            payload = SupersedePayload.from_dict(event.payload)
            validate_boundary(
                event,
                payload,
                {
                    (record.attempt, record.step)
                    for record in self.records
                    if record.attempt != payload.source_attempt
                    or payload.source_seq is None
                    or record.seq <= payload.source_seq
                },
                self.known_attempts,
                self.published,
            )
            if event.attempt in self.lineage:
                raise ProtocolError("Attempt already has a supersession boundary")
            self.lineage[event.attempt] = (
                payload.source_attempt,
                payload.step,
                payload.source_seq,
            )
            if event.attempt >= self.head_attempt:
                self._restore(event.attempt)
                self.head_attempt = event.attempt
            self.generations[event.attempt] = payload.generation
        elif event.type == "hello":
            self.known_attempts.add(event.attempt)
            self.head_attempt = max(self.head_attempt, event.attempt)
        elif event.type == "artifact":
            payload = ArtifactPayload.from_dict(event.payload)
            if payload.kind == "checkpoint":
                self.known_attempts.add(event.attempt)
        self.cursors[event.attempt] = event.seq

    def metrics(self, *, active_only: bool = True) -> list[MetricRecord]:
        return [record for record in self.records if record.active or not active_only]
