import pytest

from wavedl.core.events import SupersedePayload, validate_supersession
from wavedl.core.metric_history import MetricHistory
from wavedl.core.protocol import Envelope, ProtocolError


RUN_ID = "00000000-0000-4000-8000-000000000001"


def event(seq, *, attempt=1, kind="metric", **payload):
    if kind == "metric":
        payload = {"name": "loss", "value": 1.0, "step": seq} | payload
    return Envelope(RUN_ID, attempt, kind, seq, "2026-10-03T00:00:00Z", payload)


def test_resume_replaces_tail_and_tracks_generation():
    history = MetricHistory(RUN_ID)
    for step in (1, 2, 3):
        history.apply(event(step))
    marker = event(
        0,
        attempt=2,
        kind="supersede",
        source_attempt=1,
        step=1,
        generation="gen-000001",
    )
    history.apply(marker)
    history.apply(marker)
    history.apply(event(1, attempt=2, step=2, value=5.0))
    history.apply(event(2, attempt=2, step=2, value=6.0))
    assert [(r.step, r.value) for r in history.metrics()] == [(1, 1.0), (2, 6.0)]
    assert history.metrics()[-1].source_generation == "gen-000001"
    assert len(history.metrics(active_only=False)) == 5
    history.apply(
        event(
            0,
            attempt=3,
            kind="supersede",
            source_attempt=1,
            step=3,
            generation="gen-000002",
        )
    )
    assert [r.attempt for r in history.metrics()] == [1, 1, 1]


@pytest.mark.parametrize("bad", [True, "1", float("nan"), float("inf")])
def test_invalid_metric_is_atomic(bad):
    history = MetricHistory(RUN_ID)
    with pytest.raises(ProtocolError):
        history.apply(event(1, value=bad))
    assert history.records == []
    assert history.cursors == {}


def test_empty_checkpoint_requires_known_source_and_artifacts_are_not_authority():
    history = MetricHistory(RUN_ID)
    marker = event(
        0,
        attempt=2,
        kind="supersede",
        source_attempt=1,
        step=0,
        generation="gen-000001",
    )
    with pytest.raises(ProtocolError):
        history.apply(marker)
    history.apply(event(0, kind="hello"))
    history.apply(marker)
    with pytest.raises(ProtocolError):
        history.apply(
            event(
                0,
                attempt=3,
                kind="supersede",
                source_attempt=1,
                step=8,
                generation="gen-000008",
            )
        )
    history.acknowledge_generation(1, 8, "gen-000008")
    history.apply(
        event(
            0,
            attempt=3,
            kind="supersede",
            source_attempt=1,
            step=8,
            generation="gen-000008",
        )
    )


def test_contract_step_zero_is_explicit_and_other_absent_steps_reject():
    payload = SupersedePayload(1, 0, "gen-000001")
    with pytest.raises(ProtocolError):
        validate_supersession(payload, set())
    validate_supersession(payload, set(), known_attempts={1})
    with pytest.raises(ProtocolError):
        validate_supersession(
            SupersedePayload(1, 3, "gen-000001"), set(), known_attempts={1}
        )


def branched_events():
    return [
        event(1, step=1),
        event(2, step=2),
        event(3, step=3),
        event(
            0, attempt=2, kind="supersede", source_attempt=1, step=1, generation="gen-1"
        ),
        event(1, attempt=2, step=2, value=20.0),
        event(
            0, attempt=3, kind="supersede", source_attempt=1, step=3, generation="gen-2"
        ),
        event(1, attempt=3, step=4, value=30.0),
        event(
            0, attempt=4, kind="supersede", source_attempt=2, step=2, generation="gen-3"
        ),
        event(1, attempt=4, step=3, value=40.0),
        event(4, attempt=1, step=5, value=99.0),
        event(2, attempt=2, step=2, value=99.0),
    ]


def test_restore_follows_source_branch_and_late_points_stay_inactive():
    history = MetricHistory(RUN_ID)
    for item in branched_events():
        history.apply(item)
    assert [(r.step, r.value) for r in history.metrics()] == [
        (1, 1.0),
        (2, 20.0),
        (3, 40.0),
    ]
    assert all(not r.active for r in history.records if r.value == 99.0)


def causal_events():
    return [
        event(1, step=1),
        event(2, step=2),
        event(
            0,
            attempt=2,
            kind="supersede",
            source_attempt=1,
            step=2,
            generation="gen-1",
            source_seq=2,
        ),
        event(1, attempt=2, step=3, value=20.0),
        event(3, attempt=1, step=2, value=99.0),
        event(
            0,
            attempt=3,
            kind="supersede",
            source_attempt=2,
            step=3,
            generation="gen-2",
            source_seq=1,
        ),
        event(2, attempt=2, step=3, value=98.0),
    ]


def test_source_sequence_cutoff_makes_numeric_replay_causal():
    live, replay = MetricHistory(RUN_ID), MetricHistory(RUN_ID)
    batch = causal_events()
    for item in batch:
        live.apply(item)
    for item in sorted(batch, key=lambda item: (item.attempt, item.seq)):
        replay.apply(item)
    assert live.metrics() == replay.metrics()
    assert [(r.step, r.value) for r in live.metrics()] == [
        (1, 1.0),
        (2, 1.0),
        (3, 20.0),
    ]


@pytest.mark.parametrize("bad", [-1, True, "2", 1.5])
def test_invalid_source_sequence_is_rejected(bad):
    with pytest.raises(ProtocolError):
        SupersedePayload.from_dict(
            {"source_attempt": 1, "step": 2, "generation": "gen-1", "source_seq": bad}
        )
