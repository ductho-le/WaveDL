"""Monitoring work per lap is bounded without fleet/history alias starvation."""

from wavedl_daemon.runs.runtime_service import ServiceSession


def test_active_attempt_precedes_rotating_historical_window():
    attempts = list(range(1, 1002))
    seen = set()
    for epoch in range(501):
        batch = ServiceSession._attempt_batch(attempts, 1001, epoch)
        assert batch[0] == 1001
        assert len(batch) <= 3
        assert batch[1:] == sorted(batch[1:])
        seen.update(batch)
    assert seen == set(attempts)


def test_abandoned_attempt_gaps_and_absent_active_directory():
    assert ServiceSession._attempt_batch([1, 9, 20], 20, 0) == [20, 1, 9]
    assert ServiceSession._attempt_batch([1, 9], 20, 0) == [1, 9]
    assert ServiceSession._attempt_batch([], 20, 0) == []


def test_monitor_errors_remain_bounded_without_evicting_on_existing_key():
    session = ServiceSession(None)
    for index in range(128):
        session._monitor_error(str(index), "ProtocolError")
    session._monitor_error("127", "ValueError")
    assert len(session.errors) == 128 and "0" in session.errors
    session._monitor_error("new", "ProtocolError")
    assert len(session.errors) == 128 and "0" not in session.errors
