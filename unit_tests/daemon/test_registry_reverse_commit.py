"""Crash proof for the opposite projection/cursor write ordering."""

from pathlib import Path

from unit_tests.daemon.test_registry import RUN_ID, event
from wavedl_daemon.storage.registry import Registry


REVERSE_VICTIM = """
import sys, time
sys.path.insert(0, {src!r})
from wavedl.core.protocol import Envelope
from wavedl_daemon.storage.registry import Registry

original_write_record = Registry._write_record

def cursor_before_record(self, record):
    assert self._connection.in_transaction
    self._write_cursor(record.run_id, record.attempt, 0)
    print("cursor written, projection pending", flush=True)
    time.sleep(300)
    original_write_record(self, record)

# Exercise the existing transaction and real cursor writer, but inject the
# opposite helper ordering. This is not an assembled runtime/HTTP crash test.
Registry._write_record = cursor_before_record
registry = Registry({path!r})
registry.apply_events({run_id!r}, 1, [Envelope({run_id!r}, 1, "hello", 0,
    "2026-09-05T12:00:00Z", {{}})])
"""


def test_cursor_first_crash_rolls_back_projection_and_replays_once(
    tmp_path, paused_process
):
    """A crash after cursor-before-projection must not lose the event."""
    path = tmp_path / "registry.db"
    Registry(path).close()
    script = REVERSE_VICTIM.format(
        src=str(Path("src").resolve()), path=str(path), run_id=RUN_ID
    )
    with paused_process(script, "cursor written, projection pending") as victim:
        victim.kill()
        victim.wait(timeout=10)

    with Registry(path) as registry:
        assert registry.get(RUN_ID) is None
        assert registry.cursor(RUN_ID, 1) == -1
        assert registry.apply_events(RUN_ID, 1, [event(0, type="hello")]) == 0
        assert registry.get(RUN_ID).event_count == 1
        assert registry.cursor(RUN_ID, 1) == 0
        assert registry.apply_events(RUN_ID, 1, [event(0, type="hello")]) == 0
        assert registry.get(RUN_ID).event_count == 1
