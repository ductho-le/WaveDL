import pytest

from wavedl_daemon.execution import WorkBudget


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_workers", 0),
        ("max_workers", True),
        ("max_pending", -1),
        ("max_pending", 1.5),
        ("max_command_bytes", 0),
        ("max_output_bytes", -1),
        ("timeout_seconds", 0),
        ("timeout_seconds", float("nan")),
        ("timeout_seconds", float("inf")),
        ("timeout_seconds", True),
    ],
)
def test_invalid_budget(field, value):
    with pytest.raises(ValueError):
        WorkBudget(**{field: value})


def test_zero_pending_and_output_allowed():
    assert WorkBudget(max_pending=0, max_output_bytes=0).max_pending == 0
