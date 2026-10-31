"""CV auto-selection probes shared memory; explicit worker choices remain owned."""

import multiprocessing.shared_memory

import pytest

from unit_tests import test_cv_option_fidelity as cv_fixtures
from wavedl.utils import cross_validation


cpu_cv = cv_fixtures.cpu_cv


def observe_loaders(monkeypatch):
    """Record requested counts but use bounded genuine main-process loading."""
    counts = []
    original = cross_validation.DataLoader

    def loader(*args, **kwargs):
        counts.append(kwargs["num_workers"])
        return original(*args, **(kwargs | {"num_workers": 0}))

    monkeypatch.setattr(cross_validation, "DataLoader", loader)
    return counts


@pytest.mark.parametrize("error", [PermissionError, OSError])
def test_actual_cli_cv_auto_workers_fall_back_and_publish(
    cpu_cv, monkeypatch, caplog, error
):
    counts = observe_loaders(monkeypatch)
    probes = []

    def denied(**kwargs):
        probes.append(kwargs)
        raise error("controlled shared-memory creation denial")

    monkeypatch.setattr(multiprocessing.shared_memory, "SharedMemory", denied)
    result = cpu_cv("--workers", "-1")
    assert len(result) == 2
    assert counts == [0, 0, 0, 0]
    assert probes == [{"create": True, "size": 1}]
    assert "falling back to workers=0" in caplog.text


@pytest.mark.parametrize("workers", [0, 3])
def test_explicit_workers_preserved_without_shared_memory_probe(
    cpu_cv, monkeypatch, workers
):
    counts = observe_loaders(monkeypatch)

    def forbidden(**kwargs):
        pytest.fail("Explicit worker counts must not probe shared memory")

    monkeypatch.setattr(multiprocessing.shared_memory, "SharedMemory", forbidden)
    cpu_cv("--workers", str(workers))
    assert counts == [workers] * 4


def test_controlled_success_selects_bounded_count_and_closes_owned_probe(
    cpu_cv, monkeypatch
):
    counts = observe_loaders(monkeypatch)
    actions = []

    class Probe:
        def close(self):
            actions.append("close")

        def unlink(self):
            actions.append("unlink")

    def create(**kwargs):
        assert kwargs == {"create": True, "size": 1}
        actions.append("create")
        return Probe()

    monkeypatch.setattr(multiprocessing.shared_memory, "SharedMemory", create)
    monkeypatch.setattr(cross_validation.os, "cpu_count", lambda: 12)
    monkeypatch.setattr(cross_validation.torch.cuda, "is_available", lambda: False)
    cpu_cv("--workers", "-1")
    assert counts == [10] * 4
    assert actions == ["create", "close", "unlink"]


@pytest.mark.parametrize("failing_action", ["close", "unlink"])
def test_cleanup_error_falls_back_with_unlink_attempt(
    cpu_cv, monkeypatch, caplog, failing_action
):
    counts = observe_loaders(monkeypatch)
    actions = []

    class Probe:
        def close(self):
            actions.append("close")
            if failing_action == "close":
                raise OSError("controlled close failure")

        def unlink(self):
            actions.append("unlink")
            if failing_action == "unlink":
                raise PermissionError("controlled unlink failure")

    monkeypatch.setattr(
        multiprocessing.shared_memory, "SharedMemory", lambda **kwargs: Probe()
    )
    cpu_cv("--workers", "-1")
    assert counts == [0] * 4
    assert actions == ["close", "unlink"]
    assert "falling back to workers=0" in caplog.text
