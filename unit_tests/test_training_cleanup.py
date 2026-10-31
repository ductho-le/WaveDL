"""Prepared-model lifetime must end before distributed backend shutdown."""

import traceback
import weakref
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("failure", [None, ValueError, KeyboardInterrupt])
def test_training_cleanup_releases_frame_and_accelerator_models_first(
    monkeypatch, failure
):
    pytest.importorskip("accelerate")
    from wavedl import train

    calls = []
    references = []

    class PreparedModel:
        def __init__(self):
            self.cycle = self

    accelerator = SimpleNamespace(models=[])

    def run(args, accelerator, journal):
        model = PreparedModel()
        references.append(weakref.ref(model))
        accelerator.models.append(model)
        calls.append("training")
        if failure is not None:
            raise failure("original training failure")

    def free_memory():
        accelerator.models.clear()
        calls.append("release")

    def end_training():
        assert references[0]() is None
        calls.append("destroy")

    accelerator.free_memory = free_memory
    accelerator.end_training = end_training
    monkeypatch.setattr(train, "_run_training", run)

    if failure is None:
        train._run_training_and_cleanup(None, accelerator, None)
    else:
        with pytest.raises(failure, match="original training failure") as caught:
            train._run_training_and_cleanup(None, accelerator, None)
        # Clearing retained locals must not remove the original failure site.
        assert "run" in {
            frame.name for frame in traceback.extract_tb(caught.value.__traceback__)
        }

    assert calls == ["training", "release", "destroy"]


def test_training_cleanup_failure_is_not_silently_successful(monkeypatch):
    pytest.importorskip("accelerate")
    from wavedl import train

    monkeypatch.setattr(train, "_run_training", lambda *args: None)

    def end_training():
        raise RuntimeError("backend cleanup failed")

    accelerator = SimpleNamespace(free_memory=lambda: None, end_training=end_training)
    with pytest.raises(ExceptionGroup, match="backend cleanup failed") as caught:
        train._run_training_and_cleanup(None, accelerator, None)
    assert isinstance(caught.value.exceptions[0], RuntimeError)


@pytest.mark.parametrize("primary", [ValueError, KeyboardInterrupt])
@pytest.mark.parametrize("release_fails", [False, True])
def test_combined_failures_keep_primary_and_explicit_cleanup_evidence(
    monkeypatch, primary, release_fails
):
    pytest.importorskip("accelerate")
    from wavedl import train
    from wavedl.runtime import training_lifecycle

    calls = []
    original = primary("original training failure")
    cleanup = RuntimeError("native cleanup failure")

    def run(*args):
        raise original

    def free_memory():
        calls.append("release")
        if release_fails:
            raise cleanup

    def end_training():
        calls.append("end")
        raise cleanup

    def destroy():
        calls.append("destroy")

    events = []
    journal = SimpleNamespace(
        log=lambda *args, **kwargs: None,
        emit=lambda kind, payload: events.append((kind, payload)),
        close=lambda: calls.append("close"),
    )
    accelerator = SimpleNamespace(free_memory=free_memory, end_training=end_training)
    monkeypatch.setattr(train, "_run_training", run)
    monkeypatch.setattr(train.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(train.dist, "destroy_process_group", destroy)
    with pytest.raises(BaseExceptionGroup) as caught, training_lifecycle(journal):
        train._run_training_and_cleanup(None, accelerator, journal)

    assert caught.value.exceptions == (original, cleanup)
    message = str(caught.value)
    assert "original training failure" in message
    assert "native cleanup failure" in message
    assert ("backend shutdown skipped" in message) == release_fails
    assert calls == (
        ["release", "close"]
        if release_fails
        else ["release", "end", "destroy", "close"]
    )
    assert events[-1][0] == "exit"
    assert events[-1][1].status == "failed"
    assert events[-1][1].code == 1
    assert "original training failure" in events[-1][1].message
    assert "native cleanup failure" in events[-1][1].message


def test_end_and_fallback_shutdown_failures_are_both_retained(monkeypatch):
    pytest.importorskip("accelerate")
    from wavedl import train

    tracker_error = RuntimeError("tracker finish failed")
    backend_error = OSError("destroy backend failed")

    def end_training():
        raise tracker_error

    def destroy():
        raise backend_error

    monkeypatch.setattr(train, "_run_training", lambda *args: None)
    monkeypatch.setattr(train.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(train.dist, "destroy_process_group", destroy)
    accelerator = SimpleNamespace(free_memory=lambda: None, end_training=end_training)
    with pytest.raises(ExceptionGroup) as caught:
        train._run_training_and_cleanup(None, accelerator, None)
    assert caught.value.exceptions == (tracker_error, backend_error)
    assert "tracker finish failed" in str(caught.value)
    assert "destroy backend failed" in str(caught.value)


def test_main_does_not_destroy_group_after_prepared_release_failure(monkeypatch):
    pytest.importorskip("accelerate")
    from wavedl import train
    from wavedl.core.training_cli import parse_training_args

    calls = []

    def free_memory():
        raise RuntimeError("prepared references still retained")

    accelerator = SimpleNamespace(
        free_memory=free_memory, end_training=lambda: calls.append("end")
    )
    journal = SimpleNamespace(
        log=lambda *args, **kwargs: None,
        emit=lambda *args: None,
        close=lambda: calls.append("close"),
    )
    monkeypatch.setattr(train, "parse_args", lambda: parse_training_args([]))
    monkeypatch.setattr(train, "Accelerator", lambda **kwargs: accelerator)
    monkeypatch.setattr(train, "open_training_runtime", lambda *args: journal)
    monkeypatch.setattr(train, "_run_training", lambda *args: None)
    monkeypatch.setattr(train.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        train.dist, "destroy_process_group", lambda: calls.append("destroy")
    )
    with pytest.raises(ExceptionGroup, match="backend shutdown skipped"):
        train.main()
    assert calls == ["close"]
