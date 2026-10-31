"""Real CPU folds honour deterministic mode without leaking global state."""

import inspect
import random

import numpy as np
import pytest
import torch

from unit_tests import test_cv_option_fidelity as cv_fixtures
from wavedl.models.base import BaseModel
from wavedl.utils import cross_validation


cpu_cv = cv_fixtures.cpu_cv


class StochasticCvModel(BaseModel):
    def __init__(self, in_shape, out_size, **kwargs):
        super().__init__(in_shape, out_size, **kwargs)
        self.linear = torch.nn.Linear(in_shape[0], out_size)
        self.dropout = torch.nn.Dropout(0.25)
        self.register_buffer(
            "offset", torch.tensor(random.random() + np.random.random())
        )

    def forward(self, x):
        return self.linear(self.dropout(x.flatten(1))) + self.offset


def flags():
    return (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state()


def assert_rng_state(before):
    python_state, numpy_state, torch_state = before
    assert random.getstate() == python_state
    current = np.random.get_state()
    assert current[0] == numpy_state[0]
    np.testing.assert_array_equal(current[1], numpy_state[1])
    assert current[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def options(output):
    rows = np.arange(12, dtype=np.float32)
    return {
        "X": np.stack([rows / 12, rows**2 / 144], axis=1),
        "y": (rows * 2 + 1).reshape(-1, 1),
        "model_name": "cnn",
        "in_shape": (2,),
        "out_size": 1,
        "folds": 2,
        "epochs": 2,
        "batch_size": 2,
        "workers": 0,
        "seed": 42,
        "device": torch.device("cpu"),
        "output_dir": str(output),
    }


@pytest.mark.parametrize("workers", [0, 1])
def test_actual_two_fold_cpu_repeatability_and_caller_restoration(
    tmp_path, monkeypatch, workers
):
    from wavedl.models.registry import MODEL_REGISTRY

    monkeypatch.setitem(MODEL_REGISTRY, "cnn", StochasticCvModel)
    initial_flags = flags()
    initial_rng = rng_state()
    observed = []
    actual_fold = cross_validation.train_fold

    def train_fold(**kwargs):
        observed.append(flags())
        return actual_fold(**kwargs)

    monkeypatch.setattr(cross_validation, "train_fold", train_fold)
    first = cross_validation.run_cross_validation(
        **(options(tmp_path / "first") | {"workers": workers}), deterministic=True
    )
    assert flags() == initial_flags
    assert_rng_state(initial_rng)
    # Perturb all model RNG sources: deterministic CV must not depend on them.
    random.random()
    np.random.random()
    torch.rand(3)
    second_rng = rng_state()
    second = cross_validation.run_cross_validation(
        **(options(tmp_path / "second") | {"workers": workers}), deterministic=True
    )
    assert flags() == initial_flags
    assert_rng_state(second_rng)
    assert observed == [(True, True, False, True, False, False)] * 4
    assert first["fold_results"] == second["fold_results"]
    for fold in (1, 2):
        first_state = torch.load(
            tmp_path / "first" / f"fold_{fold}" / "model.pth", weights_only=True
        )
        second_state = torch.load(
            tmp_path / "second" / f"fold_{fold}" / "model.pth", weights_only=True
        )
        assert first_state.keys() == second_state.keys()
        for key in first_state:
            assert torch.equal(first_state[key], second_state[key])


def test_failure_restores_rng_and_backend_state(tmp_path, monkeypatch):
    import wavedl.models

    initial_flags, initial_rng = flags(), rng_state()

    def fail(*args, **kwargs):
        assert flags() == (True, True, False, True, False, False)
        random.random()
        np.random.random()
        torch.rand(3)
        raise RuntimeError("injected fold allocation failure")

    monkeypatch.setattr(wavedl.models, "build_model", fail)
    with pytest.raises(RuntimeError, match="injected fold"):
        cross_validation.run_cross_validation(**options(tmp_path), deterministic=True)
    assert flags() == initial_flags
    assert_rng_state(initial_rng)


def test_direct_helper_default_does_not_change_backend_policy(tmp_path, monkeypatch):
    import wavedl.models

    assert (
        inspect.signature(cross_validation.run_cross_validation)
        .parameters["deterministic"]
        .default
        is False
    )
    initial_flags = flags()

    def fail(*args, **kwargs):
        assert flags() == initial_flags
        raise RuntimeError("default policy observed")

    monkeypatch.setattr(wavedl.models, "build_model", fail)
    with pytest.raises(RuntimeError, match="default policy"):
        cross_validation.run_cross_validation(**options(tmp_path))
    assert flags() == initial_flags


def test_spawn_safe_worker_seeds_python_and_numpy_from_torch(monkeypatch):
    before = rng_state()
    monkeypatch.setattr(torch, "initial_seed", lambda: 2**32 + 17)
    try:
        cross_validation._seed_cv_worker(9)
        assert random.random() == random.Random(17).random()
        assert np.random.random() == np.random.RandomState(17).random_sample()
    finally:
        random.setstate(before[0])
        np.random.set_state(before[1])
        torch.set_rng_state(before[2])


def test_cli_deterministic_option_reaches_actual_folds_and_publications(
    cpu_cv, monkeypatch
):
    """CLI opt-in reaches the scoped policy and completes published CPU folds."""
    actual_context = cross_validation._deterministic_cv
    requested_seeds = []

    def observe_context(seed):
        requested_seeds.append(seed)
        return actual_context(seed)

    # Ordinary CLI setup also changes process-wide RNG/backend policy. Keep the
    # whole test scoped, while observing only contexts requested by CLI calls.
    with actual_context(13):
        monkeypatch.setattr(cross_validation, "_deterministic_cv", observe_context)
        cpu_cv("--seed", "47")
        assert requested_seeds == []
        folds = cpu_cv("--seed", "47", "--deterministic")
        assert requested_seeds == [47]
        assert len(folds) == 2
