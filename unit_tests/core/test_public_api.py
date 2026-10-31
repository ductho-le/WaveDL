import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_import_does_not_load_the_ml_stack():
    """`import wavedl` must not cost the ML stack; the whole boundary rests on it."""
    script = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "torch", "numpy", "PySide6", "wavedl_daemon", "wavedl_gui"
        }:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
import wavedl
assert wavedl.__version__
assert "build_model" in dir(wavedl)
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        env=os.environ | {"PYTHONPATH": str(Path("src").resolve())},
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("name", "owner"),
    [
        ("build_model", "wavedl.models"),
        ("get_model", "wavedl.models"),
        ("list_models", "wavedl.models"),
        ("register_model", "wavedl.models"),
        ("load_test_data", "wavedl.utils"),
        ("load_training_data", "wavedl.utils"),
        ("prepare_data", "wavedl.utils"),
    ],
)
def test_public_exports_remain_lazy(monkeypatch, name, owner):
    """The existing top-level export names still resolve to their owner."""
    import wavedl

    calls = []
    sentinel = object()

    class Module:
        pass

    setattr(Module, name, sentinel)

    def load(name):
        calls.append(name)
        return Module()

    monkeypatch.setattr(wavedl, "import_module", load)
    original = wavedl.__dict__.pop(name, None)
    try:
        assert getattr(wavedl, name) is sentinel
        assert getattr(wavedl, name) is sentinel
        assert calls == [owner]
    finally:
        if original is None:
            wavedl.__dict__.pop(name, None)
        else:
            wavedl.__dict__[name] = original
    with pytest.raises(AttributeError):
        wavedl.__getattr__("missing")
