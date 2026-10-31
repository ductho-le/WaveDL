"""Keep browser authoring metadata aligned with the dependency-light CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wavedl.core.training_cli import build_training_parser, training_defaults


def test_browser_metadata_covers_daemon_supported_training_options():
    """Every supported destination has the correct default, type and choices."""
    root = Path(__file__).resolve().parents[2]
    metadata = json.loads(
        (root / "src/wavedl_daemon/web/shared/training-options.json").read_text()
    )
    fields = {field["name"]: field for field in metadata}
    defaults = training_defaults()
    actions = {
        action.dest: action
        for action in build_training_parser()._actions
        # allow_pickle and selection_metric are CLI/HPO-only, never daemon options.
        if action.dest
        not in {
            "help",
            "list_models",
            "config",
            "output_dir",
            "allow_pickle",
            "selection_metric",
        }
        and action.help != argparse.SUPPRESS
    }
    assert set(fields) == set(actions)
    assert len(fields) == len(metadata)
    for name, action in actions.items():
        kind = (
            "boolean"
            if isinstance(action, argparse._StoreTrueAction)
            else "integer"
            if action.type is int
            else "number"
            if action.type is float
            else "string"
        )
        assert fields[name] == {
            "name": name,
            "kind": kind,
            "multiple": action.nargs == "+",
            "choices": list(action.choices or []),
            "default": defaults[name],
            "description": action.help,
        }
