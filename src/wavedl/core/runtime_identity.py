"""Exact probe-attested Python identity; never infer environments from symlinks."""

from __future__ import annotations

import json
import os

from wavedl.core.protocol import strict_loads


def load_runtime_identity_json(raw: bytes | str) -> object:
    """Reject ambiguous duplicate fields before any proof is canonicalized."""

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate runtime identity field")
            result[key] = value
        return result

    strict_loads(raw)
    return json.loads(raw, object_pairs_hook=unique)


def checked_python_identity(value: object) -> dict[str, str]:
    if type(value) is not dict or set(value) != {
        "interpreter",
        "prefix",
        "base_prefix",
    }:
        raise ValueError("Invalid Python identity fields")
    for item in value.values():
        if (
            type(item) is not str
            or not os.path.isabs(item)
            or len(item) > 8192
            or any(ord(char) < 32 for char in item)
        ):
            raise ValueError("Invalid absolute Python identity")
    return dict(value)
