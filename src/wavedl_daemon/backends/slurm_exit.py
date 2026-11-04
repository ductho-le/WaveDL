"""
The Slurm exit record: one ``batch training stage_out`` line per attempt.

The batch script writes it and the daemon reads it. It is written to a
temporary file and renamed, so a reader sees a whole line or no file.

Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""

from __future__ import annotations

import re

from wavedl_daemon.runs.layout import EXIT_RECORD_FILE
from wavedl_daemon.runs.model import ExitRecord


_RECORD = re.compile(rb"([0-9]{1,3}) ([0-9]{1,3}) ([0-9]{1,3})\n")


def render_exit_record(attempt_var: str) -> str:
    """Bash that records ``$WAVEDL_BATCH_EXIT $WAVEDL_TRAINING_EXIT $WAVEDL_STAGE_OUT_EXIT``.

    ``attempt_var`` names the shell variable holding the attempt directory.
    """
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", attempt_var):
        raise ValueError(f"Not a shell variable name: {attempt_var!r}")
    target = f'"${{{attempt_var}}}/{EXIT_RECORD_FILE}"'
    partial = f'"${{{attempt_var}}}/{EXIT_RECORD_FILE}.tmp"'
    return (
        "printf '%d %d %d\\n' "
        '"$WAVEDL_BATCH_EXIT" "$WAVEDL_TRAINING_EXIT" "$WAVEDL_STAGE_OUT_EXIT" '
        f"> {partial} && mv -f {partial} {target}\n"
    )


def parse_exit_record(data: bytes) -> ExitRecord:
    match = _RECORD.fullmatch(data)
    if match is None:
        raise ValueError("Malformed exit record")
    batch, training, stage_out = (int(code) for code in match.groups())
    if max(batch, training, stage_out) > 255:
        raise ValueError("Exit record code is out of range")
    return ExitRecord(batch, training, stage_out)
