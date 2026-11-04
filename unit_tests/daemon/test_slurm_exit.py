"""The Slurm exit record, written by bash and read by the daemon."""

import subprocess
import sys

import pytest

from wavedl_daemon.backends.slurm_exit import parse_exit_record, render_exit_record
from wavedl_daemon.runs.layout import RunPaths
from wavedl_daemon.runs.model import ExitRecord


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Executes the cluster's Bash exit writer against a real POSIX filesystem",
)
def test_the_rendered_bash_writes_a_record_the_daemon_parses(tmp_path):
    paths = RunPaths(tmp_path / "run dir")
    paths.attempt(2).mkdir(parents=True)
    script = (
        "set -eu\n"
        'WAVEDL_ATTEMPT="$1"\n'
        "WAVEDL_BATCH_EXIT=0\n"
        "WAVEDL_TRAINING_EXIT=137\n"
        "WAVEDL_STAGE_OUT_EXIT=3\n" + render_exit_record("WAVEDL_ATTEMPT")
    )
    subprocess.run(
        ["bash", "-c", script, "bash", str(paths.attempt(2))], check=True, timeout=10
    )
    assert parse_exit_record(paths.exit_record(2).read_bytes()) == ExitRecord(0, 137, 3)
    assert sorted(path.name for path in paths.attempt(2).iterdir()) == ["exit"]


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"0 0 0",
        b"0 0\n",
        b"0 0 0 0\n",
        b"0  0 0\n",
        b"0 0 x\n",
        b"0 0 256\n",
        b"0 0 0\n0 0 0\n",
    ],
)
def test_a_malformed_record_is_refused(data):
    with pytest.raises(ValueError):
        parse_exit_record(data)


def test_only_a_shell_variable_name_is_rendered():
    with pytest.raises(ValueError):
        render_exit_record('A}"; rm -rf /; echo "')
