"""
Where one run's files live.

``output/`` is the trainer's ``--output_dir``. It is stable across attempts
because upstream resumes from it, and it holds no daemon files, so upstream's
archive and resume scans never see them. Each attempt owns ``attempts/<N>/``,
claimed by ``mkdir``: the daemon claims Local attempts and the batch script
claims Slurm ones.

Author: Ngo Duc Manh Le (manh@nnsvn.me)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from wavedl_daemon.runs.model import Operation


if TYPE_CHECKING:
    from pathlib import Path


METRICS_FILE = "metrics.jsonl"
EVALUATION_FILE = "evaluation.json"
EXIT_RECORD_FILE = "exit"

# The trainer option through which an operation writes its attempt's results.
RESULT_OPTIONS = {
    Operation.TRAIN: ("--metrics_jsonl", METRICS_FILE),
    Operation.TEST: ("--metrics_json", EVALUATION_FILE),
}


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path

    @property
    def output(self) -> Path:
        return self.run_dir / "output"

    @property
    def logs(self) -> Path:
        return self.run_dir / "logs"

    @property
    def submission_script(self) -> Path:
        return self.run_dir / "submission.sh"

    def attempt(self, number: int) -> Path:
        return self.run_dir / "attempts" / str(number)

    def metrics(self, number: int) -> Path:
        return self.attempt(number) / METRICS_FILE

    def exit_record(self, number: int) -> Path:
        return self.attempt(number) / EXIT_RECORD_FILE

    def evaluation(self, number: int) -> Path:
        return self.attempt(number) / EVALUATION_FILE

    def output_log(self, number: int) -> Path:
        return self.attempt(number) / "output.log"
