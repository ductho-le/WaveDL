"""``accelerate launch --multi_gpu`` that keeps every rank in the caller's group.

torchrun starts each rank of a script or module in a new POSIX session, so the
ranks escape the process group that ``wavedl-train`` signals on preemption or
Ctrl+C and can outlive it. torchrun's ``--run-path`` route instead runs the
script in multiprocessing workers, which inherit the launcher's process group.
Accelerate still performs its normal config, topology and environment setup.

Usage (as built by ``wavedl.launcher``)::

    python -m wavedl.distributed_launch --multi_gpu ... -m wavedl.train ...
"""

from __future__ import annotations

from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    from accelerate.commands.launch import launch_command, launch_command_parser

    args = launch_command_parser().parse_args(argv)
    if args.training_script != "wavedl.train" or not args.module:
        raise ValueError("wavedl.distributed_launch requires -m wavedl.train")
    if not args.multi_gpu or args.cpu:
        raise ValueError("wavedl.distributed_launch requires --multi_gpu")

    args.training_script = str(Path(__file__).with_name("train.py").resolve())
    args.module = False
    args.run_path = True
    args.start_method = "spawn"
    if hasattr(args, "nondefault"):
        args.nondefault.update(
            {"training_script", "module", "run_path", "start_method", "multi_gpu"}
        )
    launch_command(args)


if __name__ == "__main__":
    main()
