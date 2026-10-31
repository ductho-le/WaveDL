"""Accelerate launch bridge using torchrun's contained callable-worker route.

Torchrun's binary worker route starts a new POSIX session for each rank, outside
WaveDL's supervising process group. Its supported ``run_path`` callable route
uses multiprocessing instead and inherits the owner's process group. Keep
Accelerate responsible for its normal config, topology, and environment setup.
"""

from __future__ import annotations

from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    from accelerate.commands.launch import launch_command, launch_command_parser
    from torch.distributed.run import get_args_parser

    args = launch_command_parser().parse_args(argv)
    if args.training_script != "wavedl.train" or not args.module:
        raise ValueError("contained distributed launch requires -m wavedl.train")
    if not args.multi_gpu or any(
        getattr(args, name, False)
        for name in (
            "cpu",
            "tpu",
            "tpu_use_cluster",
            "use_deepspeed",
            "use_fsdp",
            "use_megatron_lm",
            "no_python",
        )
    ):
        raise ValueError("contained distributed launch requires --multi_gpu")

    # Fail before launching if the installed torchrun lacks this public route.
    supported = vars(get_args_parser().parse_args(["--run-path", "worker.py"]))
    if not supported.get("run_path") or "start_method" not in supported:
        raise RuntimeError("installed torchrun does not support contained workers")

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
