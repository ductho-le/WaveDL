"Dependency-light shared inference and plot command-line configuration."

import argparse
from copy import deepcopy


def build_inference_parser(*, models=()) -> argparse.ArgumentParser:
    """Build the dependency-light inference and plot parser."""
    parser = argparse.ArgumentParser(
        description="WaveDL Testing & Inference Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Legacy checkpoint, published generation, or run directory (selects best)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to test data (NPZ, HDF5, or MAT v7.3 format)",
    )

    # Model specification
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model architecture (auto-detect if not specified). Available: {models}",
    )
    parser.add_argument(
        "--import",
        dest="import_modules",
        type=str,
        nargs="+",
        default=[],
        help="Python modules or .py files to import before inference (custom models)",
    )

    # Data format
    parser.add_argument(
        "--format",
        type=str,
        default="auto",
        choices=["auto", "npz", "mat", "hdf5"],
        help="Data format (auto-detect by extension)",
    )
    parser.add_argument(
        "--input_key",
        type=str,
        default=None,
        help="Custom key name for input data in MAT/HDF5/NPZ files (e.g., 'X', 'waveforms')",
    )
    parser.add_argument(
        "--output_key",
        type=str,
        default=None,
        help="Custom key name for output data in MAT/HDF5/NPZ files (e.g., 'Y', 'labels')",
    )
    parser.add_argument(
        "--param_names",
        type=str,
        nargs="+",
        default=None,
        help="Parameter/channel names for output (e.g., 'h' 'v11' 'v12')",
    )
    parser.add_argument(
        "--input_channels",
        type=int,
        default=None,
        help="Explicit number of input channels. Bypasses auto-detection heuristics "
        "for ambiguous 4D shapes (e.g., 3D volumes with small depth).",
    )
    parser.add_argument(
        "--allow_pickle",
        action="store_true",
        help="Allow trusted legacy NPZ object arrays (can execute arbitrary code).",
    )

    # Inference options
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for inference"
    )
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers")

    # Output options
    parser.add_argument(
        "--output_dir", type=str, default=".", help="Directory for saving results"
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save vector predictions to CSV or dense fields to compressed NPZ",
    )
    parser.add_argument("--plot", action="store_true", help="Generate diagnostic plots")
    parser.add_argument(
        "--plot_format",
        type=str,
        nargs="+",
        default=["png"],
        help="Output format(s) for plots: png, pdf, svg, eps, tiff (default: png)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-sample predictions"
    )

    # Export options
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        choices=["onnx"],
        help="Export model format (onnx)",
    )
    parser.add_argument(
        "--export_path",
        type=str,
        default=None,
        help="Output path for exported model (default: {model_name}.onnx)",
    )
    parser.add_argument(
        "--export_opset",
        type=int,
        default=17,
        help="ONNX opset version (11-17, higher = newer ops)",
    )
    parser.add_argument(
        "--no_denorm",
        action="store_true",
        help="Disable de-normalization in ONNX (output normalized values instead)",
    )
    parser.add_argument(
        "--out_size",
        type=int,
        default=None,
        help=(
            "Override vector target count or field channel count "
            "(inferred from scaler.pkl if targets are missing)"
        ),
    )

    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    parser.add_argument("--operation", choices=["test", "plot"], default="test")
    return parser


def parse_inference_args(argv=None, *, models=()):
    return build_inference_parser(models=models).parse_args(argv)


def inference_defaults() -> dict[str, object]:
    """Snapshot parser defaults; required inputs cannot be silently approved."""
    args = build_inference_parser().parse_args(["--checkpoint", "", "--data_path", ""])
    return deepcopy(vars(args))


def validate_import_modules(value: object) -> list[str]:
    """Validate explicit registration intent, never checkpoint metadata."""
    if (
        type(value) is not list
        or len(value) > 128
        or any(
            type(item) is not str
            or not item
            or len(item) > 8192
            or "\0" in item
            or item.startswith("-")
            for item in value
        )
    ):
        raise ValueError("Imports require a bounded list of safe module names or paths")
    return list(value)
