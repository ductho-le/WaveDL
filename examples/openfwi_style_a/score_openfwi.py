"""Score WaveDL Style-A predictions in physical and OpenFWI units."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


VELOCITY_MIN, VELOCITY_MAX = 1500.0, 4500.0
MAP_SHAPE = (1, 70, 70)


def load_velocity_maps(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the physical-unit target and prediction arrays from wavedl-test."""
    with np.load(path, allow_pickle=False) as archive:
        if not {"target", "prediction"}.issubset(archive.files):
            raise ValueError("Expected 'target' and 'prediction' from wavedl-test.")
        target, prediction = archive["target"], archive["prediction"]
    if (
        target.shape[1:] != MAP_SHAPE
        or target.shape != prediction.shape
        or not len(target)
    ):
        raise ValueError(
            "Expected matching nonempty (N, 1, 70, 70) arrays, "
            f"got {target.shape} and {prediction.shape}."
        )
    return target, prediction


def ssim_per_sample(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return a mean SSIM for each NCHW sample, with unit data range."""
    # Match lanl/OpenFWI's 11x11 Gaussian (sigma 1.5), including zero padding
    # at the image edges: https://github.com/lanl/OpenFWI/blob/main/pytorch_ssim.py
    radius = 5
    weights = torch.tensor(
        [math.exp(-0.5 * (i / 1.5) ** 2) for i in range(-radius, radius + 1)],
        dtype=torch.float32,
    )
    weights = weights / weights.sum()
    channels = x.shape[1]
    kernel = (
        torch.outer(weights, weights).to(x).expand(channels, 1, -1, -1).contiguous()
    )

    def average(values: torch.Tensor) -> torch.Tensor:
        return F.conv2d(values, kernel, padding=radius, groups=channels)

    mean_x, mean_y = average(x), average(y)
    mean_xy = mean_x * mean_y
    mean_x2, mean_y2 = mean_x.square(), mean_y.square()
    variance_x = average(x.square()) - mean_x2
    variance_y = average(y.square()) - mean_y2
    covariance = average(x * y) - mean_xy
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2 * mean_xy + c1) * (2 * covariance + c2)
    denominator = (mean_x2 + mean_y2 + c1) * (variance_x + variance_y + c2)
    return (numerator / denominator).mean(dim=(1, 2, 3))


@torch.inference_mode()
def score(path: Path, batch_size: int) -> dict[str, float | int]:
    target, prediction = load_velocity_maps(path)
    span = VELOCITY_MAX - VELOCITY_MIN
    midpoint = 0.5 * (VELOCITY_MIN + VELOCITY_MAX)
    abs_error, squared_error, ssim_total = 0.0, 0.0, 0.0
    sum_t, sum_t2, sum_p, sum_p2, sum_tp = 0.0, 0.0, 0.0, 0.0, 0.0
    for start in range(0, len(target), batch_size):
        truth = torch.from_numpy(target[start : start + batch_size]).double()
        pred = torch.from_numpy(prediction[start : start + batch_size]).double()
        if not (torch.isfinite(truth).all() and torch.isfinite(pred).all()):
            raise ValueError("Target and prediction arrays must contain finite values.")
        error = pred - truth
        abs_error += error.abs().sum().item()
        squared_error += error.square().sum().item()
        # Shift by midpoint to preserve precision in pixel-pooled variance/covariance.
        t_centered = truth - midpoint
        p_centered = pred - midpoint
        sum_t += t_centered.sum().item()
        sum_t2 += t_centered.square().sum().item()
        sum_p += p_centered.sum().item()
        sum_p2 += p_centered.square().sum().item()
        sum_tp += (t_centered * p_centered).sum().item()
        # OpenFWI test.py scores all samples together on [0, 1], without clipping.
        # Accumulate sample scores, not batch means, so the last batch is weighted correctly.
        truth_unit = ((truth - VELOCITY_MIN) / span).float()
        pred_unit = ((pred - VELOCITY_MIN) / span).float()
        ssim_total += (
            ssim_per_sample(truth_unit, pred_unit).sum(dtype=torch.float64).item()
        )

    n_pixels = target.size
    mae = abs_error / n_pixels
    rmse = math.sqrt(squared_error / n_pixels)
    ss_tot = max(0.0, sum_t2 - (sum_t * sum_t) / n_pixels)
    var_p = max(0.0, sum_p2 - (sum_p * sum_p) / n_pixels)
    cov_tp = sum_tp - (sum_t * sum_p) / n_pixels
    r2 = 1.0 - squared_error / ss_tot if ss_tot > 0.0 else 0.0
    denom = math.sqrt(ss_tot * var_p)
    pearson = cov_tp / denom if denom > 0.0 else 0.0
    return {
        "samples": len(target),
        "mae_normalized": mae / (span / 2),
        "rmse_normalized": rmse / (span / 2),
        "ssim": ssim_total / len(target),
        "mae_m_s": mae,
        "rmse_m_s": rmse,
        "r2": r2,
        "pearson": pearson,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="WaveDL predictions.npz")
    parser.add_argument("--batch-size", type=int, default=32, help="SSIM chunk size")
    parser.add_argument(
        "--output", type=Path, help="Also write metrics to this JSON file"
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    try:
        result = json.dumps(
            score(args.predictions, args.batch_size), indent=2, allow_nan=False
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
