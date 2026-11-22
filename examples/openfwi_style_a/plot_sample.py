"""Plot the test gathers, velocity maps, and signed error for one Style-A sample."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from score_openfwi import MAP_SHAPE, VELOCITY_MAX, VELOCITY_MIN, load_velocity_maps

from wavedl.utils.preprocessing import ShardManifestDataset


matplotlib.use("Agg")
import matplotlib.pyplot as plt


SHOTS = (1, 3, 5)
DT, DX = 0.001, 10


def load_sample(predictions: Path, manifest: Path, sample: int):
    targets, predictions_array = load_velocity_maps(predictions)
    if not 0 <= sample < len(targets):
        raise ValueError(f"--sample must be in [0, {len(targets) - 1}].")
    count = len(targets)
    truth = targets[sample, 0].copy()
    pred = predictions_array[sample, 0].copy()
    del targets, predictions_array

    dataset = ShardManifestDataset(manifest)
    try:
        if dataset.input_shape != (5, 1000, 70):
            raise ValueError(
                f"Expected Style-A gather shape (5, 1000, 70), got {dataset.input_shape}."
            )
        if dataset.target_shape != MAP_SHAPE:
            raise ValueError(
                f"Expected Style-A target shape {MAP_SHAPE}, got {dataset.target_shape}."
            )
        if len(dataset) != count:
            raise ValueError(
                f"The manifest contains {len(dataset)} samples but the NPZ contains {count}."
            )
        gathers, manifest_target = dataset[sample]
    finally:
        dataset.close()
    if not np.allclose(truth, manifest_target.numpy()[0], rtol=0, atol=1e-3):
        raise ValueError(
            "Saved target does not match this sample in the test manifest."
        )
    if not np.isfinite(pred).all():
        raise ValueError("Prediction contains non-finite velocities.")
    return gathers.numpy()[np.array(SHOTS) - 1], truth, pred


def plot_sample(
    gathers: np.ndarray, truth: np.ndarray, pred: np.ndarray, output: Path
) -> None:
    time = np.arange(gathers.shape[1]) * DT
    # Display gain only: training uses signed-log preprocessing, not this gain.
    gain = (np.maximum(time, 0.02) / 0.2) ** 2
    gained = gathers * gain[None, :, None]
    amplitude_limit = float(np.percentile(np.abs(gained), 99))
    if amplitude_limit == 0:
        amplitude_limit = float(np.max(np.abs(gained))) or 1.0
    gained /= amplitude_limit
    error = pred.astype(np.float64) - truth
    error_limit = float(np.max(np.abs(error))) or 1.0
    mae = np.mean(np.abs(error))

    fig = plt.figure(figsize=(12, 4.6), layout="constrained")
    grid = fig.add_gridspec(2, 4, height_ratios=(1, 0.045))
    shot_grid = grid[0, 0].subgridspec(3, 1, hspace=0.05)
    xmax = (gathers.shape[2] - 1) * DX
    # Put pixel centres at the physical sample coordinates, not the image edges.
    x_extent = (-DX / 2, xmax + DX / 2)
    for i, (shot, values) in enumerate(zip(SHOTS, gained)):
        ax = fig.add_subplot(shot_grid[i, 0])
        shot_image = ax.imshow(
            values,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            extent=(*x_extent, time[-1] + DT / 2, -DT / 2),
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_xlim(0, xmax)
        ax.set_ylim(time[-1], 0)
        ax.set_xticks([0, xmax / 2, xmax])
        ax.set_yticks([0, 0.5, time[-1]], labels=["0", "0.5", "0.999"])
        ax.tick_params(labelbottom=i == len(SHOTS) - 1, labelsize=8)
        ax.text(
            0.96,
            0.08,
            f"S{shot}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2},
        )
        if i == 0:
            ax.set_title("(a) Shot gathers", loc="left", fontsize=11)
        if i == 1:
            ax.set_ylabel("Time (s)")
        if i == len(SHOTS) - 1:
            ax.set_xlabel("Receiver position (m)")

    fig.colorbar(
        shot_image,
        cax=fig.add_subplot(grid[1, 0]),
        orientation="horizontal",
        ticks=[-1, 0, 1],
        extend="both" if np.max(np.abs(gained)) > 1 else "neither",
        label="Gained amplitude (a.u.)",
    )

    zmax = (truth.shape[0] - 1) * DX
    extent = (-DX / 2, (truth.shape[1] - 0.5) * DX, zmax + DX / 2, -DX / 2)
    panels = (
        ("(b) Ground truth", truth, "Blues", VELOCITY_MIN, VELOCITY_MAX),
        ("(c) U-Net prediction", pred, "Blues", VELOCITY_MIN, VELOCITY_MAX),
        ("(d) Signed error", error, "RdBu_r", -error_limit, error_limit),
    )
    for column, (title, values, cmap, low, high) in enumerate(panels, 1):
        ax = fig.add_subplot(grid[0, column])
        image = ax.imshow(
            values,
            cmap=cmap,
            vmin=low,
            vmax=high,
            extent=extent,
            interpolation="nearest",
        )
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlim(0, xmax)
        ax.set_ylim(zmax, 0)
        ax.set_xticks([0, xmax / 2, xmax])
        ax.set_yticks([0, zmax / 2, zmax])
        ax.set_xlabel("Position (m)")
        ax.tick_params(labelleft=column == 1, labelsize=8)
        if column == 1:
            ax.set_ylabel("Depth (m)")
        elif column == 2:
            below = min(truth.min(), pred.min()) < VELOCITY_MIN
            above = max(truth.max(), pred.max()) > VELOCITY_MAX
            extend = (
                "both"
                if below and above
                else "min"
                if below
                else "max"
                if above
                else "neither"
            )
            fig.colorbar(
                image,
                cax=fig.add_subplot(grid[1, 1:3]),
                orientation="horizontal",
                ticks=[1500, 2500, 3500, 4500],
                extend=extend,
                label="Velocity (m/s)",
            )
        else:
            ax.text(
                0.04,
                0.04,
                f"MAE: {mae:.2f} m/s",
                transform=ax.transAxes,
                fontsize=9,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.9,
                    "pad": 3,
                },
            )
            fig.colorbar(
                image,
                cax=fig.add_subplot(grid[1, 3]),
                orientation="horizontal",
                label="Prediction - ground truth (m/s)",
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for fmt in ("png", "pdf"):
            path = Path(f"{output}.{fmt}")
            fig.savefig(path, dpi=300)
            print(path)
    finally:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="WaveDL predictions.npz")
    parser.add_argument(
        "test_manifest", type=Path, help="The same manifest given to wavedl-test"
    )
    parser.add_argument(
        "--sample", type=int, default=0, help="Zero-based test manifest sample index"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output path without an extension"
    )
    args = parser.parse_args()
    if args.output.suffix.lower() in {".png", ".pdf"}:
        parser.error("--output must be a path without a .png or .pdf extension")
    try:
        plot_sample(
            *load_sample(args.predictions, args.test_manifest, args.sample), args.output
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
