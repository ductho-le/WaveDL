"""Build WaveDL shard manifests for OpenFWI Style-A."""

import argparse
from os.path import relpath
from pathlib import Path


# Train on OpenFWI's full training split (shards 1-120). The validation manifest
# reuses its last 12 shards to monitor the fit only; the final-epoch weights are
# reported, so no checkpoint is selected on held-out data.
SPLITS = {"train": range(1, 121), "val": range(109, 121), "test": range(121, 135)}


def _manifest_path(path: Path, base_dir: Path) -> str:
    """Return a path relative to base_dir, falling back to absolute across drives."""
    try:
        return relpath(path, base_dir)
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "style_a", type=Path, help="Style_A directory with data/ and model/"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use only shard pairs 1, 109, and 121 (the first of each list) for a tiny run.",
    )
    args = parser.parse_args()
    root = args.style_a.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    manifests = {}
    for split, indices in SPLITS.items():
        rows = []
        for index in indices[:1] if args.smoke_test else indices:
            pair = (
                root / "data" / f"data{index}.npy",
                root / "model" / f"model{index}.npy",
            )
            for path in pair:
                if not path.is_file():
                    parser.error(f"missing {split} shard: {path}")
            # WaveDL resolves relative paths from the manifest, not the working directory.
            rows.append("\t".join(_manifest_path(path, output_dir) for path in pair))
        manifests[split] = "\n".join(rows) + "\n"

    # Check every required pair before replacing any manifests.
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, text in manifests.items():
        path = output_dir / f"style-a_{split}.txt"
        path.write_text(text, encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
