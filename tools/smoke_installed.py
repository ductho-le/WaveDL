"""Exercise an installed wheel from outside its source checkout."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import wavedl
from wavedl.models import build_model
from wavedl.utils import SpatialTargetScaler


def main():
    checkout = Path(__file__).resolve().parents[1]
    assert checkout / "src" not in Path(wavedl.__file__).resolve().parents
    model = build_model("cnn", in_shape=(32,), out_size=2).eval()
    with torch.no_grad():
        assert model(torch.zeros(2, 1, 32)).shape == (2, 2)
    values = np.arange(32, dtype=np.float32).reshape(2, 1, 4, 4)
    scaler = SpatialTargetScaler().fit(values)
    np.testing.assert_allclose(
        scaler.inverse_transform(scaler.transform(values)), values, atol=1e-5
    )
    for module in ("train", "test", "hpo"):
        subprocess.run([sys.executable, "-m", f"wavedl.{module}", "--help"], check=True)
    for command in ("wavedl-train", "wavedl-test", "wavedl-hpo"):
        subprocess.run([command, "--help"], check=True)
    subprocess.run(["wavedl", "serve", "--help"], check=True)


if __name__ == "__main__":
    main()
