"""Exercise an installed wheel from outside its source checkout."""

import json
import os
import subprocess
import sys
import sysconfig
import threading
from http.server import ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch

import wavedl
from wavedl.gui import app as gui_app
from wavedl.models import build_model
from wavedl.utils import SpatialTargetScaler


def main():
    checkout = Path(__file__).resolve().parents[1]
    if checkout / "src" in Path(wavedl.__file__).resolve().parents:
        raise RuntimeError(
            f"wavedl imported from source checkout instead of installed wheel: {wavedl.__file__}"
        )
    model = build_model("cnn", in_shape=(32,), out_size=2).eval()
    with torch.no_grad():
        out_shape = tuple(model(torch.zeros(2, 1, 32)).shape)
    if out_shape != (2, 2):
        raise RuntimeError(f"Unexpected CNN forward output shape: {out_shape}")
    values = np.arange(32, dtype=np.float32).reshape(2, 1, 4, 4)
    scaler = SpatialTargetScaler().fit(values)
    np.testing.assert_allclose(
        scaler.inverse_transform(scaler.transform(values)), values, atol=1e-5
    )
    if not files("wavedl.gui").joinpath("index.html").is_file():
        raise RuntimeError("Packaged wavedl.gui/index.html is missing")
    server = ThreadingHTTPServer(("127.0.0.1", 0), gui_app.WaveDLGUIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        with urlopen(f"{base_url}/", timeout=5) as resp:
            html = resp.read().decode("utf-8")
        if "WaveDL" not in html:
            raise RuntimeError("GUI index page did not render expected HTML")
        with urlopen(f"{base_url}/api/info", timeout=5) as resp:
            info = json.loads(resp.read().decode("utf-8"))
        if "workspace_root" not in info:
            raise RuntimeError(f"Unexpected /api/info payload: {info}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    env = os.environ.copy()
    scripts_dir = sysconfig.get_path("scripts")
    if scripts_dir:
        env["PATH"] = f"{scripts_dir}{os.pathsep}{env.get('PATH', '')}"
    for module in ("train", "test", "hpo", "gui"):
        subprocess.run(
            [sys.executable, "-m", f"wavedl.{module}", "--help"],
            check=True,
            env=env,
        )
    for command in (
        "wavedl-train",
        "wavedl-hpc",
        "wavedl-test",
        "wavedl-hpo",
        "wavedl-gui",
    ):
        subprocess.run([command, "--help"], check=True, env=env)


if __name__ == "__main__":
    main()
