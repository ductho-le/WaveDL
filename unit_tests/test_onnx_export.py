"""Tests for self-contained ONNX export."""

import pytest


torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")

from wavedl import test  # noqa: E402


def test_export_is_single_file_loadable_after_being_moved(tmp_path):
    model = torch.nn.Linear(3, 2).eval()
    exported = tmp_path / "model.onnx"
    assert test.export_to_onnx(model, torch.randn(1, 3), str(exported), validate=False)
    assert list(tmp_path.iterdir()) == [exported]

    moved = tmp_path / "elsewhere" / "different-name.onnx"
    moved.parent.mkdir()
    exported.replace(moved)
    graph = onnx.load(str(moved), load_external_data=False)
    onnx.checker.check_model(graph)
    assert graph.graph.initializer
    for tensor in graph.graph.initializer:
        assert tensor.data_location != onnx.TensorProto.EXTERNAL
        assert not tensor.external_data
