"""Regular-file reads must not prevent a running trainer from writing."""

import os

import pytest

from wavedl_daemon import paths


def test_regular_open_refuses_a_directory(tmp_path):
    with pytest.raises(ValueError, match="regular file"):
        paths.open_regular(tmp_path)


def test_regular_open_allows_a_concurrent_writer(tmp_path):
    target = tmp_path / "training_history.csv"
    target.write_bytes(b"epoch,loss\n")
    with paths.open_regular(target) as stream:
        with target.open("ab") as writer:
            writer.write(b"1,0.5\n")
        assert target.read_bytes() == b"epoch,loss\n1,0.5\n"
        assert os.fstat(stream.fileno()).st_size == len(target.read_bytes())


def test_regular_open_closes_descriptor_if_stream_creation_fails(tmp_path, monkeypatch):
    target = tmp_path / "dataset.npz"
    target.write_bytes(b"dataset")
    descriptors = []

    def refuse(descriptor, mode):
        descriptors.append(descriptor)
        raise OSError("stream creation failed")

    monkeypatch.setattr(paths.os, "fdopen", refuse)
    with pytest.raises(OSError, match="stream creation failed"):
        paths.open_regular(target)
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
