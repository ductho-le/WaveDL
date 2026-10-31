"""Partial TeX installations must not disable ordinary CPU plots."""

import inspect
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pytest

from wavedl.utils.metrics import (
    _is_latex_available,
    configure_matplotlib_style,
    plot_error_boxplot,
)


@pytest.mark.parametrize(
    "available,expected",
    [
        (set(), False),
        ({"latex"}, False),
        ({"dvipng"}, False),
        ({"latex", "dvipng"}, True),
    ],
)
def test_agg_latex_requires_complete_executable_pair(monkeypatch, available, expected):
    monkeypatch.setattr(
        shutil, "which", lambda name: f"/tools/{name}" if name in available else None
    )
    assert _is_latex_available() is expected


def test_partial_tex_uses_mathtext_and_renders_real_png(tmp_path, monkeypatch):
    monkeypatch.setattr(
        shutil, "which", lambda name: "/tools/latex" if name == "latex" else None
    )
    with plt.rc_context():
        configure_matplotlib_style()
        assert plt.rcParams["text.usetex"] is False
        figure, axes = plt.subplots()
        try:
            axes.plot([0, 1], [1, 0])
            axes.set_xlabel(r"$R^2$ validation")
            output = tmp_path / "plot.png"
            figure.savefig(output)
            assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        finally:
            plt.close(figure)


@pytest.mark.parametrize("api", ["old", "new", "both"])
def test_boxplot_keyword_tracks_installed_signature(monkeypatch, api):
    from matplotlib.axes import Axes

    real_boxplot = Axes.boxplot
    real_keyword = (
        "tick_labels"
        if "tick_labels" in inspect.signature(real_boxplot).parameters
        else "labels"
    )
    calls = []

    def legacy(self, x, *, labels=None, **kwargs):
        calls.append(("labels", labels))
        return real_boxplot(self, x, **{real_keyword: labels}, **kwargs)

    def modern(self, x, *, tick_labels=None, **kwargs):
        calls.append(("tick_labels", tick_labels))
        return real_boxplot(self, x, **{real_keyword: tick_labels}, **kwargs)

    def transitional(self, x, *, labels=None, tick_labels=None, **kwargs):
        assert labels is None
        calls.append(("tick_labels", tick_labels))
        return real_boxplot(self, x, **{real_keyword: tick_labels}, **kwargs)

    monkeypatch.setattr(
        Axes, "boxplot", {"old": legacy, "new": modern, "both": transitional}[api]
    )
    with plt.rc_context({"text.usetex": False}):
        figure = plot_error_boxplot(
            np.array([[1.0, 2.0], [2.0, 3.0]]),
            np.array([[1.1, 2.2], [2.1, 3.2]]),
            ["a", "b"],
        )
        try:
            assert calls == [("labels" if api == "old" else "tick_labels", ["a", "b"])]
        finally:
            plt.close(figure)


def test_real_error_boxplot_renders_labels_and_png(tmp_path):
    with plt.rc_context({"text.usetex": False}):
        figure = plot_error_boxplot(
            np.array([[1.0, 2.0], [2.0, 3.0]]),
            np.array([[1.1, 2.2], [2.1, 3.2]]),
            ["alpha", "beta"],
        )
        try:
            output = tmp_path / "error-boxplot.png"
            figure.savefig(output)
            assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            assert [tick.get_text() for tick in figure.axes[0].get_xticklabels()] == [
                "alpha",
                "beta",
            ]
        finally:
            plt.close(figure)


def test_boxplot_invalid_data_typeerror_is_not_hidden(monkeypatch):
    from matplotlib.axes import Axes

    def refuses(self, x, *, tick_labels=None, **kwargs):
        raise TypeError("invalid boxplot data")

    monkeypatch.setattr(Axes, "boxplot", refuses)
    original = set(plt.get_fignums())
    try:
        with pytest.raises(TypeError, match="invalid boxplot data"):
            plot_error_boxplot(np.ones((2, 1)), np.zeros((2, 1)))
    finally:
        for number in set(plt.get_fignums()) - original:
            plt.close(number)
