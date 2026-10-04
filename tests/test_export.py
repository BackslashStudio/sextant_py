"""render_png/render_svg, savefig to paths and file objects, and the IPython reprs."""

import io
import struct
import sys

import numpy as np
import pytest

import sextant


def png_size(data):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


@pytest.fixture
def fig():
    f = sextant.Figure(width=160, height=120)
    f.axes().line([0, 1, 2], [0, 1, 0], color="red")
    return f


def test_render_png_options(fig):
    assert png_size(fig.render_png()) == (160, 120)
    assert png_size(fig.render_png(dpi=192)) == (320, 240)
    assert png_size(fig.render_png(width=80, height=60)) == (80, 60)
    with pytest.raises(TypeError, match="max_splits"):
        fig.render_png(max_splits=3)


def test_render_svg(fig):
    svg, report = fig.render_svg()
    assert "<svg" in svg and report.scene_order_exact
    assert isinstance(report, sextant.SvgSaveReport)


@pytest.mark.parametrize("name", ["out.png", "out.PNG", "out.svg"])
def test_savefig_path_matches_render(fig, tmp_path, name):
    path = tmp_path / name  # a PathLike
    r = fig.savefig(path)
    if name.lower().endswith(".png"):
        assert r is None and path.read_bytes() == fig.render_png()
    else:
        assert r.scene_order_exact and path.read_text(encoding="utf-8") == fig.render_svg()[0]


def test_savefig_format_overrides_the_extension(fig, tmp_path):
    fig.savefig(str(tmp_path / "plot.dat"), format="svg")
    assert "<svg" in (tmp_path / "plot.dat").read_text(encoding="utf-8")
    fig.savefig(str(tmp_path / "hi.png"), dpi=192)
    assert png_size((tmp_path / "hi.png").read_bytes()) == (320, 240)


@pytest.mark.parametrize("name", ["plot.jpg", "noext", "dir.v2/plot"])
def test_savefig_unknown_format(fig, tmp_path, name):
    with pytest.raises(ValueError):
        fig.savefig(str(tmp_path / name))


@pytest.mark.skipif(sys.platform != "win32", reason="':' names an NTFS data stream only on Windows")
@pytest.mark.parametrize("name", ["plot 12:30.png", "plot 12:30.svg"])
def test_savefig_refuses_a_data_stream_name(fig, tmp_path, name):
    # Once an empty "plot 12" and the bytes in its hidden stream "30.png", silently.
    with pytest.raises(OSError, match="':' is not allowed"):
        fig.savefig(str(tmp_path / name))
    assert list(tmp_path.iterdir()) == []
    fig.savefig("\\\\?\\" + str(tmp_path / "long.png"))  # a \\?\ path's drive colon is fine
    assert (tmp_path / "long.png").read_bytes() == fig.render_png()


def test_savefig_options_belong_to_the_format(fig, tmp_path):
    with pytest.raises(TypeError, match="SvgExportOptions fields: max_splits, max_tests"):
        fig.savefig(str(tmp_path / "a.svg"), dpi=192)


def test_savefig_to_file_objects(fig):
    b = io.BytesIO()
    assert fig.savefig(b) is None  # PNG by default
    assert b.getvalue() == fig.render_png()
    b = io.BytesIO()
    fig.savefig(b, format="svg")
    assert b.getvalue().decode("utf-8") == fig.render_svg()[0]
    s = io.StringIO()
    fig.savefig(s, format="svg")
    assert s.getvalue() == fig.render_svg()[0]


def test_reprs(fig):
    try:
        assert png_size(fig._repr_png_()) == (160, 120)
        assert fig._repr_svg_() is None
        sextant.set_repr_formats("svg")
        assert fig._repr_png_() is None and "<svg" in fig._repr_svg_()
        sextant.set_repr_formats()
        assert fig._repr_png_() is None and fig._repr_svg_() is None
        with pytest.raises(ValueError):
            sextant.set_repr_formats("jpeg")
    finally:
        sextant.set_repr_formats("png")


def test_inexact_svg_reports_and_warns():
    """The real diagnostic path: an SVG export whose 3D budget ran out."""
    fig = sextant.Figure(width=200, height=150)
    ax = fig.add_subplot3d(1, 1, 1)
    g = np.linspace(-1, 1, 12)
    u, v = np.meshgrid(g, g, indexing="ij")
    ax.surface("xy", g, g, u * v, alpha=0.6).surface("xy", g, g, -u * v, alpha=0.6)
    with pytest.warns(RuntimeWarning, match="scene order"):
        svg, report = fig.render_svg(max_splits=1)
    assert not report.scene_order_exact and "max_splits" in report.warning
