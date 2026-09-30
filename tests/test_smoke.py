import numpy as np
import pytest

import sextant

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def make_figure():
    fig = sextant.Figure(width=320, height=240)
    x = np.linspace(0.0, 2.0 * np.pi, 50)
    fig.axes().line(x, np.sin(x)).line(np.cos(x))
    return fig


def test_render_png_without_a_window():
    fig = make_figure()
    assert fig.axes().line_count() == 2
    png = fig.render_png()
    assert png.startswith(PNG_SIGNATURE)
    assert not fig.is_open()


def test_savefig_writes_render_png_bytes(tmp_path):
    fig = make_figure()
    path = tmp_path / "smoke.png"
    fig.savefig(str(path))
    assert path.read_bytes() == fig.render_png()


def test_chained_calls_return_the_same_axes():
    fig = sextant.Figure()
    ax = fig.axes()
    assert ax.line(np.arange(3.0)) is ax


def test_non_float64_input_is_converted():
    fig = sextant.Figure()
    ax = fig.axes()
    ax.line(np.arange(5), np.arange(10, dtype=np.float32)[::2])
    assert ax.line_count() == 1


def test_length_mismatch_raises_value_error():
    ax = sextant.Figure().axes()
    with pytest.raises(ValueError):
        ax.line(np.arange(3.0), np.arange(4.0))
