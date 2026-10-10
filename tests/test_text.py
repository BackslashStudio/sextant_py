"""Text and annotations (sextant 1.1 step 31): Axes.text()/annotate(), the Axes3D
calls, read-back and set_text_data()."""

import numpy as np
import pytest

import sextant


def near(img, rgb, tol):
    """How many pixels of an RGBA image are within `tol` of an RGB."""
    d = np.abs(img[..., :3].astype(int) - np.array(rgb))
    return int((d.max(axis=2) <= tol).sum())


def axes(width=400, height=300):
    fig = sextant.Figure(width=width, height=height)
    ax = fig.axes()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return fig, ax


# --- 2D ----------------------------------------------------------------------


def test_text_and_annotate_read_back():
    fig, ax = axes()
    assert ax.text("a", 2, 3, fontsize=14, color="red") is ax
    ax.text("b", 2.5, 0.95, ycoords="fraction")
    ax.text("c", 0.1, 0.2, coords="fraction")
    assert ax.annotate(5, 6, "peak", 0.6, 0.8, coords=sextant.Coords.FRACTION, arrow={"arc": 0.2}) is ax
    assert ax.text_count() == 4

    a = ax.text_data(0)
    assert (a.text, a.x, a.y, a.xcoords, a.ycoords, a.arrow) == ("a", 2.0, 3.0, "data", "data", False)
    assert a.xcoords is sextant.Coords.DATA

    b = ax.text_data(1)
    assert (b.xcoords, b.ycoords) == ("data", "fraction")
    c = ax.text_data(-2)
    assert (c.x, c.y, c.xcoords, c.ycoords) == (0.1, 0.2, "fraction", "fraction")

    d = ax.text_data(3)
    assert d.arrow and (d.px, d.py) == (5.0, 6.0) and (d.x, d.y) == (0.6, 0.8)
    assert "TextData(text='peak'" in repr(d)

    with pytest.raises(IndexError):
        ax.text_data(4)


def test_xcoords_overrides_coords():
    _, ax = axes()
    ax.text("t", 3, 0.5, coords="fraction", xcoords="data")
    d = ax.text_data(0)
    assert (d.xcoords, d.ycoords) == ("data", "fraction")
    ax.text("u", 0.5, 0.5, coords="axes fraction")  # matplotlib's spelling, an alias
    assert ax.text_data(1).xcoords == "fraction"


def test_set_text_data_both_forms():
    _, ax = axes()
    ax.annotate(5, 6, "peak", 0.6, 0.8, coords="fraction")

    # The short form keeps the arrow, its point and each coordinate's coords.
    ax.set_text_data(0, "moved", 0.3, 0.4)
    d = ax.text_data(0)
    assert (d.text, d.x, d.y, d.xcoords, d.ycoords, d.arrow, d.px, d.py) == (
        "moved", 0.3, 0.4, "fraction", "fraction", True, 5.0, 6.0)
    ax.set_text_data(0, "data now", 2, 3, coords="data")
    assert ax.text_data(0).xcoords == "data"

    # The object form round-trips, and can drop the arrow.
    d = ax.text_data(0)
    d.arrow = False
    d.text = "plain"
    assert ax.set_text_data(0, d) is ax
    assert not ax.text_data(0).arrow and ax.text_data(0).text == "plain"

    ax.set_text_data(0, sextant.TextData("new", 0.5, 0.5, "fraction", "fraction", True, 1, 2))
    e = ax.text_data(0)
    assert e.arrow and (e.px, e.py) == (1.0, 2.0) and e.text == "new"


def test_non_finite_is_a_value_error_and_adds_nothing():
    _, ax = axes()
    with pytest.raises(ValueError):
        ax.text("t", float("nan"), 1)
    with pytest.raises(ValueError):
        ax.annotate(float("inf"), 1, "t", 1, 1)
    assert ax.text_count() == 0


def test_bad_options():
    _, ax = axes()
    with pytest.raises(TypeError, match=r"unexpected keyword argument 'size'.*fontsize"):
        ax.text("t", 1, 1, size=3)
    with pytest.raises(TypeError, match=r"'arrow' takes a dict of ArrowOptions"):
        ax.annotate(1, 1, "t", 2, 2, arrow=True)
    with pytest.raises(TypeError, match=r"'arrow' has no field 'style'"):
        ax.annotate(1, 1, "t", 2, 2, arrow={"style": "->"})
    with pytest.raises(TypeError):
        ax.text("t", 1, 1, coords="pixels")


def test_text_draws_and_hides_out_of_view():
    fig, ax = axes()
    blank = fig.render_rgba()
    ax.text("XXXX", 5, 5, fontsize=30, color="#ff0000")
    assert near(fig.render_rgba(), (255, 0, 0), 60) > near(blank, (255, 0, 0), 60) + 20
    ax.set_xlim(20, 30)  # its x leaves the view: hidden whole
    assert near(fig.render_rgba(), (255, 0, 0), 60) == near(blank, (255, 0, 0), 60)


def test_text_never_widens_auto_limits():
    fig = sextant.Figure(width=300, height=200)
    ax = fig.axes()
    ax.line([0, 1], [0, 1])
    lim = ax.xlim()
    ax.text("far", 100, 100)
    assert ax.xlim() == lim


def test_box_and_arrow_in_svg():
    fig, ax = axes()
    ax.annotate(2, 2, "A<B", 0.7, 0.7, coords="fraction", ha="center", rotation=30,
                background="white", edge_linewidth=1.0, arrow={"head": "filled", "color": "blue"})
    svg = fig.render_svg()[0]
    assert ">A&lt;B</text>" in svg
    assert 'text-anchor="middle"' in svg
    assert "rotate(-30)" in svg
    assert "<polygon" in svg and "<polyline" in svg


def test_cla_clears_texts():
    _, ax = axes()
    ax.text("t", 1, 1)
    ax.cla()
    assert ax.text_count() == 0


# --- 3D ----------------------------------------------------------------------


def test_3d_calls_read_back():
    fig = sextant.Figure(width=400, height=300)
    ax = fig.add_subplot3d(1, 1, 1)
    ax.scatter3d([0, 1], [0, 1], [0, 1])
    assert ax.text("p", 0.5, 0.5, 0.5, color="red") is ax
    assert ax.text2d("caption", 0.02, 0.95, va="top") is ax
    assert ax.annotate(1, 1, 1, "tip", 40, 30, arrow={"tail": "bar"}) is ax
    assert ax.text_count() == 3

    p, c, t = (ax.text_data(i) for i in range(3))
    assert (p.text, p.x, p.y, p.z, p.in_frame, p.arrow) == ("p", 0.5, 0.5, 0.5, False, False)
    assert c.in_frame and (c.x, c.y) == (0.02, 0.95)
    assert t.arrow and (t.dx, t.dy) == (40.0, 30.0)
    assert "Text3DData(text='tip'" in repr(t)

    # The short form keeps the rest: a text2d stays in the frame.
    ax.set_text_data(1, "moved", 0.5, 0.5)
    assert ax.text_data(1).in_frame and ax.text_data(1).text == "moved"

    t.dx = 10
    ax.set_text_data(2, t)
    assert ax.text_data(2).dx == 10.0
    with pytest.raises(ValueError):
        ax.set_text_data(0, sextant.Text3DData("x", 0, 0, 0, in_frame=True, arrow=True))

    svg = fig.render_svg()[0]
    assert ">moved</text>" in svg and ">tip</text>" in svg
    fig.render_rgba()


def test_3d_text_size_ignores_the_camera():
    fig = sextant.Figure(width=400, height=300)
    ax = fig.add_subplot3d(1, 1, 1)
    ax.scatter3d([0, 1], [0, 1], [0, 1], color="black")
    ax.text2d("WWWW", 0.1, 0.1, fontsize=28, color="#ff0000")
    a = near(fig.render_rgba(), (255, 0, 0), 60)
    ax.set_camera(zoom=2.0)
    b = near(fig.render_rgba(), (255, 0, 0), 60)
    assert a > 50 and abs(a - b) <= a * 0.05


# --- Math in text (sextant 1.1 step 32a) ---------------------------------------


def test_math_in_every_string_reaches_the_svg_as_tspans():
    fig, ax = axes()
    ax.line([0, 1], [0, 1], name=r"$\alpha_1$")
    ax.set_title(r"$x^2$ [m]")
    ax.text(r"$\beta^2$", 5, 5)
    ax.legend()
    svg = fig.render_svg()[0]
    assert '<tspan font-style="italic">x</tspan><tspan dx=' in svg  # italic, its script kerned (32b)
    assert ">\u03b1</tspan>" in svg and ">\u03b2</tspan>" in svg


def test_parse_math_off_per_text_and_per_figure():
    fig, ax = axes()
    ax.text(r"$a^2$ raw", 2, 2, parse_math=False)
    ax.text(r"$b^2$ rich", 6, 6)
    svg = fig.render_svg()[0]
    assert ">$a^2$ raw</text>" in svg
    assert '<tspan font-style="italic">b</tspan>' in svg
    off = sextant.Figure(width=400, height=300, mathtext=False)
    off.axes().set_title(r"$x^2$")
    assert ">$x^2$</text>" in off.render_svg()[0]


def test_malformed_math_warns_and_draws_as_written():
    fig, ax = axes()
    with pytest.warns(RuntimeWarning, match=r"set_title: math not parsed at column 2: unknown command"):
        ax.set_title(r"$\alp$")
    assert ax.title() == r"$\alp$"
    assert r">$\alp$</text>" in fig.render_svg()[0]
