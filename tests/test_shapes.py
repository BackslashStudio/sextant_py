"""sextant 1.3 stage 2 through the core binding: pixel positions, the shapes (poly, circle,
wedge) on Axes and Plane2D, sloped axline, and Plane2D.axspan."""

import math
import re

import numpy as np
import pytest

import sextant


def axes(width=200, height=150, xlim=(0, 10), ylim=(0, 2)):
    fig = sextant.Figure(width=width, height=height)
    ax = fig.axes()
    ax.set_xlim(*xlim).set_ylim(*ylim)
    return fig, ax


def frame(svg):
    """(x0, x1, top, bottom) of the plot frame in SVG pixels (y down)."""
    m = re.search(r'd="M([\d.]+) ([\d.]+)H([\d.]+) M[\d.]+ ([\d.]+)H', svg)
    x0, bottom, x1, top = (float(g) for g in m.groups())
    return x0, x1, top, bottom


def fills(svg):
    """[(rgb, [(x, y), ...])] of the filled paths, in drawing order."""
    out = []
    for m in re.finditer(r'<path fill="rgb\((\d+),(\d+),(\d+)\)"[^>]* d="([^"]+)"', svg):
        pts = [tuple(float(v) for v in p.split(",")) for p in re.findall(r"[ML]([-\d.e]+,[-\d.e]+)", m.group(4))]
        out.append(((int(m.group(1)), int(m.group(2)), int(m.group(3))), pts))
    return out


def box(pts):
    xs, ys = zip(*pts)
    return min(xs), max(xs), min(ys), max(ys)


def area(pts):
    return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))) / 2


def polylines(svg):
    return [[tuple(float(v) for v in p.split(",")) for p in m.split()] for m in
            re.findall(r'<polyline fill="none"[^>]* points="([^"]+)"', svg)]


# --- Coords.PIXELS ---------------------------------------------------------------------

def test_new_enum_values():
    assert sextant.Coords("pixels") is sextant.Coords.PIXELS
    assert [m.value for m in sextant.AxLineMode] == ["value", "point_slope", "two_points"]
    assert sextant.AxLineMode("point-slope") is sextant.AxLineMode.POINT_SLOPE
    assert [sextant.PickKind(s) for s in ("poly", "circle", "wedge")] == [
        sextant.PickKind.POLY, sextant.PickKind.CIRCLE, sextant.PickKind.WEDGE]


def test_text_in_pixels_lands_from_the_frame_corner():
    fig, ax = axes()
    ax.text("A", 50, 40, coords="pixels")
    ax.text("B", 30, 0.5, xcoords="pixels")  # one coordinate in pixels, the other in data
    svg = fig.render_svg()[0]
    x0, x1, top, bottom = frame(svg)
    assert f'<g transform="translate({x0 + 50:g},{bottom - 40:g})">' in svg
    d = ax.text_data(0)
    assert (d.x, d.y, d.xcoords, d.ycoords) == (50.0, 40.0, "pixels", "pixels")
    assert ax.text_data(1).xcoords is sextant.Coords.PIXELS and ax.text_data(1).ycoords is sextant.Coords.DATA
    ax.set_text_data(0, "A", 10, 10)  # the short form keeps the coords
    assert ax.text_data(0).xcoords is sextant.Coords.PIXELS


# --- circle ------------------------------------------------------------------------------

def test_circle_radius_spaces():
    fig, ax = axes()
    ax.shape_circle(5, 1, 1, color="#ff0000")  # data radius: an ellipse, 1 unit along each axis
    ax.shape_circle(100, 50, 20, coords="pixels", color="#00ff00")
    ax.shape_circle(0.5, 0.5, 0.25, coords="fraction", color="#0000ff")
    svg = fig.render_svg()[0]
    x0, x1, top, bottom = frame(svg)
    px, py = (x1 - x0) / 10, (bottom - top) / 2
    red, green, blue = (box(p) for _, p in fills(svg))
    assert red[1] - red[0] == pytest.approx(2 * px, rel=0.02) and red[3] - red[2] == pytest.approx(2 * py, rel=0.02)
    assert (red[0] + red[1]) / 2 == pytest.approx(x0 + 5 * px, abs=0.1)
    assert green[1] - green[0] == pytest.approx(40, abs=0.5) and green[3] - green[2] == pytest.approx(40, abs=0.5)
    assert (green[0] + green[1]) / 2 == pytest.approx(x0 + 100, abs=0.1)  # pixels from the frame's left
    assert (green[2] + green[3]) / 2 == pytest.approx(bottom - 50, abs=0.1)  # and bottom, y up
    side = 0.25 * min(x1 - x0, bottom - top)  # a fraction radius: of the frame's shorter side, round
    assert blue[1] - blue[0] == pytest.approx(2 * side, abs=0.5) and blue[3] - blue[2] == pytest.approx(2 * side, abs=0.5)
    assert (blue[0] + blue[1]) / 2 == pytest.approx((x0 + x1) / 2, abs=0.1)


def test_circle_coords_spellings():
    fig, ax = axes()
    ax.shape_circle(5, 1, 20, rcoords="pixels")  # only the radius in pixels, a data centre
    ax.shape_circle(0.5, 1, 20, xcoords="fraction", rcoords=sextant.Coords.PIXELS)
    ax.shape_circle(0.5, 0.5, 0.1, coords="fraction", ycoords="pixels")  # y wins over coords: 0.5 px
    a, b, c = (ax.shape_circle_data(i) for i in range(3))
    assert (a.cxcoords, a.cycoords, a.rcoords) == ("data", "data", "pixels")
    assert (b.cxcoords, b.cycoords, b.rcoords) == ("fraction", "data", "pixels")
    assert (c.cxcoords, c.cycoords, c.rcoords) == ("fraction", "pixels", "fraction")
    # the centre of a and b is the same point: a data 5 of 10 is half way
    svg = fig.render_svg()[0]
    pa, pb, _ = (box(p) for _, p in fills(svg))
    assert (pa[0] + pa[1]) == pytest.approx(pb[0] + pb[1], abs=0.1)


def test_circle_read_back_and_edit():
    _, ax = axes()
    assert ax.shape_circle(5, 1, 1, name="c") is ax
    d = ax.shape_circle_data(0)
    assert (d.cx, d.cy, d.r) == (5.0, 1.0, 1.0)
    assert d.cxcoords is sextant.Coords.DATA
    assert "CircleData(cx=5.0" in repr(d)
    ax.set_shape_circle_data(0, 6, 1.5, 2)  # keeps each coords
    assert (ax.shape_circle_data(0).cx, ax.shape_circle_data(0).rcoords) == (6.0, "data")
    ax.set_shape_circle_data(0, 6, 1.5, 20, rcoords="pixels")
    assert ax.shape_circle_data(0).rcoords is sextant.Coords.PIXELS
    ax.set_shape_circle_data(0, 6, 1.5, 3)  # the pixels radius stays pixels
    assert (ax.shape_circle_data(0).r, ax.shape_circle_data(0).rcoords) == (3.0, "pixels")
    ax.set_shape_circle_data(-1, sextant.CircleData(1, 1, 0.5, "fraction", "fraction", "fraction"))
    assert ax.shape_circle_data(0).cxcoords is sextant.Coords.FRACTION
    with pytest.raises(IndexError):
        ax.shape_circle_data(1)


# --- wedge ---------------------------------------------------------------------------------

def test_wedge_areas():
    fig, ax = axes()
    ax.shape_wedge(100, 60, 40, 0, 90, coords="pixels", color="#ff0000")
    ax.shape_wedge(100, 60, 40, 90, 360, coords="pixels", inner=20, color="#00ff00")
    ax.shape_wedge(100, 60, 40, 270, 90, coords="pixels", color="#0000ff")  # theta2 < theta1 wraps
    ax.shape_wedge(100, 60, 40, 0, 360, coords="pixels", color="#ffff00")
    quarter, ring, half, whole = (area(p) for _, p in fills(fig.render_svg()[0]))
    full = math.pi * 40 ** 2
    assert quarter == pytest.approx(full / 4, rel=0.02)
    assert ring == pytest.approx(full * 0.75 * (1 - 0.25), rel=0.02)  # three quarters of an annulus
    assert half == pytest.approx(full / 2, rel=0.02)
    assert whole == pytest.approx(full, rel=0.02)


def test_wedge_read_back_and_edit():
    _, ax = axes()
    ax.shape_wedge(5, 1, 2, 10, 80, inner=0.5, rcoords="data")
    d = ax.shape_wedge_data(0)
    assert (d.cx, d.cy, d.r, d.theta1, d.theta2, d.inner) == (5.0, 1.0, 2.0, 10.0, 80.0, 0.5)
    ax.set_shape_wedge_data(0, 5, 1, 3, 0, 90)  # inner is kept when left out
    assert (ax.shape_wedge_data(0).r, ax.shape_wedge_data(0).inner) == (3.0, 0.5)
    ax.set_shape_wedge_data(0, 5, 1, 3, 0, 90, inner=0)
    assert ax.shape_wedge_data(0).inner == 0.0
    ax.set_shape_wedge_data(0, sextant.WedgeData(1, 1, 2, 0, 45, 1.0, "pixels", "pixels", "pixels"))
    assert ax.shape_wedge_data(0).cxcoords is sextant.Coords.PIXELS and ax.shape_wedge_data(0).inner == 1.0
    assert "WedgeData(cx=1.0" in repr(ax.shape_wedge_data(0))


# --- poly ------------------------------------------------------------------------------------

def test_poly_spaces_and_read_back():
    fig, ax = axes()
    ax.shape_poly([1, 3, 2], [0.5, 0.5, 1.5], color="#ff0000")
    ax.shape_poly([0, 1, 0], [0, 0, 1], space="fraction", color="#00ff00")
    ax.shape_poly([10, 60, 10], [10, 10, 40], space="pixels", color="#0000ff")
    svg = fig.render_svg()[0]
    x0, x1, top, bottom = frame(svg)
    red, green, blue = (box(p) for _, p in fills(svg))
    px, py = (x1 - x0) / 10, (bottom - top) / 2
    assert red[0] == pytest.approx(x0 + px, abs=0.1) and red[1] == pytest.approx(x0 + 3 * px, abs=0.1)
    assert green[:2] == pytest.approx((x0, x0 + (x1 - x0)), abs=0.1) and green[3] == pytest.approx(bottom, abs=0.1)
    assert blue == pytest.approx((x0 + 10, x0 + 60, bottom - 40, bottom - 10), abs=0.1)
    d = ax.shape_poly_data(0)
    np.testing.assert_array_equal(d.x, [1, 3, 2])
    assert ax.shape_poly_count() == 3 and "PolyData(x=array(" in repr(d)
    ax.set_shape_poly_data(0, [0, 1, 1], [0, 0, 1])
    ax.set_shape_poly_data(1, sextant.PolyData([0, 2, 1, 1], [0, 0, 1, 0.5]))
    np.testing.assert_array_equal(ax.shape_poly_data(0).y, [0, 0, 1])
    assert len(ax.shape_poly_data(1).x) == 4


def test_poly_takes_array_likes():
    _, ax = axes()
    ax.shape_poly(np.array([1, 2, 3], dtype=np.float32), (0, 1, 0))
    ax.shape_poly(np.arange(12.0)[::4], np.arange(3.0))  # a strided view
    assert ax.shape_poly_count() == 2


# --- options, identity, errors -----------------------------------------------------------------

def test_style_options_and_visible():
    fig, ax = axes()
    before = fig.render_rgba()
    ax.shape_circle(5, 1, 1, color="red", alpha=0.5, edgecolor="blue", linewidth=2, linestyle="dashed",
                    name="disc", zorder=3, visible=False)
    assert np.array_equal(fig.render_rgba(), before)  # hidden: nothing drawn
    assert ax.shape_circle_visible(0) is False
    ax.set_shape_circle_visible(0, True)
    assert not np.array_equal(fig.render_rgba(), before)
    ax.shape_wedge(5, 1, 1, 0, 90, show_legend=False, name="w")
    ax.shape_poly([1, 2, 1.5], [0, 0, 1], name="p")
    ax.set_shape_poly_visible(0, False)
    assert ax.shape_poly_visible(0) is False and ax.shape_wedge_visible(0) is True
    with pytest.raises(IndexError):
        ax.set_shape_wedge_visible(3, True)


def test_cycle_colour_ids_zorder_and_remove():
    fig, ax = axes()
    ax.line([0, 1], [0, 1])
    ax.shape_circle(5, 1, 1)
    c = ax.last_object()
    ax.shape_wedge(5, 1, 1, 0, 90)
    w = ax.last_object()
    ax.shape_poly([1, 2, 1.5], [0, 0, 1])
    p = ax.last_object()
    assert ax.find_object(c) == (sextant.PickKind.CIRCLE, 0)
    assert ax.find_object(w) == (sextant.PickKind.WEDGE, 0)
    assert ax.find_object(p) == (sextant.PickKind.POLY, 0)
    colours = {rgb for rgb, _ in fills(fig.render_svg()[0])}
    assert len(colours) == 3  # each took the next colour of the cycle
    assert ax.set_zorder(c, 5.0) is True and ax.zorder(c) == 5.0
    assert ax.remove(w) is True
    assert ax.shape_wedge_count() == 0 and ax.shape_circle_count() == 1
    assert ax.find_object(w) is None
    ax.cla()
    assert ax.shape_circle_count() == ax.shape_poly_count() == 0


def test_legend_entry():
    fig, ax = axes()
    ax.shape_circle(5, 1, 1, name="a disc")
    ax.legend()
    assert "a disc" in fig.render_svg()[0]


def test_limits_take_data_shapes_only():
    def limits(*shapes):
        ax = sextant.Figure(width=200, height=150).axes()
        ax.line([0, 1], [0, 1])
        for make in shapes:
            make(ax)
        return ax.xlim()

    base = limits()
    assert limits(lambda ax: ax.shape_circle(100, 100, 5))[1] >= 105
    assert limits(lambda ax: ax.shape_poly([50, 60, 55], [0, 0, 1]))[1] >= 60
    assert limits(lambda ax: ax.shape_circle(100, 100, 5, coords="pixels", rcoords="pixels")) == base
    assert limits(lambda ax: ax.shape_circle(0.5, 0.5, 0.2, coords="fraction")) == base
    assert limits(lambda ax: ax.shape_poly([50, 60, 55], [0, 0, 1], space="pixels")) == base
    assert limits(lambda ax: ax.shape_wedge(100, 100, 5, 0, 90))[1] >= 100


@pytest.mark.parametrize("call", [
    lambda ax: ax.shape_circle(1, 1, 0),
    lambda ax: ax.shape_circle(1, 1, -2),
    lambda ax: ax.shape_circle(math.nan, 1, 1),
    lambda ax: ax.shape_circle(0.5, 1, 1, xcoords="fraction"),  # a data radius needs a data centre
    lambda ax: ax.shape_wedge(1, 1, 1, 0, 90, inner=1),
    lambda ax: ax.shape_wedge(1, 1, 1, 0, math.inf),
    lambda ax: ax.shape_wedge(1, 1, 2, 0, 90, inner=-1),
    lambda ax: ax.shape_poly([0, 1], [0, 1]),
    lambda ax: ax.shape_poly([0, 1, 2], [0, 1]),
    lambda ax: ax.shape_poly([0, 1, math.nan], [0, 1, 2]),
])
def test_invalid_shapes_add_nothing(call):
    _, ax = axes()
    with pytest.raises(ValueError):
        call(ax)
    assert ax.shape_circle_count() == ax.shape_wedge_count() == ax.shape_poly_count() == 0


def test_invalid_edits_keep_the_shape():
    _, ax = axes()
    ax.shape_circle(5, 1, 1)
    with pytest.raises(ValueError):
        ax.set_shape_circle_data(0, 5, 1, -1)
    assert ax.shape_circle_data(0).r == 1.0


def test_bad_option_names():
    _, ax = axes()
    with pytest.raises(TypeError, match="unexpected keyword argument 'colour'"):
        ax.shape_circle(1, 1, 1, colour="red")
    with pytest.raises(TypeError):
        ax.shape_circle(1, 1, 1, coords="inches")
    with pytest.raises(TypeError, match="inner"):
        ax.shape_circle(1, 1, 1, inner=0.5)  # a wedge's field


# --- Plane2D -------------------------------------------------------------------------------------

def plane():
    fig = sextant.Figure(width=200, height=150)
    ax3 = fig.add_subplot3d(1, 1, 1)
    return fig, ax3, ax3.plane("xy", 0.0)


def test_plane_shapes_and_axspan():
    fig, ax3, p = plane()
    before = fig.render_rgba()
    assert p.shape_circle(0, 0, 1, color="red") is p
    p.shape_wedge(0, 0, 1, 0, 90, inner=0.2).shape_poly([0, 1, 1], [0, 0, 1])
    p.axspan(0, 1, color="blue", orientation="vertical")
    assert (p.shape_circle_count(), p.shape_wedge_count(), p.shape_poly_count(), p.axspan_count()) == (1, 1, 1, 1)
    assert not np.array_equal(fig.render_rgba(), before)
    assert p.shape_circle_data(0).r == 1.0 and p.shape_wedge_data(0).inner == 0.2
    assert (p.axspan_data(0).lo, p.axspan_data(0).hi) == (0.0, 1.0)
    p.set_axspan_data(0, 2, 1)  # stored swapped
    assert (p.axspan_data(0).lo, p.axspan_data(0).hi) == (1.0, 2.0)
    p.set_axspan_data(0, sextant.AxSpanData(0, 3))
    assert p.axspan_data(0).hi == 3.0
    p.set_shape_circle_data(0, 1, 1, 2).set_shape_poly_data(0, [0, 2, 2], [0, 0, 2])
    p.set_shape_wedge_data(0, sextant.WedgeData(0, 0, 2, 0, 180))
    assert p.shape_circle_data(0).r == 2.0
    p.set_axspan_visible(0, False).set_shape_poly_visible(0, False)
    assert p.axspan_visible(0) is False and p.shape_poly_visible(0) is False
    c = p.shape_circle_count()
    p.shape_circle(0, 0, 1)
    assert p.find_object(p.last_object()) == (sextant.PickKind.CIRCLE, c)
    with pytest.raises(ValueError):
        p.shape_circle(0, 0, 0)
    assert not hasattr(p, "axline")  # no axline on a plane


def test_plane_pixels_are_on_the_planes_sheet():
    fig, ax3, p = plane()
    p.shape_circle(30, 30, 10, coords="pixels", rcoords="pixels")
    assert p.shape_circle_data(0).rcoords is sextant.Coords.PIXELS
    fig.render_rgba()
    assert "<svg" in fig.render_svg()[0]


# --- sloped axline --------------------------------------------------------------------------------

def test_axline_forms_and_read_back():
    _, ax = axes(xlim=(0, 10), ylim=(0, 10))
    assert ax.axline(5) is ax
    ax.axline(1, 2, slope=0.5)
    ax.axline(1, 2, 3, 4, color="red")
    assert ax.axline_count() == 3
    v, ps, tp = (ax.axline_data(i) for i in range(3))
    assert (v.mode, v.value) == ("value", 5.0)
    assert v.mode is sextant.AxLineMode.VALUE
    assert (ps.mode, ps.x, ps.y, ps.slope, ps.xcoords) == ("point_slope", 1.0, 2.0, 0.5, "data")
    assert (tp.mode, tp.x, tp.y, tp.x2, tp.y2) == ("two_points", 1.0, 2.0, 3.0, 4.0)
    assert "AxLineData(value=" in repr(tp)
    # the value form still moves a value line
    ax.set_axline_data(0, 6.0)
    assert ax.axline_data(0).value == 6.0
    ax.set_axline_data(1, sextant.AxLineData(mode="point_slope", x=2, y=2, slope=1))
    assert ax.axline_data(1).x == 2.0 and ax.axline_data(1).slope == 1.0
    ax.set_axline_data(2, ps)  # a data object read back goes back in
    assert ax.axline_data(2).mode is sextant.AxLineMode.POINT_SLOPE
    ax.set_axline_visible(1, False)
    assert ax.axline_visible(1) is False


def test_axline_coords_spellings():
    _, ax = axes()
    ax.axline(0.5, 20, slope=1, xcoords="fraction", ycoords="pixels")
    ax.axline(0.1, 0.1, 0.9, 0.8, coords="fraction")
    a, b = ax.axline_data(0), ax.axline_data(1)
    assert (a.xcoords, a.ycoords) == ("fraction", "pixels")
    assert (b.xcoords, b.ycoords, b.x2coords, b.y2coords) == ("fraction",) * 4


def test_axline_screen_and_data_lines():
    fig, ax = axes(xlim=(0, 10), ylim=(0, 10))
    ax.axline(0, 0, slope=1, color="#ff0000")  # data: the diagonal of the frame
    ax.axline(50, 20, slope=1, coords="pixels", color="#00ff00")  # screen: 45 degrees, y up
    ax.axline(2, 0, 2, 5, color="#0000ff")  # equal x: vertical
    svg = fig.render_svg()[0]
    x0, x1, top, bottom = frame(svg)
    red, green, blue = polylines(svg)
    # the infinite lines run past the frame (clipped there): test points on them
    ratio = (bottom - top) / (x1 - x0)  # one y unit over one x unit, in pixels
    assert len(red) == 2 and all(bottom - y == pytest.approx((x - x0) * ratio, abs=0.05) for x, y in red)
    assert all(bottom - y == pytest.approx((x - x0) - 50 + 20, abs=0.05) for x, y in green)  # 45 degrees, y up
    assert blue[0][0] == pytest.approx(blue[-1][0], abs=1e-3) == pytest.approx(x0 + 0.2 * (x1 - x0), abs=0.05)
    assert blue[0][1] != blue[-1][1]


def test_axline_bends_on_log_axes():
    fig, ax = axes(xlim=(1, 1000), ylim=(1, 1000))
    ax.set_xscale("log").set_yscale("log")
    ax.axline(1, 1, slope=1, color="#ff0000")  # y = x - 0: not straight on log axes
    ax.axline(10, 10, slope=1, coords="data", color="#00ff00")
    pl = polylines(fig.render_svg()[0])
    assert all(len(p) > 2 for p in pl)
    lin = axes(xlim=(0, 10), ylim=(0, 10))
    lin[1].axline(1, 1, slope=1)
    assert len(polylines(lin[0].render_svg()[0])[0]) == 2  # straight on linear axes


def test_axline_limits_take_the_data_points():
    fresh = sextant.Figure(width=200, height=150).axes()
    fresh.axline(100, 200, slope=1)
    lo, hi = fresh.xlim()
    assert lo <= 100 <= hi


@pytest.mark.parametrize("call", [
    lambda ax: ax.axline(1, 2),  # a point with neither slope nor second point
    lambda ax: ax.axline(1, 2, 3),
    lambda ax: ax.axline(1, 2, 3, 4, slope=1),  # both
    lambda ax: ax.axline(1, slope=1),  # a slope with no second coordinate
    lambda ax: ax.axline(slope=1),
    lambda ax: ax.axline(1, 2, 3, 4, 5),
    lambda ax: ax.axline(1, from_=0.1, slope=1),
])
def test_axline_bad_shapes_of_call(call):
    _, ax = axes()
    with pytest.raises(TypeError):
        call(ax)
    assert ax.axline_count() == 0


@pytest.mark.parametrize("call", [
    lambda ax: ax.axline(1, 1, 1, 1),  # equal points
    lambda ax: ax.axline(1, 1, slope=math.nan),
    lambda ax: ax.axline(math.inf, 1, slope=1),
    lambda ax: ax.axline(1, 1, 2, math.nan),
])
def test_axline_invalid_numbers(call):
    _, ax = axes()
    with pytest.raises(ValueError):
        call(ax)
    assert ax.axline_count() == 0
