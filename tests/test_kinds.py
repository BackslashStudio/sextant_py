"""Plot-kind options through the core binding: the marker shapes, step
drawstyles, per-point scatter sizes, bar bases and horizontal bars, and the
reference lines and spans (axline/axspan)."""

import math

import numpy as np
import pytest

import sextant

RED = (1.0, 0.0, 0.0, 1.0)


@pytest.fixture
def fig():
    return sextant.Figure(width=320, height=240)


@pytest.fixture
def ax(fig):
    return fig.axes()


def red_pixels(img):
    r, g, b = (img[..., i].astype(int) for i in range(3))
    return int(((r > 180) & (g < 80) & (b < 80)).sum())


def red_box(img):
    """(x0, x1, y0, y1) of the red pixels, y down; None without any."""
    r, g, b = (img[..., i].astype(int) for i in range(3))
    ys, xs = np.nonzero((r > 180) & (g < 80) & (b < 80))
    return None if not len(xs) else (xs.min(), xs.max(), ys.min(), ys.max())


# --- markers -------------------------------------------------------------------------

def test_marker_names_and_matplotlib_codes():
    M = sextant.MarkerStyle
    assert [M(c) for c in "v<>*ph"] == [M.TRIANGLE_DOWN, M.TRIANGLE_LEFT, M.TRIANGLE_RIGHT, M.STAR,
                                        M.PENTAGON, M.HEXAGON]
    # matplotlib's "x" and "+" are strokes; its filled ones are "X" and "P".
    assert [M(c) for c in "x+XP|_"] == [M.THIN_CROSS, M.THIN_PLUS, M.CROSS, M.PLUS, M.VLINE, M.HLINE]
    assert M("Thin-Cross") is M.THIN_CROSS


@pytest.mark.parametrize("marker", ["triangle_down", "triangle_left", "triangle_right", "star", "pentagon",
                                    "hexagon", "thin_cross", "thin_plus", "vline", "hline", "x", "|"])
def test_every_new_marker_draws(fig, ax, marker):
    ax.scatter([0.5], [0.5], marker=marker, size=30, color="red", alpha=1.0).set_xlim(0, 1).set_ylim(0, 1)
    assert red_pixels(fig.render_rgba()) > 20


def test_line_marker_stroke_is_edge_linewidth(fig, ax):
    ax.set_xlim(0, 1).set_ylim(0, 1)
    ax.scatter([0.5], [0.5], marker="hline", size=60, color="red", alpha=1.0)
    thin = red_box(fig.render_rgba())
    ax.cla()
    ax.set_xlim(0, 1).set_ylim(0, 1)
    ax.scatter([0.5], [0.5], marker="hline", size=60, color="red", alpha=1.0, edge_linewidth=8)
    thick = red_box(fig.render_rgba())
    assert thin[3] - thin[2] <= 3 < thick[3] - thick[2]  # the stroke's height, not the length
    assert abs((thin[1] - thin[0]) - (thick[1] - thick[0])) <= 2


# --- drawstyle ------------------------------------------------------------------------

def test_step_drawstyles(fig, ax):
    x, y = [0, 1, 2], [0, 2, 1]

    def picture(**kw):
        ax.cla()
        ax.line(x, y, color="red", **kw).set_xlim(-0.5, 2.5).set_ylim(-0.5, 2.5)
        return fig.render_rgba()

    plain = picture()
    pre = picture(drawstyle="steps_pre")
    assert not np.array_equal(plain, pre)
    np.testing.assert_array_equal(picture(drawstyle="steps"), pre)  # matplotlib's "steps"
    np.testing.assert_array_equal(picture(drawstyle="steps-pre"), pre)
    mid, post = picture(drawstyle=sextant.DrawStyle.STEPS_MID), picture(drawstyle="steps-post")
    assert not np.array_equal(mid, pre) and not np.array_equal(post, pre) and not np.array_equal(mid, post)
    np.testing.assert_array_equal(ax.line_data(0).y, y)  # the data stay the points
    with pytest.raises(TypeError, match="drawstyle"):
        ax.line(x, y, drawstyle="zigzag")


# --- per-point sizes -------------------------------------------------------------------

def test_scatter_sizes_read_back_and_update(ax):
    ax.scatter([0, 1, 2], [0, 1, 2], sizes=np.array([10.0, 20.0, 30.0]))
    ax.scatter([0, 1, 2], [0, 1, 2])
    np.testing.assert_array_equal(ax.scatter_data(0).sizes, [10, 20, 30])
    assert ax.scatter_data(1).sizes is None

    ax.set_scatter_data(0, [5, 6, 7], [5, 6, 7])  # the (x, y) form keeps them while the count holds
    np.testing.assert_array_equal(ax.scatter_data(0).sizes, [10, 20, 30])
    d = ax.scatter_data(0)
    d.sizes = [1, 2, 3]
    ax.set_scatter_data(0, d)  # the object form takes them as given
    np.testing.assert_array_equal(ax.scatter_data(0).sizes, [1, 2, 3])
    ax.set_scatter_data(0, sextant.ScatterData([0, 1], [0, 1]))  # None clears them
    assert ax.scatter_data(0).sizes is None
    with pytest.raises(ValueError):
        ax.set_scatter_data(1, sextant.ScatterData([0, 1], [0, 1], sizes=[1, 2, 3]))


def test_sizes_take_arrays_lists_and_none(ax):
    ax.scatter([0, 1], [0, 1], sizes=[4, 8]).scatter([0, 1], [0, 1], sizes=None)
    ax.scatter([0, 1], [0, 1], sizes=np.arange(4.0)[::2])  # a strided view
    np.testing.assert_array_equal(ax.scatter_data(2).sizes, [0, 2])
    assert ax.scatter_data(1).sizes is None
    with pytest.raises(ValueError):
        ax.scatter([0, 1, 2], [0, 1, 2], sizes=[4, 8])  # neither one per point nor empty
    with pytest.raises(TypeError, match="bad value for 'sizes'"):
        ax.scatter([0, 1], [0, 1], sizes="big")
    with pytest.raises(TypeError, match="bad value for 'sizes'"):
        ax.scatter([0, 1], [0, 1], sizes=[[1, 2]])


def test_sizes_reach_the_pixels(fig, ax):
    ax.set_xlim(0, 4).set_ylim(0, 1)
    ax.scatter([1, 3], [0.5, 0.5], sizes=[10, 40], color="red", alpha=1.0, marker="square")
    img = fig.render_rgba()
    half = img.shape[1] // 2
    assert 10 * red_pixels(img[:, :half]) < red_pixels(img[:, half:])
    ax.set_scatter_data(0, sextant.ScatterData([1, 3], [0.5, 0.5], sizes=[40, float("nan")]))
    img = fig.render_rgba()
    assert red_pixels(img[:, :half]) > 0 and red_pixels(img[:, half:]) == 0  # NaN hides a point


def test_scatter_z_and_scatter3d_sizes(fig):
    ax = fig.axes()
    ax.scatter_z([0, 1], [0, 1], [0, 1], sizes=[5, 9])
    np.testing.assert_array_equal(ax.scatter_z_data(0).sizes, [5, 9])
    ax.set_scatter_z_data(0, sextant.ScatterZData([0, 1], [0, 1], [1, 0]))
    assert ax.scatter_z_data(0).sizes is None

    ax3 = sextant.Figure(width=320, height=240).add_subplot3d(1, 1, 1)
    ax3.scatter3d([0, 1], [0, 1], [0, 1], sizes=[6, 12])
    d = ax3.scatter3d_data(0)
    np.testing.assert_array_equal(d.sizes, [6, 12])
    assert d.colors is None
    ax3.set_scatter3d_data(0, sextant.Scatter3DData([0, 1], [0, 1], [0, 1], colors=[0, 1], sizes=[1, 2]))
    d = ax3.scatter3d_data(0)
    np.testing.assert_array_equal(d.sizes, [1, 2])
    np.testing.assert_array_equal(d.colors, [0, 1])


def test_plane_scatter_sizes():
    ax3 = sextant.Figure(width=320, height=240).add_subplot3d(1, 1, 1)
    pl = ax3.plane("xy", 0.0)
    pl.scatter([0, 1], [0, 1], sizes=[3, 4]).bar([0, 1], [1, 2], bottom=0.5, orientation="horizontal")
    np.testing.assert_array_equal(pl.scatter_data(0).sizes, [3, 4])
    np.testing.assert_array_equal(pl.bar_data(0).bottom, [0.5, 0.5])
    pl.set_bar_data(0, sextant.BarData([0, 1], [1, 2]))
    assert pl.bar_data(0).bottom is None


# --- bars ------------------------------------------------------------------------------

def test_bar_bottom_read_back_and_update(ax):
    ax.bar([0, 1, 2], [1, 2, 3], bottom=2.0)  # one value: every bar
    ax.bar([0, 1, 2], [1, 2, 3], bottom=[0, 1, 2])
    ax.bar([0, 1, 2], [1, 2, 3])
    np.testing.assert_array_equal(ax.bar_data(0).bottom, [2, 2, 2])
    np.testing.assert_array_equal(ax.bar_data(1).bottom, [0, 1, 2])
    assert ax.bar_data(2).bottom is None
    assert ax.ylim()[1] >= 5  # the tallest bar ends at 2 + 3
    ax.set_bar_data(1, [0, 1, 2], [3, 3, 3])
    np.testing.assert_array_equal(ax.bar_data(1).bottom, [0, 1, 2])
    ax.set_bar_data(1, sextant.BarData([0, 1, 2], [3, 3, 3], bottom=[1, 1, 1]))
    np.testing.assert_array_equal(ax.bar_data(1).bottom, [1, 1, 1])
    with pytest.raises(ValueError):
        ax.bar([0, 1, 2], [1, 2, 3], bottom=[0, 1])


def test_horizontal_bars_lie_along_x(fig, ax):
    ax.bar([1, 2, 3], [10, 20, 30], orientation="horizontal", color="red")
    assert ax.xlim()[1] >= 30 and ax.ylim()[1] < 10  # lengths along x, positions along y
    img = fig.render_rgba()
    x0, x1, y0, y1 = red_box(img)
    row = lambda y: red_pixels(img[y:y + 1])  # noqa: E731
    assert row(y0 + 3) > 2.5 * row(y1 - 3)  # the top bar (30 long) against the bottom one (10)
    ax.cla()
    ax.hist([1, 2, 2, 3, 3, 3], bins=3, orientation=sextant.Orientation.HORIZONTAL)
    assert ax.xlim()[1] >= 3


# --- reference lines and spans -----------------------------------------------------------

def test_axline_and_axspan_read_back_and_update(ax):
    ax.axline(2.5).axline(1.0, orientation="vertical")
    ax.axspan(4, 3)  # stored ordered
    assert (ax.axline_count(), ax.axspan_count()) == (2, 1)
    assert ax.axline_data(0).value == 2.5 and ax.axline_data(-1).value == 1.0
    d = ax.axspan_data(0)
    assert (d.lo, d.hi) == (3, 4)
    ax.set_axline_data(0, 7.0).set_axline_data(1, sextant.AxLineData(8.0))
    assert [ax.axline_data(i).value for i in range(2)] == [7.0, 8.0]
    ax.set_axspan_data(0, 9, 5)
    assert (ax.axspan_data(0).lo, ax.axspan_data(0).hi) == (5, 9)
    ax.set_axspan_data(0, sextant.AxSpanData(1, 2))
    assert repr(ax.axspan_data(0)) == "AxSpanData(lo=1.0, hi=2.0)"
    with pytest.raises(IndexError):
        ax.axline_data(2)
    ax.cla()
    assert (ax.axline_count(), ax.axspan_count()) == (0, 0)


def test_reference_values_are_checked(ax):
    for bad in (lambda: ax.axline(math.nan), lambda: ax.axspan(0, math.inf),
                lambda: ax.axline(1, from_=math.nan)):
        with pytest.raises(ValueError):
            bad()
    assert ax.axline_count() == 0 and ax.axspan_count() == 0


def test_from_is_from_underscore(ax):
    ax.axline(0.5, from_=0.25, to=0.75).axspan(0, 1, from_=0.1)
    with pytest.raises(TypeError, match="unexpected keyword argument 'from'.*from_, to"):
        ax.axline(0.5, **{"from": 0.25})


def test_limits_take_the_reference_values(ax):
    ax.line([0, 1], [0, 1]).axline(5.0).axspan(-3, -2, orientation="vertical")
    assert ax.ylim()[1] >= 5 and ax.xlim()[0] <= -3


def test_reference_pixels(fig, ax):
    ax.set_xlim(0, 10).set_ylim(0, 10)
    ax.axline(5, color="red", linewidth=3, from_=0.5)  # the right half of the frame only
    x0, x1, y0, y1 = red_box(fig.render_rgba())
    assert y1 - y0 <= 4 and x1 - x0 > 60
    ax.cla()
    ax.set_xlim(0, 10).set_ylim(0, 10)
    ax.axspan(2, 4, orientation="vertical", color="red", alpha=1.0)
    x0, x1, y0, y1 = red_box(fig.render_rgba())
    assert y1 - y0 > 2 * (x1 - x0)  # a band across y, two units wide along x


def test_axline_takes_the_color_cycle_and_axspan_does_not(fig, ax):
    ax.set_color_cycle(["red", "blue"]).set_xlim(0, 1).set_ylim(0, 1)
    ax.axline(0.5)
    assert red_pixels(fig.render_rgba()) > 50
    ax.cla()
    ax.set_color_cycle(["red"]).axspan(0.2, 0.8, alpha=1.0)  # unset: blue
    assert red_pixels(fig.render_rgba()) == 0


def test_legend_keys_reference_objects(fig, ax):
    ax.line([0, 1], [0, 1])
    before, _ = fig.render_svg()
    ax.axline(0.5, name="limit").axspan(0.1, 0.2, name="band").legend()
    svg, _ = fig.render_svg()
    assert "limit" in svg and "band" in svg and "limit" not in before
