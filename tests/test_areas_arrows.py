"""sextant 1.2 steps 36d and 36e through the core binding: filled areas (fill,
stack) and arrow fields (quiver), their options, read-back and setters."""

import math

import numpy as np
import pytest

import sextant

X = np.arange(5.0)


@pytest.fixture
def fig():
    return sextant.Figure(width=320, height=240)


@pytest.fixture
def ax(fig):
    return fig.axes()


def red_mask(img):
    r, g, b = (img[..., i].astype(int) for i in range(3))
    return (r > 180) & (g < 80) & (b < 80)


def red_box(img):
    """(x0, x1, y0, y1) of the red pixels, y down; None without any."""
    ys, xs = np.nonzero(red_mask(img))
    return None if not len(xs) else (xs.min(), xs.max(), ys.min(), ys.max())


# --- enums ---------------------------------------------------------------------------

def test_new_enums():
    assert [sextant.StackBaseline(s) for s in ("zero", "sym", "wiggle", "weighted_wiggle")] == list(
        sextant.StackBaseline)
    assert sextant.QuiverPivot("mid") is sextant.QuiverPivot.MIDDLE  # matplotlib's name
    assert sextant.QuiverPivot("Tip") is sextant.QuiverPivot.TIP
    assert sextant.PickKind("quiver") is sextant.PickKind.QUIVER


# --- fill ----------------------------------------------------------------------------

def test_fill_read_back_and_update(ax):
    ax.fill(X, X, 2.0)
    ax.fill(X, X, X * 2, exclude=[False, True, False, False, False])
    assert ax.fill_count() == 2
    d = ax.fill_data(0)
    np.testing.assert_array_equal(d.y2, np.full(5, 2.0))  # a number comes back repeated
    assert d.exclude is None
    mask = ax.fill_data(-1).exclude
    assert mask.dtype == np.bool_ and mask.tolist() == [False, True, False, False, False]
    # The (x, y1, y2) form keeps the mask while the count holds, drops it otherwise.
    ax.set_fill_data(1, X, X, X + 1)
    assert ax.fill_data(1).exclude.tolist() == mask.tolist()
    np.testing.assert_array_equal(ax.fill_data(1).y2, X + 1)
    ax.set_fill_data(1, X[:3], X[:3], X[:3])
    assert ax.fill_data(1).exclude is None
    # The object form takes it as given: a mask of numbers, then None clears it.
    ax.set_fill_data(1, sextant.FillData(X, X, X, exclude=np.array([1, 0, 0, 0, 1])))
    assert ax.fill_data(1).exclude.tolist() == [True, False, False, False, True]
    d = ax.fill_data(1)
    d.exclude = None
    ax.set_fill_data(1, d)
    assert ax.fill_data(1).exclude is None
    assert "FillData(x=array(" in repr(d)
    ax.cla()
    assert ax.fill_count() == 0


def test_fill_y2_forms(ax):
    ax.fill(X, X)  # down to 0
    ax.fill(X, X, np.float64(1.5))  # a numpy scalar is a number
    ax.fill(X, X, [1, 1, 1, 1, 1])  # a list is per point
    np.testing.assert_array_equal(ax.fill_data(0).y2, np.zeros(5))
    np.testing.assert_array_equal(ax.fill_data(1).y2, np.full(5, 1.5))
    np.testing.assert_array_equal(ax.fill_data(2).y2, np.ones(5))


def test_fill_refusals(ax):
    for bad in (lambda: ax.fill(X, X, X[:3]), lambda: ax.fill(X, X, exclude=[True, False]),
                lambda: ax.set_fill_data(0, X, X, X)):
        with pytest.raises((ValueError, IndexError)):
            bad()
    for bad in (lambda: ax.fill(X, X, exclude="yes"), lambda: ax.fill(X, X, exclude=np.ones((2, 2))),
                lambda: ax.fill(X, X, exclude=["a", "b", "c", "d", "e"])):
        with pytest.raises(TypeError, match="exclude"):
            bad()
    assert ax.fill_count() == 0


def test_fill_pixels(fig, ax):
    ax.set_xlim(0, 4).set_ylim(0, 4)
    ax.fill(X, np.full(5, 3.0), 1.0, color="red")
    whole = red_mask(fig.render_rgba())
    x0, x1, y0, y1 = red_box(fig.render_rgba())
    assert x1 - x0 > 1.4 * (y1 - y0)  # a band 4 wide and 2 high
    # Excluding the middle point splits the band: its middle columns stay empty.
    ax.set_fill_data(0, sextant.FillData(X, np.full(5, 3.0), np.ones(5), exclude=[False, False, True, False,
                                                                                    False]))
    split = red_mask(fig.render_rgba())
    mid = (x0 + x1) // 2
    assert whole[:, mid].sum() > 10 and split[:, mid].sum() == 0
    # orientation='vertical' reads x as y: the band stands.
    ax.cla()
    ax.set_xlim(0, 4).set_ylim(0, 4)
    ax.fill(X, np.full(5, 3.0), 1.0, color="red", orientation="vertical")
    x0, x1, y0, y1 = red_box(fig.render_rgba())
    assert y1 - y0 > 1.4 * (x1 - x0) * 0.75  # 4 high and 2 wide, on a 4:3 frame


def test_fill_takes_the_color_cycle(fig, ax):
    ax.set_color_cycle(["red", "blue"]).set_xlim(0, 4).set_ylim(0, 4)
    ax.fill(X, X)
    assert red_mask(fig.render_rgba()).sum() > 100


def test_plane_fill_and_stack():
    f = sextant.Figure(width=200, height=150)
    pl = f.add_subplot3d(1, 1, 1).plane("xy", 0.0)
    pl.fill(X, X, 1.0).stack(X, [X, X])
    assert pl.fill_count() == 3
    np.testing.assert_array_equal(pl.fill_data(2).y2, 2 * X)
    pl.set_fill_data(0, X, X, X)
    f.render_rgba()


# --- stack ---------------------------------------------------------------------------

def test_stack_adds_one_fill_per_layer(ax):
    ys = np.array([[1.0, 2, 3, 4, 5], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2]])
    ax.stack(X, ys)
    assert ax.fill_count() == 3
    np.testing.assert_array_equal(ax.fill_data(0).y1, np.zeros(5))
    for k in range(3):  # each layer starts where the one below ends
        d = ax.fill_data(k)
        np.testing.assert_allclose(d.y2 - d.y1, ys[k])
        if k:
            np.testing.assert_allclose(d.y1, ax.fill_data(k - 1).y2)
    ax.stack(X, [list(ys[0]), ys[1]])  # a sequence of 1-D arrays
    ax.stack(X, ys[2])  # a 1-D array is one layer
    assert ax.fill_count() == 6


def test_stack_baselines(ax):
    ys = np.ones((2, 5))
    ax.stack(X, ys, baseline="sym")
    np.testing.assert_allclose(ax.fill_data(0).y1, np.full(5, -1.0))  # centred on 0
    np.testing.assert_allclose(ax.fill_data(1).y2, np.full(5, 1.0))
    ax.stack(X, ys, baseline=sextant.StackBaseline.WEIGHTED_WIGGLE)
    assert ax.fill_count() == 4


def test_stack_colors_names_and_refusals(fig, ax):
    ax.set_xlim(0, 4).set_ylim(0, 4)
    ax.stack(X, np.ones((2, 5)), colors=["red", (0, 0, 1)], names=["low", "high"]).legend()
    assert red_mask(fig.render_rgba()).sum() > 100
    svg, _ = fig.render_svg()
    assert "low" in svg and "high" in svg
    n = ax.fill_count()
    for bad in (lambda: ax.stack(X, np.ones((2, 5)), colors=["red"]),
                lambda: ax.stack(X, np.ones((2, 5)), names=["a", "b", "c"]),
                lambda: ax.stack(X, np.ones((2, 3)))):
        with pytest.raises(ValueError):
            bad()
    for bad in (lambda: ax.stack(X, np.ones((2, 2, 5))), lambda: ax.stack(X, [[1, 2], [1, 2, 3]])):
        with pytest.raises(TypeError):
            bad()
    assert ax.fill_count() == n


# --- quiver --------------------------------------------------------------------------

def test_quiver_read_back_and_update(ax):
    ax.quiver(X, X, np.ones(5), np.zeros(5))
    ax.quiver(X, X, np.ones(5), np.zeros(5), X, cmap="plasma")
    assert ax.quiver_count() == 2
    assert ax.quiver_data(0).c is None
    np.testing.assert_array_equal(ax.quiver_data(1).c, X)
    # The (x, y, u, v) form keeps c while the count holds, drops it otherwise.
    ax.set_quiver_data(1, X, X, X, X)
    np.testing.assert_array_equal(ax.quiver_data(1).c, X)
    np.testing.assert_array_equal(ax.quiver_data(1).u, X)
    ax.set_quiver_data(1, X[:2], X[:2], X[:2], X[:2])
    assert ax.quiver_data(1).c is None
    # The object form takes c as given; None makes it flat.
    ax.set_quiver_data(0, sextant.QuiverData(X, X, X, X, c=X * 2))
    np.testing.assert_array_equal(ax.quiver_data(0).c, X * 2)
    d = ax.quiver_data(0)
    d.c = None
    ax.set_quiver_data(0, d)
    assert ax.quiver_data(0).c is None
    assert repr(d).startswith("QuiverData(x=array(")
    ax.cla()
    assert ax.quiver_count() == 0


def test_quiver_refusals(ax):
    for bad in (lambda: ax.quiver(X, X, X, X[:3]), lambda: ax.quiver(X, X, X, X, X[:2]),
                lambda: ax.quiver(X, X, X, X, amp=0), lambda: ax.quiver(X, X, X, X, amp=math.nan)):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(TypeError, match="pivot"):
        ax.quiver(X, X, X, X, pivot="side")
    ax.quiver(X, X, X, X)
    with pytest.raises(ValueError):
        ax.set_quiver_data(0, sextant.QuiverData(X, X, X, X, c=X[:1]))
    assert ax.quiver_count() == 1 and ax.quiver_data(0).c is None


def arrow_box(fig, ax, **kw):
    ax.cla()
    ax.set_xlim(0, 10).set_ylim(0, 10)
    ax.quiver([5.0], [5.0], [1.0], [0.0], color="red", width=4, head_width=10, head_length=10, **kw)
    return red_box(fig.render_rgba())


def test_quiver_pixels_amp_and_pivot(fig, ax):
    x0, x1, _, _ = arrow_box(fig, ax, amp=4)
    w4 = x1 - x0
    x0b, x1b, _, _ = arrow_box(fig, ax, amp=2)
    assert abs(w4 - 2 * (x1b - x0b)) <= 3 and x0 == x0b  # amp data units per unit of vector, from the tail
    xt0, xt1, _, _ = arrow_box(fig, ax, amp=4, pivot="tip")
    assert abs(xt1 - x0) <= 4 and abs((xt1 - xt0) - w4) <= 4  # the tip on (x, y), its point anti-aliased
    xm0, xm1, _, _ = arrow_box(fig, ax, amp=4, pivot="mid")
    assert abs((xm0 + xm1) / 2 - x0) <= 3


def test_quiver_auto_amp_is_kept(fig, ax):
    ax.set_xlim(-1, 10).set_ylim(-1, 10)
    x = np.array([0.0, 8.0])
    ax.quiver(x, [5.0, 5.0], [1.0, 1.0], [0.0, 0.0], color="red", width=4, head_width=10, head_length=10)
    a0, a1, _, _ = red_box(fig.render_rgba())
    ax.set_quiver_data(0, x, [5.0, 5.0], [2.0, 2.0], [0.0, 0.0])  # twice the vectors: twice the arrows
    b0, b1, _, _ = red_box(fig.render_rgba())
    assert a0 == b0 and b1 - a1 > 10


def test_quiver_color_cycle_and_map(fig, ax):
    ax.set_color_cycle(["red", "blue"]).set_xlim(0, 10).set_ylim(0, 10)
    ax.quiver([1.0], [5.0], [1.0], [0.0], amp=5, width=6, c=[0.0], cmap="gray", vmin=-1, vmax=1)  # holds red
    ax.line([0, 10], [8, 8], linewidth=4)  # still red: the colormapped quiver did not advance the cycle
    img = fig.render_rgba()
    assert red_mask(img).sum() > 50
    gray = (np.abs(img[..., 0].astype(int) - img[..., 1]) < 6) & (np.abs(img[..., 1].astype(int) - 128) < 20)
    assert gray.sum() > 30  # c = 0 is the middle of [-1, 1]: mid gray


def test_quiver_legend_and_colorbar(fig, ax):
    ax.quiver(X, X, X, X, X, name="speed", colorbar=True)
    ax.quiver(X, X, X, X, name="flat").legend()
    svg, _ = fig.render_svg()
    assert svg.count("speed") >= 2 and "flat" in svg  # in the legend and on the colorbar


def test_plane_quiver():
    f = sextant.Figure(width=200, height=150)
    pl = f.add_subplot3d(1, 1, 1).plane("xy", 0.0)
    pl.quiver(X, X, X, X).quiver(X, X, X, X, X, amp=0.1)
    assert pl.quiver_count() == 2
    pl.set_quiver_data(1, sextant.QuiverData(X, X, X, X))
    assert pl.quiver_data(1).c is None
    f.render_rgba()
