"""Hiding plot objects. visible= on every plotting call,
set_<kind>_visible(i, b) / <kind>_visible(i) on Axes, Plane2D and Axes3D, and
matplotlib's set_visible()/get_visible() and visible= in sextant.pyplot. A hidden
object draws as if it were absent."""

import numpy as np
import pytest

import sextant
import sextant.pyplot as plt
from sextant import _mpl

X = np.array([0.0, 1.0, 2.0, 3.0])
Y = np.array([1.0, 3.0, 2.0, 4.0])
GRID = np.arange(6.0).reshape(2, 3)


@pytest.fixture(autouse=True)
def fresh_state():
    plt.close("all")
    yield
    plt.close("all")


def figure():
    return sextant.Figure(width=240, height=180)


def same_picture(a, b):
    """The same SVG, and pixels equal up to raster noise: Apple's software renderer
    (the macOS runners') does not render one figure the same twice, off by a level
    here and there. A hidden object that drew would differ far more."""
    assert a.render_svg()[0] == b.render_svg()[0]
    ia, ib = a.render_rgba(), b.render_rgba()
    d = np.abs(ia.astype(int) - ib.astype(int)).max(axis=2)
    ys, xs = np.nonzero(d)
    if d.max() > 16 or len(xs) > d.size // 1000:
        pytest.fail(f"{len(xs)} pixels differ (max {d.max()}), x {xs.min()}..{xs.max()}, y {ys.min()}..{ys.max()}; "
                    f"a repeats: {np.array_equal(ia, a.render_rgba())}, b repeats: {np.array_equal(ib, b.render_rgba())}")


# Each 2D kind: (count name, getter/setter stem, the call adding one).
KINDS_2D = [
    ("line", lambda ax, **o: ax.line(X + 10, Y, name="l", **o)),
    ("scatter", lambda ax, **o: ax.scatter(X, Y - 10, name="s", **o)),
    ("scatter_z", lambda ax, **o: ax.scatter_z(X, Y, Y, colorbar=True, **o)),
    ("bar", lambda ax, **o: ax.bar(X, Y + 5, name="b", **o)),
    ("heatmap", lambda ax, **o: ax.heatmap(GRID, (-20, -15), (0, 2), colorbar=True, **o)),
    ("fill", lambda ax, **o: ax.fill(X, Y, Y + 8, name="f", **o)),
    ("quiver", lambda ax, **o: ax.quiver(X, Y, Y, X, name="q", **o)),
]
KINDS_AXES = KINDS_2D + [
    ("axline", lambda ax, **o: ax.axline(30.0, name="h", **o)),
    ("axspan", lambda ax, **o: ax.axspan(-30.0, -25.0, orientation="vertical", **o)),
    ("text", lambda ax, **o: ax.text("t", 1.0, 2.0, **o)),
]


def base(ax):
    ax.line(X, Y, color="red", name="kept")
    ax.legend()


@pytest.mark.parametrize("kind,add", KINDS_AXES, ids=[k for k, _ in KINDS_AXES])
def test_hidden_2d_object_draws_as_absent(kind, add):
    with_it, without = figure(), figure()
    base(with_it.axes())
    base(without.axes())
    ax = with_it.axes()
    add(ax, visible=False)
    assert not getattr(ax, f"{kind}_visible")(-1)
    same_picture(with_it, without)
    assert ax.xlim() == without.axes().xlim() and ax.ylim() == without.axes().ylim()

    # Shown again it draws, and the limits take it in where it widens them.
    shown = figure()
    base(shown.axes())
    add(shown.axes())
    assert getattr(ax, f"set_{kind}_visible")(-1, True) is ax
    assert getattr(ax, f"{kind}_visible")(-1)
    same_picture(with_it, shown)


def test_set_visible_hides_after_the_call_and_index_errors():
    fig = figure()
    ax = fig.axes()
    ax.line(X, Y).line(X, Y * 10)
    lo, hi = ax.ylim()
    assert hi > 30
    ax.set_line_visible(-1, False)
    assert ax.line_visible(0) and not ax.line_visible(1)
    assert ax.ylim()[1] < 10  # the auto limits leave it out
    ax.set_line_visible(1, True)
    assert ax.ylim() == (lo, hi)
    with pytest.raises(IndexError):
        ax.set_line_visible(2, False)
    with pytest.raises(IndexError):
        ax.scatter_visible(0)
    with pytest.raises(TypeError):
        ax.line(X, Y, visible="no")


def test_stack_and_bar_stack_hide_as_one():
    ax = figure().axes()
    ax.stack(X, [Y, Y], visible=False).bar_stack(X, [Y, Y], visible=False)
    assert ax.stack_count() == 1 and not ax.stack_visible(0)
    assert ax.bar_stack_count() == 1 and not ax.bar_stack_visible(0)


def axes3d(fig):
    return fig.add_subplot3d(1, 1, 1)


KINDS_3D = [
    ("bar3d", lambda ax, **o: ax.bar3d("xy", [0.0, 1.0], [0.0, 1.0], [[1.0, 2.0], [3.0, 4.0]], **o)),
    ("surface", lambda ax, **o: ax.surface("xy", [0.0, 1.0], [0.0, 1.0], [[0.0, 1.0], [1.0, 9.0]],
                                          colorbar=True, **o)),
    ("surface_tri", lambda ax, **o: ax.surface_tri([0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0],
                                                   [0.0, 1.0, 2.0, 7.0], orient="xy", **o)),
    ("scatter3d", lambda ax, **o: ax.scatter3d(X, Y, X - 5, name="s3", **o)),
    ("line3d", lambda ax, **o: ax.line3d(X, Y, X + 5, name="l3", **o)),
    ("text", lambda ax, **o: ax.text("t", 0.5, 0.5, 0.5, **o)),
]


def base3d(ax):
    ax.scatter3d([0.0, 2.0], [0.0, 2.0], [0.0, 2.0], color="red", name="kept")
    ax.legend()


@pytest.mark.parametrize("kind,add", KINDS_3D, ids=[k for k, _ in KINDS_3D])
def test_hidden_3d_object_draws_as_absent(kind, add):
    with_it, without = figure(), figure()
    ax = axes3d(with_it)
    base3d(ax)
    base3d(axes3d(without))
    add(ax, visible=False)
    assert not getattr(ax, f"{kind}_visible")(-1)
    same_picture(with_it, without)

    shown = figure()
    a2 = axes3d(shown)
    base3d(a2)
    add(a2)
    assert getattr(ax, f"set_{kind}_visible")(-1, True) is ax
    same_picture(with_it, shown)


@pytest.mark.parametrize("kind,add", KINDS_2D, ids=[k for k, _ in KINDS_2D])
def test_hidden_plane_object_draws_as_absent(kind, add):
    with_it, without = figure(), figure()
    p = axes3d(with_it).plane("xy", 0.0)
    p.line([0.0, 1.0], [0.0, 1.0], color="red")
    axes3d(without).plane("xy", 0.0).line([0.0, 1.0], [0.0, 1.0], color="red")
    add(p, visible=False)
    assert not getattr(p, f"{kind}_visible")(-1)
    same_picture(with_it, without)
    assert getattr(p, f"set_{kind}_visible")(-1, True) is p
    assert getattr(p, f"{kind}_visible")(-1)


def test_plane_flag_is_ink_only_object_flag_is_absence():
    # Plane2DOptions.visible hides the plane's ink and keeps it in the limits; a
    # plane whose objects are all hidden counts as empty, its offset with it.
    def zlim(plane_visible, line_visible):
        ax = axes3d(figure())
        ax.scatter3d([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
        ax.plane("xy", 50.0, visible=plane_visible).line([0.0, 1.0], [0.0, 1.0], visible=line_visible)
        return ax.zlim()

    assert zlim(False, True)[1] >= 50
    assert zlim(True, True)[1] >= 50
    assert zlim(True, False)[1] < 50


# --- sextant.pyplot ------------------------------------------------------------------

def mpl_pair():
    figs = []
    for _ in range(2):
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot([0, 1], [0, 1], color="red", label="kept")
        figs.append((fig, ax))
    return figs


def same_core(a, b):
    same_picture(a.core, b.core)


@pytest.mark.parametrize("call", [
    lambda ax, **o: ax.plot([0, 1, 2], [5, 6, 9], "o-", label="both", **o),
    lambda ax, **o: ax.scatter([0, 1], [3, 4], **o),
    lambda ax, **o: ax.bar([0, 1], [3, 4], **o),
    lambda ax, **o: ax.hist([1, 2, 2, 3], **o),
    lambda ax, **o: ax.errorbar([0, 1], [2, 3], yerr=0.5, fmt="s-", **o),
    lambda ax, **o: ax.axhline(7, **o),
    lambda ax, **o: ax.axvspan(3, 4, **o),
    lambda ax, **o: ax.fill_between([0, 1], [0, 1], [2, 3], **o),
    lambda ax, **o: ax.stackplot([0, 1], [1, 1], [2, 2], **o),
    lambda ax, **o: ax.quiver([0, 1], [0, 1], [1, 1], [1, 0], **o),
    lambda ax, **o: ax.imshow(GRID, extent=(5, 6, 5, 6), **o),
    lambda ax, **o: ax.text(0.5, 0.5, "t", **o),
    lambda ax, **o: ax.annotate("a", (0.2, 0.2), (0.8, 0.8), arrowprops={}, **o),
], ids=["plot", "scatter", "bar", "hist", "errorbar", "axhline", "axvspan", "fill_between",
        "stackplot", "quiver", "imshow", "text", "annotate"])
def test_pyplot_visible_kw_and_set_visible(call):
    (fig, ax), (ref, rax) = mpl_pair()
    ax.legend()
    rax.legend()
    out = call(ax, visible=False)
    same_core(fig, ref)
    artists = [a for a in (out if isinstance(out, (list, tuple)) else [out])
               if isinstance(a, (_mpl._Artist, _mpl.StackLayer))]
    assert artists and not any(a.get_visible() for a in artists)

    # A later style call replays it hidden; set_visible(True) brings it back.
    ax.set_title("x")
    rax.set_title("x")
    same_core(fig, ref)
    for a in artists:
        a.set_visible(True)
    assert all(a.get_visible() for a in artists)
    shown, sax = mpl_pair()[0]
    sax.legend()
    call(sax)
    sax.set_title("x")
    same_core(fig, shown)


def test_pyplot_line_with_markers_hides_both():
    (fig, ax), (ref, _) = mpl_pair()
    (line,) = ax.plot([0, 1, 2], [5, 6, 9], "o-")
    line.set_visible(False)
    assert not ax.core.line_visible(1) and not ax.core.scatter_visible(0)
    same_core(fig, ref)
    line.set_color("blue")  # a rebuild keeps it hidden
    same_core(fig, ref)


def test_pyplot_3d_visible():
    figs = []
    for hide in (False, True):
        fig = plt.figure(figsize=(3, 2))
        ax = fig.add_subplot(projection="3d")
        ax.scatter([0, 1], [0, 1], [0, 1], color="red")
        if hide:
            ax.plot([0, 1], [1, 0], [0, 5], visible=False)
            s = ax.plot_surface(np.array([[0.0, 1.0], [0.0, 1.0]]), np.array([[0.0, 0.0], [1.0, 1.0]]),
                                np.array([[0.0, 1.0], [2.0, 3.0]]))
            s.set_visible(False)
            assert not s.get_visible() and not ax.core.surface_visible(0)
        figs.append(fig)
    same_core(*figs)
