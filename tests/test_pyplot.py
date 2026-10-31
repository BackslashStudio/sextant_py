"""sextant.pyplot and the matplotlib-shaped adapters (sextant/_mpl.py)."""

import io
import math
import re
import struct
import sys
import textwrap
import types

import numpy as np
import pytest
from _subproc import run_python

import sextant
import sextant.pyplot as plt
from sextant import _mpl

window = pytest.mark.window


@pytest.fixture(autouse=True)
def fresh_state():
    plt.close("all")
    plt.ioff()
    yield
    plt.close("all")
    plt.ioff()


def png_size(data):
    return struct.unpack(">II", data[16:24])


def color_of(entry):
    return tuple(round(c, 3) for c in entry[2]["color"])


def hexrgba(h):
    return tuple(round(c, 3) for c in _mpl._hex(h))


# --- colours and format strings ------------------------------------------------------

@pytest.mark.parametrize("spec, want", [
    ("C1", "#ff7f0e"), ("C13", "#d62728"), ("tab:red", "#d62728"), ("r", "#ff0000"),
    ("#abc", "#aabbcc"), ("#11223344", "#11223344"),
])
def test_to_rgba(spec, want):
    assert tuple(round(c, 3) for c in _mpl.to_rgba(spec)) == hexrgba(want)


def test_to_rgba_other_forms():
    assert _mpl.to_rgba("0.5") == (0.5, 0.5, 0.5, 1.0)
    assert _mpl.to_rgba((1, 0, 0)) == (1.0, 0.0, 0.0, 1.0)
    assert _mpl.to_rgba("r", alpha=0.3) == (1.0, 0.0, 0.0, 0.3)
    assert _mpl.to_rgba("none")[3] == 0.0
    assert tuple(round(c, 3) for c in _mpl.to_rgba("Green")) == hexrgba("#008000")  # matplotlib's green
    assert _mpl.to_rgba("gray", alpha=0.5)[3] == 0.5
    with pytest.raises(ValueError):
        _mpl.to_rgba((1, 0))


def test_unlisted_names_reach_sextant():
    if "matplotlib" in sys.modules or _has_matplotlib():
        pytest.skip("matplotlib resolves every name")
    assert _mpl.to_rgba("rebeccapurple") == "rebeccapurple"  # through, for sextant to refuse
    with pytest.raises(TypeError, match="expected a color"):
        plt.plot([0, 1], color="rebeccapurple")


def _has_matplotlib():
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.parametrize("fmt, want", [
    ("r--o", ("--", "o", "r")), ("C3:", (":", None, "C3")), (".", (None, ".", None)),
    ("-.k", ("-.", None, "k")), ("", (None, None, None)),
])
def test_parse_fmt(fmt, want):
    assert _mpl.parse_fmt(fmt) == want


def test_parse_fmt_rejects_unknown():
    with pytest.raises(ValueError, match="unrecognized character 'q'"):
        _mpl.parse_fmt("rq")


# --- plot ------------------------------------------------------------------------------

def test_plot_forms():
    lines = plt.plot([3.0, 4.0, 5.0])
    ax = plt.gca()
    assert len(lines) == 1 and isinstance(lines[0], _mpl.Line2D)
    np.testing.assert_array_equal(ax.core.line_data(0).x, [0, 1, 2])
    plt.plot([0, 1], [0, 1], "r", [0, 1], [1, 0], "g--")
    y2 = np.arange(6.0).reshape(3, 2)
    assert len(plt.plot([0, 1, 2], y2)) == 2
    assert ax.core.line_count() == 5


def test_plot_is_an_alias_of_line():
    assert plt.plot is plt.line
    assert _mpl.Axes.plot is _mpl.Axes.line and _mpl.Axes3D.plot is _mpl.Axes3D.line
    plt.line([0, 1], [1, 0], "r--")
    plt.gca().line([0, 1])
    assert plt.gca().core.line_count() == 2
    a3 = plt.figure().add_subplot(projection="3d")
    a3.line([0, 1], [0, 1], [0, 1])
    assert a3.core.line3d_count() == 1


def test_markers_only_and_line_with_markers():
    ax = plt.gca()
    h = ax.plot([0, 1, 2], [1, 2, 3], "ro")[0]
    assert ax.core.line_count() == 0 and ax.core.scatter_count() == 1 and h._kind == "scatter"
    h2 = ax.plot([0, 1, 2], [1, 2, 3], "b-s", markersize=4)[0]
    assert ax.core.line_count() == 1 and ax.core.scatter_count() == 2
    assert ax._log[-1][2]["size"] == pytest.approx(4 * _mpl.PT)
    h2.set_ydata([5, 6, 7])
    np.testing.assert_array_equal(ax.core.line_data(0).y, [5, 6, 7])
    np.testing.assert_array_equal(ax.core.scatter_data(1).y, [5, 6, 7])  # the markers follow


def test_color_cycles():
    ax = plt.gca()
    a = ax.plot([0, 1])[0]
    b = ax.plot([0, 1])[0]
    s = ax.scatter([0], [0])
    assert color_of(a._entry) == hexrgba("#1f77b4") and color_of(b._entry) == hexrgba("#ff7f0e")
    assert color_of(s._entry) == hexrgba("#1f77b4")  # patches cycle on their own


def test_kwarg_aliases_and_passthrough():
    ax = plt.gca()
    h = ax.plot([0, 1], [0, 1], c="k", lw=3, ls=":", label="x", zorder=5, loop=True)[0]
    kw = h._entry[2]
    assert kw["linestyle"] == "dotted" and kw["linewidth"] == pytest.approx(3 * _mpl.PT)
    assert kw["name"] == "x" and kw["loop"] is True  # sextant's own options pass through
    with pytest.raises(TypeError, match="drawstyle"):
        ax.plot([0, 1], drawstyle="wavy")


def test_line_setters_rebuild_everything():
    ax = plt.gca()
    h = ax.plot([0, 1, 2], [0, 1, 0])[0]
    ax.set_title("kept")
    ax.set_xlim(-1, 3)
    h.set_ydata([2, 2, 2])
    h.set_color("red")
    assert ax.core.title() == "kept" and ax.core.xlim() == (-1.0, 3.0)
    np.testing.assert_array_equal(ax.core.line_data(0).y, [2, 2, 2])  # the update survived the replay
    assert color_of(h._entry) == (1.0, 0.0, 0.0, 1.0)


def test_legend():
    ax = plt.gca()
    ax.plot([0, 1], label="a")
    ax.plot([1, 0])
    ax.legend(loc="upper left")
    assert ax._log[-1] == ["legend", [], {"anchor": "inside_tl"}]
    ax.legend(["first", "second"])
    names = [e[2].get("name") for e in ax._log if e[0] == "line"]
    assert names == ["first", "second"]
    ax.legend(loc=3)
    assert ax._log[-1][2]["anchor"] == "inside_bl"


# --- scatter, bar, hist, errorbar -------------------------------------------------------

def test_scatter_colormapped():
    ax = plt.gca()
    sc = ax.scatter([0, 1, 2], [0, 1, 2], c=[5.0, 7.0, 9.0], cmap="plasma", s=36)
    assert ax.core.scatter_z_count() == 1
    kw = sc._entry[2]
    assert (kw["vmin"], kw["vmax"]) == (5.0, 9.0) and kw["cmap"] == "plasma"
    assert kw["size"] == pytest.approx(6 * _mpl.PT)  # s is points^2
    sc.set_offsets([[1, 1], [2, 2], [3, 3]])
    np.testing.assert_array_equal(ax.core.scatter_z_data(0).x, [1, 2, 3])
    with pytest.raises(ValueError, match="one value per point"):
        ax.scatter([0, 1], [0, 1], s=[10, 20, 30])


def test_bar_width_errors_and_tick_labels():
    ax = plt.gca()
    b = ax.bar([0, 2, 4], [1, 2, 3], width=1.6, yerr=[0.1, 0.2, 0.3], capsize=4, tick_label=["a", "b", "c"])
    kw = b._entry[2]
    assert kw["width"] == pytest.approx(0.8)  # data units -> fraction of the spacing
    assert kw["errorbar"]["capsize"] == pytest.approx(4 * _mpl.PT)
    assert b._entry[2]["linewidth"] == 0.0  # no edge by default, as matplotlib
    assert ax._log[-1][0] == "set_xticks"
    np.testing.assert_array_equal(b.datavalues, [1, 2, 3])


def test_hist_returns_numpys_counts():
    data = np.random.default_rng(0).normal(size=300)
    n, bins, bars = plt.hist(data, bins=12, density=True)
    n2, bins2 = np.histogram(data, bins=12, density=True)
    np.testing.assert_allclose(n, n2)
    np.testing.assert_allclose(bins, bins2)
    np.testing.assert_allclose(plt.gca().core.bar_data(0).height, n)
    n, _, _ = plt.hist(data, bins=5, cumulative=True)
    assert n[-1] == len(data)


def test_errorbar_forms():
    ax = plt.gca()
    ax.errorbar([0, 1, 2], [1, 2, 3], yerr=0.5)
    assert ax.core.line_count() == 1
    ax.errorbar([0, 1, 2], [1, 2, 3], yerr=[[0.1, 0.1, 0.1], [0.3, 0.3, 0.3]], fmt="o")
    assert ax.core.scatter_count() == 1
    h = ax.errorbar([0, 1, 2], [1, 2, 3], xerr=0.2, fmt="none")
    assert h._entry[2]["linestyle"] == "none"
    assert ax._log[0][2]["errorbar"]["capsize"] == 0.0  # matplotlib's default


# --- images --------------------------------------------------------------------------

def test_imshow_like_matplotlib():
    ax = plt.gca()
    im = ax.imshow(np.arange(12.0).reshape(3, 4))
    assert ax.core.ylim() == (2.5, -0.5)  # row 0 at the top, y pointing down
    assert ax.core.xlim() == (-0.5, 3.5)
    kw = im._entry[2]
    assert (kw["vmin"], kw["vmax"]) == (0.0, 11.0)
    im.set_data(np.zeros((3, 4)))
    assert ax.core.heatmap_data(0).data.sum() == 0
    with pytest.raises(ValueError, match="rows, cols"):
        ax.imshow(np.zeros((2, 2, 5)))


def test_imshow_rgb():
    ax = plt.gca()
    im = ax.imshow(np.full((3, 4, 3), 255, dtype=np.uint8))
    assert ax.core.heatmap_data(0).data.shape == (3, 4, 3)
    assert ax.core.heatmap_data(0).data.max() == 1.0
    im.set_data(np.zeros((3, 4, 3)))
    assert ax.core.heatmap_data(0).data.sum() == 0
    ax.imshow(np.random.rand(2, 2, 4))  # RGBA


def test_imshow_extent_and_lower_origin():
    ax = plt.gca()
    im = ax.imshow(np.ones((2, 2)), origin="lower", extent=(0, 4, 10, 20))
    assert im.get_extent() == (0.0, 4.0, 10.0, 20.0)
    assert ax.core.ylim()[0] < ax.core.ylim()[1]


def test_imshow_draws_y_ticks():
    """The inverted axis once had no tick labels (sextant generate_ticks, lo > hi)."""
    fig = plt.figure(figsize=(3, 3))
    plt.imshow(np.arange(100.0).reshape(10, 10))
    img = fig.core.render_rgba()
    left = img[:, :35, :3]
    assert int((left.max(axis=2) < 100).sum()) > 40


def test_pcolormesh():
    ax = plt.gca()
    ax.pcolormesh(np.ones((2, 3)))
    assert ax.core.heatmap_data(0).xrange == (0.0, 3.0)
    ax.pcolormesh([0, 1, 2], [0, 10], np.ones((2, 3)))  # centres
    assert ax.core.heatmap_data(1).xrange == (-0.5, 2.5)
    ax.pcolormesh([0, 1, 5, 6], [0, 1], np.ones((1, 3)))  # uneven: a rectilinear mesh
    d = ax.core.heatmap_data(2)
    assert list(d.x_edges) == [0, 1, 5, 6]


def test_norm_on_colormapped_calls():
    from sextant import colors
    ax = plt.gca()
    z = np.linspace(1, 100, 20).reshape(4, 5)
    ax.imshow(z, norm=colors.LogNorm())
    assert ax.core.heatmap_data(0) is not None
    ax.scatter([1, 2, 3], [1, 2, 3], c=[1, 10, 100], norm=colors.LogNorm(1, 100))
    ax.quiver([0, 1], [0, 1], [1, 1], [1, 1], [1, 5], norm="log")
    ax.imshow(z, norm=colors.TwoSlopeNorm(50, 1, 100))
    ax.imshow(z, norm=colors.BoundaryNorm([1, 10, 50, 100], 3))
    assert colors.resolve_norm("log")[0] is not None


def test_tripcolor_hist2d_contour():
    rng = np.random.default_rng(1)
    ax = plt.gca()
    x, y = rng.random(30), rng.random(30)
    tc = ax.tripcolor(x, y, x + y)
    assert len(tc.get_array()) == 30
    counts, xe, ye, im = ax.hist2d(rng.normal(size=300), rng.normal(size=300), bins=6)
    assert counts.shape == (6, 6) and len(xe) == 7 and counts.sum() == 300
    g = np.linspace(-1, 1, 15)
    X, Y = np.meshgrid(g, g)
    cs = ax.contour(X, Y, X * X + Y * Y, 4)
    ax.clabel(cs)
    assert len(cs.levels) >= 3
    cf = ax.contourf(X * X + Y * Y, levels=[0, 0.5, 1, 2])
    assert cf.filled and not cs.filled
    cf.remove()
    assert ax.core.contour_count() == 1
    plt.close("all")
    plt.tripcolor(x, y, x)
    plt.hist2d(x, y)
    plt.contour(np.eye(4))
    ax = plt.figure().add_subplot(projection="3d")
    q = ax.quiver(*rng.random((6, 5)), length=0.5)
    assert q.get_segments().shape == (5, 2, 3)


def test_colorbar_replays_the_axes():
    ax = plt.gca()
    im = plt.imshow(np.eye(3))
    ax.set_title("t")
    im.set_data(np.full((3, 3), 0.5))
    cb = plt.colorbar(label="level")
    assert im._entry[2]["colorbar"] is True and im._entry[2]["name"] == "level"
    assert ax.core.heatmap_count() == 1 and ax.core.title() == "t"
    assert float(ax.core.heatmap_data(0).data.mean()) == pytest.approx(0.5)
    cb.set_label("renamed")
    assert im._entry[2]["name"] == "renamed"
    im.set_data(np.zeros((3, 3)))  # still addresses the replayed object
    assert ax.core.heatmap_data(0).data.sum() == 0


def test_colorbar_needs_a_mappable():
    plt.plot([0, 1])
    with pytest.raises(RuntimeError, match="colormapped"):
        plt.colorbar()


# --- axes decoration and state functions -----------------------------------------------

def test_state_functions():
    plt.plot([0, 10], [0, 5])
    plt.title("T", fontsize=12)
    plt.xlabel("X")
    plt.ylabel("Y")
    ax = plt.gca()
    assert (ax.get_title(), ax.get_xlabel(), ax.get_ylabel()) == ("T", "X", "Y")
    assert plt.xlim(0, 20) == (0.0, 20.0) and plt.xlim() == (0.0, 20.0)
    assert plt.ylim(bottom=-1)[0] == -1.0
    plt.xticks([0, 10, 20], ["a", "b", "c"])
    assert list(plt.xticks()) == [0, 10, 20]
    plt.grid(True, color="gray", alpha=0.5)
    ax.set(title="via set", xlim=(1, 2))
    assert ax.core.title() == "via set" and ax.core.xlim() == (1.0, 2.0)


def test_title_fontdict():
    plt.gca().set_title("x", {"fontsize": 20})
    assert plt.gca()._log[-1][1][1] == pytest.approx(20 * _mpl.PT)


# --- figures and subplots ---------------------------------------------------------------

def test_subplots_shapes():
    fig, ax = plt.subplots()
    assert isinstance(ax, _mpl.Axes)
    _, axs = plt.subplots(1, 2)
    assert axs.shape == (2,)
    _, axs = plt.subplots(2, 3, squeeze=False)
    assert axs.shape == (2, 3)
    _, a3 = plt.subplots(subplot_kw={"projection": "3d"})
    assert isinstance(a3, _mpl.Axes3D)


def test_add_subplot_positions():
    fig = plt.figure()
    a = fig.add_subplot(221)
    assert fig.add_subplot(2, 2, 1) is a
    wide = fig.add_subplot(2, 1, 2)  # half the grid: cells 3-4 of the 2x2
    assert fig.core.add_subplot((3, 4)) is wide.core
    with pytest.raises(ValueError, match="grid is 2x2"):
        fig.add_subplot(3, 3, 1)
    assert plt.gca() is not None


def test_figure_numbers_and_close():
    f1 = plt.figure()
    f2 = plt.figure("named")
    assert plt.figure(1) is f1 and plt.figure("named") is f2
    assert plt.get_fignums() == [1]
    plt.close(f2)
    plt.close(1)
    assert plt.get_fignums() == []


def test_figsize_and_dpi():
    fig = plt.figure()
    fig.gca().plot([0, 1])
    b = io.BytesIO()
    fig.savefig(b)
    assert png_size(b.getvalue()) == (640, 480)  # matplotlib's 6.4x4.8 at 100 dpi
    b = io.BytesIO()
    fig.savefig(b, dpi=50)
    assert png_size(b.getvalue()) == (320, 240)
    big = plt.figure(figsize=(2, 1), dpi=200)
    b = io.BytesIO()
    big.savefig(b)
    assert png_size(b.getvalue()) == (400, 200)
    assert big.get_size_inches().tolist() == [2.0, 1.0]


def test_savefig_svg(tmp_path):
    plt.plot([0, 1])
    plt.savefig(tmp_path / "a.svg", bbox_inches="tight")
    assert "<svg" in (tmp_path / "a.svg").read_text(encoding="utf-8")


def test_sca_and_layout_noops():
    fig, (a, b) = plt.subplots(1, 2)
    plt.sca(a)
    assert plt.gca() is a
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)


# --- 3D ----------------------------------------------------------------------------------

def test_3d_kinds():
    ax = plt.figure().add_subplot(projection="3d")
    t = np.linspace(0, 1, 10)
    ax.plot(t, t, t, label="path")
    ax.scatter(t, t, t, c=t, cmap="plasma")
    X, Y = np.meshgrid(np.arange(4.0), np.arange(3.0))  # 'xy'
    s = ax.plot_surface(X, Y, X + Y, cmap="viridis")
    Xi, Yi = np.meshgrid(np.arange(4.0), np.arange(3.0), indexing="ij")
    ax.plot_surface(Xi, Yi, Xi * Yi)
    assert ax.core.line3d_count() == 1 and ax.core.scatter3d_count() == 1 and ax.core.surface_count() == 2
    np.testing.assert_array_equal(ax.core.surface_data(0).heights, (X + Y).T)
    assert s._entry[2]["colormap"] is True
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 8), np.linspace(0, np.pi, 5))
    ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v))  # not a grid
    assert ax.core.surface_tri_count() == 1
    assert ax.core.surface_tri_data(0).tri.shape == (2 * 7 * 4, 3)
    ax.plot_trisurf([0, 1, 0, 1], [0, 0, 1, 1], [0, 1, 1, 0], cmap="magma")
    assert ax.core.surface_tri_count() == 2


def test_bar3d_on_a_grid():
    ax = plt.figure().add_subplot(projection="3d")
    xs, ys = np.meshgrid([0.0, 1.0, 2.0], [0.0, 1.0], indexing="ij")
    x, y = xs.ravel(), ys.ravel()
    dz = np.arange(6.0) + 1
    ax.bar3d(x, y, np.full(6, 0.5), 0.8, 0.8, dz)
    d = ax.core.bar3d_data(0)
    np.testing.assert_array_equal(d.heights, dz.reshape(3, 2))
    np.testing.assert_array_equal(d.bottoms, np.full((3, 2), 0.5))
    np.testing.assert_allclose(d.u, [0.4, 1.4, 2.4])
    with pytest.raises(ValueError, match="grid"):
        ax.bar3d([0, 1, 5], [0, 3, 1], 0, 1, 1, [1, 2, 3])


def test_3d_view_and_labels():
    ax = plt.figure().add_subplot(111, projection="3d")
    ax.scatter([0, 1], [0, 1], [0, 1])
    ax.view_init(elev=20, azim=10)
    ax.set_proj_type("persp")
    ax.set_zlabel("Z")
    ax.set_zlim(-1, 1)
    cam = ax.core.camera()
    assert (cam["elevation"], cam["azimuth"]) == (20, 10) and cam["projection"] == "perspective"
    assert ax.get_zlabel() == "Z" and ax.get_zlim() == (-1.0, 1.0)
    with pytest.raises(ValueError, match="roll"):
        ax.view_init(roll=30)


def test_3d_colorbar():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    X, Y = np.meshgrid(np.arange(3.0), np.arange(3.0))
    s = ax.plot_surface(X, Y, X * Y, cmap="viridis")
    ax.view_init(10, 20)
    fig.colorbar(s)
    assert s._entry[2]["colorbar"] is True and ax.core.surface_count() == 1
    assert ax.core.camera()["elevation"] == 10  # the replay kept the view


# --- events -------------------------------------------------------------------------------

def test_mpl_event_translation():
    fig, ax = plt.subplots()
    h = ax.plot([0, 1, 2], [0, 1, 0])[0]
    e = types.SimpleNamespace(
        kind=sextant.EventKind.PICK, x=10.0, y=100.0, inaxes=ax.core, has_data=True, xdata=1.0,
        ydata=1.0, key="", double_click=False, button=1, scroll_y=0.0, width=0, height=0,
        pick_kind=sextant.PickKind.LINE, pick_object=ax.core.last_object(), pick_index=1,
        pick_plane=0)
    m = _mpl.MplEvent(fig, "pick_event", e)
    assert m.inaxes is ax and m.artist is h and list(m.ind) == [1]
    assert m.button == 3  # sextant's right button is matplotlib's 3
    assert m.y == 480 - 100  # from the bottom
    e.kind, e.scroll_y = sextant.EventKind.SCROLL, -2.0
    m = _mpl.MplEvent(fig, "scroll_event", e)
    assert m.button == "down" and m.step == -2.0


def test_pick_finds_fills_and_stack_layers():
    # A press inside a fill is a pick of kind "fill" with the segment as its index;
    # inside a stack, of kind "stack" with the layer.
    fig, ax = plt.subplots()
    band = ax.fill_between([0, 1, 2], [0, 1, 0], [2, 3, 2])
    layers = ax.stackplot([0, 1, 2], [1, 1, 1], [2, 2, 2])
    stack_id = ax.core.last_object()
    picked = twin_event(fig, ax, pick_kind=sextant.PickKind.FILL, pick_object=stack_id - 1, pick_index=1)
    assert picked.artist is band and list(picked.ind) == [1]
    for k in range(2):
        m = twin_event(fig, ax, pick_kind=sextant.PickKind.STACK, pick_object=stack_id, pick_index=k)
        assert m.artist is layers[k] and list(m.ind) == [k]


@window
def test_canvas_mpl_connect():
    fig, ax = plt.subplots()
    got = []
    cid = fig.canvas.mpl_connect("close_event", got.append)
    fig.show(block=False)
    fig.core.close()
    fig.canvas.flush_events()
    assert len(got) == 1 and got[0].name == "close_event" and got[0].canvas is fig.canvas
    fig.canvas.mpl_disconnect(cid)
    with pytest.raises(TypeError):
        fig.canvas.mpl_connect("draw_event", print)


def twin_event(fig, host, **fields):
    e = dict(kind=sextant.EventKind.PICK, x=10.0, y=100.0, inaxes=host.core, has_data=True, xdata=1.0,
             ydata=2.0, x2data=math.nan, y2data=math.nan, key="", double_click=False, button=0,
             scroll_y=0.0, width=0, height=0, pick_kind=sextant.PickKind.LINE, pick_object=0,
             pick_index=0, pick_plane=0, pick_twin=0)
    e.update(fields)
    return _mpl.MplEvent(fig, "pick_event", types.SimpleNamespace(**e))


def test_events_over_a_twin_report_the_twin():
    fig, ax = plt.subplots()
    own = ax.plot([0, 1], [0, 1])[0]
    own_id = ax.core.last_object()
    tw = ax.twinx()
    theirs = tw.plot([0, 1], [5, 6])[0]
    their_id = tw.core.last_object()
    # matplotlib's inaxes is the topmost axes, with ydata in its coordinates.
    m = twin_event(fig, ax, y2data=5.5, pick_object=own_id)
    assert m.inaxes is tw and m.xdata == 1.0 and m.ydata == 5.5
    # A pick on the twin's object (pick_twin=1) finds the twin's artist, not the host's.
    assert m.artist is own
    assert twin_event(fig, ax, pick_twin=1, pick_object=their_id).artist is theirs
    # An id the axes no longer has (or a plane's object) finds no artist.
    assert twin_event(fig, ax, pick_object=their_id).artist is None
    assert twin_event(fig, ax, pick_object=own_id, pick_plane=7).artist is None
    ty = ax.twiny()
    m = twin_event(fig, ax, x2data=7.0, y2data=5.5)
    assert m.inaxes is ty and m.xdata == 7.0 and m.ydata == 2.0  # the last twin made is on top


# --- scales, ticks, sharing, twins, aspect (sextant 1.2) -------------------------------------

def test_scales_like_matplotlib():
    fig, ax = plt.subplots()
    ax.set_xscale("log", base=2, nonpositive="clip", subs=[2])
    assert ax.get_xscale() == "log" and ax.xaxis.get_scale() == "log"
    plt.yscale("symlog", linthresh=0.1)
    assert ax.get_yscale() == "symlog" and ax.core.yscale() == "symlog"
    with pytest.raises(ValueError, match="'linear', 'log' and 'symlog'"):
        ax.set_xscale("logit")
    with pytest.raises(TypeError):
        ax.set_xscale("log", basex=2)


@pytest.mark.parametrize("fn, scales", [("semilogx", ("log", "linear")), ("semilogy", ("linear", "log")),
                                        ("loglog", ("log", "log"))])
def test_semilog_and_loglog(fn, scales):
    lines = getattr(plt, fn)([1, 10, 100], [1, 10, 100], "r--", base=10)
    ax = plt.gca()
    assert (ax.get_xscale(), ax.get_yscale()) == scales and ax.core.line_count() == 1
    assert len(lines) == 1


@pytest.mark.parametrize("fmt, printf", [("{x:.1f}", "%.1f"), ("{x:+.2e} V", "%+.2e V"), ("{x}%", "%g%%"),
                                         ("{x:d}", "%.0f"), ("%.3g", "%.3g"), ("{{{x:g}}}", "{%g}")])
def test_tick_formats_become_printf(fmt, printf):
    assert _mpl._printf_format(fmt) == printf


def test_major_formatter():
    fig, ax = plt.subplots()
    class FormatStrFormatter:  # matplotlib's, by name and .fmt
        fmt = "%.1f"

    ax.xaxis.set_major_formatter("{x:.2f}")
    ax.yaxis.set_major_formatter(FormatStrFormatter())
    calls = [e for e in ax._log if e[0].endswith("tick_format")]
    assert calls == [["set_xtick_format", ["%.2f"], {}], ["set_ytick_format", ["%.1f"], {}]]

    class FuncFormatter:
        pass

    with pytest.raises(TypeError, match="FuncFormatter"):
        ax.xaxis.set_major_formatter(FuncFormatter())
    with pytest.raises(ValueError):
        ax.xaxis.set_major_formatter("{pos}")
    with pytest.raises(ValueError):
        ax.xaxis.set_major_formatter("{x:,}")


def test_minorticks_and_minor_grid():
    fig, ax = plt.subplots()
    plt.minorticks_on()
    assert ax._style["minor_xticks"] is True and ax._style["minor_yticks"] is True
    ax.grid(True)
    ax.grid(which="minor", color="red")
    assert ax._log[-1] == ["grid", [True], {"minor": True, "minor_color": (1.0, 0.0, 0.0, 1.0)}]
    ax.grid(color="0.5")  # a later major grid() keeps the minor lines
    assert ax._log[-1][2]["minor"] is True and ax._log[-1][2]["color"] == (0.5, 0.5, 0.5, 1.0)
    ax.grid(False, which="both")
    assert ax._log[-1][1] == [False] and ax._log[-1][2]["minor"] is False
    plt.minorticks_off()
    assert ax._style["minor_xticks"] is False
    with pytest.raises(ValueError):
        ax.grid(which="sub")


@pytest.mark.parametrize("how, x_groups", [
    (True, [[(0, 0), (0, 1), (1, 0), (1, 1)]]), ("all", [[(0, 0), (0, 1), (1, 0), (1, 1)]]),
    ("col", [[(0, 0), (1, 0)], [(0, 1), (1, 1)]]), ("row", [[(0, 0), (0, 1)], [(1, 0), (1, 1)]])])
def test_subplots_share(how, x_groups, recwarn):
    fig, axs = plt.subplots(2, 2, sharex=how)
    for i, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        axs[r, c].set_xlim(i, i + 10)
        for group in x_groups:
            if (r, c) in group:
                assert all(axs[g].get_xlim() == (i, i + 10) for g in group)
            else:
                assert all(axs[g].get_xlim() != (i, i + 10) for g in group)
    assert not recwarn.list  # no more "not supported" warning


def test_subplots_share_y_and_bad_value():
    fig, axs = plt.subplots(1, 3, sharey=True)
    axs[2].set_ylim(-1, 1)
    assert axs[0].get_ylim() == (-1, 1)
    with pytest.raises(ValueError, match="sharex"):
        plt.subplots(2, 2, sharex="diagonal")


def test_add_subplot_and_axes_share():
    fig = plt.figure()
    a = fig.add_subplot(211)
    b = fig.add_subplot(212, sharex=a)
    c = plt.figure().add_subplot(111)
    a.set_xlim(3, 4)
    assert b.get_xlim() == (3, 4)
    b.sharey(a)
    a.set_ylim(5, 6)
    assert b.get_ylim() == (5, 6)
    with pytest.raises(ValueError):
        c.sharex(a)
    with pytest.raises(ValueError, match="2D axes only"):
        plt.figure().add_subplot(111, projection="3d", sharex=a)




def test_tick_params_and_label_outer():
    fig, axs = plt.subplots(2, 2)
    axs[0, 1].label_outer()
    assert axs[0, 1]._style == {"show_xtick_labels": False, "show_ytick_labels": False}
    axs[1, 0].label_outer()
    assert axs[1, 0]._style == {}
    axs[1, 1].tick_params(axis="x", labelbottom=False, labelleft=False)
    assert axs[1, 1]._style == {"show_xtick_labels": False}
    plt.sca(axs[0, 0])
    plt.tick_params(labelleft=False, labelright=True)  # one y side in sextant: the last wins
    assert axs[0, 0]._style == {"show_ytick_labels": True}
    with pytest.raises(TypeError, match="direction"):
        axs[0, 0].tick_params(direction="in")
    with pytest.raises(ValueError, match="which"):
        axs[0, 0].tick_params(which="all")


def test_tick_params_styles_each_axis():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.tick_params(axis="y", colors="red", labelsize=14, length=6, width=2)
    st = ax._style
    assert st["ytick_color"] == st["ylabel_color"] == _mpl.to_rgba("red")
    assert st["ylabel_fontsize"] == pytest.approx(14 * _mpl.PT)
    assert st["ytick_length"] == pytest.approx(6 * _mpl.PT)
    assert st["ytick_linewidth"] == pytest.approx(2 * _mpl.PT)
    assert not any(k.startswith("x") for k in st)
    ax.tick_params(axis="both", which="minor", length=2)
    assert "minor_xtick_length" in ax._style and "minor_ytick_length" in ax._style
    assert "xtick_length" not in ax._style
    ax.tick_params(axis="x", color="blue", labelcolor="green")
    assert ax._style["xtick_color"] == _mpl.to_rgba("blue")
    assert ax._style["xlabel_color"] == _mpl.to_rgba("green")
    svg = fig.core.render_svg()[0]
    assert 'fill="rgb(255,0,0)"' in svg and 'fill="rgb(0,128,0)"' in svg


def test_style_calls_keep_title_sizes():
    # The sizes live in sextant's axes style, which a later style call sends whole.
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("XLAB", fontsize=30)
    ax.set_title("TITLE", fontsize=20)
    ax.tick_params(labelbottom=False)
    ax.set_facecolor("white")
    svg = fig.core.render_svg()[0]
    assert re.search(r'font-size="41\.6667"[^>]*>XLAB<', svg)
    assert re.search(r'font-size="27\.7778"[^>]*>TITLE<', svg)
    ax.set_xlabel("XLAB")  # the default size again, as matplotlib's rcParams
    assert "xtitle_fontsize" not in ax._style


def test_twinx_like_matplotlib():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    tw = plt.twinx()
    assert plt.gca() is tw and tw in fig.axes and tw is ax.twinx()
    assert tw.core is ax.core.twinx()
    # A matplotlib twin restarts the colour cycle at C0.
    tw.plot([0, 1], [3, 2])
    assert color_of(tw._log[-1]) == hexrgba(_mpl.TAB10[0])
    with pytest.raises(ValueError):
        tw.twinx()


def test_twin_calls_reach_the_host():
    # sextant forwards a twin's shared-axis and cell calls to its host; the adapters
    # keep each record where the library keeps the value.
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_xticks([0.5], ["MID"])
    ax.set_xlabel("XL", fontsize=20)
    tw = ax.twinx()
    tw.plot([0, 1], [3, 2])
    # Style on the twin: the frame, background and x axis are the host's, its y its own.
    tw.set_facecolor("white")
    tw.tick_params(axis="both", colors="red")
    assert "background" in ax._style and "background" not in tw._style
    assert "xtick_color" in ax._style and "xtick_color" not in tw._style
    assert "ytick_color" in tw._style and "ytick_color" not in ax._style
    svg = fig.core.render_svg()[0]
    # The host's x axis is still drawn, its label keeps its size, and it is drawn once.
    assert ">MID<" in svg and svg.count(">XL<") == 1
    assert re.search(r'font-size="27\.7778"[^>]*>XL<', svg)
    # The twin's x label, ticks and title are the host's.
    tw.set_xlabel("XT")
    tw.set_xticks([0.25], ["Q"])
    tw.set_title("T2")
    assert ax.get_xlabel() == tw.get_xlabel() == "XT" and ax.get_title() == "T2"
    assert ax._xticks is not None and tw._xticks is None
    svg = fig.core.render_svg()[0]
    assert svg.count(">XT<") == 1 and ">XL<" not in svg and ">Q<" in svg and ">MID<" not in svg
    # axis('off') on the twin hides its own axis only.
    tw.axis("off")
    assert not tw.axison and ax.axison and "show_xticks" not in ax._style
    # cla() on the twin keeps the host's.
    tw.cla()
    assert tw._style == {}
    svg = fig.core.render_svg()[0]
    assert ">Q<" in svg and svg.count(">XT<") == 1
    # A twiny's y calls likewise.
    ty = ax.twiny()
    ty.set_yticks([])
    assert ax._style["show_yticks"] is False and "show_yticks" not in ty._style


def test_twin_rebuild_keeps_the_host_style():
    # A rebuild replays the twin's style calls, which reset the host's fields.
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    tw = ax.twinx()
    tw.plot([0, 1], [3, 2])
    tw.tick_params(axis="y", colors="red")
    ax.set_facecolor((0.0, 0.0, 1.0))
    tw._rebuild()
    assert 'fill="#0000ff"' in fig.core.render_svg()[0]


def test_set_aspect_like_matplotlib():
    fig, ax = plt.subplots()
    assert ax.get_aspect() == "auto" and ax.get_adjustable() == "box"
    ax.set_aspect("equal")
    assert ax.get_aspect() == 1.0 and ax.core.aspect() == 1.0
    ax.set_aspect(2.5, adjustable="datalim")
    assert ax.get_aspect() == 2.5 and ax.get_adjustable() == "datalim"
    ax.set_aspect("equal")  # keeps the adjustable, as matplotlib does
    assert ax.get_adjustable() == "datalim"
    ax.set_adjustable("box")
    assert ax.get_adjustable() == "box" and ax.get_aspect() == 1.0
    ax.set_aspect("auto")
    assert ax.get_aspect() == "auto"
    with pytest.raises(ValueError):
        ax.set_aspect("square")
    with pytest.raises(ValueError):
        ax.set_aspect(1, anchor="SW")
    with pytest.raises(ValueError):
        ax.twinx().set_aspect("equal")


def test_axis_modes():
    fig, ax = plt.subplots()
    ax.plot([0, 4], [0, 1])
    plt.axis("equal")
    assert (ax.get_aspect(), ax.get_adjustable()) == (1.0, "datalim")
    plt.axis("scaled")
    assert (ax.get_aspect(), ax.get_adjustable()) == (1.0, "box")
    lims = plt.axis("square")
    assert lims[1] - lims[0] == pytest.approx(lims[3] - lims[2])
    plt.axis("auto")
    assert ax.get_aspect() == "auto"
    assert plt.axis([0, 2, -1, 1]) == (0, 2, -1, 1)
    assert ax.axis(xmax=3) == (0, 3, -1, 1)
    plt.axis("off")
    assert not ax.axison and "show_xticks" not in ax._style  # the axis switch, not the style
    plt.axis(True)
    assert ax.axison
    ax.set_axis_off()
    assert not ax.core.axis_visible()
    ax.set_axis_on()
    assert ax.axison
    with pytest.raises(ValueError):
        plt.axis("tight")


def test_imshow_keeps_the_frame_unless_aspect_is_given():
    fig, ax = plt.subplots()
    ax.imshow(np.ones((2, 8)))
    assert ax.get_aspect() == "auto"
    fig, ax = plt.subplots()
    ax.imshow(np.ones((2, 8)), aspect="equal")
    assert ax.get_aspect() == 1.0
    fig, ax = plt.subplots()
    ax.imshow(np.ones((2, 8)), aspect=0.5)
    assert ax.get_aspect() == 0.5


def test_set_prop_cycle():
    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=["red", "k"])
    ax.plot([0, 1])
    ax.bar([0], [1])
    ax.plot([0, 1])
    ax.plot([0, 1])
    lines = [color_of(e) for e in ax._log if e[0] == "line"]
    assert lines == [(1, 0, 0, 1), (0, 0, 0, 1), (1, 0, 0, 1)]
    assert color_of(next(e for e in ax._log if e[0] == "bar")) == (1, 0, 0, 1)

    class Cycler:
        def by_key(self):
            return {"color": ["#00ff00"]}

    ax.set_prop_cycle(Cycler())
    ax.plot([0, 1])
    assert color_of(ax._log[-1]) == (0, 1, 0, 1)
    ax.set_prop_cycle(None)
    ax.plot([0, 1])
    assert color_of(ax._log[-1]) == hexrgba(_mpl.TAB10[0])
    with pytest.raises(TypeError, match="colors only"):
        ax.set_prop_cycle(linestyle=["-", "--"])
    ax.set_prop_cycle(color=["red"])
    ax.cla()
    ax.plot([0, 1])
    assert color_of(ax._log[-1]) == hexrgba(_mpl.TAB10[0])


# --- markers, steps, sizes, bar bases, barh, reference lines ------------------------------

@pytest.mark.parametrize("code, shape", [
    ("v", "triangle_down"), ("<", "triangle_left"), (">", "triangle_right"), ("*", "star"),
    ("p", "pentagon"), ("h", "hexagon"), ("x", "thin_cross"), ("+", "thin_plus"), ("X", "cross"),
    ("P", "plus"), ("|", "vline"), ("_", "hline"), ("1", "triangle_down"), ("o", "circle"),
])
def test_matplotlib_markers(code, shape):
    ax = plt.gca()
    ax.plot([0, 1], [0, 1], code)
    ax.scatter([0], [0], marker=code)
    assert ax._log[-2][2]["marker"] == shape and ax._log[-1][2]["marker"] == shape
    ax.figure.core.render_rgba()


def test_drawstyle_and_step():
    ax = plt.gca()
    h = ax.plot([0, 1, 2], [0, 1, 0], ds="steps-post")[0]
    assert h._entry[2]["drawstyle"] == "steps-post" and h.get_drawstyle() == "steps-post"
    h.set_drawstyle("default")
    assert h._entry[2]["drawstyle"] == "default"
    s = ax.step([0, 1, 2], [0, 1, 0], "r--", where="mid", label="s")[0]
    kw = s._entry[2]
    assert kw["drawstyle"] == "steps-mid" and kw["linestyle"] == "dashed" and kw["name"] == "s"
    assert ax.step([0, 1], [1, 0])[0]._entry[2]["drawstyle"] == "steps-pre"  # matplotlib's default
    plt.step([0, 1], [0, 1], where="post")
    assert ax._log[-1][2]["drawstyle"] == "steps-post"
    with pytest.raises(ValueError, match="where"):
        ax.step([0, 1], [0, 1], where="center")
    ax.figure.core.render_rgba()


def test_scatter_sizes_per_point():
    ax = plt.gca()
    sc = ax.scatter([0, 1, 2], [0, 1, 2], s=[4, 16, 64])
    np.testing.assert_allclose(sc._entry[2]["sizes"], np.array([2, 4, 8]) * _mpl.PT)  # s: points^2
    np.testing.assert_allclose(ax.core.scatter_data(0).sizes, np.array([2, 4, 8]) * _mpl.PT)
    np.testing.assert_allclose(sc.get_sizes(), [4, 16, 64])
    assert "sizes" not in ax.scatter([0, 1], [0, 1], s=[25])._entry[2]  # one value: uniform
    sc.set_sizes([1, 1, 1])
    np.testing.assert_allclose(ax.core.scatter_data(0).sizes, np.full(3, _mpl.PT))
    sc.set_sizes(9)
    assert ax.core.scatter_data(0).sizes is None and sc.get_sizes() == pytest.approx([9])
    sc.set_sizes([4, 16, 64])
    sc.set_offsets([[0, 0], [1, 1]])  # a new count drops the per-point sizes, as sextant does
    sc.set_label("x")  # a rebuild must not resend three sizes for two points
    assert ax.core.scatter_data(0).sizes is None and len(ax.core.scatter_data(0).x) == 2
    z = ax.scatter([0, 1], [0, 1], c=[1, 2], s=[9, 36])
    np.testing.assert_allclose(ax.core.scatter_z_data(0).sizes, np.array([3, 6]) * _mpl.PT)
    np.testing.assert_allclose(z.get_sizes(), [9, 36])

    a3 = plt.figure().add_subplot(projection="3d")
    a3.scatter([0, 1], [0, 1], [0, 1], s=[4, 16])
    np.testing.assert_allclose(a3.core.scatter3d_data(0).sizes, np.array([2, 4]) * _mpl.PT)


def test_bar_bottom_barh_and_hist_orientation():
    ax = plt.gca()
    b = ax.bar([0, 1, 2], [1, 2, 3], bottom=[3, 2, 1])
    np.testing.assert_array_equal(ax.core.bar_data(0).bottom, [3, 2, 1])
    ax.bar([0, 1, 2], [1, 1, 1], bottom=4)
    np.testing.assert_array_equal(ax.core.bar_data(1).bottom, [4, 4, 4])
    b.set_heights([2, 2, 2])
    np.testing.assert_array_equal(ax.core.bar_data(0).bottom, [3, 2, 1])  # same count: bases kept

    h = ax.barh([1, 2, 3], [10, 20, 30], height=0.5, left=1, xerr=[1, 2, 3], tick_label=["a", "b", "c"])
    kw = h._entry[2]
    assert kw["orientation"] == "horizontal" and kw["width"] == pytest.approx(0.5)
    np.testing.assert_array_equal(ax.core.bar_data(2).bottom, [1, 1, 1])
    assert ax._log[-1][0] == "set_yticks" and ax._log[-1][1][1] == ["a", "b", "c"]
    np.testing.assert_array_equal(h.datavalues, [10, 20, 30])
    plt.barh([0], [5])
    assert ax._log[-1][2]["orientation"] == "horizontal"
    with pytest.raises(ValueError, match="one height per series"):
        ax.barh([0, 1], [1, 1], height=[0.5, 0.6])

    n, _, hb = ax.hist([1, 2, 2, 3, 3, 3], bins=3, orientation="horizontal", bottom=1)
    assert hb._entry[2]["orientation"] == "horizontal"
    np.testing.assert_array_equal(ax.core.bar_data(4).bottom, [1, 1, 1])
    with pytest.raises(ValueError, match="orientation"):
        ax.hist([1, 2], orientation="sideways")
    ax.figure.core.render_rgba()


def test_axhline_axvline():
    ax = plt.gca()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    h = ax.axhline(5, xmin=0.25, xmax=0.75, ls="--", label="limit")
    kw = h._entry[2]
    assert ax.core.axline_count() == 1 and ax.core.axline_data(0).value == 5
    assert kw["orientation"] == "horizontal" and (kw["from_"], kw["to"]) == (0.25, 0.75)
    assert kw["linestyle"] == "dashed" and kw["name"] == "limit"
    assert kw["linewidth"] == pytest.approx(1.5 * _mpl.PT)
    assert color_of(h._entry) == hexrgba(_mpl.TAB10[0])  # 'C0', as matplotlib ...
    assert color_of(ax.plot([0, 1])[0]._entry) == hexrgba(_mpl.TAB10[0])  # ... without moving the cycle
    assert h.get_ydata() == [5, 5] and h.get_xdata() == [0.25, 0.75]
    h.set_ydata(6)
    assert ax.core.axline_data(0).value == 6
    h.set_xdata([0, 1])
    assert (h._entry[2]["from_"], h._entry[2]["to"]) == (0, 1)
    h.set_color("red")  # a rebuild replays the move
    assert ax.core.axline_data(0).value == 6 and color_of(h._entry) == (1, 0, 0, 1)

    v = plt.axvline(2, color="k", lw=3, alpha=0.5)
    kw = v._entry[2]
    assert kw["orientation"] == "vertical" and kw["alpha"] == 0.5 and kw["linewidth"] == pytest.approx(3 * _mpl.PT)
    assert v.get_xdata() == [2, 2]
    ax.set_prop_cycle(color=["#00ff00"])
    assert color_of(ax.axhline(1)._entry) == (0, 1, 0, 1)  # C0 is the cycle's first
    ax.figure.core.render_rgba()


def test_axhspan_axvspan():
    ax = plt.gca()
    s = ax.axhspan(1, 2, xmin=0.1, label="band")
    kw = s._entry[2]
    assert ax.core.axspan_count() == 1 and s.get_span() == (1, 2)
    assert kw["orientation"] == "horizontal" and kw["from_"] == 0.1 and kw["name"] == "band"
    assert kw["alpha"] == 1.0 and "linewidth" not in kw  # opaque, no outline, as matplotlib
    assert color_of(s._entry) == hexrgba(_mpl.TAB10[0])
    assert color_of(ax.bar([0], [1])._entry) == hexrgba(_mpl.TAB10[0])  # the patch cycle did not move
    s.set_span(3, 4)
    assert (ax.core.axspan_data(0).lo, ax.core.axspan_data(0).hi) == (3, 4)

    v = plt.axvspan(5, 4, fc="red", ec="k", alpha=0.3)
    kw = v._entry[2]
    assert kw["orientation"] == "vertical" and kw["alpha"] == 0.3
    assert color_of(v._entry) == (1, 0, 0, 1) and kw["linewidth"] == pytest.approx(_mpl.PT)
    assert (ax.core.axspan_data(1).lo, ax.core.axspan_data(1).hi) == (4, 5)
    assert "linewidth" not in ax.axvspan(0, 1, ec="none")._entry[2]
    v.set_alpha(None)
    v.set_facecolor("blue")
    assert v._entry[2]["alpha"] == 1.0 and ax.core.axspan_data(0).lo == 3  # replay kept the move
    ax.legend()
    svg, _ = ax.figure.core.render_svg()
    assert "band" in svg


def test_fill_between():
    ax = plt.gca()
    x = np.arange(5.0)
    f = ax.fill_between(x, x, label="area")
    kw = f._entry[2]
    assert f._entry[0] == "fill" and f._entry[1][2] == 0.0  # y2 = 0, a number
    assert color_of(f._entry) == hexrgba(_mpl.TAB10[0]) and kw["name"] == "area"
    assert "linewidth" not in kw and "edgecolor" not in kw and "exclude" not in kw  # no outline, as matplotlib
    # The patch cycle moved, the line cycle did not (matplotlib 3.10).
    assert color_of(ax.bar([0], [1])._entry) == hexrgba(_mpl.TAB10[1])
    assert color_of(ax.plot([0, 1])[0]._entry) == hexrgba(_mpl.TAB10[0])

    g = ax.fill_between(x, x, x + 1, where=x != 2, step="mid", interpolate=True, color="r", alpha=0.5)
    kw = g._entry[2]
    assert kw["exclude"].tolist() == [False, False, True, False, False]  # the inverse of where
    assert kw["drawstyle"] == "steps-mid" and kw["interpolate"] is True and kw["alpha"] == 0.5
    assert color_of(g._entry) == (1, 0, 0, 1)
    # color= is the outline's too, at 1 pt; sextant's alpha fades it, as matplotlib's.
    assert kw["edgecolor"] == (1, 0, 0, 1) and kw["linewidth"] == pytest.approx(_mpl.PT)
    assert "exclude" not in ax.fill_between(x, x, where=np.ones(5, bool))._entry[2]
    e = ax.fill_between(x, x, fc="b", ec="k", lw=2, ls="--")
    kw = e._entry[2]
    assert kw["edgecolor"] == (0, 0, 0, 1) and kw["linewidth"] == pytest.approx(2 * _mpl.PT)
    assert kw["linestyle"] == "dashed"
    assert ax.core.fill_count() == 4
    for bad in (lambda: ax.fill_between(x, x, step="side"), lambda: ax.fill_between(x, x, where=[True])):
        with pytest.raises(ValueError):
            bad()

    # Handles: set_data as matplotlib 3.10's (where included); a rebuild replays it.
    g.set_data(x, 2 * x, 0, where=x < 3)
    d = ax.core.fill_data(1)
    np.testing.assert_array_equal(d.y1, 2 * x)
    assert d.exclude.tolist() == [False, False, False, True, True]
    g.set_alpha(0.25)
    assert g._entry[2]["alpha"] == 0.25 and g._entry[2]["edgecolor"] == (1, 0, 0, 1)  # alpha fades both
    g.set_facecolor("g")
    d = ax.core.fill_data(1)
    np.testing.assert_array_equal(d.y1, 2 * x)
    assert d.exclude.tolist() == [False, False, False, True, True]
    g.set_data(x, x, 1)
    assert ax.core.fill_data(1).exclude is None and "exclude" not in g._entry[2]
    g.set_edgecolor("none")
    assert "linewidth" not in g._entry[2]
    np.testing.assert_array_equal(g.get_data()[2], np.ones(5))
    ax.legend()
    svg, _ = ax.figure.core.render_svg()
    assert "area" in svg


def test_fill_betweenx():
    ax = plt.gca()
    y = np.arange(4.0)
    f = ax.fill_betweenx(y, y, 3, where=[True, True, False, True])
    kw = f._entry[2]
    assert kw["orientation"] == "vertical" and kw["exclude"].tolist() == [False, False, True, False]
    d = ax.core.fill_data(0)  # the core's arrays read y, x1, x2
    np.testing.assert_array_equal(d.x, y)
    np.testing.assert_array_equal(d.y2, np.full(4, 3.0))
    ax.figure.core.render_rgba()


def test_stackplot():
    ax = plt.gca()
    x = np.arange(4.0)
    ys = np.array([[1.0, 2, 3, 4], [1, 1, 1, 1], [2, 0, 2, 0]])
    layers = ax.stackplot(x, *ys, labels=["a", "b"])
    assert len(layers) == 3 and ax.core.stack_count() == 1 and ax.core.fill_count() == 0
    assert all(isinstance(h, _mpl.StackLayer) for h in layers)
    # One line-cycle colour per layer; the patch cycle untouched (matplotlib 3.10).
    assert [tuple(round(c, 3) for c in h.get_facecolor()) for h in layers] == [hexrgba(c) for c in _mpl.TAB10[:3]]
    assert color_of(ax.plot([0, 1])[0]._entry) == hexrgba(_mpl.TAB10[3])
    assert color_of(ax.bar([0], [1])._entry) == hexrgba(_mpl.TAB10[0])
    # Labels past the given ones are unnamed.
    assert [h.get_label() for h in layers] == ["a", "b", ""]
    np.testing.assert_array_equal(ax.core.stack_data(0).ys, ys)
    # A layer's values and colour are its own; the stack restacks, and a replay keeps them.
    layers[0].set_data(x, 0, 9)
    layers[2].set_color("k")
    layers[1].set_label("B")
    np.testing.assert_array_equal(layers[0].get_data()[1], np.full(4, 9.0))
    np.testing.assert_array_equal(layers[1].get_data()[1], ys[1])
    assert layers[2].get_facecolor() == (0, 0, 0, 1) and layers[1].get_label() == "B"
    ax._rebuild()
    np.testing.assert_array_equal(ax.core.stack_data(0).ys[0], np.full(4, 9.0))
    ax.legend()
    svg, _ = ax.figure.core.render_svg()
    assert ">a<" in svg and ">B<" in svg

    ax.cla()
    sym = ax.stackplot(x, ys[:2], colors=["r"], baseline="sym", alpha=0.5, ec="k", step="post")
    assert [h.get_facecolor() for h in sym] == [(1, 0, 0, 1)] * 2  # colors repeat
    kw = sym[0]._opts()
    assert kw["alpha"] == 0.5 and kw["edgecolor"] == (0, 0, 0, 1) and kw["drawstyle"] == "steps-post"
    assert kw["baseline"] == "sym" and sym[1].get_edgecolor() == (0, 0, 0, 1)
    with pytest.raises(ValueError, match="hatch"):
        ax.stackplot(x, ys, hatch="//")
    with pytest.raises(TypeError):
        ax.stackplot(x)
    ax.figure.core.render_rgba()


def test_quiver():
    ax = plt.gca()
    x, y = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    u, v = np.array([1.0, 1.0]), np.array([1.0, 0.0])
    q = ax.quiver(x, y, u, v, label="flow")
    kw = q._entry[2]
    assert q._entry[0] == "quiver" and len(q._entry[1]) == 4
    assert color_of(q._entry) == (0, 0, 0, 1)  # matplotlib's black ...
    assert color_of(ax.plot([0, 1])[0]._entry) == hexrgba(_mpl.TAB10[0])  # ... taking no cycle colour
    # The amp: sextant's rule (matplotlib's mean arrow), computed here and passed.
    mean = (math.sqrt(2) + 1) / 2
    assert kw["amp"] == pytest.approx(1.0 / (1.8 * 10 * mean))
    # Shape: matplotlib's default shaft, 0.06 / clip(sqrt(N), 8, 25) of the frame
    # (0.0075 of 496 px for a lone 6.4 in axes), a head 3 wide and 5 long.
    assert kw["width"] == pytest.approx(0.0075 * 496)
    assert kw["head_width"] == pytest.approx(3 * kw["width"]) and kw["head_length"] == pytest.approx(5 * kw["width"])
    assert kw["pivot"] == "tail" and kw["name"] == "flow"
    assert ax._artists[("quiver", 0)] is q  # pick events find it
    np.testing.assert_array_equal(q.U, u)
    assert q.N == 2 and q.get_array() is None

    # set_UVC keeps the anchors and the amp; a rebuild replays the new vectors.
    q.set_UVC([2, 2], [0, 0])
    q.set_color("r")
    d = ax.core.quiver_data(0)
    np.testing.assert_array_equal(d.u, [2, 2])
    assert q._entry[2]["amp"] == kw["amp"] and color_of(q._entry) == (1, 0, 0, 1)
    q.set_offsets([[5, 5], [6, 6]])
    np.testing.assert_array_equal(ax.core.quiver_data(0).x, [5, 6])

    # scale in its units; width in its units.
    assert ax.quiver(x, y, u, v, scale=4, scale_units="xy")._entry[2]["amp"] == 0.25
    assert ax.quiver(x, y, u, v, scale=4)._entry[2]["amp"] == 0.25  # the anchors span 1 in x
    assert ax.quiver(x * 3, y, u, v, scale=4, scale_units="height")._entry[2]["amp"] == 0.25
    assert ax.quiver(x, y, u, v, width=0.01)._entry[2]["width"] == pytest.approx(4.96)
    assert ax.quiver(x, y, u, v, units="dots", width=2, headwidth=4)._entry[2]["head_width"] == pytest.approx(8)
    assert ax.quiver(x, y, u, v, pivot="mid")._entry[2]["pivot"] == "mid"
    for bad in (lambda: ax.quiver(x, y, u, v, scale=4, scale_units="inches"),
                lambda: ax.quiver(x, y, u, v, units="xy", width=0.1),
                lambda: ax.quiver(x, y, u, v, angles=[0, 90]),
                lambda: ax.quiver(x, y, u, v, linewidth=1)):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(TypeError):
        ax.quiver(u)
    ax.legend()
    svg, _ = ax.figure.core.render_svg()
    assert "flow" in svg


def test_quiver_grids_and_colors():
    fig, ax = plt.subplots()
    U = np.ones((3, 4))
    q = ax.quiver(U, U)  # anchors on U's index grid
    np.testing.assert_array_equal(q.X, np.tile(np.arange(4.0), 3))
    np.testing.assert_array_equal(q.Y, np.repeat(np.arange(3.0), 4))
    m = ax.quiver(np.arange(4.0) * 2, np.arange(3.0), U, U * 2, np.arange(12.0), cmap="plasma")  # a meshgrid
    np.testing.assert_array_equal(m.X, np.tile(np.arange(4.0) * 2, 3))
    kw = m._entry[2]
    assert len(m._entry[1]) == 5 and (kw["vmin"], kw["vmax"]) == (0, 11) and kw["cmap"] == "plasma"
    assert "color" not in kw
    np.testing.assert_array_equal(m.get_array(), np.arange(12.0))
    assert ax.quiver(U, U, np.arange(12.0), clim=(2, 4))._entry[2]["vmin"] == 2
    cb = fig.colorbar(m, label="speed")
    assert m._entry[2]["colorbar"] is True and m._entry[2]["name"] == "speed" and cb.mappable is m
    m.set_UVC(U, U, np.zeros(12))  # C replaced; a replay keeps it
    m.set_clim(0, 1)
    np.testing.assert_array_equal(ax.core.quiver_data(1).c, np.zeros(12))
    assert ax.core.quiver_data(1).c is not None and m._entry[2]["vmax"] == 1
    svg, _ = fig.core.render_svg()
    assert "speed" in svg


def test_quiver_frame_units_follow_the_grid():
    # matplotlib's axes in a 2x2 grid of a 6.4 x 4.8 in figure: 225.45 x 168 px; one
    # spanning a row: 496 px wide.
    fig, axs = plt.subplots(2, 2)
    kw = axs[0, 0].quiver([0, 1], [0, 1], [1, 1], [1, 0], width=0.01)._entry[2]
    assert kw["width"] == pytest.approx(2.2545, abs=1e-3)
    kw = axs[1, 1].quiver([0, 1], [0, 1], [1, 1], [1, 0], units="height", width=0.01)._entry[2]
    assert kw["width"] == pytest.approx(1.68, abs=1e-3)
    wide = plt.figure().add_subplot(2, 2, (1, 2))
    assert wide.quiver([0], [0], [1], [0], width=0.01)._entry[2]["width"] == pytest.approx(4.96)


# --- showing ------------------------------------------------------------------------------

@window
def test_show_blocks_until_closed_in_a_script():
    code = textwrap.dedent("""
        import threading, sextant.pyplot as plt
        plt.plot([0, 1])
        fig2 = plt.figure()
        plt.plot([1, 0])
        threading.Timer(0.4, lambda: [f.core.close() for f in list(plt._figs.values())]).start()
        plt.show()
        print("returned", plt.get_fignums())
    """)
    p = run_python(["-c", code])
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "returned []"


@window
def test_pause_shows_and_returns():
    plt.plot([0, 1])
    import time
    t0 = time.monotonic()
    plt.pause(0.2)
    assert time.monotonic() - t0 >= 0.19
    assert plt.gcf().core.is_open()


@window
def test_ion_does_not_block():
    plt.ion()
    plt.plot([0, 1])
    plt.show()
    fig = plt.gcf()
    assert fig.core.is_open()
    plt.plot([1, 0])  # reaches the open window at once (refresh)
    assert fig.gca().core.line_count() == 2


# --- Jupyter ------------------------------------------------------------------------------

def test_inline_display_in_a_kernel(monkeypatch):
    displayed, hooks = [], {}

    class ZMQInteractiveShell:
        events = types.SimpleNamespace(register=lambda name, fn: hooks.setdefault(name, fn))

    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: ZMQInteractiveShell()
    display_mod = types.ModuleType("IPython.display")
    display_mod.display = displayed.append
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display_mod)
    monkeypatch.setattr(plt, "_inline_registered", False)

    plt._install_inline()
    assert "post_execute" in hooks
    plt.plot([0, 1])
    plt.figure()  # empty: not displayed
    hooks["post_execute"]()
    assert len(displayed) == 1 and isinstance(displayed[0], sextant.Figure)
    assert plt.get_fignums() == []
    plt.plot([1, 2])
    plt.show()  # in a kernel: display now
    assert len(displayed) == 2


# --- sextant 1.1: backgrounds, hidden ticks, marker outlines ----------------------


def rgba_of(fig):
    return fig.core.render_rgba()


def test_figure_and_axes_facecolor():
    fig = plt.figure(figsize=(2, 1.5), facecolor="#0000ff")
    ax = fig.add_subplot(111, facecolor="#00ff00")
    ax.plot([0, 1], [0, 1])
    img = rgba_of(fig)
    assert tuple(img[1, 1]) == (0, 0, 255, 255)
    assert ((img[..., 1] == 255) & (img[..., 0] == 0) & (img[..., 2] == 0)).sum() > 1000  # the plot area
    fig.set_facecolor("#ff0000")
    assert tuple(rgba_of(fig)[1, 1]) == (255, 0, 0, 255)
    assert fig.get_facecolor() == (1.0, 0.0, 0.0, 1.0)
    ax.set_facecolor("#ffff00")
    assert (rgba_of(fig)[..., 2] == 0).sum() > 1000


def test_savefig_facecolor_and_transparent_are_for_that_file_only(tmp_path):
    fig, ax = plt.subplots(figsize=(2, 1.5))
    ax.plot([0, 1], [0, 1])
    before = tuple(rgba_of(fig)[1, 1])
    fig.savefig(tmp_path / "t.svg", transparent=True)
    assert "<rect width=" not in (tmp_path / "t.svg").read_text()
    fig.savefig(tmp_path / "f.svg", facecolor="#0000ff")
    assert 'fill="#0000ff"' in (tmp_path / "f.svg").read_text()
    assert tuple(rgba_of(fig)[1, 1]) == before   # restored


def text_count(fig):
    return fig.core.render_svg()[0].count("<text")


def test_set_xticks_empty_hides_the_ticks_as_in_matplotlib():
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1, 2], [0, 3, 1])
    full = text_count(fig)
    ax.set_xticks([])
    hidden = text_count(fig)
    assert 0 < hidden < full
    ax.set_xticks([0, 1, 2])             # explicit ticks show again
    assert text_count(fig) > hidden
    ax.set_xticks([])
    ax.set_yticks([])
    assert text_count(fig) == 0
    ax.cla()                             # clearing the axes brings the ticks back
    ax.plot([0, 1, 2], [0, 3, 1])
    assert text_count(fig) == full


def ink(fig, rgb, tol=60):
    d = np.abs(rgba_of(fig)[..., :3].astype(int) - np.array(rgb)).max(axis=2)
    return int((d <= tol).sum())


def test_scatter_edgecolors_linewidths_and_facecolors():
    def fig_with(**kw):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.scatter([5], [5], s=900, color="#0000ff", **kw)   # 30 pt across
        return fig

    filled = ink(fig_with(), (0, 0, 255))
    ring = ink(fig_with(facecolors="none", edgecolors="#ff0000", linewidths=3), (255, 0, 0))
    assert filled > 1000 and 150 < ring < filled
    assert ink(fig_with(facecolors="none", edgecolors="#ff0000", linewidths=3), (0, 0, 255)) == 0
    # no outline unless asked for: edgecolors='none' and the default draw the same picture
    assert ink(fig_with(edgecolors="none"), (0, 0, 255)) == filled
    # edgecolors='face' with a width outlines in the marker's own color
    assert ink(fig_with(facecolors="none", edgecolors="face", linewidths=3), (0, 0, 255)) > 150


# --- text and annotate (sextant 1.1) ------------------------------------------------


def test_text_matplotlib_signature_and_transforms():
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 10], [0, 10])
    t = ax.text(2, 3, "a", fontsize=12, color="r", ha="center", va="center_baseline", rotation="vertical")
    assert isinstance(t, _mpl.Text)
    d = ax.core.text_data(0)
    assert (d.text, d.x, d.y, d.xcoords) == ("a", 2.0, 3.0, "data")
    ax.text(0.05, 0.95, "corner", transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="k", alpha=0.8))
    assert ax.core.text_data(1).xcoords == "fraction"
    plt.text(1, 1, "via pyplot")
    assert ax.core.text_count() == 3
    with pytest.raises(ValueError, match="transData or ax.transAxes"):
        ax.text(0, 0, "x", transform=object())
    with pytest.raises(TypeError, match="bbox"):
        ax.text(0, 0, "x", bbox=dict(shadow=True))


def test_text_artist_setters_survive_a_replay():
    fig, ax = plt.subplots(figsize=(4, 3))
    t = ax.text(1, 2, "old")
    t.set_text("new")
    t.set_position((3, 4))
    assert t.get_text() == "new" and t.get_position() == (3.0, 4.0)
    t.set_color("blue")  # an option change replays the record
    d = ax.core.text_data(0)
    assert (d.text, d.x, d.y) == ("new", 3.0, 4.0)


def test_annotate_text_coordinates_and_arrowprops():
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 10], [0, 10])
    a = ax.annotate("peak", xy=(5, 6), xytext=(0.6, 0.8), textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color="red", lw=1.5))
    d = ax.core.text_data(0)
    assert d.arrow and (d.px, d.py) == (5.0, 6.0) and (d.x, d.y, d.xcoords) == (0.6, 0.8, "fraction")
    assert a.get_text() == "peak"

    # 'offset points': the text sits at the point, moved by the offset.
    ax.annotate("off", xy=(2, 2), xytext=(10, 5), textcoords="offset points", arrowprops={})
    d = ax.core.text_data(1)
    assert d.arrow and (d.x, d.y) == (2.0, 2.0)

    # No arrowprops: just the text.
    ax.annotate("plain", (1, 1))
    assert not ax.core.text_data(2).arrow

    svg = fig.core.render_svg()[0]
    assert ">peak</text>" in svg and ">off</text>" in svg

    with pytest.raises(ValueError, match="arrowstyle"):
        ax.annotate("x", (1, 1), xytext=(2, 2), arrowprops=dict(arrowstyle="curve"))
    with pytest.raises(ValueError, match="arc3"):
        ax.annotate("x", (1, 1), xytext=(2, 2), arrowprops=dict(connectionstyle="angle3"))
    with pytest.raises(ValueError, match="xycoords"):
        ax.annotate("x", (0.5, 0.5), xytext=(2, 2), xycoords="axes fraction", arrowprops={})
    with pytest.raises(ValueError, match="textcoords"):
        ax.annotate("x", (1, 1), xytext=(2, 2), textcoords="figure pixels")


def test_3d_text_and_text2D():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter([0, 1], [0, 1], [0, 1])
    t = ax.text(0.5, 0.5, 0.5, "p", color="k")
    c = ax.text2D(0.05, 0.95, "caption", transform=ax.transAxes)
    assert ax.core.text_count() == 2
    assert ax.core.text_data(1).in_frame
    t.set_position((1, 1, 1))
    assert (ax.core.text_data(0).x, ax.core.text_data(0).z) == (1.0, 1.0)
    c.set_text("moved")
    assert ax.core.text_data(1).text == "moved" and ax.core.text_data(1).in_frame
    with pytest.raises(ValueError, match="zdir"):
        ax.text(0, 0, 0, "x", zdir="x")


# --- math in text (sextant 1.1) ---------------------------------------------------


def test_matplotlib_math_labels_draw_as_math():
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1], label=r"$\sin(\omega t)$")
    ax.set_xlabel(r"$\theta$ [rad]")
    ax.set_ylabel(r"$\sigma^2$ (m$^2$)")
    ax.legend()
    svg = fig.core.render_svg()[0]
    # Math letters are italic, so a letter and the upright text after it are two runs.
    assert '<tspan font-style="italic">\u03b8</tspan><tspan> [rad]</tspan>' in svg
    assert '<tspan font-style="italic">\u03c3</tspan>' in svg and "<tspan>sin(</tspan>" in svg
    assert "font-size=" in svg


def test_parse_math_and_usetex():
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.text(0.1, 0.1, r"$x^2$", parse_math=False)
    assert ax.core.text_count() == 1
    ax.annotate(r"$y^2$", (0.5, 0.5), xytext=(0.2, 0.8), parse_math=True)
    svg = fig.core.render_svg()[0]
    assert ">$x^2$</text>" in svg and '<tspan font-style="italic">y</tspan>' in svg
    with pytest.raises(ValueError, match="usetex"):
        ax.text(0, 0, "x", usetex=True)
    with pytest.raises(ValueError, match="figure\\(mathtext=False\\)"):
        ax.set_title(r"$x$", parse_math=False)
    ax.set_title(r"$x$", parse_math=True)
    off = plt.figure(mathtext=False)
    off.add_subplot().set_title(r"$x^2$")
    assert ">$x^2$</text>" in off.core.render_svg()[0]


# --- Artist.remove() (sextant 1.3) --------------------------------------------------------

def test_artist_remove_drops_it_and_relocates_the_rest():
    fig, ax = plt.subplots()
    a = ax.plot([0, 1, 2], [0, 1, 0], label="a")[0]
    b = ax.plot([0, 1, 2], [5, 6, 5], label="b")[0]
    c = ax.plot([0, 1, 2], [9, 8, 9], label="c")[0]
    a.remove()
    assert ax.core.line_count() == 2
    assert list(b.get_ydata()) == [5, 6, 5] and list(c.get_ydata()) == [9, 8, 9]
    b.set_ydata([7, 7, 7])  # lands on b's own object after the indices moved
    assert list(ax.core.line_data(0).y) == [7, 7, 7] and list(ax.core.line_data(1).y) == [9, 8, 9]
    # Picks still find the artists: by the (new) id, through find_object.
    assert twin_event(fig, ax, pick_object=ax.core.last_object()).artist is c
    with pytest.raises(ValueError, match="not in its axes"):
        a.remove()
    with pytest.raises(ValueError, match="removed"):
        a.set_label("x")


def test_artist_remove_takes_markers_and_stack_layers_whole():
    fig, ax = plt.subplots()
    ln = ax.plot([0, 1, 2], [0, 1, 0], "-o")[0]
    other = ax.plot([0, 1, 2], [3, 3, 3])[0]
    assert ax.core.line_count() == 2 and ax.core.scatter_count() == 1
    ln.remove()
    assert ax.core.line_count() == 1 and ax.core.scatter_count() == 0
    assert list(other.get_ydata()) == [3, 3, 3]

    band = ax.fill_between([0, 1, 2], [0, 0, 0], [1, 1, 1])
    layers = ax.stackplot([0, 1, 2], [1, 1, 1], [2, 2, 2], [3, 3, 3])
    after = ax.fill_between([0, 1, 2], [5, 5, 5], [6, 6, 6])
    layers[2].set_visible(False)  # a layer's visibility is the whole stack's
    assert ax.core.stack_visible(0) is False and layers[0].get_visible() is False
    layers[1].remove()  # and remove() takes the whole stack out
    assert ax.core.stack_count() == 0 and ax.core.fill_count() == 2
    after.set_visible(False)  # the others still address their own objects
    assert ax.core.fill_visible(1) is False and ax.core.fill_visible(0) is True
    with pytest.raises(ValueError):
        layers[0].remove()


def test_artist_remove_on_a_twin_and_after_cla():
    fig, ax = plt.subplots()
    own = ax.plot([0, 1], [0, 1])[0]
    tw = ax.twinx()
    t1 = tw.plot([0, 1], [5, 6])[0]
    t2 = tw.plot([0, 1], [7, 8])[0]
    t1.remove()
    assert tw.core.line_count() == 1 and ax.core.line_count() == 1
    assert list(t2.get_ydata()) == [7, 8] and list(own.get_ydata()) == [0, 1]
    ax.cla()
    with pytest.raises(ValueError):
        own.remove()


# --- axline with a slope or two points, and the patches ----------------------------------

def test_axline_point_slope_and_two_points():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    a = ax.axline((1, 0), slope=1, color="r", lw=3, ls="--", alpha=0.5, label="s")
    b = ax.axline((0, 1), (2, 1))
    d = ax.core.axline_data(0)
    assert (d.mode, d.x, d.y, d.slope) == ("point_slope", 1.0, 0.0, 1.0)
    d = ax.core.axline_data(1)
    assert (d.mode, d.x2, d.y2) == ("two_points", 2.0, 1.0)
    kw = a._entry[2]
    assert kw["linestyle"] == "dashed" and kw["alpha"] == 0.5 and kw["name"] == "s"
    assert kw["linewidth"] == pytest.approx(3 * _mpl.PT) and "orientation" not in kw and "from_" not in kw
    assert color_of(b._entry) == hexrgba(_mpl.TAB10[0]) and b.get_zorder() == 2.0  # C0, a line's zorder
    assert a.get_slope() == 1.0 and a.get_xy1() == (1.0, 0.0) and a.get_xy2() is None
    assert b.get_slope() == 0.0 and b.get_xy2() == (2.0, 1.0)
    a.set_slope(2)
    a.set_xy1(0, 0)
    b.set_xy2(3, 2)
    assert (ax.core.axline_data(0).slope, ax.core.axline_data(0).x) == (2.0, 0.0)
    assert ax.core.axline_data(1).y2 == 2.0 and b.get_slope() == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="'xy2' is set"):
        b.set_slope(1)
    with pytest.raises(ValueError, match="'slope' is set"):
        a.set_xy2(1, 1)
    with pytest.raises(NotImplementedError):
        a.get_xdata()
    a.set_color("blue")  # replays with the move kept
    assert ax.core.axline_data(0).slope == 2.0 and color_of(a._entry) == (0, 0, 1, 1)
    a.remove()
    assert ax.core.axline_count() == 1 and b.get_xy1() == (0.0, 1.0)
    fig.core.render_rgba()


def test_axline_follows_matplotlibs_argument_rules():
    ax = plt.gca()
    with pytest.raises(TypeError, match="Must specify either"):
        ax.axline((0, 0))
    with pytest.raises(TypeError, match="Cannot specify both"):
        ax.axline((0, 0), (1, 1), slope=1)
    with pytest.raises(NotImplementedError):
        ax.axline((0, 0), slope=1, transform=ax.transAxes)
    with pytest.raises(ValueError):
        ax.axline((1, 1), (1, 1))  # the core refuses two equal points
    assert ax.core.axline_count() == 0 and not ax._log
    ax.axline((0, 0), slope=1, transform=ax.transData)
    assert plt.axline((0, 1), slope=2).get_slope() == 2.0  # the pyplot function
    hidden = ax.axline((0, 0), (1, 1), visible=False)
    assert hidden.get_visible() is False and ax.core.axline_visible(2) is False
    assert ax.axline((0, 0), (1, 1), zorder=7).get_zorder() == 7.0


def test_patches_circle_wedge_polygon():
    from sextant.patches import Circle, Polygon, Wedge

    fig, ax = plt.subplots()
    ax.plot([0, 4], [0, 4], label="line")
    c = Circle((2, 2), 1, fc="green", ec="k", lw=2, alpha=0.5, label="disc", zorder=4)
    assert c.axes is None and c.get_zorder() == 4.0
    assert ax.add_patch(c) is c and c.axes is ax
    assert ax.core.shape_circle_data(0).r == 1.0
    kw = c._entry[2]
    assert kw["alpha"] == 0.5 and kw["name"] == "disc" and kw["zorder"] == 4.0
    assert color_of(c._entry) == hexrgba("#008000") and kw["linewidth"] == pytest.approx(2 * _mpl.PT)

    w = ax.add_patch(Wedge((1, 1), 1, 0, 90, width=0.25))
    assert ax.core.shape_wedge_data(0).inner == pytest.approx(0.75) and w.get_zorder() == 1.0
    assert color_of(w._entry) == hexrgba(_mpl.TAB10[0])  # patch.facecolor, C0
    p = ax.add_patch(Polygon([[0, 0], [1, 0], [0.5, 1]]))
    np.testing.assert_array_equal(ax.core.shape_poly_data(0).y, [0, 0, 1])
    assert "linewidth" not in p._entry[2]  # no outline unless one is asked for
    ax.legend()
    svg = fig.core.render_svg()[0]
    assert "disc" in svg
    fig.core.render_rgba()


def test_patch_setters_before_and_after_add():
    from sextant.patches import Circle

    fig, ax = plt.subplots()
    c = Circle((0, 0), 1)
    c.set_facecolor("red")
    c.set_zorder(3)
    c.set_visible(False)
    c.set_label("x")
    c.set_alpha(0.25)
    assert ax.core.shape_circle_count() == 0
    ax.add_patch(c)
    assert color_of(c._entry) == (1, 0, 0, 1) and c.get_zorder() == 3.0 and c.get_visible() is False
    assert ax.core.shape_circle_visible(0) is False and ax.core.shape_circle_data(0).r == 1.0
    c.set_visible(True)
    c.set_facecolor("blue")  # replay
    c.set_zorder(0.5)
    c.set_edgecolor("k")
    c.set_linewidth(2)
    c.set_linestyle("--")
    assert ax.core.shape_circle_visible(0) is True and color_of(c._entry) == (0, 0, 1, 1)
    assert c._entry[2]["linestyle"] == "dashed" and c.get_zorder() == 0.5
    with pytest.raises(ValueError, match="re-use an artist"):
        plt.figure().gca().add_patch(c)


def test_patch_remove_and_pick():
    from sextant.patches import Circle, Polygon, Wedge

    fig, ax = plt.subplots()
    c = ax.add_patch(Circle((0, 0), 1))
    w = ax.add_patch(Wedge((0, 0), 2, 0, 90))
    p = ax.add_patch(Polygon([[0, 0], [1, 0], [0, 1]]))
    picked = [twin_event(fig, ax, pick_kind=kind, pick_object=oid, pick_index=0)
              for kind, oid in ((sextant.PickKind.CIRCLE, ax.core.last_object() - 2),
                                (sextant.PickKind.WEDGE, ax.core.last_object() - 1),
                                (sextant.PickKind.POLY, ax.core.last_object()))]
    assert [m.artist for m in picked] == [c, w, p]
    c.remove()
    assert ax.core.shape_circle_count() == 0 and ax.core.shape_wedge_count() == 1
    with pytest.raises(ValueError):
        c.remove()
    w.set_facecolor("red")  # the others still work after the replay
    assert color_of(w._entry) == (1, 0, 0, 1) and ax.core.shape_wedge_count() == 1
    ax.cla()
    assert ax.core.shape_poly_count() == 0


def test_patch_limits_and_errors():
    from sextant.patches import Circle, Polygon, Wedge

    fig, ax = plt.subplots()
    with pytest.raises(NotImplementedError, match="fill=False"):
        Circle((0, 0), 1, fill=False)
    with pytest.raises(NotImplementedError, match="data coordinates"):
        Circle((0, 0), 1, transform=ax.transAxes)
    Circle((0, 0), 1, transform=ax.transData)
    with pytest.raises(ValueError, match="hatching"):
        Circle((0, 0), 1, hatch="//")
    with pytest.raises(NotImplementedError, match="closed"):
        Polygon([[0, 0], [1, 0], [0, 1]], closed=False)
    with pytest.raises(ValueError, match="shape"):
        Polygon([0, 1, 2])
    with pytest.raises(TypeError):
        ax.add_patch(object())
    for bad in (Circle((0, 0), 0), Wedge((0, 0), 1, 0, 90, width=2), Polygon([[0, 0], [1, 1]])):
        with pytest.raises(ValueError):
            ax.add_patch(bad)
        assert bad.axes is None
    assert not ax._log and not ax._artists
    ax.add_patch(Circle((10, 20), 5))
    lo, hi = ax.core.xlim()
    assert lo <= 5 and hi >= 15  # autoscale takes the shape


# --- review fixes: dropped keywords, replay, weights, alpha, arrows --------------------

def test_contour_origin_extent_and_rejected_keywords():
    ax = plt.gca()
    Z = np.arange(12.0).reshape(3, 4)
    ax.contour(Z, origin="lower", extent=(0, 8, 0, 6))
    d = ax.core.contour_data(0)
    np.testing.assert_allclose(d.x, [1, 3, 5, 7])  # pixel centres of the extent
    np.testing.assert_allclose(d.y, [1, 3, 5])
    ax.contour(Z, origin="upper")
    d = ax.core.contour_data(1)
    np.testing.assert_allclose(d.y, [0, 1, 2])
    np.testing.assert_array_equal(d.z, Z[::-1])  # row 0 is at the top, so y ascends over the flipped rows
    for bad in (lambda: ax.contour(Z, linestyles="dashed"), lambda: ax.contourf(Z, extend="both"),
                lambda: ax.contour(Z, colors=["r", "g"]), lambda: ax.contour(*np.meshgrid(range(4), range(3)), Z, origin="lower")):
        with pytest.raises(ValueError):
            bad()
    X, Y = np.meshgrid(range(4), range(3))
    with pytest.raises(ValueError, match="rectilinear"):
        ax.contour(X + 0.3 * Y, Y, Z)
    with pytest.raises(ValueError, match="rectilinear"):
        ax.pcolormesh(X + 0.3 * Y, Y, Z)


def test_hist2d_weights_and_set_data_guard():
    ax = plt.gca()
    x = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0.1, 0.2, 0.8, 0.9])
    w = np.array([1.0, 2.0, 3.0, 4.0])
    counts, xe, ye, im = ax.hist2d(x, y, bins=2, range=[[0, 1], [0, 1]], weights=w)
    np.testing.assert_allclose(counts, [[3, 0], [0, 7]])
    d = ax.core.heatmap_data(0)
    np.testing.assert_allclose(d.data, counts.T)
    im2 = ax.hist2d(x, y, bins=2)[3]
    with pytest.raises(NotImplementedError):
        im2.set_data(np.zeros((2, 2)))
    assert im2._entry[0] == "hist2d" and len(im2._entry[1]) == 2  # replay log untouched
    with pytest.raises(ValueError):
        ax.hist2d(x, y, weights=w[:2])


def test_imshow_rgb_alpha_and_uint8_set_data():
    ax = plt.gca()
    im = ax.imshow(np.full((2, 2, 3), 255, dtype=np.uint8), alpha=0.5)
    d = ax.core.heatmap_data(0)
    assert d.data.shape == (2, 2, 4) and d.data[..., 3].max() == pytest.approx(0.5)
    sc = ax.imshow(np.full((2, 2), 200, dtype=np.uint8))
    sc.set_data(np.full((2, 2), 200, dtype=np.uint8))
    assert ax.core.heatmap_data(1).data.max() == 200.0  # not rescaled as colours
    with pytest.warns(UserWarning, match="alpha"):
        ax.imshow(np.eye(2), alpha=0.5)


def test_log_norm_string_is_case_insensitive_and_uneven_pcolormesh_kwargs():
    ax = plt.gca()
    ax.imshow(np.array([[1.0, 10.0], [100.0, 1000.0]]), norm="LOG")
    with pytest.warns(UserWarning, match="alpha"):
        ax.pcolormesh([0, 1, 3], [0, 2, 3], np.ones((2, 2)), alpha=0.5)
    assert ax.core.heatmap_count() == 2


def test_quiver3d_arrow_length_ratio():
    ax = plt.figure().add_subplot(projection="3d")
    pts = np.zeros(3), np.zeros(3), np.zeros(3), np.ones(3), np.zeros(3), np.zeros(3)
    short = ax.quiver(*pts, arrow_length_ratio=0.1)._entry[2]["head_length"]
    long = ax.quiver(*pts, arrow_length_ratio=0.5)._entry[2]["head_length"]
    assert long == pytest.approx(5 * short) or long > short
    assert "head_length" not in ax.quiver(*pts)._entry[2]


# --- subplots figure keywords, spines, axis colours ---------------------------------

def _frame(fig):
    """The frame's strokes in the SVG: {stroke colour: its path data}."""
    svg = fig.core.render_svg()[0]
    return {m.group(2): m.group(1).strip() for m in
            re.finditer(r'<path d="([^"]*)" fill="none" stroke="(rgb\([^)]*\))"', svg)}


def _segments(d):
    return sorted(s for s in re.split(r"(?=M)", d) if s)


RED, GREEN, BLUE, GREY = "rgb(255,0,0)", "rgb(0,255,0)", "rgb(0,0,255)", "rgb(76,76,76)"


def test_subplots_takes_figure_keywords():
    fig, ax = plt.subplots(facecolor="#102030", sharex=True)
    assert fig.get_facecolor() == pytest.approx((16 / 255, 32 / 255, 48 / 255, 1.0))
    assert '<rect width="640" height="480" fill="#102030"' in fig.core.render_svg()[0]
    # As in matplotlib, a figure that exists keeps its own.
    fig2, _ = plt.subplots(num=fig.number, facecolor="#ff0000")
    assert fig2 is fig and '"#102030"' in fig.core.render_svg()[0]
    with pytest.raises(TypeError):
        plt.subplots(num="other", bogus=1)


def test_spines_visibility_and_width():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    assert list(ax.spines) == ["left", "right", "bottom", "top"] and ax.spines.keys() == list(ax.spines)
    assert len(_segments(_frame(fig)[GREY])) == 4
    ax.spines["top"].set_visible(False)
    ax.spines.right.set_visible(False)
    assert not ax.spines["top"].get_visible() and ax.spines["left"].get_visible()
    assert len(_segments(_frame(fig)[GREY])) == 2
    ax.spines[["top", "right"]].set_visible(True)
    ax.spines[:].set_linewidth(3)
    assert ax.spines["left"].get_linewidth() == pytest.approx(3)
    assert 'stroke-width="4.16667"' in fig.core.render_svg()[0]
    with pytest.raises(KeyError):
        ax.spines["middle"]


def test_spine_colour_without_a_twin_recolours_the_frame():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.spines["top"].set_color("#ff0000")
    frame = _frame(fig)
    assert list(frame) == [RED] and len(_segments(frame[RED])) == 4
    assert ax.spines["left"].get_edgecolor() == (1.0, 0.0, 0.0, 1.0)
    ax.spines["bottom"].set_edgecolor("#0000ff")
    assert list(_frame(fig)) == [BLUE]
    ax.spines[:].set_color("#00ff00")
    assert list(_frame(fig)) == [GREEN]
    ax.cla()
    ax.plot([0, 1], [0, 1])
    assert list(_frame(fig)) == [GREY]


def test_spine_colour_follows_the_twin_axes():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    tw = ax.twinx()
    tw.plot([0, 1], [3, 2])
    ax.spines["left"].set_color("#00ff00")
    tw.spines["right"].set_color("#0000ff")
    frame = _frame(fig)
    # the left edge's x depends on the tick label font (34 px here, 36 on CI's Linux), so only its shape is pinned
    assert re.fullmatch(r"M\d+ 15\.5V450", _segments(frame[GREEN])[0]) and len(_segments(frame[GREEN])) == 1
    assert len(_segments(frame[BLUE])) == 1
    assert len(_segments(frame[GREY])) == 2  # top and bottom keep the cell's colour
    # Same edge through either object; the edge with no axis of its own is the cell's.
    assert ax.spines["right"].get_edgecolor() == tw.spines["right"].get_edgecolor() == (0.0, 0.0, 1.0, 1.0)
    ax.spines["top"].set_color("#ff0000")
    frame = _frame(fig)
    assert len(_segments(frame[RED])) == 2 and GREY not in frame
    assert GREEN in frame and BLUE in frame
    # twiny: bottom is the host's x axis, top the twin's.
    fig, ax = plt.subplots()
    (line,) = ax.plot([0, 1], [0, 1])
    ty = ax.twiny()
    ty.plot([0, 1], [3, 2])
    ax.spines["bottom"].set_color("#00ff00")
    ty.spines["top"].set_color("#0000ff")
    frame = _frame(fig)
    assert len(_segments(frame[GREEN])) == 1 and len(_segments(frame[BLUE])) == 1
    assert "H" in frame[GREEN] and "H" in frame[BLUE] and len(_segments(frame[GREY])) == 2
    # Visibility is the cell's, whichever object it is asked on.
    ty.spines["left"].set_visible(False)
    assert not ax.spines["left"].get_visible()
    # A rebuild (here: a removed artist) keeps them.
    line.remove()
    assert GREEN in _frame(fig) and BLUE in _frame(fig)


def test_axis_label_and_tick_params():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("XL")
    ax.xaxis.label.set_color("#ff0000")
    ax.yaxis.label.set_fontsize(20)
    assert ax.xaxis.label.get_color() == (1.0, 0.0, 0.0, 1.0)
    assert ax.yaxis.label.get_fontsize() == pytest.approx(20)
    assert ax.xaxis.label.get_text() == "XL"
    assert re.search(r'fill="rgb\(255,0,0\)"[^>]*>XL<', fig.core.render_svg()[0])
    ax.set_ylabel("YL", color="#00ff00", fontsize=12)
    ax.set_title("TT", color="#0000ff")
    svg = fig.core.render_svg()[0]
    assert re.search(r'fill="rgb\(0,255,0\)"[^>]*>YL<', svg) and re.search(r'fill="rgb\(0,0,255\)"[^>]*>TT<', svg)
    assert ax.yaxis.label.get_color() == (0.0, 1.0, 0.0, 1.0)
    ax.xaxis.set_tick_params(labelcolor="#ff00ff", length=9)
    assert ax._style["xlabel_color"] == (1.0, 0.0, 1.0, 1.0) and "ylabel_color" not in ax._style
    assert ax._style["xtick_length"] == pytest.approx(9 * _mpl.PT)
    # A twin's own label is its own; the shared axis' is the host's.
    tw = ax.twinx()
    tw.plot([0, 1], [3, 2])
    tw.yaxis.label.set_color("#0000ff")
    tw.xaxis.label.set_color("#ffff00")
    assert "ytitle_color" in tw._style and tw._style["ytitle_color"] == (0.0, 0.0, 1.0, 1.0)
    assert ax._style["xtitle_color"] == (1.0, 1.0, 0.0, 1.0) and "xtitle_color" not in tw._style
    assert ax._style["ytitle_color"] == (0.0, 1.0, 0.0, 1.0)
