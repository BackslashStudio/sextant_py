"""sextant.pyplot and the matplotlib-shaped adapters (sextant/_mpl.py)."""

import io
import math
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
    with pytest.raises(TypeError):
        ax.plot([0, 1], drawstyle="steps")


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
    with pytest.raises(ValueError, match="one marker size"):
        ax.scatter([0, 1], [0, 1], s=[10, 20])


def test_bar_width_errors_and_tick_labels():
    ax = plt.gca()
    b = ax.bar([0, 2, 4], [1, 2, 3], width=1.6, yerr=[0.1, 0.2, 0.3], capsize=4, tick_label=["a", "b", "c"])
    kw = b._entry[2]
    assert kw["width"] == pytest.approx(0.8)  # data units -> fraction of the spacing
    assert kw["errorbar"]["capsize"] == pytest.approx(4 * _mpl.PT)
    assert b._entry[2]["linewidth"] == 0.0  # no edge by default, as matplotlib
    assert ax._log[-1][0] == "set_xticks"
    np.testing.assert_array_equal(b.datavalues, [1, 2, 3])
    with pytest.raises(ValueError, match="bottom"):
        ax.bar([0], [1], bottom=[1])


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
    with pytest.raises(ValueError, match="RGB"):
        ax.imshow(np.zeros((2, 2, 3)))


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
    with pytest.raises(ValueError, match="evenly spaced"):
        ax.pcolormesh([0, 1, 5, 6], [0, 1], np.ones((1, 3)))


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
        pick_kind=sextant.PickKind.LINE, pick_object=0, pick_index=1, pick_plane=-1)
    m = _mpl.MplEvent(fig, "pick_event", e)
    assert m.inaxes is ax and m.artist is h and list(m.ind) == [1]
    assert m.button == 3  # sextant's right button is matplotlib's 3
    assert m.y == 480 - 100  # from the bottom
    e.kind, e.scroll_y = sextant.EventKind.SCROLL, -2.0
    m = _mpl.MplEvent(fig, "scroll_event", e)
    assert m.button == "down" and m.step == -2.0


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
