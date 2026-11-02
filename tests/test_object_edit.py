"""Editing a plotted object after the call: error bars read back and set on their own
(set_*_errorbar, set_*_data(..., err=)), options read back and changed
(*_options / set_*_options), markers on a line (marker=, markersize=), and the
triangulated kinds re-triangulating when their points move. On Axes, Plane2D and
Axes3D."""

import numpy as np
import pytest

import sextant

X = np.array([0.0, 1.0, 2.0])
Y = np.array([1.0, 3.0, 2.0])
Z = np.array([0.5, 1.5, 1.0])


def axes():
    return sextant.Figure(width=240, height=180).add_subplot(1, 1, 1)


def axes3d():
    return sextant.Figure(width=240, height=180).add_subplot3d(1, 1, 1)


def plane():
    return axes3d().plane("xy")


# --- error bars --------------------------------------------------------------

KINDS_2D = [
    ("line", lambda t: t.line(X, Y)),
    ("scatter", lambda t: t.scatter(X, Y)),
    ("scatter_z", lambda t: t.scatter_z(X, Y, Z)),
    ("bar", lambda t: t.bar(X, Y)),
]


@pytest.mark.parametrize("make", [axes, plane], ids=["axes", "plane"])
@pytest.mark.parametrize("kind,plot", KINDS_2D, ids=[k for k, _ in KINDS_2D])
def test_errorbar_add_read_remove(make, kind, plot):
    t = make()
    plot(t)
    get = getattr(t, f"{kind}_errorbar")
    setter = getattr(t, f"set_{kind}_errorbar")
    e = get(0)
    assert not e.any() and e.y_cap_lo is None
    assert setter(0, sextant.ErrorBar(y_cap_lo=[0.1, 0.2, 0.3], x_box_hi=[1, 1, 1])) is t
    e = get(0)
    assert e.any()
    np.testing.assert_array_equal(e.y_cap_lo, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(e.x_box_hi, [1.0, 1.0, 1.0])
    assert e.y_cap_hi is None and e.x_cap_lo is None
    # What it reads back passes straight back.
    setter(0, e)
    np.testing.assert_array_equal(get(0).y_cap_lo, [0.1, 0.2, 0.3])
    setter(0, None)
    assert not get(0).any()
    setter(0, e)
    setter(0, sextant.ErrorBar())
    assert not get(0).any()


def test_errorbar_must_match_the_point_count():
    ax = axes()
    ax.line(X, Y, err=sextant.ErrorBar(y_cap_lo=[1, 1, 1]))
    with pytest.raises(ValueError):
        ax.set_line_errorbar(0, sextant.ErrorBar(y_cap_lo=[1, 1]))
    np.testing.assert_array_equal(ax.line_errorbar(0).y_cap_lo, [1.0, 1.0, 1.0])


def test_errorbar_from_the_plotting_call_reads_back():
    ax = axes()
    ax.scatter(X, Y, err=sextant.ErrorBar(y_cap_lo=[1, 2, 3]))
    np.testing.assert_array_equal(ax.scatter_errorbar(0).y_cap_lo, [1.0, 2.0, 3.0])


def test_set_data_with_err_changes_the_count():
    ax = axes()
    ax.line(X, Y, err=sextant.ErrorBar(y_cap_lo=[1, 1, 1]))
    oid = ax.last_object()
    ax.set_line_data(0, sextant.LineData(np.array([0.0, 1.0]), np.array([2.0, 3.0])),
                     err=sextant.ErrorBar(y_cap_lo=[0.5, 0.5]))
    np.testing.assert_array_equal(ax.line_data(0).y, [2.0, 3.0])
    np.testing.assert_array_equal(ax.line_errorbar(0).y_cap_lo, [0.5, 0.5])
    assert ax.find_object(oid) is not None
    # Without err a count change drops them, as before.
    ax.set_line_data(0, [0.0, 1.0, 2.0], [1.0, 1.0, 1.0])
    assert not ax.line_errorbar(0).any()
    # The bar data form too.
    ax.bar(X, Y)
    ax.set_bar_data(0, sextant.BarData(np.array([0.0, 1.0]), np.array([1.0, 2.0])),
                    err=sextant.ErrorBar(y_cap_hi=[0.1, 0.2]))
    np.testing.assert_array_equal(ax.bar_errorbar(0).y_cap_hi, [0.1, 0.2])


@pytest.mark.parametrize("kind", ["scatter3d", "line3d"])
def test_errorbar_3d(kind):
    ax = axes3d()
    getattr(ax, kind)(X, Y, Z)
    get = getattr(ax, f"{kind}_errorbar")
    setter = getattr(ax, f"set_{kind}_errorbar")
    assert not get(0).any()
    setter(0, sextant.ErrorBar3D(z_cap_lo=[0.1, 0.2, 0.3]))
    e = get(0)
    np.testing.assert_array_equal(e.z_cap_lo, [0.1, 0.2, 0.3])
    assert e.x_cap_lo is None
    setter(0, None)
    assert not get(0).any()
    data = getattr(ax, f"{kind}_data")(0)
    data.x, data.y, data.z = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])
    if kind == "scatter3d":
        data.sizes = None
        data.colors = None
    getattr(ax, f"set_{kind}_data")(0, data, err=sextant.ErrorBar3D(z_cap_hi=[1, 2]))
    np.testing.assert_array_equal(get(0).z_cap_hi, [1.0, 2.0])


# --- options -------------------------------------------------------------------

OPTION_KINDS_2D = [
    ("line", lambda t: t.line(X, Y)),
    ("scatter", lambda t: t.scatter(X, Y)),
    ("scatter_z", lambda t: t.scatter_z(X, Y, Z)),
    ("bar", lambda t: t.bar(X, Y)),
    ("heatmap", lambda t: t.heatmap(np.eye(2), (0.0, 1.0), (0.0, 1.0))),
    ("heatmap_tri", lambda t: t.heatmap_tri(X, Y, Z)),
    ("contour", lambda t: t.contour(X, X, np.outer(X, X))),
]


@pytest.mark.parametrize("make", [axes, plane], ids=["axes", "plane"])
@pytest.mark.parametrize("kind,plot", OPTION_KINDS_2D, ids=[k for k, _ in OPTION_KINDS_2D])
def test_options_round_trip(make, kind, plot):
    t = make()
    plot(t)
    opts = getattr(t, f"{kind}_options")(0)
    assert isinstance(opts, dict) and "name" in opts
    assert getattr(t, f"set_{kind}_options")(0, name="renamed") is t
    after = getattr(t, f"{kind}_options")(0)
    assert after["name"] == "renamed"
    # Only the field given changed.
    for k, v in opts.items():
        if k != "name":
            assert repr(after[k]) == repr(v), k
    # The whole dict goes back as keywords, to the setter and to the plotting call.
    getattr(t, f"set_{kind}_options")(0, **opts)
    assert getattr(t, f"{kind}_options")(0)["name"] == opts["name"]


def test_options_report_the_cycle_color():
    ax = axes()
    ax.line(X, Y)
    ax.line(X, Y)
    c0 = ax.line_options(0)["color"]
    c1 = ax.line_options(1)["color"]
    assert c0 is not None and c1 is not None and c0 != c1
    ax.line(X, Y, **ax.line_options(1))
    assert ax.line_options(2)["color"] == c1


def test_set_options_checks_and_keeps_data():
    ax = axes()
    ax.line(X, Y)
    oid = ax.last_object()
    with pytest.raises(ValueError):
        ax.set_line_options(0, marker="circle", markersize=-1.0)
    assert ax.line_options(0)["marker"] == sextant.MarkerStyle.NONE
    with pytest.raises(TypeError):
        ax.set_line_options(0, no_such_field=1)
    ax.set_line_options(0, color="red", zorder=2.0)
    np.testing.assert_array_equal(ax.line_data(0).y, Y)
    assert ax.find_object(oid) is not None
    assert ax.zorder(oid) == 2.0


@pytest.mark.parametrize("kind,plot", [
    ("surface", lambda a: a.surface("xy", X, X, np.outer(X, X))),
    ("surface_tri", lambda a: a.surface_tri(X, Y, Z, orient="xy")),
    ("scatter3d", lambda a: a.scatter3d(X, Y, Z)),
    ("line3d", lambda a: a.line3d(X, Y, Z)),
])
def test_options_3d(kind, plot):
    ax = axes3d()
    plot(ax)
    opts = getattr(ax, f"{kind}_options")(0)
    getattr(ax, f"set_{kind}_options")(0, name="n", alpha=0.5)
    after = getattr(ax, f"{kind}_options")(0)
    assert after["name"] == "n" and after["alpha"] == 0.5
    getattr(ax, f"set_{kind}_options")(0, **opts)
    assert getattr(ax, f"{kind}_options")(0)["alpha"] == opts["alpha"]


# --- markers on a line ---------------------------------------------------------------

def test_line_markers_draw_and_read_back():
    def svg(**kw):
        fig = sextant.Figure(width=240, height=180)
        fig.add_subplot(1, 1, 1).line(X, Y, **kw)
        return fig.render_svg()[0]

    plain, marked = svg(), svg(marker="circle", markersize=10)
    assert marked.count("<circle") >= plain.count("<circle") + 3
    ax = axes()
    ax.line(X, Y)
    assert ax.line_options(0)["marker"] == sextant.MarkerStyle.NONE
    ax.set_line_options(0, marker="square", markersize=12)
    o = ax.line_options(0)
    assert o["marker"] == sextant.MarkerStyle.SQUARE and o["markersize"] == 12.0
    with pytest.raises(ValueError):
        ax.line(X, Y, marker="circle", markersize=-1)


# --- re-triangulation ---------------------------------------------------------------

X4 = np.array([0.0, 1.0, 0.0, 1.0])
Y4 = np.array([0.0, 0.0, 1.0, 1.0])


def test_heatmap_tri_retriangulates_when_points_move():
    ax = axes()
    ax.heatmap_tri(X4, Y4, [0.0, 1.0, 2.0, 3.0], [[0, 1, 2], [1, 3, 2]])
    # Values only: the given triangles stay.
    ax.set_heatmap_tri_data(0, X4, Y4, [3.0, 2.0, 1.0, 0.0])
    np.testing.assert_array_equal(np.asarray(ax.heatmap_tri_data(0).tri).ravel(), [0, 1, 2, 1, 3, 2])
    # A new point count: triangulated again.
    x6 = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    y6 = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    ax.set_heatmap_tri_data(0, x6, y6, np.arange(6.0))
    assert np.asarray(ax.heatmap_tri_data(0).tri).size == 12
    with pytest.raises(ValueError):
        ax.set_heatmap_tri_data(0, [0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [1.0, 1.0, 1.0])


def test_surface_tri_retriangulates_in_its_plane():
    ax = axes3d()
    ax.surface_tri(X4, Y4, [0.0, 1.0, 2.0, 3.0], [[0, 1, 2], [1, 3, 2]])
    ax.set_surface_tri_data(0, X4, Y4, [3.0, 2.0, 1.0, 0.0], "xy")
    np.testing.assert_array_equal(np.asarray(ax.surface_tri_data(0).tri).ravel(), [0, 1, 2, 1, 3, 2])
    x6 = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    y6 = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    ax.set_surface_tri_data(0, x6, y6, np.arange(6.0), orient=sextant.PlaneOrientation.XY)
    assert np.asarray(ax.surface_tri_data(0).tri).size == 12
    # The triangles form still works next to it.
    ax.set_surface_tri_data(0, X4, Y4, [0.0, 0.0, 0.0, 0.0], [[0, 1, 2], [1, 3, 2]])
    assert np.asarray(ax.surface_tri_data(0).tri).size == 6
