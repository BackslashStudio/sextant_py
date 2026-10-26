"""sextant 1.3 stage 4: PolarAxes and Figure.add_subplot_polar(), through the core binding and pyplot."""

import math

import numpy as np
import pytest

from test_pyplot import twin_event

import sextant
import sextant.pyplot as plt
from sextant import Coords, ThetaZero
from test_objects import messages  # noqa: F401  (fixture)


def polar():
    fig = sextant.Figure(width=200, height=200)
    return fig, fig.add_subplot_polar(1, 1, 1)


# --- the subplot --------------------------------------------------------------------------

def test_add_subplot_polar_overloads_return_the_same_object():
    fig = sextant.Figure(width=300, height=200)
    p = fig.add_subplot_polar(1, 2, 1)
    assert isinstance(p, sextant.PolarAxes)
    assert fig.add_subplot_polar(1, 2, 1) is p and fig.add_subplot_polar(1) is p
    q = fig.add_subplot_polar(1, 2, (2, 2))
    assert fig.add_subplot_polar((2, 2)) is q and q is not p


def test_a_cell_holds_one_kind():
    fig = sextant.Figure(width=300, height=200)
    fig.add_subplot_polar(1, 3, 1)
    fig.add_subplot(1, 3, 2)
    fig.add_subplot3d(1, 3, 3)
    with pytest.raises(ValueError, match="polar"):
        fig.add_subplot(1, 3, 1)
    with pytest.raises(ValueError, match="polar"):
        fig.add_subplot3d(1, 3, 1)
    for i in (2, 3):
        with pytest.raises(ValueError):
            fig.add_subplot_polar(1, 3, i)


def test_theta_zero_names():
    assert [z.value for z in ThetaZero] == ["e", "n", "w", "s", "ne", "nw", "sw", "se"]
    assert ThetaZero("NW") is ThetaZero.NW


# --- plot kinds -----------------------------------------------------------------------------

def test_plot_both_forms_read_back_theta_and_r():
    fig, p = polar()
    assert p.plot([0.0, 1.0], [2.0, 3.0]) is p
    p.plot([1.0, 2.0, 3.0])
    assert p.line_count() == 2
    np.testing.assert_array_equal(p.line_data(0).x, [0, 1])
    np.testing.assert_array_equal(p.line_data(0).y, [2, 3])
    np.testing.assert_array_equal(p.line_data(1).x, [0, 1, 2])
    p.set_line_data(0, [0.5, 1.5], [1.0, 1.0])
    np.testing.assert_array_equal(p.line_data(0).x, [0.5, 1.5])
    p.set_line_data(0, sextant.LineData(x=[0.0], y=[4.0]))
    np.testing.assert_array_equal(p.line_data(0).y, [4])
    with pytest.raises(TypeError, match="LineOptions"):
        p.plot([1.0], bogus=1)


def test_each_kind_counts_reads_back_and_updates():
    fig, p = polar()
    t = np.array([0.0, 1.0, 2.0])
    p.scatter(t, [1, 2, 3]).scatter_z(t, [1, 2, 3], [0, 1, 2], colorbar=True)
    p.bar(t, [1, 2, 3]).fill(t, [1, 2, 1]).fill(t, [2, 3, 2], [1, 1, 1]).shape_poly(t, [1, 1, 1])
    assert (p.scatter_count(), p.scatter_z_count(), p.bar_count(), p.fill_count(), p.shape_poly_count()) == (
        1, 1, 1, 2, 1)
    np.testing.assert_array_equal(p.fill_data(0).y2, [0, 0, 0])  # r2 defaults to the centre
    np.testing.assert_array_equal(p.fill_data(1).y2, [1, 1, 1])
    np.testing.assert_array_equal(p.bar_data(0).height, [1, 2, 3])

    p.set_scatter_data(0, t, [3, 3, 3])
    np.testing.assert_array_equal(p.scatter_data(0).y, [3, 3, 3])
    p.set_scatter_z_data(0, t, [1, 1, 1], [5, 5, 5])
    np.testing.assert_array_equal(p.scatter_z_data(0).z, [5, 5, 5])
    p.set_bar_data(0, t, [4, 4, 4])
    np.testing.assert_array_equal(p.bar_data(0).height, [4, 4, 4])
    p.set_fill_data(0, t, [2, 2, 2], [1, 1, 1])
    np.testing.assert_array_equal(p.fill_data(0).y1, [2, 2, 2])
    p.set_shape_poly_data(0, t, [2, 2, 2])
    np.testing.assert_array_equal(p.shape_poly_data(0).y, [2, 2, 2])
    for kind in ("scatter", "scatter_z", "bar", "fill", "shape_poly"):
        d = getattr(p, f"{kind}_data")(0)
        assert getattr(p, f"set_{kind}_data")(0, d) is p
    svg = fig.render_svg()[0]
    assert svg.count("<path") > 5


def test_plot_kind_errors():
    fig, p = polar()
    with pytest.raises(ValueError, match="Vertical"):
        p.bar([0.0], [1.0], orientation="horizontal")
    with pytest.raises(ValueError, match="Data only"):
        p.shape_poly([0, 1, 2], [1, 1, 1], space="fraction")
    assert p.bar_count() == p.shape_poly_count() == 0


def test_text_and_annotate():
    fig, p = polar()
    p.text("a", 0.5, 1.0).text("b", 0.5, 0.5, coords="fraction")
    p.annotate(0.0, 1.0, "c", 1.0, 2.0, arrow={"head_length": 8})
    assert p.text_count() == 3
    d = p.text_data(1)
    assert (d.text, d.xcoords, d.ycoords) == ("b", Coords.FRACTION, Coords.FRACTION)
    assert p.text_data(2).arrow and (p.text_data(2).px, p.text_data(2).py) == (0.0, 1.0)
    p.set_text_data(1, "B", 0.2, 0.3)  # keeps fraction
    d = p.text_data(1)
    assert (d.text, d.x, d.xcoords) == ("B", 0.2, Coords.FRACTION)
    p.set_text_data(1, "B", 0.2, 0.3, coords="data")
    assert p.text_data(1).ycoords == Coords.DATA
    d.xcoords = Coords.DATA
    with pytest.raises(ValueError):  # mixed spaces
        p.set_text_data(1, d)
    assert "a" in fig.render_svg()[0]


def test_visible_and_objects(messages):
    fig, p = polar()
    p.plot([0.0, 1.0], [1.0, 2.0])
    line = p.last_object()
    p.scatter([0.0], [1.0])
    assert p.find_object(line) == (sextant.PickKind.LINE, 0)
    assert p.find_object(p.last_object()) == (sextant.PickKind.SCATTER, 0)
    assert p.set_line_visible(0, False) is p and not p.line_visible(0)
    assert p.set_zorder(line, 5) and p.zorder(line) == 5
    assert p.remove(line) and p.line_count() == 0 and messages == []
    assert not p.remove(line) and len(messages) == 1
    for kind in ("line", "scatter", "scatter_z", "bar", "fill", "shape_poly", "text"):
        assert hasattr(p, f"set_{kind}_visible") and hasattr(p, f"{kind}_visible")


# --- the axes ------------------------------------------------------------------------------

def test_r_axis():
    fig, p = polar()
    assert p.rlim() == (0.0, 1.0)
    p.plot([1.0, 2.0, 3.0])
    assert p.rlim() == (0.0, 3.0)
    assert p.set_rmax(5) is p and p.rlim() == (0.0, 5.0)
    p.set_rmin(1)
    assert p.rlim() == (1.0, 5.0)
    p.set_rlim(4, 0)
    assert p.rlim() == (4.0, 0.0)
    with pytest.raises(ValueError):
        p.set_rlim(1, 1)
    with pytest.raises(ValueError):
        p.set_rlim(0, math.inf)
    p.set_rticks([1, 2], ["one", "two"]).set_rtick_format("%.1f")
    assert "one" in fig.render_svg()[0]
    with pytest.raises(ValueError):
        p.set_rtick_format("x")
    assert p.rlabel_position() == 22.5
    p.set_rlabel_position(60)
    assert p.rlabel_position() == 60


def test_theta_axis():
    fig, p = polar()
    assert (p.theta_offset(), p.theta_direction(), p.thetalim()) == (0.0, 1, (0.0, 360.0))
    assert p.set_theta_zero_location("n") is p and p.theta_offset() == 90
    p.set_theta_zero_location(ThetaZero.N, 10)
    assert p.theta_offset() == 100
    p.set_theta_zero_location("sw")
    assert p.theta_offset() == 225
    p.set_theta_offset(30)
    assert p.theta_offset() == 30
    p.set_theta_direction(-1)
    assert p.theta_direction() == -1
    with pytest.raises(ValueError):
        p.set_theta_direction(2)
    p.set_thetalim(0, 180)
    assert p.thetalim() == (0.0, 180.0)
    p.set_thetamin(-45).set_thetamax(90)
    assert p.thetalim() == (-45.0, 90.0)
    for lo, hi in ((10, 0), (0, 400)):
        with pytest.raises(ValueError):
            p.set_thetalim(lo, hi)
    p.set_thetagrids([0, 90], ["E", "N"])
    svg = fig.render_svg()[0]
    assert ">E<" in svg and ">N<" in svg


def test_decoration_and_cla():
    fig, p = polar()
    p.plot([1.0, 2.0], name="s").legend().set_title("polar").grid(False).grid(True, color="red")
    p.set_axes_style(background="#eeeeee").set_colorbar_style().set_color_cycle(["red", "blue"])
    assert p.title() == "polar"
    p.set_theta_offset(90).set_rlim(0, 5).text("t", 0, 1)
    assert p.cla() is p
    assert (p.line_count(), p.text_count(), p.theta_offset(), p.rlim(), p.title()) == (0, 0, 0.0, (0.0, 1.0), "")


# --- pyplot -------------------------------------------------------------------------------

@pytest.fixture
def mpl():
    plt.close("all")
    yield
    plt.close("all")


def test_every_route_makes_a_polar_axes(mpl):
    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1, projection="polar")
    b = fig.add_subplot(2, 2, 2, polar=True)
    assert isinstance(a, plt.PolarAxes) and isinstance(b, plt.PolarAxes)
    assert isinstance(a.core, sextant.PolarAxes) and a.core is fig.core.add_subplot_polar(2, 2, 1)
    assert isinstance(plt.subplot(2, 2, 3, projection="polar"), plt.PolarAxes)
    _, axs = plt.subplots(1, 2, subplot_kw={"projection": "polar"})
    assert all(isinstance(ax, plt.PolarAxes) for ax in axs)
    plt.figure()
    assert isinstance(plt.axes(polar=True), plt.PolarAxes)
    plt.figure()
    lines = plt.polar([0, 1], [1, 2], "r-")
    assert isinstance(plt.gca(), plt.PolarAxes) and len(lines) == 1


def test_route_errors(mpl):
    fig = plt.figure()
    with pytest.raises(ValueError, match="polar=True"):
        fig.add_subplot(1, 2, 1, projection="3d", polar=True)
    with pytest.raises(ValueError, match="'polar'"):
        fig.add_subplot(1, 2, 1, projection="hammer")
    a = fig.add_subplot(1, 2, 1)
    with pytest.raises(ValueError, match="2D axes only"):
        fig.add_subplot(1, 2, 2, projection="polar", sharex=a)
    with pytest.raises(ValueError, match="2D axes only"):
        plt.subplots(1, 2, sharey=True, subplot_kw={"polar": True})
    plt.figure()
    plt.plot([0, 1], [1, 2])
    with pytest.raises(ValueError, match="not polar"):
        plt.polar([0, 1], [1, 2])
    for f in (plt.thetagrids, plt.rgrids):
        with pytest.raises(RuntimeError, match="only defined for polar"):
            f([1, 2])


def test_plotting_calls_reach_the_core(mpl):
    ax = plt.subplot(projection="polar")
    t = np.array([0.0, 1.0, 2.0])
    ax.plot(t, [1, 2, 3], "-", label="a")
    ax.scatter(t, [1, 2, 3])
    ax.scatter(t, [1, 2, 3], c=[0, 1, 2])
    bars = ax.bar(t, [1, 2, 3], width=0.5)
    ax.fill_between(t, 1, 2)
    polys = ax.fill(t, [1, 1, 1], "red", t, [2, 2, 2])
    ax.text(0, 1, "t")
    ax.annotate("a", (0, 1), xytext=(1, 2), arrowprops={})
    c = ax.core
    assert (c.line_count(), c.scatter_count(), c.scatter_z_count(), c.bar_count(), c.fill_count(),
            c.shape_poly_count(), c.text_count()) == (1, 1, 1, 1, 1, 2, 2)
    assert bars is not None and len(polys) == 2
    np.testing.assert_array_equal(c.line_data(0).x, t)
    np.testing.assert_array_equal(c.fill_data(0).y1, [1, 1, 1])
    np.testing.assert_array_equal(c.fill_data(0).y2, [2, 2, 2])
    assert ax._grid_on  # on by default, as matplotlib's


def test_unsupported_raise(mpl):
    ax = plt.subplot(projection="polar")
    with pytest.raises(ValueError, match="error bars"):
        ax.bar([0.0], [1.0], yerr=[0.1])
    with pytest.raises(ValueError, match="error bars"):
        ax.errorbar([0.0], [1.0], yerr=[0.1])
    with pytest.raises(ValueError, match="centre"):
        ax.set_rorigin(-1)
    for f in (ax.set_rscale, ax.set_yscale):
        with pytest.raises(ValueError, match="linear"):
            f("log")
    with pytest.raises(ValueError, match="axis titles"):
        ax.set_xlabel("theta")
    assert ax.core.bar_count() == 0


def test_r_axis_mapping(mpl):
    ax = plt.subplot(projection="polar")
    ax.set_rlim(0, 4)
    assert ax.get_ylim() == (0.0, 4.0)
    ax.set_rlim(rmin=1)
    assert (ax.get_rmin(), ax.get_rmax()) == (1.0, 4.0)
    ax.set_ylim(top=5)
    assert ax.core.rlim() == (1.0, 5.0)
    ax.set_rmax(6)
    assert ax.core.rlim() == (1.0, 6.0)
    ax.invert_yaxis()
    assert ax.core.rlim() == (6.0, 1.0)
    ax.set_rgrids([2, 4], ["two", "four"], angle=60)
    assert ax.get_rlabel_position() == 60
    assert "four" in ax.figure.core.render_svg()[0]
    ax.set_rlabel_position(30)
    assert ax.core.rlabel_position() == 30
    ax.set_rgrids([2, 4], fmt="%.2f")
    assert "2.00" in ax.figure.core.render_svg()[0]
    assert ax.get_rscale() == ax.get_yscale() == "linear"
    ax.set_rticks([])
    assert ax._style.get("show_yticks") is False


def test_theta_units_are_matplotlibs(mpl):
    ax = plt.subplot(projection="polar")
    ax.set_theta_offset(math.pi / 2)
    assert ax.core.theta_offset() == pytest.approx(90)
    assert ax.get_theta_offset() == pytest.approx(math.pi / 2)
    ax.set_theta_zero_location("W", offset=10)
    assert ax.core.theta_offset() == pytest.approx(190)
    ax.set_theta_direction("clockwise")
    assert ax.get_theta_direction() == -1
    ax.set_theta_direction(1)
    assert ax.core.theta_direction() == 1
    with pytest.raises(ValueError):
        ax.set_theta_direction("sideways")
    assert ax.set_thetalim(0, math.pi) == pytest.approx((0, math.pi))
    assert ax.core.thetalim() == pytest.approx((0, 180))
    ax.set_thetalim(thetamin=-90, thetamax=90)
    assert ax.core.thetalim() == pytest.approx((-90, 90))
    ax.set_xlim(0, math.pi / 2)
    assert ax.get_xlim() == pytest.approx((0, math.pi / 2))
    ax.set_thetamax(270)
    ax.set_thetamin(10)
    assert (ax.get_thetamin(), ax.get_thetamax()) == pytest.approx((10, 270))
    ax.set_thetagrids([0, 45, 90], fmt="%d deg")
    assert "45 deg" in ax.figure.core.render_svg()[0]
    ax.set_xticks([0, math.pi / 2], ["E", "N"])
    svg = ax.figure.core.render_svg()[0]
    assert ">N<" in svg and "45 deg" not in svg
    plt.thetagrids([0, 180], ["a", "b"])
    plt.rgrids([0.5], ["half"])
    svg = ax.figure.core.render_svg()[0]
    assert ">b<" in svg and "half" in svg


def test_cla_resets(mpl):
    ax = plt.subplot(projection="polar")
    ax.plot([0, 1], [1, 2], label="s")
    ax.set_title("p")
    ax.legend()
    ax.set_theta_offset(1.0)
    assert ax.core.title() == "p"
    ax.grid(False)
    ax.cla()
    assert ax.core.line_count() == 0 and ax.core.theta_offset() == 0.0 and ax._grid_on


def test_artists_remove_zorder_visible(mpl):
    ax = plt.subplot(projection="polar")
    (line,) = ax.plot([0, 1], [1, 2])
    sc = ax.scatter([0], [1], zorder=5)
    assert ax.core.zorder(ax.core.last_object()) == 5
    line.remove()
    assert ax.core.line_count() == 0
    sc.set_visible(False)
    assert not ax.core.scatter_visible(0)


def test_a_pick_over_a_polar_axes_finds_its_artist(mpl):
    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")
    (line,) = ax.plot([0, 1], [1, 2])
    e = twin_event(fig, ax, pick_kind=sextant.PickKind.LINE, pick_object=ax.core.last_object())
    assert e.inaxes is ax and e.artist is line


@pytest.mark.parametrize("fmt", ["png", "svg"])
def test_export(mpl, tmp_path, fmt):
    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")
    ax.plot(np.linspace(0, 6, 30), np.linspace(0, 1, 30))
    out = tmp_path / f"p.{fmt}"
    fig.savefig(out)
    assert out.stat().st_size > 100
