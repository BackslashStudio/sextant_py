"""Axes.pie() and the axis switch, through the core binding and pyplot."""

import math

import numpy as np
import pytest
from test_pyplot import twin_event
from test_shapes import area, fills

import sextant
import sextant.pyplot as plt
from sextant import _mpl


def axes():
    fig = sextant.Figure(width=200, height=200)
    return fig, fig.axes()


def test_pie_draws_one_slice_per_value_by_share():
    fig, ax = axes()
    assert ax.pie([1, 1, 2]) is ax
    slices = fills(fig.render_svg()[0])
    assert [c for c, _ in slices] == [(31, 119, 180), (255, 126, 14), (44, 159, 44)]  # tab10 in turn
    a = [area(p) for _, p in slices]
    assert a[0] == pytest.approx(a[1], rel=0.01) and a[2] == pytest.approx(2 * a[0], rel=0.01)


def test_pie_turns_the_axis_off_and_sets_aspect_one():
    fig, ax = axes()
    assert ax.axis_visible()
    ax.pie([1, 2])
    assert not ax.axis_visible() and ax.aspect() == 1.0
    assert ax.set_axis_visible() is ax and ax.axis_visible()
    ax.set_axis_visible(False)
    ax.cla()
    assert ax.axis_visible() and ax.pie_count() == 0


def test_axis_off_hides_the_spines():
    fig, ax = axes()
    ax.line([0, 1], [0, 1])
    with_axis = fig.render_svg()[0]
    ax.set_axis_visible(False)
    without = fig.render_svg()[0]
    assert with_axis.count("<text") > without.count("<text")


def test_pie_data_round_trip():
    fig, ax = axes()
    ax.pie([1, 2, 3], labels=["a", "b", "c"], explode=[0, 0.1, 0], autopct="%1.1f%%")
    assert ax.pie_count() == 1
    d = ax.pie_data(0)
    assert isinstance(d, sextant.PieData)
    np.testing.assert_array_equal(d.values, [1, 2, 3])
    assert d.labels == ["a", "b", "c"]
    np.testing.assert_array_equal(d.explode, [0, 0.1, 0])
    assert ax.pie_data(-1).labels == d.labels
    with pytest.raises(IndexError):
        ax.pie_data(1)
    assert ax.find_object(ax.last_object()) == (sextant.PickKind.PIE, 0)


def test_pie_data_without_labels_or_explode():
    fig, ax = axes()
    ax.pie([1, 2])
    d = ax.pie_data(0)
    assert d.labels == [] and d.explode is None
    assert "PieData(values=" in repr(d)


def test_set_pie_data_both_forms():
    fig, ax = axes()
    ax.pie([1, 2, 3], labels=["a", "b", "c"])
    assert ax.set_pie_data(0, [3, 2, 1]) is ax
    d = ax.pie_data(0)
    np.testing.assert_array_equal(d.values, [3, 2, 1])
    assert d.labels == ["a", "b", "c"]  # kept
    ax.set_pie_data(0, [1, 1], labels=["x", "y"], explode=[0.2, 0])
    d = ax.pie_data(0)
    assert d.labels == ["x", "y"]
    np.testing.assert_array_equal(d.explode, [0.2, 0])
    ax.set_pie_data(0, sextant.PieData([4, 5, 6]))
    d = ax.pie_data(0)
    np.testing.assert_array_equal(d.values, [4, 5, 6])
    assert d.labels == [] and d.explode is None
    with pytest.raises(ValueError, match="labels"):
        ax.set_pie_data(0, [1, 2], labels=["a"])
    ax.set_pie_data(0, sextant.PieData([1, 2], ["a", "b"]))
    with pytest.raises(ValueError, match="labels"):  # the kept labels no longer fit
        ax.set_pie_data(0, [1, 2, 3])


@pytest.mark.parametrize("values, kw, msg", [
    ([-1, 2], {}, "finite"),
    ([1, float("nan")], {}, "finite"),
    ([0, 0], {}, "sum"),
    ([1, 2], {"labels": ["a"]}, "labels"),
    ([1, 2], {"explode": [1]}, "explode"),
    ([1, 2], {"explode": [-1, 0]}, "explode"),
    ([1, 2], {"radius": 0}, "radius"),
    ([1, 2], {"inner": 1}, "inner"),
    ([1, 2], {"autopct": "x"}, "autopct"),
    ([1, 2], {"normalize": False}, "normalize"),
])
def test_pie_errors_add_nothing(values, kw, msg):
    fig, ax = axes()
    with pytest.raises(ValueError, match=msg):
        ax.pie(values, **kw)
    assert ax.pie_count() == 0


def test_pie_options_are_checked_by_name():
    fig, ax = axes()
    with pytest.raises(TypeError, match="wedgeprops"):
        ax.pie([1, 2], wedgeprops={})


def test_pie_visible_and_remove():
    fig, ax = axes()
    ax.pie([1, 2])
    assert ax.pie_visible(0)
    assert ax.set_pie_visible(0, False) is ax and not ax.pie_visible(0)
    assert fills(fig.render_svg()[0]) == []
    ax.set_pie_visible(0, True)
    assert ax.remove(ax.last_object()) and ax.pie_count() == 0


def test_pie_labels_and_percents_are_drawn():
    fig, ax = axes()
    ax.pie([1, 3], labels=["apples", "pears"], autopct="%.0f%%")
    svg = fig.render_svg()[0]
    for s in ("apples", "pears", "25%", "75%"):
        assert s in svg


# --- sextant.pyplot -------------------------------------------------------------------------

@pytest.fixture
def mpl():
    plt.close("all")
    yield
    plt.close("all")


def test_mpl_pie_returns_wedges_and_texts(mpl):
    fig, ax = plt.subplots()
    wedges, texts = ax.pie([1, 2, 3], labels=["a", "b", "c"])
    assert len(wedges) == len(texts) == 3 and isinstance(wedges[0], _mpl.PieWedge)
    assert [t.get_text() for t in texts] == ["a", "b", "c"] and wedges[1].get_label() == "b"
    wedges, texts, autotexts = ax.pie([1, 3], autopct="%1.1f%%", labeldistance=None)
    assert texts == [] and [t.get_text() for t in autotexts] == ["25.0%", "75.0%"]
    assert ax.core.pie_count() == 2 and not ax.axison and ax.get_aspect() == 1.0


def test_mpl_pie_takes_the_patch_cycle(mpl):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])  # the line cycle is its own
    ax.bar([0], [1])  # C0
    wedges, _ = ax.pie([1, 1])
    assert [w.get_facecolor() for w in wedges] == [_mpl._hex(_mpl.TAB10[1]), _mpl._hex(_mpl.TAB10[2])]
    wedges, _ = ax.pie([1, 1, 1], colors=["red", "blue"])
    assert [w.get_facecolor()[:3] for w in wedges] == [(1, 0, 0), (0, 0, 1), (1, 0, 0)]


def test_mpl_pie_maps_matplotlib_units(mpl):
    fig, ax = plt.subplots()
    ax.pie([1, 1], radius=2, explode=[0.2, 0], center=(1, -1), startangle=90, counterclock=False,
           wedgeprops={"width": 0.5, "edgecolor": "white", "linewidth": 2, "alpha": 0.5},
           textprops={"color": "red", "fontsize": 12})
    opts = ax._log[-1][2]
    np.testing.assert_allclose(ax.core.pie_data(0).explode, [0.1, 0])  # data units -> fraction of the radius
    assert opts["inner"] == 1.5  # width -> the hole
    assert (opts["cx"], opts["cy"], opts["startangle"], opts["counterclock"]) == (1, -1, 90, False)
    assert opts["linewidth"] == pytest.approx(2 * _mpl.PT) and opts["alpha"] == 0.5
    assert opts["edgecolor"] == (1, 1, 1, 1) and opts["textcolor"][:3] == (1, 0, 0)
    assert opts["fontsize"] == pytest.approx(12 * _mpl.PT)


def test_mpl_pie_wedge_geometry(mpl):
    fig, ax = plt.subplots()
    wedges, _ = ax.pie([1, 2, 3], explode=[0, 0.1, 0], startangle=90)
    assert (wedges[1].theta1, wedges[1].theta2) == pytest.approx((150, 270))
    assert wedges[1].center == pytest.approx((0.1 * math.cos(math.radians(210)), 0.1 * math.sin(math.radians(210))))
    assert wedges[0].center == (0, 0) and wedges[0].r == 1
    wedges, _ = ax.pie([1, 1], counterclock=False)
    assert (wedges[0].theta1, wedges[0].theta2) == (-180, 0)


def test_mpl_pie_frame_keeps_the_axis(mpl):
    fig, ax = plt.subplots()
    ax.pie([1, 2], frame=True)
    assert ax.axison
    ax.cla()
    assert ax.axison and ax.core.pie_count() == 0


@pytest.mark.parametrize("kw, exc", [
    ({"autopct": lambda p: f"{p:.0f}"}, TypeError),
    ({"autopct": 3}, TypeError),
    ({"shadow": True}, ValueError),
    ({"hatch": "/"}, ValueError),
    ({"rotatelabels": True}, ValueError),
    ({"wedgeprops": {"hatch": "/"}}, TypeError),
    ({"textprops": {"weight": "bold"}}, TypeError),
    ({"explode": [-1, 0]}, ValueError),
])
def test_mpl_pie_raises(mpl, kw, exc):
    fig, ax = plt.subplots()
    with pytest.raises(exc):
        ax.pie([1, 2], **kw)
    assert ax.core.pie_count() == 0


def test_mpl_pie_parts_restyle_the_pie(mpl):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    wedges, texts, autotexts = ax.pie([1, 2], labels=["a", "b"], autopct="%d%%")
    wedges[1].set_facecolor("red")
    assert ax._log[-1][2]["colors"][1] == (1, 0, 0, 1) and wedges[0].get_facecolor() != (1, 0, 0, 1)
    wedges[0].set_label("A")
    assert ax.core.pie_data(0).labels == ["A", "b"] and texts[0].get_text() == "A"
    texts[1].set_color("blue")
    autotexts[0].set_fontsize(14)
    assert ax._log[-1][2]["textcolor"][:3] == (0, 0, 1) and ax._log[-1][2]["fontsize"] == pytest.approx(14 * _mpl.PT)
    wedges[0].set_edgecolor("black")
    assert ax._log[-1][2]["linewidth"] == pytest.approx(_mpl.PT)
    assert wedges[0].get_zorder() == 1
    wedges[1].set_zorder(5)
    assert wedges[0].get_zorder() == 5
    texts[0].set_visible(False)
    assert not ax.core.pie_visible(0) and not wedges[1].get_visible()
    wedges[0].remove()
    assert ax.core.pie_count() == 0 and ax.core.line_count() == 1
    with pytest.raises(ValueError):
        wedges[1].remove()


def test_mpl_pie_pick_names_the_slice(mpl):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    wedges, _ = ax.pie([1, 2, 3])
    m = twin_event(fig, ax, pick_kind=sextant.PickKind.PIE, pick_object=ax.core.last_object(), pick_index=2)
    assert m.artist is wedges[2] and list(m.ind) == [2]


def test_plt_pie_and_export(mpl, tmp_path):
    plt.pie([1, 2], labels=["x", "y"], autopct="%.0f%%")
    plt.savefig(tmp_path / "pie.png")
    plt.savefig(tmp_path / "pie.svg")
    svg = (tmp_path / "pie.svg").read_text()
    assert (tmp_path / "pie.png").stat().st_size > 0 and "67%" in svg and ">x<" in svg
