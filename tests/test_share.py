"""Shared axes, twins and equal aspect (sextant 1.2)."""

import gc
import math

import pytest

import sextant


def grid(rows=2, cols=1):
    fig = sextant.Figure(width=300, height=300)
    return fig, [fig.add_subplot(rows, cols, i + 1) for i in range(rows * cols)]


# --- share groups ------------------------------------------------------------------------


def test_sharex_returns_self_and_shares_limits_and_scale():
    _, (a, b) = grid()
    assert b.sharex(a) is b
    a.set_xlim(2, 5)
    assert b.xlim() == (2, 5)
    b.set_xscale("log")
    assert a.xscale() == "log"
    assert b.ylim() != (2, 5) or a.ylim() != (2, 5)  # y stays per axes


def test_shared_auto_limits_cover_every_member():
    _, (a, b) = grid()
    b.sharey(a)
    a.line([0, 1], [0, 1])
    b.line([0, 1], [0, 100])
    assert a.ylim()[1] >= 100 and a.ylim() == b.ylim()


def test_sharing_refusals():
    _, (a, b) = grid()
    other = sextant.Figure().axes()
    with pytest.raises(ValueError, match="different figures"):
        a.sharex(other)


def test_heatmap_shares_a_log_scale():
    _, (a, b) = grid()
    a.imshow([[1.0, 2.0]])
    b.set_xscale("log")
    a.sharex(b)  # heatmaps draw on every scale since 1.3
    assert a.xscale() == "log"


def test_cla_keeps_membership():
    _, (a, b) = grid()
    b.sharex(a)
    b.cla()
    a.set_xlim(3, 4)
    assert b.xlim() == (3, 4)


def test_inner_tick_labels_hide_unless_set():
    fig, (a, b) = grid()
    a.line([0, 1], [0, 1])
    b.line([0, 1], [0, 1])
    alone = fig.render_rgba()
    b.sharex(a)
    shared = fig.render_rgba()
    assert (shared != alone).any()
    a.set_axes_style(show_xtick_labels=True)
    assert (fig.render_rgba() != shared).any()


# --- twins ---------------------------------------------------------------------------------


def test_twin_is_one_object():
    _, (a,) = grid(1)
    t = a.twinx()
    assert t is a.twinx() and isinstance(t, sextant.Axes)
    assert a.twiny() is a.twiny() and a.twiny() is not t


def test_twin_shares_one_axis():
    _, (a,) = grid(1)
    t = a.twinx()
    a.set_xlim(0, 7)
    assert t.xlim() == (0, 7)
    t.set_ylim(100, 200)
    assert a.ylim() != (100, 200)


def test_twin_of_a_twin_is_refused():
    _, (a,) = grid(1)
    with pytest.raises(ValueError):
        a.twinx().twinx()
    with pytest.raises(ValueError):
        a.twiny().twinx()


def test_twin_title_is_the_cells():
    _, (a,) = grid(1)
    a.twinx().set_title("both")
    assert a.title() == "both"


def test_twin_plots_draw():
    fig, (a,) = grid(1)
    a.line([0, 1], [0, 1])
    before = fig.render_rgba()
    a.twinx().line([0, 1], [5, 3], color="red", linewidth=4)
    assert (fig.render_rgba() != before).any()


def test_twin_lives_with_its_host():
    # Dropping the Python object does not drop the twin: the host keeps it.
    _, (a,) = grid(1)
    a.twinx().set_ylim(5, 6)
    gc.collect()
    assert a.twinx().ylim() == (5, 6)


# --- aspect ------------------------------------------------------------------------------


def test_aspect_read_back():
    _, (a,) = grid(1)
    assert a.aspect() == 0 and a.adjustable() is sextant.Adjustable.BOX
    a.set_aspect(2, "datalim")
    assert a.aspect() == 2 and a.adjustable() == "datalim"
    a.set_aspect(1)
    assert a.adjustable() == "box"
    a.cla()
    assert a.aspect() == 0


def test_aspect_changes_the_frame():
    fig, (a,) = grid(1)
    a.line([0, 1], [0, 1])
    auto = fig.render_rgba()
    a.set_xlim(0, 4).set_ylim(0, 1).set_aspect(1)
    assert (fig.render_rgba() != auto).any()


@pytest.mark.parametrize("ratio", [-1.0, math.inf, math.nan])
def test_bad_aspect_is_a_value_error(ratio):
    with pytest.raises(ValueError):
        grid(1)[1][0].set_aspect(ratio)


def test_aspect_refusals():
    _, (a,) = grid(1)
    with pytest.raises(ValueError):
        a.twinx().set_aspect(1)
    _, (a, b) = grid()
    b.sharex(a)
    b.sharey(a)
    with pytest.raises(ValueError):
        b.set_aspect(1, "datalim")
    b.set_aspect(1, "box")
    _, (a, b) = grid()
    a.set_aspect(1, "datalim")
    b.sharey(a)
    with pytest.raises(ValueError):
        a.twinx()  # shares x: a datalim axes with both axes shared


def test_bad_adjustable_is_a_type_error():
    with pytest.raises(TypeError):
        grid(1)[1][0].set_aspect(1, "frame")


# --- events ---------------------------------------------------------------------------------


def test_event_has_the_twin_fields():
    for name in ("x2data", "y2data", "pick_twin"):
        assert isinstance(getattr(sextant.Event, name), property)
