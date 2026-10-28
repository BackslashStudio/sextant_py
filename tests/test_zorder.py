"""Per-object zorder. The core takes zorder= on every 2D plotting
call and set_zorder(id, z) / zorder(id) on Axes and Plane2D; sextant.pyplot gives
matplotlib's per-kind defaults and zorder=, set_zorder(), get_zorder() on artists."""

import math

import numpy as np
import pytest

import sextant
import sextant.pyplot as plt

X = np.array([0.0, 1.0, 2.0, 3.0])
Y = np.array([1.0, 3.0, 2.0, 4.0])


@pytest.fixture(autouse=True)
def fresh_state():
    plt.close("all")
    yield
    plt.close("all")


def figure():
    return sextant.Figure(width=240, height=180)


def test_options_take_zorder():
    fig = figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.line(X, Y, zorder=2.0)
    ax.scatter(X, Y, zorder=-1.5)
    assert ax.line_count() == 1 and ax.scatter_count() == 1
    with pytest.raises(ValueError):
        ax.line(X, Y, zorder=float("nan"))
    with pytest.raises(ValueError):
        ax.scatter(X, Y, zorder=math.inf)
    assert ax.line_count() == 1
    with pytest.raises((TypeError, ValueError), match="zorder"):
        ax.line(X, Y, not_an_option=1)  # the field list names zorder


@pytest.fixture
def messages():
    got = []
    prev = sextant.set_message_handler(got.append)
    yield got
    sextant.set_message_handler(prev)


def kinds(h):
    """One object of each kind a holder takes; their ids."""
    cells = np.arange(4.0).reshape(2, 2)
    ids = []
    for call in (lambda: h.line(X, Y), lambda: h.scatter(X, Y), lambda: h.scatter_z(X, Y, Y),
                 lambda: h.bar(X, Y), lambda: h.heatmap(cells, (0.0, 1.0), (0.0, 1.0)),
                 lambda: h.fill(X, Y, 0.0), lambda: h.quiver(X, Y, Y, X)):
        call()
        ids.append(h.last_object())
    return ids


def test_core_set_zorder_round_trips_on_every_2d_kind():
    ax = figure().add_subplot(1, 1, 1)
    ids = kinds(ax)
    ax.axline(1.0)
    ids.append(ax.last_object())
    ax.axspan(1.0, 2.0)
    ids.append(ax.last_object())
    for n, i in enumerate(ids):
        assert ax.zorder(i) == 0.0
        assert ax.set_zorder(i, 1.5 + n) is True
        assert ax.zorder(i) == 1.5 + n

    plane = figure().add_subplot3d(1, 1, 1).plane(sextant.PlaneOrientation.XY, 0.0)
    for n, i in enumerate(kinds(plane)):
        assert plane.set_zorder(i, -2.0 - n) is True
        assert plane.zorder(i) == -2.0 - n


def test_core_set_zorder_stale_ids_and_values(messages):
    ax = figure().add_subplot(1, 1, 1)
    ax.line(X, Y)
    ln = ax.last_object()
    ax.scatter(X, Y)
    sc = ax.last_object()
    ax.text("t", 0.0, 0.0)
    tx = ax.last_object()
    assert ax.remove(sc)
    messages.clear()
    assert ax.set_zorder(sc, 3.0) is False and len(messages) == 1
    assert ax.set_zorder(tx, 3.0) is False and len(messages) == 2
    assert ax.zorder(sc) is None and ax.zorder(tx) is None
    assert ax.zorder(ln) == 0.0
    with pytest.raises(ValueError):
        ax.set_zorder(ln, float("nan"))
    with pytest.raises(ValueError):
        ax.set_zorder(ln, math.inf)
    assert ax.zorder(ln) == 0.0


def test_core_axes3d_has_no_zorder():
    ax3 = figure().add_subplot3d(1, 1, 1)
    assert not hasattr(ax3, "set_zorder") and not hasattr(ax3, "zorder")


def red_pixels(fig):
    img = fig.render_rgba()
    return int(((img[..., 0] > 240) & (img[..., 1] < 20) & (img[..., 2] < 20)).sum())


def test_core_set_zorder_changes_the_picture():
    """A red square made first, a viridis one over it: raising the red one puts it on top."""
    def render(raise_red):
        fig = figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(0.0, 10.0)
        ax.set_ylim(0.0, 10.0)
        ax.scatter([5.0], [5.0], color="#ff0000", size=40.0, marker="square", alpha=1.0)
        red = ax.last_object()
        ax.scatter_z([5.0], [5.0], [0.0], size=40.0, marker="square", alpha=1.0, vmin=0.0, vmax=1.0)
        if raise_red:
            ax.set_zorder(red, 1.0)
        return red_pixels(fig)

    assert render(False) < 20 and render(True) > 500


# --- sextant.pyplot ------------------------------------------------------------------

def new_axes():
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 10.0)
    return fig, ax


def reds(fig):
    return red_pixels(fig.core)


LINE_RED = dict(color="#ff0000", linewidth=8)


def test_pyplot_imshow_after_plot_leaves_the_line_on_top():
    """matplotlib's defaults: images 0, lines 2 (it used to be call order)."""
    fig, ax = new_axes()
    ax.plot([0, 10], [5, 5], **LINE_RED)
    ax.imshow(np.zeros((2, 2)), extent=(0, 10, 0, 10), cmap="Blues")
    assert reds(fig) > 100


def test_pyplot_bar_after_plot_leaves_the_line_on_top():
    fig, ax = new_axes()
    ax.plot([0, 10], [2, 2], **LINE_RED)
    ax.bar([5], [6], width=8, color="#0000ff")
    assert reds(fig) > 100


def test_pyplot_explicit_zorder_wins_and_none_is_the_default():
    fig, ax = new_axes()
    ax.plot([0, 10], [5, 5], zorder=0.5, **LINE_RED)
    ax.imshow(np.zeros((2, 2)), extent=(0, 10, 0, 10), zorder=1.0)
    assert reds(fig) < 20  # an image above a line, by explicit zorder
    fig, ax = new_axes()
    ax.plot([0, 10], [5, 5], zorder=None, **LINE_RED)
    ax.imshow(np.zeros((2, 2)), extent=(0, 10, 0, 10))
    assert reds(fig) > 100


@pytest.mark.parametrize("make, expected", [
    (lambda ax: ax.plot([0, 1], [0, 1])[0], 2.0),
    (lambda ax: ax.plot([0, 1], [0, 1], "o")[0], 2.0),
    (lambda ax: ax.scatter([0, 1], [0, 1]), 1.0),
    (lambda ax: ax.bar([0, 1], [1, 2]), 1.0),
    (lambda ax: ax.fill_between([0, 1], [0, 1], 0), 1.0),
    (lambda ax: ax.axhline(0.5), 2.0),
    (lambda ax: ax.axvspan(0.2, 0.4), 1.0),
    (lambda ax: ax.imshow(np.zeros((2, 2))), 0.0),
    (lambda ax: ax.pcolormesh(np.zeros((2, 2))), 1.0),
    (lambda ax: ax.quiver([0, 1], [0, 1], [1, 1], [1, 1]), 1.0),
])
def test_pyplot_default_zorders_are_matplotlibs(make, expected):
    fig, ax = new_axes()
    assert make(ax).get_zorder() == expected


def test_pyplot_set_zorder_changes_the_picture_and_survives_replays():
    fig, ax = new_axes()
    line = ax.plot([0, 10], [5, 5], **LINE_RED)[0]
    ax.imshow(np.zeros((2, 2)), extent=(0, 10, 0, 10), zorder=3.0)
    other = ax.axhline(9.0)
    assert reds(fig) < 20
    line.set_zorder(5.0)
    assert line.get_zorder() == 5.0 and reds(fig) > 100
    other.remove()  # replays the log
    assert line.get_zorder() == 5.0 and reds(fig) > 100


def test_pyplot_set_zorder_sets_every_part_of_a_handle():
    fig, ax = new_axes()
    h = ax.plot([1, 2], [1, 2], "o-")[0]
    h.set_zorder(4.0)
    assert ax.core.zorder(ax.core.last_object()) == 4.0  # the marker scatter
    assert ax.core.line_count() == 1 and h.get_zorder() == 4.0
    with pytest.raises(ValueError):
        h.set_zorder(float("nan"))


def test_pyplot_marker_and_line_share_an_explicit_zorder():
    fig, ax = new_axes()
    h = ax.plot([1, 2], [1, 2], "o-", zorder=7)[0]
    assert h.get_zorder() == 7.0
    assert ax.core.zorder(ax.core.last_object()) == 7.0


def test_pyplot_nan_zorder_raises_at_the_call():
    fig, ax = new_axes()
    with pytest.raises(ValueError):
        ax.plot([0, 1], [0, 1], zorder=float("nan"))


def test_pyplot_3d_still_ignores_zorder():
    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_subplot(projection="3d")
    ax.plot([0, 1], [0, 1], [0, 1], zorder=5)
    ax.scatter([0, 1], [0, 1], [0, 1], zorder=5)


def test_pyplot_text_zorder_is_read_back_only():
    fig, ax = new_axes()
    t = ax.text(1, 1, "hi")
    assert t.get_zorder() == 3.0
    t.set_zorder(9)
    assert t.get_zorder() == 9.0 and ax.core.text_count() == 1
