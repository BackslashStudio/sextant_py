"""What the adapter reads back -- bounds, ticks, the artist lists, the legend, figure geometry."""
import numpy as np
import pytest

import sextant.pyplot as plt


@pytest.fixture
def fig_ax():
    fig, ax = plt.subplots(figsize=(4, 3))
    yield fig, ax
    plt.close("all")


def test_bounds_are_sorted_and_keep_the_direction(fig_ax):
    _, ax = fig_ax
    ax.set_xlim(0, 10)
    assert ax.get_xbound() == (0, 10) and not ax.xaxis_inverted()
    ax.set_xbound(8, 2)  # arguments sorted
    assert ax.get_xlim() == (2, 8)
    ax.invert_xaxis()
    assert ax.xaxis_inverted() and ax.get_xbound() == (2, 8)
    ax.set_xbound(1, 9)  # an inverted axis stays inverted
    assert ax.get_xlim() == (9, 1) and ax.xaxis_inverted()
    ax.set_xbound(upper=5)
    assert ax.get_xbound() == (1, 5)
    ax.set_ylim(3, 1)
    assert ax.yaxis_inverted() and ax.get_ybound() == (1, 3)
    ax.set_ybound(0, 4)
    assert ax.get_ylim() == (4, 0)


def test_explicit_ticks_read_back_and_automatic_ones_raise(fig_ax):
    _, ax = fig_ax
    with pytest.raises(RuntimeError, match="automatic"):
        ax.get_xticks()
    with pytest.raises(RuntimeError, match="automatic"):
        ax.get_yticklabels()
    assert plt.xticks() is None
    ax.set_xticks([0, 1, 2], ["a", "b", "c"])
    assert list(ax.get_xticks()) == [0, 1, 2]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c"]
    ax.set_yticks([0.5, 1.5])
    assert [t.get_text() for t in ax.get_yticklabels()] == ["0.5", "1.5"]
    ax.set_yticklabels(["lo", "hi"])
    assert [t.get_text() for t in ax.get_yticklabels()] == ["lo", "hi"]
    ax.cla()
    with pytest.raises(RuntimeError):
        ax.get_xticks()


def test_artist_lists_follow_creation_order(fig_ax):
    _, ax = fig_ax
    x = np.arange(4.0)
    l1, = ax.plot(x, x)
    sc = ax.scatter(x, x)
    t = ax.text(1, 1, "hi")
    l2, = ax.plot(x, x + 1)
    ax.imshow(np.eye(3))
    ax.axhline(1)
    assert ax.lines[:2] == (l1, l2) and len(ax.lines) == 3 and ax.get_lines() == list(ax.lines)
    assert ax.collections == (sc,) and ax.texts == (t,) and len(ax.images) == 1
    assert ax.lines[2] in ax.get_children() and t in ax.get_children()
    l1.remove()
    assert ax.lines[0] is l2 and len(ax.lines) == 2


def test_legend_handles_and_legend_object(fig_ax):
    _, ax = fig_ax
    x = np.arange(3.0)
    ax.plot(x, x, label="b")
    ax.plot(x, x, label="_hidden")
    ax.scatter(x, x, label="a")
    ax.plot(x, x)
    handles, labels = ax.get_legend_handles_labels()
    assert labels == ["b", "a"] and handles[0] is ax.lines[0]
    assert ax.get_legend() is None
    ax.legend(frameon=False)
    leg = ax.get_legend()
    assert [t.get_text() for t in leg.get_texts()] == ["b", "a"] and not leg.get_frame_on()


def test_face_colours_and_figure_geometry(fig_ax):
    fig, ax = fig_ax
    assert ax.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    ax.set_facecolor("red")
    assert ax.get_facecolor() == (1.0, 0.0, 0.0, 1.0)
    fig.set_facecolor("blue")
    assert fig.get_facecolor() == (0.0, 0.0, 1.0, 1.0)
    assert (fig.get_figwidth(), fig.get_figheight()) == (4.0, 3.0) and fig.get_dpi() == 100.0
    fig.set_figwidth(5)
    fig.set_figheight(2)
    assert list(fig.get_size_inches()) == [5.0, 2.0]
    with pytest.raises(NotImplementedError):
        fig.set_dpi(50)
    assert ax.get_visible() and fig.get_visible()
