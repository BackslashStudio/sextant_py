"""matplotlib's layout calls (accepted, no effect), figure registry queries, figaspect."""
import io

import numpy as np
import pytest

import sextant.pyplot as plt


@pytest.fixture(autouse=True)
def fresh_state():
    plt.close("all")
    plt.ioff()
    yield
    plt.close("all")


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return buf.getvalue()


def _make(calls):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1, 2], [1, 3, 2])
    ax.set_xlabel("x")
    calls(fig, ax)
    png = _png(fig)
    plt.close(fig)
    return png


def test_layout_calls_change_nothing():
    def calls(fig, ax):
        assert fig.tight_layout(pad=2) is None
        fig.subplots_adjust(left=0.3, wspace=0.5)
        fig.align_labels()
        fig.align_xlabels()
        fig.align_ylabels([ax])
        fig.align_titles()
        fig.set_tight_layout(True)
        fig.set_constrained_layout(True)
        plt.tight_layout()
        plt.subplots_adjust(top=0.8)

    assert _make(calls) == _make(lambda fig, ax: None)


def test_layout_getters():
    fig = plt.figure()
    assert fig.get_tight_layout() is False and fig.get_constrained_layout() is False


def test_sca_picks_the_current_axes_of_the_figure_and_of_pyplot():
    fig, (a, b) = plt.subplots(1, 2)
    assert plt.gca() is b
    fig.sca(a)
    assert fig.gca() is a and plt.gca() is a
    other = plt.figure()
    other.add_subplot(1, 1, 1)
    plt.sca(b)
    assert plt.gcf() is fig and plt.gca() is b
    with pytest.raises(ValueError):
        fig.sca(other.axes[0])


def test_figure_registry_queries():
    plt.figure(3)
    plt.figure("named")
    plt.figure("other")
    assert plt.fignum_exists(3) and plt.fignum_exists("named") and not plt.fignum_exists(4)
    assert plt.get_figlabels() == ["named", "other"]
    plt.close("named")
    assert not plt.fignum_exists("named") and plt.get_figlabels() == ["other"]


@pytest.mark.parametrize("arg, expect", [
    (0.5, (9.6, 4.8)), (2, (4.0, 8.0)), (10, (4.0, 16.0)), (0.05, (16.0, 2.0)),
    (np.zeros((3, 6)), (9.6, 4.8)), (np.zeros((20, 2)), (4.0, 16.0)),
])
def test_figaspect_matches_matplotlib(arg, expect):  # values taken from matplotlib 3.10.6
    assert tuple(plt.figaspect(arg)) == pytest.approx(expect)
    with pytest.raises(ValueError):
        plt.figaspect(0)
