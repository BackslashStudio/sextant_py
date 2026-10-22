"""Axis scales, tick formats, minor ticks and the minor grid (sextant 1.2)."""

import numpy as np
import pytest

import sextant


def axes():
    fig = sextant.Figure(width=300, height=200)
    return fig, fig.axes()


def test_scale_read_back_and_enum():
    _, ax = axes()
    assert ax.xscale() is sextant.Scale.LINEAR
    ax.set_xscale("log").set_yscale(sextant.Scale.SYMLOG, linthresh=0.5, linscale=2, base=2)
    assert ax.xscale() == "log" and ax.yscale() is sextant.Scale.SYMLOG


def test_scale_options_are_checked():
    _, ax = axes()
    with pytest.raises(ValueError, match="base"):
        ax.set_xscale("log", base=1)
    with pytest.raises(ValueError):
        ax.set_yscale("symlog", linthresh=0)
    with pytest.raises(TypeError, match="ScaleOptions fields"):
        ax.set_xscale("log", subs=[2, 5])
    with pytest.raises(TypeError):
        ax.set_xscale("logit")


def test_log_axis_refuses_limits_at_or_below_zero():
    _, ax = axes()
    ax.set_xscale("log")
    with pytest.raises(ValueError):
        ax.set_xlim(0, 10)
    ax.set_xlim(1, 10)
    assert ax.xlim() == (1, 10)


def test_log_and_heatmap_coexist():
    _, ax = axes()
    ax.imshow(np.ones((2, 2)))
    ax.set_yscale("log")
    assert ax.yscale() == "log"
    _, ax = axes()
    ax.set_xscale("symlog")
    ax.heatmap(np.ones((2, 2)), (0, 1), (0, 1))
    assert ax.heatmap_count() == 1


def test_log_scale_changes_the_picture():
    fig, ax = axes()
    ax.line([1, 10, 100, 1000], [1, 2, 3, 4], linewidth=3)
    linear = fig.render_rgba()
    ax.set_xscale("log")
    assert (fig.render_rgba() != linear).any()


def test_cla_resets_the_scale():
    _, ax = axes()
    ax.set_xscale("log").cla()
    assert ax.xscale() == "linear"


def test_tick_format():
    fig, ax = axes()
    ax.line([0, 1], [0, 1])
    before = fig.render_rgba()
    ax.set_xtick_format("%.2f").set_ytick_format("$%.1e$ s")
    assert (fig.render_rgba() != before).any()
    ax.set_xtick_format("").set_ytick_format("")
    assert (fig.render_rgba() == before).all()


@pytest.mark.parametrize("fmt", ["%d", "%.1f %.1f", "no conversion", "%s"])
def test_bad_tick_format_is_a_value_error(fmt):
    with pytest.raises(ValueError):
        axes()[1].set_xtick_format(fmt)


def test_minor_ticks_and_minor_grid():
    fig, ax = axes()
    ax.line([0, 10], [0, 10])
    plain = fig.render_rgba()
    ax.set_axes_style(minor_xticks=True, minor_tick_length=4)
    ticks = fig.render_rgba()
    assert (ticks != plain).any()
    ax.grid(True)
    grid = fig.render_rgba()
    ax.grid(True, minor=True, minor_color="red")
    assert (fig.render_rgba() != grid).any()
    # Unset follows the scale: on for log, off for linear.
    ax.set_axes_style(minor_xticks=None)
    ax.grid(False)
    assert (fig.render_rgba() == plain).all()
