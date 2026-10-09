"""Figure and plot-area backgrounds, hidden ticks, and marker outlines (sextant 1.1)."""

import numpy as np
import pytest

import sextant


def near(img, rgb, tol):
    """How many pixels of an RGBA image are within `tol` of an RGB."""
    d = np.abs(img[..., :3].astype(int) - np.array(rgb))
    return int((d.max(axis=2) <= tol).sum())


def one_marker(**opts):
    fig = sextant.Figure(width=400, height=300)
    ax = fig.axes()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.scatter([5], [5], **{"size": 40, "color": "#0000ff", "alpha": 1.0, **opts})
    return fig


# --- backgrounds --------------------------------------------------------------


def test_figure_background_default_option_and_setter():
    fig = sextant.Figure(width=200, height=150)
    fig.axes().line([0, 1], [0, 1])
    assert tuple(fig.render_rgba()[1, 1]) == (237, 237, 237, 255)

    blue = sextant.Figure(width=200, height=150, background="#0000ff")
    blue.axes().line([0, 1], [0, 1])
    assert tuple(blue.render_rgba()[1, 1]) == (0, 0, 255, 255)

    fig.set_background((0.0, 1.0, 0.0))
    assert tuple(fig.render_rgba()[1, 1]) == (0, 255, 0, 255)
    fig.set_background("red")
    assert fig.render_rgba()[1, 1, 3] == 255


def test_background_alpha_zero_is_transparent_in_png_and_svg():
    fig = sextant.Figure(width=200, height=150, background=(0, 0, 0, 0))
    fig.axes().line([0, 1], [0, 1])
    assert fig.render_rgba()[1, 1, 3] == 0
    assert "<rect width=" not in fig.render_svg()[0]
    assert "<rect width=" in sextant.Figure().render_svg()[0]


def test_bad_background_is_a_type_error():
    with pytest.raises(TypeError, match="expected a color"):
        sextant.Figure(background="crimson")
    with pytest.raises(TypeError):
        sextant.Figure().set_background(3)


def test_plot_area_background():
    fig = sextant.Figure(width=300, height=200)
    ax = fig.axes()
    ax.line([0, 1], [0, 1])
    white = near(fig.render_rgba(), (255, 255, 255), 0)
    assert white > 5000
    ax.set_axes_style(background=(1, 0, 0))
    img = fig.render_rgba()
    assert near(img, (255, 0, 0), 0) > 5000 and near(img, (255, 255, 255), 0) < 100
    assert 'fill="#ff0000"' in fig.render_svg()[0]


# --- tick visibility ----------------------------------------------------------


def n_text(svg):
    return svg.count("<text")


def test_hidden_ticks_per_axis_keep_the_grid():
    def svg(**style):
        fig = sextant.Figure(width=300, height=200)
        ax = fig.axes()
        ax.line([0, 1, 2], [0, 3, 1])
        ax.grid(True)
        ax.set_axes_style(**style)
        return fig.render_svg()[0]

    all_ticks = svg()
    no_x, no_y = svg(show_xticks=False), svg(show_yticks=False)
    none = svg(show_xticks=False, show_yticks=False)
    assert n_text(no_x) + n_text(no_y) == n_text(all_ticks) > 0
    assert n_text(none) == 0
    # the grid lines are still there
    assert none.count("<line") == all_ticks.count("<line") - no_x.count("<line") - no_y.count("<line") + none.count("<line") \
        or none.count("<line") > 0
    assert none.count("<line") > 0


def test_set_xticks_empty_still_means_automatic():
    fig = sextant.Figure(width=300, height=200)
    ax = fig.axes()
    ax.line([0, 1, 2], [0, 3, 1])
    before = n_text(fig.render_svg()[0])
    ax.set_xticks([])
    assert n_text(fig.render_svg()[0]) == before


def test_hidden_ticks_3d():
    def labels(**style):
        fig = sextant.Figure(width=300, height=200)
        ax = fig.add_subplot3d(1, 1, 1)
        g = np.arange(3.0)
        ax.surface("xy", g, np.arange(2.0), np.arange(6.0).reshape(3, 2))
        ax.set_axes_style(**style)
        return n_text(fig.render_svg()[0])

    full = labels()
    assert full > 0
    assert 0 < labels(show_zticks=False) < full
    assert labels(show_xticks=False, show_yticks=False, show_zticks=False) == 0


# --- marker outlines ----------------------------------------------------------


def test_hollow_marker_leaves_only_its_outline():
    filled = near(one_marker().render_rgba(), (0, 0, 255), 8)
    assert 1100 < filled < 1400
    ring = near(one_marker(alpha=0.0, edge_linewidth=3.0).render_rgba(), (0, 0, 255), 60)
    assert 220 < ring < 480 and ring < filled / 2


def test_outline_in_its_own_color_lies_inside_the_marker():
    img = one_marker(edgecolor="#ff0000", edge_linewidth=4.0).render_rgba()
    red, blue = near(img, (255, 0, 0), 40), near(img, (0, 0, 255), 8)
    assert 300 < red < 600 and 500 < blue < 900
    filled = near(one_marker().render_rgba(), (0, 0, 255), 8)
    assert red + blue < filled + 100


def test_edge_alpha_is_independent_of_the_fill_alpha():
    img = one_marker(alpha=0.0, edgecolor="#ff0000", edge_alpha=0.5, edge_linewidth=4.0).render_rgba()
    assert near(img, (255, 128, 128), 25) > 250


@pytest.mark.parametrize("marker", ["circle", "square", "triangle", "cross", "plus", "diamond"])
def test_every_shape_takes_an_outline(marker):
    full = near(one_marker(marker=marker).render_rgba(), (0, 0, 255), 60)
    ring = near(one_marker(marker=marker, alpha=0.0, edge_linewidth=2.0).render_rgba(), (0, 0, 255), 60)
    assert full > 100 and 40 < ring < full


def test_scatter_z_and_scatter3d_outlines():
    fig = sextant.Figure(width=400, height=300)
    ax = fig.axes()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.scatter_z([5], [5], [0.5], size=40, alpha=0.0, edgecolor="#ff0000", edge_linewidth=3.0)
    ring = near(fig.render_rgba(), (255, 0, 0), 60)
    assert 220 < ring < 480
    assert 'stroke="rgb(255,0,0)"' in fig.render_svg()[0]

    fig3 = sextant.Figure()
    ax3 = fig3.add_subplot3d(1, 1, 1)
    x = np.array([0.0, 1.0, 2.0])
    ax3.scatter3d(x, [0, 1, 0.5], x, size=30, alpha=0.0, edgecolor="#ff0000", edge_linewidth=3.0)
    img = fig3.render_rgba()
    assert near(img, (255, 0, 0), 60) > 150 and near(img, (0, 0, 255), 60) == 0
    assert fig3.render_svg()[0].count('stroke="rgb(255,0,0)"') >= 3


def test_svg_outline_is_inset_and_legend_follows():
    fig = one_marker(alpha=0.0, edgecolor="#ff0000", edge_alpha=0.5, edge_linewidth=3.0, name="pts")
    fig.axes().legend()
    svg = fig.render_svg()[0]
    assert 'fill-opacity="0"' in svg and 'stroke-opacity="0.5"' in svg and 'stroke-width="3"' in svg
    assert 'r="18.5"' in svg
    assert svg.count('stroke="rgb(255,0,0)"') >= 2   # the marker and its legend key


def test_bad_outline_values_are_type_errors():
    ax = sextant.Figure().axes()
    with pytest.raises(TypeError, match="edge_linewidth"):
        ax.scatter([0], [0], edge_linewidth="thick")
    with pytest.raises(TypeError, match="expected a color"):
        ax.scatter([0], [0], edgecolor="crimson")
