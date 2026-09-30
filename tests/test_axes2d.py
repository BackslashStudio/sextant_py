"""The 2D Axes surface: plotting, read-back, data updates, subplots and layout."""

import numpy as np
import pytest

import sextant


@pytest.fixture
def ax():
    return sextant.Figure(width=320, height=240).axes()


def red_pixels(img):
    r, g, b = (img[..., i].astype(int) for i in range(3))
    return int(((r > 180) & (g < 80) & (b < 80)).sum())


# --- plotting ----------------------------------------------------------------

def test_every_kind_takes_lists(ax):
    ax.line([0, 1, 2], [1, 0, 1]).line([3, 1, 2])
    ax.scatter([0, 1], [1, 2]).scatter_z([0, 1], [1, 2], [0.2, 0.8])
    ax.bar([1, 2, 3], [4, 5, 6]).hist([1, 2, 2, 3, 3, 3], bins=3)
    ax.heatmap([[0, 1], [1, 0]], (0, 1), (0, 1)).imshow([[0.5]])
    assert (ax.line_count(), ax.scatter_count(), ax.scatter_z_count(),
            ax.bar_count(), ax.heatmap_count()) == (2, 1, 1, 2, 2)


def test_line_of_y_alone_uses_indices(ax):
    ax.line([5.0, 6.0, 7.0])
    np.testing.assert_array_equal(ax.line_data(0).x, [0, 1, 2])


def test_errorbar_arrays_are_checked(ax):
    ax.scatter([0, 1, 2], [0, 1, 2], err=sextant.ErrorBar(y_cap_lo=[0.1, 0.2, 0.3], x_box_hi=[1, 1, 1]))
    with pytest.raises(ValueError):
        ax.line([0, 1, 2], [0, 1, 2], err=sextant.ErrorBar(y_cap_lo=[0.1]))


def test_hist_density_and_bins(ax):
    data = np.random.default_rng(0).normal(size=500)
    ax.hist(data, bins=20, density=True)
    d = ax.bar_data(0)
    assert len(d.x) == 20
    width = d.x[1] - d.x[0]
    assert d.height.sum() * width == pytest.approx(1.0)


def test_heatmap_needs_2d(ax):
    with pytest.raises(TypeError):
        ax.heatmap([1.0, 2.0], (0, 1), (0, 1))
    with pytest.raises(ValueError):
        ax.heatmap([[1.0]], (0, 1), (0, 1), origin="middle")


def test_mismatched_lengths_raise(ax):
    with pytest.raises(ValueError):
        ax.bar([1, 2], [1, 2, 3])


# --- options reach the picture --------------------------------------------------

def test_color_reaches_the_pixels():
    def render(**kw):
        fig = sextant.Figure(width=200, height=150)
        fig.axes().line([0, 1, 2], [0, 1, 0], linewidth=4, **kw)
        return fig.render_rgba()

    assert red_pixels(render()) == 0
    assert red_pixels(render(color="red")) > 100
    assert red_pixels(render(color="red", linestyle="none")) == 0
    assert red_pixels(render(color=(1, 0, 0))) == red_pixels(render(color="#ff0000"))


def test_render_rgba_size_and_dpi():
    fig = sextant.Figure(width=120, height=80)
    img = fig.render_rgba()
    assert img.shape == (80, 120, 4) and img.dtype == np.uint8
    assert fig.render_rgba(dpi=192).shape == (160, 240, 4)
    assert fig.render_rgba(width=60, height=40).shape == (40, 60, 4)
    assert sextant.Figure(width=120, height=80, dpi=192).render_rgba().shape == (160, 240, 4)


# --- read-back ------------------------------------------------------------------

def test_titles_and_limits(ax):
    ax.line([0, 10], [0, 5]).set_title("T").set_xtitle("X").set_ytitle("Y")
    assert (ax.title(), ax.xtitle(), ax.ytitle()) == ("T", "X", "Y")
    lo, hi = ax.xlim()
    assert lo <= 0 and hi >= 10
    ax.set_xlim(2, 3).set_ylim(-1, 1)
    assert ax.xlim() == (2.0, 3.0) and ax.ylim() == (-1.0, 1.0)


def test_data_round_trip(ax):
    x = np.linspace(0, 1, 5)
    ax.line(x, x ** 2).scatter(x, -x).scatter_z(x, x, 2 * x).bar(x, x + 1)
    np.testing.assert_array_equal(ax.line_data(0).y, x ** 2)
    np.testing.assert_array_equal(ax.scatter_data(0).y, -x)
    np.testing.assert_array_equal(ax.scatter_z_data(0).z, 2 * x)
    np.testing.assert_array_equal(ax.bar_data(0).height, x + 1)


def test_heatmap_round_trip_is_float32(ax):
    data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    ax.heatmap(data, (0, 3), (1, 2)).imshow(data)
    d = ax.heatmap_data(0)
    assert d.data.shape == (2, 3)
    np.testing.assert_array_equal(d.data, data.astype(np.float32))
    assert d.xrange == (0.0, 3.0) and d.yrange == (1.0, 2.0)
    assert ax.heatmap_data(1).xrange == (0.0, 3.0) and ax.heatmap_data(1).yrange == (0.0, 2.0)


def test_negative_and_out_of_range_indices(ax):
    ax.line([1.0, 2.0]).line([3.0, 4.0])
    np.testing.assert_array_equal(ax.line_data(-1).y, [3, 4])
    with pytest.raises(IndexError):
        ax.line_data(2)
    with pytest.raises(IndexError):
        ax.line_data(-3)
    with pytest.raises(IndexError):
        ax.set_line_data(5, [0.0], [0.0])


# --- updating plotted data --------------------------------------------------

def test_set_data_takes_the_read_back_object(ax):
    ax.line([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    d = ax.line_data(0)
    d.y *= 2
    ax.set_line_data(0, d)
    np.testing.assert_array_equal(ax.line_data(0).y, [0, 2, 8])


def test_set_data_takes_arrays_and_changes_length(ax):
    ax.scatter([0.0], [0.0]).bar([1.0, 2.0], [1.0, 1.0])
    ax.set_scatter_data(0, [1, 2, 3], [4, 5, 6]).set_bar_data(-1, [1, 2, 3], [3, 2, 1])
    np.testing.assert_array_equal(ax.scatter_data(0).x, [1, 2, 3])
    np.testing.assert_array_equal(ax.bar_data(0).height, [3, 2, 1])


def test_set_heatmap_data_both_ways(ax):
    ax.imshow(np.zeros((2, 2)))
    ax.set_heatmap_data(0, np.ones((3, 4)), (0, 4), (0, 3))
    assert ax.heatmap_data(0).data.shape == (3, 4)
    d = ax.heatmap_data(0)
    d.data = np.full((1, 2), 0.5)
    d.xrange = (-1, 1)
    ax.set_heatmap_data(0, d)
    assert ax.heatmap_data(0).xrange == (-1.0, 1.0)
    assert ax.heatmap_data(0).data.shape == (1, 2)


def test_bad_update_changes_nothing(ax):
    ax.line([0.0, 1.0], [0.0, 1.0])
    with pytest.raises(ValueError):
        ax.set_line_data(0, [0.0, 1.0], [0.0])
    np.testing.assert_array_equal(ax.line_data(0).y, [0, 1])
    with pytest.raises(TypeError):
        ax.set_line_data(0, sextant.LineData(x="nope", y=[1.0]))


def test_cla_resets(ax):
    ax.line([0.0, 1.0]).bar([1.0], [1.0]).set_xlim(5, 6)
    ax.cla()
    assert ax.line_count() == 0 and ax.bar_count() == 0


# --- decoration -----------------------------------------------------------------

def test_decoration_calls_chain(ax):
    r = (ax.line([0, 1], [0, 1], name="a")
         .grid(linestyle=":", color="gray")
         .legend(anchor="inside_tl", frameon=False)
         .set_axes_style(spine_top=False, origin_x=0.5, label_fontsize=9)
         .set_colorbar_style(anchor="left")
         .set_xticks([0, 0.5, 1], ["lo", "mid", "hi"])
         .set_yticks([0, 1]))
    assert r is ax


# --- subplots and layout -----------------------------------------------------------

def test_subplots_are_the_same_objects():
    fig = sextant.Figure()
    a = fig.add_subplot(2, 2, 1)
    assert fig.add_subplot(1) is a
    wide = fig.add_subplot(2, 2, (3, 4))
    assert fig.add_subplot((3, 4)) is wide
    with pytest.raises(Exception):
        fig.add_subplot(3, 3, 1)  # the grid is fixed


def test_ratios_margins_and_sizes():
    fig = sextant.Figure(width=400, height=300)
    fig.add_subplot(1, 2, 1)
    assert fig.col_ratios() == []
    fig.set_col_ratios([2, 1])
    assert fig.col_ratios() == [2.0, 1.0]
    with pytest.raises(ValueError):
        fig.set_col_ratios([1, 2, 3])
    fig.set_margins(left=30, top=5)
    w, h = fig.size_for_frame(200, 100)
    assert w > 200 and h > 100
    fig.resize_to_frame(200, 100)
    assert fig.render_rgba().shape[:2] == (h, w)
    fig.resize(300, 200)
    assert fig.render_rgba().shape[:2] == (200, 300)


def test_suptitle_changes_the_picture():
    fig = sextant.Figure(width=200, height=150)
    before = fig.render_rgba()
    fig.suptitle("Hello", fontsize=30)
    fig.set_suptitle_style(fontsize=30, color="red", align="left")
    assert red_pixels(fig.render_rgba()) > 0 and red_pixels(before) == 0


def test_frame_stats_before_show():
    s = sextant.Figure().frame_stats()
    assert s.frames == 0 and s.max_ms == 0.0
