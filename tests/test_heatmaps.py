"""The heatmap family through the core binding -- imshow_rgb,
heatmap_rec, heatmap_tri, hist2d, ColorNorm, and the read-back / setters."""

import numpy as np
import pytest

import sextant


@pytest.fixture
def fig():
    return sextant.Figure(width=160, height=120)


@pytest.fixture
def ax(fig):
    return fig.axes()


def px(fig):
    return fig.render_rgba()


# --- ColorNorm -------------------------------------------------------------

def test_colornorm_factories_and_fields():
    n = sextant.ColorNorm.two_slope(1.5)
    assert n.kind == sextant.NormKind.TWO_SLOPE and n.center == 1.5
    b = sextant.ColorNorm.boundaries([3.0, 1.0, 2.0])
    assert b.kind == "boundaries" and list(b.bounds) == [3.0, 1.0, 2.0]
    assert sextant.ColorNorm.log().kind == "log"
    assert sextant.ColorNorm.linear() == sextant.ColorNorm.linear()
    assert sextant.ColorNorm.linear() != sextant.ColorNorm.log()
    assert sextant.ColorNorm.linear() != 3
    assert repr(sextant.ColorNorm.two_slope(1.5)) == "ColorNorm.two_slope(1.5)"


def test_norm_takes_a_string_or_an_object(ax):
    data = np.arange(1.0, 5.0).reshape(2, 2)
    ax.heatmap(data, (0, 1), (0, 1), norm="log", vmin=1, vmax=4)
    ax.heatmap(data, (0, 1), (0, 1), norm=sextant.ColorNorm.log(), vmin=1, vmax=4)
    ax.heatmap(data, (0, 1), (0, 1), norm="Linear")
    with pytest.raises(TypeError):
        ax.heatmap(data, (0, 1), (0, 1), norm="bogus")
    with pytest.raises(TypeError):
        ax.heatmap(data, (0, 1), (0, 1), norm=3)


def test_norm_changes_the_picture():
    data = np.array([[1.0, 10.0], [100.0, 1000.0]])
    imgs = []
    for norm in ("linear", "log"):
        f = sextant.Figure(width=120, height=100)
        f.axes().heatmap(data, (0, 1), (0, 1), norm=norm, vmin=1, vmax=1000)
        imgs.append(px(f))
    assert not np.array_equal(imgs[0], imgs[1])


def test_norm_that_does_not_fit_raises(ax):
    with pytest.raises(ValueError):
        ax.heatmap(np.ones((2, 2)), (0, 1), (0, 1), norm=sextant.ColorNorm.log(), vmin=-1, vmax=1)


# --- imshow_rgb --------------------------------------------------------------

def test_imshow_rgb_uint8_and_float_agree(ax, fig):
    u8 = np.zeros((2, 2, 3), np.uint8)
    u8[..., 0] = 255
    ax.imshow_rgb(u8)
    f2 = sextant.Figure(width=160, height=120)
    f2.axes().imshow_rgb(u8 / 255.0)
    a, b = px(fig).astype(int), px(f2).astype(int)
    assert np.abs(a - b).max() <= 2
    assert ax.heatmap_count() == 1
    d = ax.heatmap_data(0)
    assert d.data.shape == (2, 2, 3)


def test_imshow_rgba_has_four_channels(ax):
    ax.imshow_rgb(np.ones((3, 2, 4)))
    assert ax.heatmap_data(0).data.shape == (3, 2, 4)


def test_imshow_rgb_requires_3d(ax):
    with pytest.raises(ValueError):
        ax.imshow_rgb(np.ones((2, 2)))
    with pytest.raises(ValueError):
        ax.imshow_rgb(np.ones((2, 2, 2)))  # not 3 or 4 channels


def test_imshow_rgb_is_red(fig, ax):
    ax.imshow_rgb(np.broadcast_to(np.array([1.0, 0.0, 0.0]), (2, 2, 3)))
    img = px(fig)
    red = (img[..., 0] > 200) & (img[..., 1] < 60) & (img[..., 2] < 60)
    assert red.sum() > 500


# --- heatmap_rec -------------------------------------------------------------

def test_heatmap_rec_roundtrip(ax):
    z = np.arange(6.0).reshape(2, 3)
    ax.heatmap_rec(z, [0, 1, 3, 7], [0, 2, 3], name="r")
    d = ax.heatmap_data(0)
    assert np.array_equal(d.data, z)
    assert list(d.x_edges) == [0, 1, 3, 7] and list(d.y_edges) == [0, 2, 3]
    d.data = z * 2
    ax.set_heatmap_data(0, d)
    assert np.array_equal(ax.heatmap_data(0).data, z * 2)


def test_heatmap_rec_checks_edges(ax):
    with pytest.raises(ValueError):
        ax.heatmap_rec(np.ones((2, 2)), [0, 1], [0, 1, 2])
    with pytest.raises(ValueError):
        ax.heatmap_rec(np.ones((2, 2)), [0, 2, 1], [0, 1, 2])


def test_even_heatmap_has_no_edges(ax):
    ax.heatmap(np.ones((2, 2)), (0, 1), (0, 1))
    assert ax.heatmap_data(0).x_edges is None


# --- heatmap_tri -------------------------------------------------------------

X3, Y3, V3 = [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 2.0, 3.0]


def test_heatmap_tri_explicit_and_delaunay(ax):
    ax.heatmap_tri(X3, Y3, V3, [[0, 1, 2], [1, 3, 2]], name="t")
    ax.heatmap_tri(X3, Y3, V3)
    assert ax.heatmap_tri_count() == 2
    d = ax.heatmap_tri_data(0)
    assert d.tri.shape == (2, 3) and d.tri.dtype == np.uint32
    assert ax.heatmap_tri_data(1).tri.shape[1] == 3
    assert list(d.values) == V3


def test_heatmap_tri_flat_and_int64_indices(ax):
    ax.heatmap_tri(X3, Y3, V3, np.array([0, 1, 2], dtype=np.int64))
    assert ax.heatmap_tri_data(0).tri.shape == (1, 3)


def test_heatmap_tri_refusals(ax):
    with pytest.raises(TypeError):
        ax.heatmap_tri(X3, Y3, V3, [[0.5, 1, 2]])
    with pytest.raises(ValueError):
        ax.heatmap_tri(X3, Y3, V3, [[0, 1, -1]])
    with pytest.raises(ValueError):
        ax.heatmap_tri(X3, Y3, V3, [[0, 1, 9]])


def test_heatmap_tri_setters_and_visible(fig, ax):
    ax.heatmap_tri(X3, Y3, V3, [[0, 1, 2], [1, 3, 2]])
    ax.set_heatmap_tri_data(0, X3, Y3, [3.0, 2.0, 1.0, 0.0])
    assert list(ax.heatmap_tri_data(0).values) == [3.0, 2.0, 1.0, 0.0]
    d = ax.heatmap_tri_data(0)
    d.values = np.array(V3)
    ax.set_heatmap_tri_data(0, d)
    assert list(ax.heatmap_tri_data(0).values) == V3
    shown = px(fig).copy()
    ax.set_heatmap_tri_visible(0, False)
    assert ax.heatmap_tri_visible(0) is False
    assert not np.array_equal(shown, px(fig))
    with pytest.raises(IndexError):
        ax.heatmap_tri_data(5)


# --- hist2d ------------------------------------------------------------------

def test_hist2d_counts_into_a_heatmap(ax):
    x = np.array([0.1, 0.2, 0.9, 0.8, 0.85])
    y = np.array([0.1, 0.15, 0.9, 0.8, 0.85])
    ax.hist2d(x, y, xbins=2, ybins=2, xrange=(0, 1), yrange=(0, 1))
    assert ax.heatmap_count() == 1
    d = ax.heatmap_data(0)
    assert d.data.shape == (2, 2) and d.data.sum() == 5
    assert sorted(d.data.ravel()) == [0, 0, 2, 3]


def test_hist2d_edges_make_a_rec(ax):
    ax.hist2d([0.5, 1.5, 2.5], [0.5, 0.5, 0.5], x_edges=[0, 1, 2, 4], ybins=1)
    assert list(ax.heatmap_data(0).x_edges) == [0, 1, 2, 4]


def test_hist2d_density_and_cmin(ax):
    ax.hist2d([0.1, 0.9], [0.1, 0.9], xbins=2, ybins=2, density=True, xrange=(0, 1), yrange=(0, 1))
    assert np.isclose(ax.heatmap_data(0).data.sum() * 0.25, 1.0)


def test_hist2d_is_not_on_planes():
    f = sextant.Figure()
    p = f.add_subplot3d(1, 1, 1).plane("xy", 0.0)
    assert not hasattr(p, "hist2d")
    for name in ("imshow_rgb", "heatmap_rec", "heatmap_tri", "contour", "contourf"):
        assert hasattr(p, name)


def test_heatmaps_on_a_plane_and_log_axes():
    f = sextant.Figure(width=120, height=100)
    ax3 = f.add_subplot3d(1, 1, 1)
    ax3.plane("xy", 0.0).heatmap_rec(np.arange(4.0).reshape(2, 2), [0, 1, 3], [0, 2, 3])
    f.render_rgba()
    f2 = sextant.Figure(width=120, height=100)
    a = f2.axes()
    a.set_xscale("log")
    a.heatmap_rec(np.ones((2, 2)), [1, 10, 100], [0, 1, 2])
    f2.render_rgba()


# --- picking -----------------------------------------------------------------

def test_new_pick_kinds_exist():
    for name in ("HEATMAP_TRI", "CONTOUR", "QUIVER3D"):
        assert hasattr(sextant.PickKind, name)
