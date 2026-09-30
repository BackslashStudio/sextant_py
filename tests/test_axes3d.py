"""Axes3D and Plane2D: plotting, planes, camera, read-back and updates."""

import numpy as np
import pytest

import sextant
from sextant import PlaneOrientation, Projection


@pytest.fixture
def fig():
    return sextant.Figure(width=240, height=180)


@pytest.fixture
def ax(fig):
    return fig.add_subplot3d(1, 1, 1)


U, V = np.arange(3.0), np.arange(2.0)
H = np.arange(6.0).reshape(3, 2) + 1  # (len(U), len(V))


# --- plotting ----------------------------------------------------------------

def test_every_kind(ax):
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    (ax.bar3d("xy", U, V, H)
       .surface("yz", U, V, H)
       .surface_tri(x, y, x + y, [[0, 1, 2], [1, 3, 2]])
       .surface_tri(x, y, x * y, orient="xy", colors=x)
       .scatter3d(x, y, x, colors=y)
       .line3d(x, y, x))
    assert (ax.bar3d_count(), ax.surface_count(), ax.surface_tri_count(),
            ax.scatter3d_count(), ax.line3d_count()) == (1, 1, 2, 1, 1)


def test_heights_flat_or_u_by_v(ax):
    ax.bar3d("xy", U, V, H.ravel())
    ax.bar3d("xy", U, V, H)
    np.testing.assert_array_equal(ax.bar3d_data(0).heights, ax.bar3d_data(1).heights)
    with pytest.raises(ValueError, match=r"expected \(len\(u\), len\(v\)\) = \(3, 2\)"):
        ax.bar3d("xy", U, V, H.T)  # right length, wrong layout
    with pytest.raises(ValueError, match=r"bottoms"):
        ax.bar3d("xy", U, V, H, bottoms=np.zeros((2, 3)))


def test_surface_tri_needs_exactly_one_topology(ax):
    x, y, z = [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]
    with pytest.raises(TypeError, match="exactly one"):
        ax.surface_tri(x, y, z)
    with pytest.raises(TypeError, match="exactly one"):
        ax.surface_tri(x, y, z, [0, 1, 2], orient="xy")
    with pytest.raises(TypeError, match="integers"):
        ax.surface_tri(x, y, z, [0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="negative"):
        ax.surface_tri(x, y, z, [0, -1, 2])


@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.int64, np.uint16, np.uint64])
def test_triangle_index_dtypes(ax, dtype):
    ax.surface_tri([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 2.0], np.array([[0, 1, 2]], dtype=dtype))
    tri = ax.surface_tri_data(0).tri
    assert tri.dtype == np.uint32 and tri.shape == (1, 3)


def test_errorbar3d(ax):
    x = np.arange(4.0)
    ax.scatter3d(x, x, x, err=sextant.ErrorBar3D(z_cap_lo=np.full(4, 0.2), x_box_hi=np.ones(4)))
    ax.line3d(x, x, x, colors=x, err=sextant.ErrorBar3D(y_cap_hi=np.ones(4)))
    with pytest.raises(ValueError):
        ax.scatter3d(x, x, x, err=sextant.ErrorBar3D(z_cap_lo=[1.0]))


def test_2d_and_3d_share_a_grid(fig):
    a2 = fig.add_subplot(1, 2, 1)
    a3 = fig.add_subplot3d(1, 2, 2)
    assert isinstance(a2, sextant.Axes) and isinstance(a3, sextant.Axes3D)
    assert fig.add_subplot3d(2) is a3
    with pytest.raises(Exception):
        fig.add_subplot3d(1)  # holds the 2D axes


# --- planes --------------------------------------------------------------------

def test_plane_is_one_object(ax):
    p = ax.plane("zx", 0.5, alpha=0.8)
    assert ax.plane_count() == 1
    assert ax.plane_at(0) is p and ax.plane_at(-1) is p
    assert p.orientation() is PlaneOrientation.ZX and p.orientation() == "zx"
    assert p.offset() == 0.5
    p.set_offset(1.5).set_alpha(0.5)
    assert p.offset() == 1.5
    with pytest.raises(IndexError):
        ax.plane_at(1)


def test_plane_carries_the_2d_kinds(ax):
    p = ax.plane(PlaneOrientation.XY, 0.0)
    x = np.arange(3.0)
    (p.line(x, x).scatter(x, x).scatter_z(x, x, x).bar(x, x + 1)
      .heatmap(np.ones((2, 2)), (0, 1), (0, 1)).imshow(np.eye(2)))
    assert (p.line_count(), p.scatter_count(), p.scatter_z_count(),
            p.bar_count(), p.heatmap_count()) == (1, 1, 1, 1, 2)
    p.set_line_data(0, [0, 1], [5, 6])
    np.testing.assert_array_equal(p.line_data(0).y, [5, 6])
    p.cla()
    assert p.line_count() == 0


def test_plane_keeps_its_figure_alive():
    p = sextant.Figure(width=64, height=48).add_subplot3d(1, 1, 1).plane("xy", 0)
    p.line([0.0, 1.0], [0.0, 1.0])
    assert p.line_count() == 1


# --- camera, box and decoration ---------------------------------------------------

def test_camera_round_trip(ax):
    ax.set_view(10, 20).set_projection("perspective").set_fov(60)
    cam = ax.camera()
    assert cam["azimuth"] == 10 and cam["elevation"] == 20 and cam["fov"] == 60
    assert cam["projection"] is Projection.PERSPECTIVE
    ax.set_camera(azimuth=0, target=(0.1, 0, 0), zoom=2)
    assert ax.camera()["target"] == (0.1, 0.0, 0.0) and ax.camera()["projection"] == "orthographic"
    ax.set_camera(**cam)
    assert ax.camera() == cam
    ax.set_default_camera(elevation=45)
    ax.set_view(0, 200)
    assert ax.camera()["elevation"] == 89  # clamped


def test_decoration_and_limits(ax):
    ax.scatter3d([0, 1], [0, 2], [0, 3])
    r = (ax.set_title("T").set_xtitle("X").set_ytitle("Y").set_ztitle("Z")
           .set_xlim(-1, 1).set_zlim(0, 10).set_xticks([0], ["zero"]).set_zticks([0, 5])
           .grid(False).set_axes_style(label_fontsize=8).set_box_style(panes=False, pane_color="white")
           .set_box_aspect((2, 1, 1)).legend(anchor="inside_tl").set_colorbar_style(anchor="left"))
    assert r is ax
    assert (ax.title(), ax.xtitle(), ax.ytitle(), ax.ztitle()) == ("T", "X", "Y", "Z")
    assert ax.xlim() == (-1.0, 1.0) and ax.zlim() == (0.0, 10.0)
    lo, hi = ax.ylim()
    assert lo <= 0 and hi >= 2


def test_cla(ax):
    ax.bar3d("xy", U, V, H).plane("xy", 0)
    ax.cla()
    assert ax.bar3d_count() == 0


# --- read-back and updates ----------------------------------------------------------

def test_bar3d_round_trip_with_bottoms(ax):
    ax.bar3d("yz", U, V, H, bottoms=-H)
    d = ax.bar3d_data(0)
    assert d.orient is PlaneOrientation.YZ
    np.testing.assert_array_equal(d.heights, H)
    np.testing.assert_array_equal(d.bottoms, -H)
    ax.bar3d("xy", U, V, H)
    assert ax.bar3d_data(1).bottoms is None
    d.heights = d.heights * 2
    ax.set_bar3d_data(0, d)
    np.testing.assert_array_equal(ax.bar3d_data(0).heights, 2 * H)
    ax.set_bar3d_data(0, "xy", [0, 1], [0, 1], [[1, 2], [3, 4]])
    d = ax.bar3d_data(0)
    assert d.orient == "xy" and d.heights.shape == (2, 2) and d.bottoms is None


def test_surface_round_trip(ax):
    ax.surface("xy", U, V, H)
    d = ax.surface_data(0)
    np.testing.assert_array_equal(d.heights, H)
    ax.set_surface_data(0, "zx", U, V, H + 1)
    assert ax.surface_data(0).orient == "zx"
    d = ax.surface_data(0)
    d.heights = np.zeros((3, 2))
    ax.set_surface_data(0, d)
    assert ax.surface_data(0).heights.sum() == 0


def test_surface_tri_round_trip(ax):
    x, y, z = [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 2.0, 3.0]
    ax.surface_tri(x, y, z, orient="xy")
    d = ax.surface_tri_data(0)
    assert d.tri.shape == (2, 3) and d.colors is None
    ax.set_surface_tri_data(0, x, y, z, [0, 1, 2], colors=[1, 2, 3, 4])
    d = ax.surface_tri_data(0)
    assert d.tri.shape == (1, 3)
    np.testing.assert_array_equal(d.colors, [1, 2, 3, 4])
    d.colors = None
    ax.set_surface_tri_data(0, d)
    assert ax.surface_tri_data(0).colors is None


@pytest.mark.parametrize("kind", ["scatter3d", "line3d"])
def test_point_kinds_round_trip(ax, kind):
    x = np.arange(3.0)
    getattr(ax, kind)(x, x, x, colors=x)
    get = getattr(ax, f"{kind}_data")
    put = getattr(ax, f"set_{kind}_data")
    np.testing.assert_array_equal(get(0).colors, x)
    put(0, x, x, x + 1)  # without colors: flat
    assert get(0).colors is None
    d = get(-1)
    d.colors = x * 2
    put(0, d)
    np.testing.assert_array_equal(get(0).colors, x * 2)


def test_optional_parts_take_none(ax):
    x = np.arange(3.0)
    ax.bar3d("xy", U, V, H, bottoms=None).scatter3d(x, x, x, colors=None, err=None)
    ax.set_bar3d_data(0, sextant.Bar3DData("xy", U, V, H, bottoms=None))
    ax.set_scatter3d_data(0, sextant.Scatter3DData(x, x, x, colors=None))
    ax.line3d(x, x, x).set_line3d_data(0, sextant.Line3DData(x, x, x, colors=None))
    ax.surface_tri(x, [0, 1, 0], x, [0, 1, 2], colors=None)
    ax.set_surface_tri_data(0, sextant.SurfaceTriData(x, [0, 1, 0], x, [[0, 1, 2]], colors=None))
    assert ax.bar3d_data(0).bottoms is None and ax.scatter3d_data(0).colors is None


def test_3d_renders(fig, ax):
    before = fig.render_rgba()
    ax.bar3d("xy", U, V, H, color="red")
    after = fig.render_rgba()
    red = ((after[..., 0] > 150) & (after[..., 1] < 90) & (after[..., 2] < 90)).sum()
    assert red > 100 and not np.array_equal(before, after)
