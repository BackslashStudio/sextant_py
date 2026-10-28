"""contour / contourf, the grid form of quiver, and the 3D
quiver through the core binding."""

import numpy as np
import pytest

import sextant

X = np.linspace(-1, 1, 5)
Y = np.linspace(-1, 1, 4)
Z = X[None, :] ** 2 + Y[:, None] ** 2  # (len(y), len(x))


@pytest.fixture
def fig():
    return sextant.Figure(width=160, height=120)


@pytest.fixture
def ax(fig):
    return fig.axes()


def test_contour_and_contourf_read_back(ax):
    ax.contour(X, Y, Z, nlevels=3)
    ax.contourf(X, Y, Z, levels=[0.5, 1.0, 1.5])
    assert ax.contour_count() == 2
    d = ax.contour_data(0)
    assert d.z.shape == (4, 5) and not d.filled
    assert np.array_equal(d.z, Z) and np.array_equal(d.x, X) and np.array_equal(d.y, Y)
    assert len(d.levels) >= 2
    f = ax.contour_data(1)
    assert f.filled and list(f.levels) == [0.5, 1.0, 1.5]
    assert ax.heatmap_count() == 0


def test_contour_z_shape_is_checked(ax):
    with pytest.raises(ValueError, match=r"\(len\(y\), len\(x\)\)"):
        ax.contour(X, Y, Z.T)
    with pytest.raises(ValueError, match="len"):
        ax.contourf(X, Y, Z.T)


def test_contour_setters(ax):
    ax.contourf(X, Y, Z, levels=[0.5, 1.0])
    ax.set_contour_data(0, X, Y, Z * 2)
    d = ax.contour_data(0)
    assert d.z.max() == 2 * Z.max() and d.filled and list(d.levels) == [0.5, 1.0]
    d.z = Z
    d.levels = [0.25, 0.75]
    ax.set_contour_data(0, d)
    assert list(ax.contour_data(0).levels) == [0.25, 0.75]
    with pytest.raises(ValueError):
        ax.set_contour_data(0, X, Y, Z.T)
    with pytest.raises(IndexError):
        ax.set_contour_data(3, X, Y, Z)


def test_contour_data_class():
    d = sextant.ContourData(x=X, y=Y, z=Z)
    assert d.filled is False and len(d.levels) == 0
    ax = sextant.Figure().axes()
    ax.contour(X, Y, Z)
    ax.set_contour_data(0, d)


def test_contour_visible_and_pixels(fig, ax):
    empty = fig.render_rgba().copy()
    ax.contourf(X, Y, Z, levels=[0.2, 1.0, 2.0], cmap="viridis")
    drawn = fig.render_rgba().copy()
    assert not np.array_equal(empty, drawn)
    ax.set_contour_visible(0, False)
    assert ax.contour_visible(0) is False
    assert np.array_equal(fig.render_rgba(), empty)


def test_contour_on_a_plane_and_with_norm(ax):
    f = sextant.Figure(width=100, height=80)
    f.add_subplot3d(1, 1, 1).plane("xy", 0.0).contourf(X, Y, Z, levels=[0.5, 1.0])
    f.render_rgba()
    ax.contour(X, Y, Z + 1, levels=[1.5, 2.0, 3.0], norm="log", vmin=1, vmax=4)


def test_contour_cmap_comes_through_the_picture():
    imgs = []
    for cmap in ("viridis", "plasma"):
        f = sextant.Figure(width=120, height=100)
        f.axes().contourf(X, Y, Z, levels=[0.0, 1.0, 2.0], cmap=cmap)
        imgs.append(f.render_rgba())
    assert not np.array_equal(*imgs)


# --- quiver, grid form -------------------------------------------------------

def test_quiver_accepts_a_grid(ax):
    u = np.ones((4, 5))
    ax.quiver(X, Y, u, u, c=Z)
    assert ax.quiver_count() == 1
    assert ax.quiver_data(0).x.shape == (20,)


def test_quiver_flat_form_still_works(ax):
    ax.quiver([0, 1], [0, 1], [1, 1], [0, 1])
    assert ax.quiver_data(0).x.shape == (2,)


# --- quiver3d ----------------------------------------------------------------

@pytest.fixture
def ax3():
    return sextant.Figure(width=120, height=100).add_subplot3d(1, 1, 1)


def test_quiver3d_read_back_and_setters(ax3):
    ax3.quiver3d([0, 1], [0, 1], [0, 1], [1, 1], [0, 1], [0, 1], c=[1, 2], cmap="plasma", name="q")
    assert ax3.quiver3d_count() == 1
    d = ax3.quiver3d_data(0)
    assert list(d.u) == [1, 1] and list(d.c) == [1, 2]
    ax3.set_quiver3d_data(0, [0, 1], [0, 1], [0, 1], [2, 2], [0, 1], [0, 1])
    d = ax3.quiver3d_data(0)
    assert list(d.u) == [2, 2] and list(d.c) == [1, 2]  # c stays
    d.u = [3, 3]
    ax3.set_quiver3d_data(0, d)
    assert list(ax3.quiver3d_data(0).u) == [3, 3]


def test_quiver3d_without_c(ax3):
    ax3.quiver3d([0], [0], [0], [1], [1], [1], color="red", amp=2.0, pivot="middle")
    assert ax3.quiver3d_data(0).c is None


def test_quiver3d_refusals(ax3):
    with pytest.raises(ValueError):
        ax3.quiver3d([0, 1], [0], [0], [1], [1], [1])
    with pytest.raises(ValueError):
        ax3.quiver3d([0], [0], [0], [1], [1], [1], c=[1, 2])
    with pytest.raises(IndexError):
        ax3.quiver3d_data(0)


def test_quiver3d_visible_changes_pixels():
    f = sextant.Figure(width=120, height=100)
    a = f.add_subplot3d(1, 1, 1)
    a.quiver3d([0, 1], [0, 1], [0, 1], [1, 1], [1, 0], [0, 1], color="red")
    shown = f.render_rgba().copy()
    a.set_quiver3d_visible(0, False)
    assert a.quiver3d_visible(0) is False
    assert not np.array_equal(shown, f.render_rgba())


def test_quiver3d_options_are_checked(ax3):
    with pytest.raises(TypeError, match="colour"):
        ax3.quiver3d([0], [0], [0], [1], [1], [1], colour="red")
