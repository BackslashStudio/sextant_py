"""The default color cycle and colormaps as values: names, reversed, registered, cyclic (sextant 1.2)."""

import types

import numpy as np
import pytest

import sextant


def near(img, rgb, tol=40):
    d = np.abs(img[..., :3].astype(int) - np.array(rgb))
    return int((d.max(axis=2) <= tol).sum())


def two_lines(setup=None, **opts):
    fig = sextant.Figure(width=200, height=150)
    ax = fig.axes()
    if setup:
        setup(ax)
    ax.line([0, 1], [0, 1], linewidth=6, **opts)
    ax.line([0, 1], [1, 0], linewidth=6, **opts)
    return fig, ax


# --- the color cycle ------------------------------------------------------------------


def test_unset_color_takes_the_cycle():
    fig, ax = two_lines(lambda ax: ax.set_color_cycle(["#ff0000", "#00ff00"]))
    img = fig.render_rgba()
    assert near(img, (255, 0, 0)) > 50 and near(img, (0, 255, 0)) > 50


def test_none_is_unset():
    fig, _ = two_lines(lambda ax: ax.set_color_cycle(["#ff0000", "#00ff00"]), color=None)
    img = fig.render_rgba()
    assert near(img, (255, 0, 0)) > 50 and near(img, (0, 255, 0)) > 50


def test_default_cycle_is_tab10():
    img = two_lines()[0].render_rgba()
    assert near(img, (0x1F, 0x77, 0xB4), 20) > 50 and near(img, (0xFF, 0x7F, 0x0E), 20) > 50


def test_empty_cycle_and_cla_restore_tab10():
    for reset in (lambda ax: ax.set_color_cycle([]), lambda ax: ax.cla()):
        fig = sextant.Figure(width=200, height=150)
        ax = fig.axes().set_color_cycle(["#ff0000"])
        reset(ax)
        ax.line([0, 1], [0, 1], linewidth=6)
        img = fig.render_rgba()
        assert near(img, (255, 0, 0)) == 0 and near(img, (0x1F, 0x77, 0xB4), 20) > 50


def test_explicit_color_does_not_advance_the_cycle():
    fig = sextant.Figure(width=200, height=150)
    ax = fig.axes().set_color_cycle(["#ff0000", "#00ff00"])
    ax.line([0, 1], [0, 1], linewidth=6, color="blue").line([0, 1], [1, 0], linewidth=6)
    assert near(fig.render_rgba(), (255, 0, 0)) > 50


def test_axes3d_color_cycle_reaches_planes():
    fig = sextant.Figure(width=200, height=150)
    a3 = fig.add_subplot3d(1, 1, 1).set_color_cycle(["#ff0000"])
    a3.plane("xy", 0).bar([0.5], [1.0])
    assert near(fig.render_rgba(), (255, 0, 0), 60) > 0


def test_bad_cycle_color_is_a_type_error():
    with pytest.raises(TypeError):
        sextant.Figure().axes().set_color_cycle(["crimson"])


# --- colormaps ----------------------------------------------------------------------------


def heat(cmap, value=0.25, vmin=0.0, vmax=1.0):
    fig = sextant.Figure(width=120, height=90)
    fig.axes().imshow(np.full((1, 1), value), cmap=cmap, vmin=vmin, vmax=vmax)
    img = fig.render_rgba()
    return tuple(img[img.shape[0] // 2, img.shape[1] // 2, :3])


def test_builtins_are_the_enum_members():
    names = sextant.colormaps()
    assert [m.value for m in sextant.Colormap] == names[:26]
    assert sextant.Colormap.RDBU == "RdBu" and sextant.Colormap.TWILIGHT_SHIFTED == "twilight_shifted"


@pytest.mark.parametrize("spelling", ["RdBu", "rdbu", "RDBU", sextant.Colormap.RDBU])
def test_names_in_any_case(spelling):
    assert heat(spelling) == heat("RdBu")


def close(a, b, tol=4):
    # Colormaps go through a lookup table: a mirrored value may land a step off.
    return max(abs(int(x) - int(y)) for x, y in zip(a, b)) <= tol


def test_reversed_maps():
    assert close(heat("viridis_r", 0.25), heat("viridis", 0.75))
    assert heat("RdBu_r", 0.1) != heat("RdBu", 0.1)


def test_grey_is_gray():
    assert heat("grey") == heat("gray") and heat("grey_r") == heat("gray_r")


def test_an_object_with_a_name_is_taken():
    # A matplotlib Colormap, by its .name.
    assert heat(types.SimpleNamespace(name="magma_r")) == heat("magma_r")


def test_unknown_name_is_a_type_error():
    with pytest.raises(TypeError, match=r"expected a colormap name"):
        heat("nope")
    with pytest.raises(TypeError):
        heat(types.SimpleNamespace(name=3))


def test_cyclic_maps_wrap_and_others_clamp():
    assert heat("twilight", 1.25) == heat("twilight", 0.25)
    assert heat("viridis", 1.25) == heat("viridis", 1.0) != heat("viridis", 0.25)


def test_register_colormap():
    assert sextant.register_colormap("test_redblue", ["red", (0.0, 0.0, 1.0)]) == "test_redblue"
    assert "test_redblue" in sextant.colormaps()
    lo, hi = heat("test_redblue", 0.0), heat("test_redblue", 1.0)
    assert lo[0] > 200 and lo[2] < 50 and hi[2] > 200 and hi[0] < 50
    assert close(heat("Test_RedBlue_r", 0.0), hi)
    with pytest.raises(ValueError):
        sextant.register_colormap("test_redblue", ["red", "blue"])  # taken
    with pytest.raises(ValueError):
        sextant.register_colormap("VIRIDIS", ["red", "blue"])  # a built-in, any case
    with pytest.raises(ValueError):
        sextant.register_colormap("test_one", ["red"])
    with pytest.raises(ValueError):
        sextant.register_colormap("test_x_r", ["red", "blue"])


def test_registered_cyclic_map_wraps():
    sextant.register_colormap("test_cyc", ["red", "blue", "red"], cyclic=True)
    assert heat("test_cyc", 1.25) == heat("test_cyc", 0.25)
