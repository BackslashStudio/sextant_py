"""C11: colormaps, the current image, rcParams, setp/getp/findobj."""
import io
import warnings

import numpy as np
import pytest

import sextant
import sextant.pyplot as plt
from sextant import _rc


@pytest.fixture(autouse=True)
def fresh_state():
    plt.close("all")
    plt.ioff()
    plt.rcdefaults()
    _rc.rcParams._warned.clear()
    yield
    plt.close("all")
    plt.rcdefaults()


def _svg(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="svg")
    return buf.getvalue()


# --- colormaps ---

def test_colormaps_registry_matches_the_core():
    assert list(plt.colormaps) == list(sextant.colormaps()) and len(plt.colormaps) == len(sextant.colormaps()) >= 26
    assert plt.colormaps["viridis"] == "viridis" and plt.cm.Blues.name == "Blues"
    assert plt.colormaps["blues"] == "Blues"  # the core's spelling
    assert plt.get_cmap("jet_r").reversed() == "jet" and plt.get_cmap("jet").reversed() == "jet_r"
    assert "plasma_r" in plt.colormaps and "nosuch" not in plt.colormaps
    assert plt.get_cmap() == "viridis"
    with pytest.raises(ValueError, match="sextant.colormaps"):
        plt.get_cmap("nosuch")
    with pytest.raises(KeyError):
        plt.colormaps["nosuch"]
    with pytest.raises(NotImplementedError):
        plt.get_cmap("viridis")(0.5)


def test_a_registered_colormap_is_in_the_registry():
    name = sextant.register_colormap("c11_test_map", ["#000000", "#ffffff"])
    assert name in plt.colormaps and plt.get_cmap("C11_TEST_MAP") == name


def test_colormap_handle_goes_to_cmap():
    fig, ax = plt.subplots()
    im = ax.imshow(np.eye(3), cmap=plt.cm.magma)
    assert im._entry[2]["cmap"] == "magma"


# --- the current image ---

def test_gci_sci_clim_set_cmap():
    fig, ax = plt.subplots()
    assert plt.gci() is None
    with pytest.raises(RuntimeError, match="define an image"):
        plt.clim(0, 1)
    a = ax.imshow(np.arange(9.0).reshape(3, 3))
    b = ax.scatter([0, 1], [0, 1], c=[1.0, 2.0])
    assert plt.gci() is b
    plt.sci(a)
    assert plt.gci() is a
    before = _svg(fig)
    plt.clim(0, 4)
    clim = _svg(fig)
    plt.set_cmap("hot")
    assert before != clim != _svg(fig)
    assert a._entry[2]["cmap"] == "hot" and plt.rcParams["image.cmap"] == "hot"
    plt.plasma()
    assert a._entry[2]["cmap"] == "plasma"
    with pytest.raises(AttributeError):
        plt.nosuchfunction


# --- rcParams ---

def test_rc_figure_and_axes_keys():
    plt.rcParams["figure.figsize"] = (3, 2)
    plt.rcParams["figure.dpi"] = 50
    plt.rcParams["figure.facecolor"] = "red"
    plt.rcParams["axes.facecolor"] = "blue"
    fig, ax = plt.subplots()
    assert list(fig.get_size_inches()) == [3.0, 2.0] and fig.get_dpi() == 50.0
    assert fig.get_facecolor() == (1.0, 0.0, 0.0, 1.0) and ax.get_facecolor() == (0.0, 0.0, 1.0, 1.0)
    ax.cla()
    assert ax.get_facecolor() == (0.0, 0.0, 1.0, 1.0)
    plt.rcdefaults()
    fig2, ax2 = plt.subplots()
    assert list(fig2.get_size_inches()) == [6.4, 4.8] and ax2.get_facecolor() == (1.0, 1.0, 1.0, 1.0)


def test_rc_grid_cycle_and_cmap():
    plt.rc("axes", grid=True, prop_cycle={"color": ["#ff0000", "#00ff00"]})
    plt.rcParams["image.cmap"] = "inferno"
    fig, ax = plt.subplots()
    assert ax._grid_on
    l1, = ax.plot([0, 1], [0, 1])
    l2, = ax.plot([0, 1], [1, 0])
    l3, = ax.plot([0, 1], [2, 2])
    assert [l.get_color() for l in (l1, l2, l3)] == [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0)]
    assert ax.imshow(np.eye(2))._entry[2]["cmap"] == "inferno"
    assert ax.imshow(np.eye(2), cmap="gray")._entry[2]["cmap"] == "gray"
    with pytest.raises(ValueError):
        plt.rcParams["axes.prop_cycle"] = []


def test_rc_unhonoured_keys_warn_once_per_key():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        plt.rcParams["font.size"] = 14
        plt.rcParams["font.size"] = 15
        plt.rc("lines", lw=3)
        plt.rcParams["not.a.key"] = 1
        plt.rcParams["figure.dpi"] = 60  # honoured: silent
    texts = [str(x.message) for x in w]
    assert len(texts) == 3 and "font.size" in texts[0] and "lines.linewidth" in texts[1] and "not.a.key" in texts[2]
    assert plt.rcParams["font.size"] == 15 and plt.rcParams["lines.linewidth"] == 3


def test_rc_context_restores_also_after_an_exception():
    plt.rcParams["axes.grid"] = False
    with plt.rc_context({"axes.grid": True}):
        assert plt.rcParams["axes.grid"] is True
    assert plt.rcParams["axes.grid"] is False
    with pytest.raises(RuntimeError):
        with plt.rc_context({"figure.dpi": 33}):
            raise RuntimeError
    assert plt.rcParams["figure.dpi"] == 100.0

    @plt.rc_context({"figure.dpi": 44})
    def make():
        return plt.figure().get_dpi()

    assert make() == 44.0 and plt.rcParams["figure.dpi"] == 100.0
    with pytest.raises(NotImplementedError):
        plt.rc_context(fname="x")


# --- setp / getp / findobj ---

def test_setp_getp_findobj():
    fig, (a, b) = plt.subplots(1, 2)
    l1, = a.plot([0, 1], [0, 1])
    l2, = b.plot([0, 1], [1, 0])
    plt.setp([l1, l2], color="r", linewidth=3)
    assert l1.get_color() == l2.get_color() == (1.0, 0.0, 0.0, 1.0)
    plt.setp(l1, "color", "b")
    plt.setp(l2, {"color": "g"})
    assert plt.getp(l1, "color") == (0.0, 0.0, 1.0, 1.0) and plt.getp(l2, "color") == (0.0, 0.5, 0.0, 1.0)
    assert plt.getp(l1)["color"] == l1.get_color()
    with pytest.raises(AttributeError):
        plt.setp(l1, nosuch=1)
    with pytest.raises(TypeError):
        plt.setp(l1)
    assert plt.findobj(fig, plt.Axes) == [a, b]
    assert plt.findobj(fig, lambda o: o is l2) == [l2]
    assert plt.findobj(fig)[0] is fig and l1 in plt.findobj(a) and plt.findobj(a, include_self=False)[0] is not a
