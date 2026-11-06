"""Style files: Figure.style_yaml / apply_style_yaml / save_style / load_style.
Headless; the SVG is compared exactly (pixels differ across renderers)."""

import numpy as np
import pytest

import sextant

X = np.linspace(0.0, 4.0, 9)


def styled(y):
    fig = sextant.Figure(width=320, height=240)
    ax = fig.add_subplot(1, 1, 1)
    ax.line(X, y, color="#c0392b", linewidth=3.5, name="a")
    ax.scatter(X, y * 0.5, color="#2980b9")
    ax.grid(True)
    ax.legend()
    ax.set_yscale("log") if y.min() > 0 else None
    return fig, ax


def plain(y):
    fig = sextant.Figure(width=320, height=240)
    ax = fig.add_subplot(1, 1, 1)
    ax.line(X, y)
    ax.scatter(X, y * 0.5)
    return fig, ax


def svg(fig):
    return fig.render_svg()[0]


def test_style_yaml_header_and_detail():
    fig, ax = styled(X + 1.0)
    ax.set_title("Hello")
    ax.set_xlim(0.5, 3.5)
    style = fig.style_yaml()
    assert style.startswith("sextant_style: 1")
    assert "Hello" not in style
    state = fig.style_yaml("state")
    assert "Hello" in state and "3.5" in state
    assert fig.style_yaml("STATE") == state


def test_copy_the_look():
    a, _ = styled(X + 1.0)
    y2 = (X + 2.0) ** 1.5
    b, _ = plain(y2)
    assert svg(b) != svg(styled(y2)[0])
    assert b.apply_style_yaml(a.style_yaml()) == []
    assert svg(b) == svg(styled(y2)[0])


def test_state_round_trip_through_a_file(tmp_path):
    a, ax = styled(X + 1.0)
    ax.set_title("T")
    ax.set_xtitle("xx")
    ax.set_xlim(0.5, 3.5)
    ax.set_xticks([1.0, 2.0, 3.0], ["one", "two", "three"])
    path = tmp_path / "s.yaml"
    a.save_style(path, "state")
    b, bx = styled(X + 1.0)
    # same data, different titles/limits
    bx.set_title("other")
    msgs = b.load_style(str(path))
    assert msgs == []
    assert bx.title() == "T" and bx.xlim() == (0.5, 3.5)
    assert svg(a) == svg(b)


def test_other_shape_skips_with_messages():
    fig = sextant.Figure(width=320, height=240)
    a1 = fig.add_subplot(1, 2, 1)
    a1.line(X, X, color="#c0392b", linewidth=4.0)
    a2 = fig.add_subplot(1, 2, 2)
    a2.line(X, X)
    text = fig.style_yaml()
    one, ax = plain(X + 1.0)
    msgs = one.apply_style_yaml(text)
    assert msgs and all(isinstance(m, str) for m in msgs)
    assert any("1x2" in m for m in msgs)
    assert ax.line_options(0)["linewidth"] == 4.0   # the matching cell applied


def test_3d_and_polar():
    def make3d(az):
        fig = sextant.Figure(width=320, height=240)
        ax = fig.add_subplot3d(1, 1, 1)
        ax.set_camera(azimuth=az)
        return fig

    a, b = make3d(33.0), make3d(0.0)
    assert b.apply_style_yaml(a.style_yaml()) == []
    assert svg(a) == svg(b)

    def makepolar(off):
        fig = sextant.Figure(width=320, height=240)
        ax = fig.add_subplot_polar(1, 1, 1)
        ax.set_theta_offset(off)
        return fig

    pa, pb = makepolar(40.0), makepolar(0.0)
    assert pb.apply_style_yaml(pa.style_yaml()) == []
    assert svg(pa) == svg(pb)


def test_errors(tmp_path):
    fig, _ = plain(X)
    with pytest.raises(ValueError):
        fig.apply_style_yaml("a: [unclosed")
    with pytest.raises(ValueError):
        fig.apply_style_yaml("figure: {}\n")
    with pytest.raises(ValueError):
        fig.apply_style_yaml("sextant_style: 99\n")
    with pytest.raises(ValueError, match="detail"):
        fig.style_yaml("loud")
    with pytest.raises(FileNotFoundError):
        fig.load_style(tmp_path / "missing.yaml")
    with pytest.raises(OSError):
        fig.save_style(tmp_path / "no_dir" / "s.yaml")
    with pytest.raises(TypeError):
        fig.load_style(3)
    with pytest.raises(TypeError):
        fig.apply_style_yaml(b"sextant_style: 1")


def test_bad_values_are_messages_not_errors():
    fig, ax = plain(X)
    text = fig.style_yaml().replace("sextant_style: 1", "sextant_style: 1\nfigure:\n  background: nonsense")
    msgs = fig.apply_style_yaml(text)
    assert msgs and any("background" in m for m in msgs)
