"""Every option struct's fields reach sextant as keyword arguments.

The field lists are compared with the C++ headers, so a field added to sextant
and not to src/ext/options.h fails here. Then every field is set through the
call that takes it, with a value of its C++ type.
"""

import pathlib
import re

import numpy as np
import pytest

import sextant

HEADERS = pathlib.Path(__file__).parents[1] / "extern" / "sextant" / "include" / "sextant"


def _strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _split_top(s, sep):
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "({<[":
            depth += 1
        elif ch in ")}>]":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


def header_structs():
    """{struct name: [(field, C++ type), ...]} for the data members of every struct."""
    out = {}
    for h in HEADERS.glob("*.h"):
        text = _strip_comments(h.read_text(encoding="utf-8"))
        for m in re.finditer(r"struct\s+(?:SEXTANT_API\s+)?(\w+)\s*\{", text):
            depth, i = 1, m.end()
            while depth:
                depth += {"{": 1, "}": -1}.get(text[i], 0)
                i += 1
            body = text[m.end():i - 1]
            # Inline method bodies -> ";", so a method is one statement.
            body = re.sub(r"\)\s*(const\s*)?(noexcept\s*)?\{(?:[^{}]|\{[^{}]*\})*\}", ");", body)
            fields = []
            for stmt in _split_top(body, ";"):
                stmt = stmt.strip()
                if not stmt or stmt.startswith(("static", "using", "friend")) or "(" in stmt.split("=")[0]:
                    continue
                decls = _split_top(stmt, ",")

                def declarator(d):
                    # "Color color = {...}", "Vec3 target{0, 0, 0}", "bool a = true"
                    return re.sub(r"\{.*\}\s*$", "", d.split("=")[0].strip(), flags=re.S).strip()

                ctype = re.sub(r"\s*\w+$", "", declarator(decls[0]))
                for d in decls:
                    fields.append((re.findall(r"\w+", declarator(d))[-1], ctype))
            out[m.group(1)] = fields
    return out


BINDING = {k: list(v) for k, v in sextant._sextant._option_fields().items()}
HEADER = header_structs()
THREE_D = {"Plane2DOptions", "Bar3DOptions", "SurfaceOptions", "SurfaceTriOptions", "Scatter3DOptions",
           "Line3DOptions", "ErrorBar3DOptions", "Box3DStyle", "Camera3D"}


@pytest.mark.parametrize("struct", sorted(BINDING))
def test_binding_lists_every_header_field(struct):
    assert BINDING[struct] == [name for name, _ in HEADER[struct]]


ENUM_SAMPLE = {
    "LineStyle": "dashed", "MarkerStyle": "square", "Colormap": "plasma", "CapStyle": "arrow",
    "AxisPosition": "mid", "LegendAnchor": "inside_br", "ColorbarAnchor": "top", "HAlign": "left",
    "PanelTheme": "dark", "Projection": "perspective", "VAlign": "top", "ArrowHead": "open",
}
BY_NAME = {  # values a generic one would make invalid
    "width": 64, "height": 48, "supersample": 1, "peel_layers": 2, "dpi": 96, "origin": "upper",
    "font_path": "", "title": "t",
}


def sample(name, ctype):
    if name in BY_NAME:
        return BY_NAME[name]
    t = ctype.replace("sextant::", "")
    if t in ENUM_SAMPLE:
        return ENUM_SAMPLE[t]
    if t == "Color":
        return "red"
    if t == "std::optional<Color>":
        return (0.1, 0.2, 0.3)
    if t == "std::optional<double>":
        return 0.5
    if t == "std::string":
        return "x"
    if t == "std::vector<std::string>":
        return ["a", "b"]
    if t == "std::vector<double>":
        return [0.5]
    if t == "bool":
        return True
    if t in ("float", "double", "int", "std::size_t"):
        return 2
    if t == "Vec3":
        return (0.1, 0.2, 0.3)
    if t in HEADER:  # a nested option struct
        return {}
    raise AssertionError(f"no sample for {name}: {ctype}")


def apply(struct, kwargs):
    """Make the call that takes `struct` as keyword options, then render."""
    x = np.arange(3.0)
    fig = sextant.Figure(width=64, height=48)
    three = struct in THREE_D
    ax = fig.add_subplot3d(1, 1, 1) if three else fig.axes()
    g = np.arange(2.0)
    calls = {
        "Plane2DOptions": lambda: ax.plane("xy", 0.5, **kwargs).line(x, x),
        "Bar3DOptions": lambda: ax.bar3d("xy", g, g, np.ones((2, 2)), **kwargs),
        "SurfaceOptions": lambda: ax.surface("xy", g, g, np.ones((2, 2)), **kwargs),
        "SurfaceTriOptions": lambda: ax.surface_tri(x, [0, 1, 0], x, [0, 1, 2], colors=x, **kwargs),
        "Scatter3DOptions": lambda: ax.scatter3d(x, x, x, colors=x, **kwargs),
        "Line3DOptions": lambda: ax.line3d(x, x, x, **kwargs),
        "ErrorBar3DOptions": lambda: ax.scatter3d(x, x, x, err=sextant.ErrorBar3D(z_cap_lo=x), errorbar=kwargs),
        "Box3DStyle": lambda: ax.set_box_style(**kwargs),
        "Camera3D": lambda: ax.set_camera(**kwargs),
    } if three else {
        "LineOptions": lambda: ax.line(x, x, **kwargs),
        "ScatterOptions": lambda: ax.scatter(x, x, **kwargs),
        "ScatterZOptions": lambda: ax.scatter_z(x, x, x, **kwargs),
        "BarOptions": lambda: ax.bar(x, x, **kwargs),
        "HeatmapOptions": lambda: ax.heatmap(np.ones((2, 2)), (0, 1), (0, 1), **kwargs),
        "GridOptions": lambda: ax.grid(True, **kwargs),
        "AxesStyle": lambda: ax.set_axes_style(**kwargs),
        "LegendOptions": lambda: ax.legend(**kwargs),
        "ColorbarOptions": lambda: ax.set_colorbar_style(**kwargs),
        "SuptitleOptions": lambda: fig.set_suptitle_style(**kwargs),
        "TextOptions": lambda: ax.text("t", 1, 1, **kwargs),
        "ArrowOptions": lambda: ax.annotate(1, 1, "t", 0.2, 0.8, coords="fraction", arrow=kwargs),
        "FigureMargins": lambda: fig.set_margins(**kwargs),
        "PngExportOptions": lambda: fig.render_rgba(**kwargs),
        "SvgExportOptions": lambda: fig.render_svg(**kwargs),
        "ErrorBarOptions": lambda: ax.line(x, x, err=sextant.ErrorBar(y_cap_lo=x), errorbar=kwargs),
        "FigureOptions": lambda: sextant.Figure(**kwargs).render_rgba(),
    }
    calls[struct]()
    fig.render_rgba()


@pytest.mark.parametrize("struct", sorted(BINDING))
def test_every_field_is_accepted(struct):
    for name, ctype in HEADER[struct]:
        apply(struct, {name: sample(name, ctype)})


@pytest.mark.parametrize("struct", sorted(BINDING))
def test_all_fields_at_once(struct):
    apply(struct, {name: sample(name, ctype) for name, ctype in HEADER[struct]})


def test_unknown_keyword_lists_the_fields():
    ax = sextant.Figure().axes()
    with pytest.raises(TypeError, match=r"unexpected keyword argument 'colour'.*color, linewidth"):
        ax.line([0, 1], [0, 1], colour="red")
    with pytest.raises(TypeError, match=r"Figure\(\) got an unexpected keyword argument 'size'"):
        sextant.Figure(size=3)


def test_nested_options_take_a_dict():
    ax = sextant.Figure().axes()
    with pytest.raises(TypeError, match=r"'errorbar' takes a dict of ErrorBarOptions"):
        ax.line([0, 1], [0, 1], errorbar=3)
    with pytest.raises(TypeError, match=r"'errorbar' has no field 'capsze'"):
        ax.line([0, 1], [0, 1], errorbar={"capsze": 3})
    with pytest.raises(TypeError, match=r"bad value for 'errorbar.capsize'"):
        ax.line([0, 1], [0, 1], errorbar={"capsize": "big"})


@pytest.mark.parametrize("value", ["--", "dashed", "Dashed", "DASHED"])
def test_enum_spellings(value):
    sextant.Figure().axes().line([0, 1], [0, 1], linestyle=value)


def test_bad_enum_lists_the_choices():
    with pytest.raises(TypeError, match=r"expected one of 'viridis', 'plasma'"):
        sextant.Figure().axes().scatter_z([0], [0], [0], cmap="jet")


@pytest.mark.parametrize("color", ["red", "grey", "#ff000080", (1, 0, 0), [1.0, 0.0, 0.0, 0.5]])
def test_color_spellings(color):
    sextant.Figure().axes().line([0, 1], [0, 1], color=color)


@pytest.mark.parametrize("color", ["crimson", "#12zz56", "#fff", (1, 0), (1, 0, "x"), 3])
def test_bad_color(color):
    with pytest.raises(TypeError, match=r"expected a color"):
        sextant.Figure().axes().line([0, 1], [0, 1], color=color)
