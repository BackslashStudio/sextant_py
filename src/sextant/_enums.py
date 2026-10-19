"""sextant's enums as str-valued Enum classes.

Every call that takes one also takes a plain string: ``linestyle=LineStyle.DASHED``,
``linestyle="dashed"`` and ``linestyle="--"`` are the same. Calls that return
one (``Plane2D.orientation()``, ``Axes3D.camera()["projection"]``, ...) return
a member, which is a ``str`` and compares equal to its value.

The members mirror the C++ name tables in src/ext/casters.h;
tests/test_enums.py keeps the two identical.
"""

import enum

from . import _sextant


def _fold(s):
    return "".join(c for c in s.lower() if c not in "_- ")


class _Named(str, enum.Enum):
    # The value, not "LineStyle.DASHED", wherever a string is expected.
    __str__ = str.__str__
    __format__ = str.__format__

    @classmethod
    def _missing_(cls, value):
        # Any spelling a call would take: "Dashed", "dash-dot", "--".
        if not isinstance(value, str):
            return None
        table = _sextant._enum_tables()[cls.__name__]
        if value in table["aliases"]:
            return cls(table["aliases"][value])
        folded = _fold(value)
        for member in cls:
            if _fold(member.value) == folded:
                return member
        return None


class LineStyle(_Named):
    SOLID = "solid"
    DASHED = "dashed"
    DOTTED = "dotted"
    DASHDOT = "dashdot"
    NONE = "none"


class MarkerStyle(_Named):
    """CROSS and PLUS are filled bars (matplotlib's "X", "P"); THIN_CROSS, THIN_PLUS,
    VLINE and HLINE are strokes in the fill's color, edge_linewidth thick
    (matplotlib's "x", "+", "|", "_")."""

    NONE = "none"
    CIRCLE = "circle"
    SQUARE = "square"
    TRIANGLE = "triangle"
    CROSS = "cross"
    PLUS = "plus"
    DIAMOND = "diamond"
    TRIANGLE_DOWN = "triangle_down"
    TRIANGLE_LEFT = "triangle_left"
    TRIANGLE_RIGHT = "triangle_right"
    STAR = "star"
    PENTAGON = "pentagon"
    HEXAGON = "hexagon"
    THIN_CROSS = "thin_cross"
    THIN_PLUS = "thin_plus"
    VLINE = "vline"
    HLINE = "hline"


class DrawStyle(_Named):
    """How a line runs between its points (LineOptions.drawstyle): straight, or held
    flat in steps; matplotlib's "steps" and "steps-pre" are STEPS_PRE."""

    DEFAULT = "default"
    STEPS_PRE = "steps_pre"
    STEPS_MID = "steps_mid"
    STEPS_POST = "steps_post"


class Orientation(_Named):
    """Which way a bar stands (BarOptions.orientation; HORIZONTAL is matplotlib's barh),
    a reference line or span runs (HORIZONTAL is axhline/axhspan, VERTICAL
    axvline/axvspan), or a fill runs (HORIZONTAL is fill_between, VERTICAL
    fill_betweenx)."""

    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"


class StackBaseline(_Named):
    """Where the bottom layer of a stack() starts (StackOptions.baseline), as
    matplotlib's stackplot baselines: ZERO at 0; SYM centred on 0; WIGGLE and
    WEIGHTED_WIGGLE (a streamgraph) minimising the layers' slopes."""

    ZERO = "zero"
    SYM = "sym"
    WIGGLE = "wiggle"
    WEIGHTED_WIGGLE = "weighted_wiggle"


class QuiverPivot(_Named):
    """Which part of a quiver arrow sits on its (x, y) (QuiverOptions.pivot);
    matplotlib's "mid" is MIDDLE."""

    TAIL = "tail"
    MIDDLE = "middle"
    TIP = "tip"


class Colormap(_Named):
    """The built-in colormaps, by matplotlib's names.

    The set is open: ``cmap=`` also takes any name ``sextant.colormaps()`` lists
    (registered ones included, see ``sextant.register_colormap()``), any of them
    with ``"_r"`` for reversed (``cmap="RdBu_r"``), and a matplotlib Colormap
    object by its ``.name``. Cyclic: TWILIGHT, TWILIGHT_SHIFTED, HSV, PHASE.
    """

    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    CIVIDIS = "cividis"
    TURBO = "turbo"
    COOLWARM = "coolwarm"
    GRAY = "gray"
    BLUES = "Blues"
    GREENS = "Greens"
    REDS = "Reds"
    ORANGES = "Oranges"
    PURPLES = "Purples"
    GREYS = "Greys"
    YLORRD = "YlOrRd"
    YLGNBU = "YlGnBu"
    HOT = "hot"
    JET = "jet"
    RDBU = "RdBu"
    SPECTRAL = "Spectral"
    BRBG = "BrBG"
    PIYG = "PiYG"
    TWILIGHT = "twilight"
    TWILIGHT_SHIFTED = "twilight_shifted"
    HSV = "hsv"
    PHASE = "phase"


class CapStyle(_Named):
    FLAT = "flat"
    ARROW = "arrow"


class AxisPosition(_Named):
    AUTO = "auto"
    LOW = "low"
    MID = "mid"
    HIGH = "high"


class Scale(_Named):
    """An axis' scale (Axes.set_xscale())."""

    LINEAR = "linear"
    LOG = "log"
    SYMLOG = "symlog"


class Adjustable(_Named):
    """What Axes.set_aspect() changes to keep its ratio: the frame or the limits."""

    BOX = "box"
    DATALIM = "datalim"


class LegendAnchor(_Named):
    INSIDE_TL = "inside_tl"
    INSIDE_TR = "inside_tr"
    INSIDE_BL = "inside_bl"
    INSIDE_BR = "inside_br"
    OUTSIDE_TL = "outside_tl"
    OUTSIDE_TR = "outside_tr"
    OUTSIDE_BL = "outside_bl"
    OUTSIDE_BR = "outside_br"
    OUTSIDE_LT = "outside_lt"
    OUTSIDE_LB = "outside_lb"
    OUTSIDE_RT = "outside_rt"
    OUTSIDE_RB = "outside_rb"


class ColorbarAnchor(_Named):
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


class HAlign(_Named):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class VAlign(_Named):
    """A text's block against its anchor; BASELINE is the last line's baseline."""

    TOP = "top"
    CENTER = "center"
    BASELINE = "baseline"
    BOTTOM = "bottom"


class Coords(_Named):
    """What a 2D text coordinate is in: data units, or a fraction of the plot frame."""

    DATA = "data"
    FRACTION = "fraction"


class ArrowHead(_Named):
    NONE = "none"
    OPEN = "open"
    FILLED = "filled"
    BAR = "bar"


class PanelTheme(_Named):
    DARK = "dark"
    LIGHT = "light"
    CLASSIC = "classic"


class MathFontset(_Named):
    TEXT = "text"
    STIX = "stix"
    CM = "cm"
    CUSTOM = "custom"


class Projection(_Named):
    ORTHOGRAPHIC = "orthographic"
    PERSPECTIVE = "perspective"


class PlaneOrientation(_Named):
    XY = "xy"
    YZ = "yz"
    ZX = "zx"


class EventKind(_Named):
    """Figure.connect() kinds; matplotlib's names ('button_press_event', ...) are aliases."""

    CLOSE = "close"
    MOUSE_DOWN = "mouse_down"
    MOUSE_UP = "mouse_up"
    MOUSE_MOVE = "mouse_move"
    SCROLL = "scroll"
    KEY_DOWN = "key_down"
    KEY_UP = "key_up"
    RESIZE = "resize"
    PICK = "pick"


class PickKind(_Named):
    NONE = "none"
    LINE = "line"
    SCATTER = "scatter"
    SCATTER_Z = "scatter_z"
    BAR = "bar"
    HEATMAP = "heatmap"
    BAR3D = "bar3d"
    SURFACE = "surface"
    SURFACE_TRI = "surface_tri"
    SCATTER3D = "scatter3d"
    LINE3D = "line3d"
    QUIVER = "quiver"
    FILL = "fill"


class EventConsumed(_Named):
    NONE = "none"
    SELECT = "select"
    NAVIGATE = "navigate"
    GRID_DRAG = "grid_drag"


__all__ = [
    "Adjustable",
    "ArrowHead",
    "AxisPosition",
    "CapStyle",
    "ColorbarAnchor",
    "Colormap",
    "Coords",
    "DrawStyle",
    "EventConsumed",
    "EventKind",
    "HAlign",
    "PickKind",
    "LegendAnchor",
    "LineStyle",
    "MarkerStyle",
    "MathFontset",
    "Orientation",
    "PanelTheme",
    "PlaneOrientation",
    "Projection",
    "QuiverPivot",
    "Scale",
    "StackBaseline",
    "VAlign",
]
