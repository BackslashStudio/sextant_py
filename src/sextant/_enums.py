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
    NONE = "none"
    CIRCLE = "circle"
    SQUARE = "square"
    TRIANGLE = "triangle"
    CROSS = "cross"
    PLUS = "plus"
    DIAMOND = "diamond"


class Colormap(_Named):
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    CIVIDIS = "cividis"
    TURBO = "turbo"
    COOLWARM = "coolwarm"
    GRAY = "gray"


class CapStyle(_Named):
    FLAT = "flat"
    ARROW = "arrow"


class AxisPosition(_Named):
    AUTO = "auto"
    LOW = "low"
    MID = "mid"
    HIGH = "high"


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


class EventConsumed(_Named):
    NONE = "none"
    SELECT = "select"
    NAVIGATE = "navigate"
    GRID_DRAG = "grid_drag"


__all__ = [
    "ArrowHead",
    "AxisPosition",
    "CapStyle",
    "ColorbarAnchor",
    "Colormap",
    "Coords",
    "EventConsumed",
    "EventKind",
    "HAlign",
    "PickKind",
    "LegendAnchor",
    "LineStyle",
    "MarkerStyle",
    "PanelTheme",
    "PlaneOrientation",
    "Projection",
    "VAlign",
]
